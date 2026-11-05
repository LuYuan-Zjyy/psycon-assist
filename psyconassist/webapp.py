"""PsyConAssist Web 服务（FastAPI）。

提供：数据 API + 反馈提交 + 视频流 + 静态前端。
启动：python -m psyconassist.webapp
"""
from __future__ import annotations

import datetime
import re
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, ids, pipeline
from .core.errors import StageError
from .core.io import read_json_optional, write_json
from .core.logging_utils import get_logger
from .core.paths import SessionPaths, SubjectPaths
from .stages import s5_feedback, s6_profile

log = get_logger(__name__)
app = FastAPI(title="PsyConAssist", version="0.1.0")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """静态资源禁用缓存，避免前端更新后浏览器仍用旧 HTML/JS。"""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


WEB_DIR = Path(__file__).resolve().parent / "web"

# 启动时确保目录存在
config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
config.SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
config.DATA_ROOT.joinpath("uploads").mkdir(parents=True, exist_ok=True)

# 后台分析任务注册表：task_id -> {status, stage, detail, session_id, error, summary}
TASKS: dict[str, dict] = {}
_TASKS_LOCK = threading.Lock()

# 允许的上传扩展名（保存为 uploads/<task_id><ext>，扩展名必须来自白名单）
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".ts", ".mpeg", ".mpg"}


def _safe_session(session_id: str) -> SessionPaths:
    """校验 session_id 后再构造路径，避免 `../` 穿越到 data/ 之外。"""
    try:
        config.session_dir(session_id)
    except ValueError:
        raise HTTPException(404, "session 不存在")
    return SessionPaths(session_id)


def _safe_subject(subject_id: str) -> SubjectPaths:
    try:
        config.subject_dir(subject_id)
    except ValueError:
        raise HTTPException(400, "subject_id 只能包含字母、数字、下划线和连字符（如 P-0001）")
    return SubjectPaths(subject_id)



# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def _list_sessions() -> list[str]:
    out = []
    for d in config.DATA_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("S"):
            out.append(d.name)
    return sorted(out)


def _list_subjects() -> list[str]:
    if not config.SUBJECTS_DIR.exists():
        return []
    return sorted(d.name for d in config.SUBJECTS_DIR.iterdir() if d.is_dir() and d.name.startswith("P"))


def _session_summary(session_id: str) -> dict:
    p = SessionPaths(session_id)
    media = read_json_optional(p.media_json, default={}) or {}
    events = read_json_optional(p.events_json, default=[]) or []
    claims = read_json_optional(p.claims_json, default=[]) or []
    fb = read_json_optional(p.feedback_json, default=[]) or []
    chunks = read_json_optional(p.chunks_json, default=[]) or []
    return {
        "session_id": session_id,
        "duration": media.get("duration"),
        "has_video": bool(media.get("normalized_path") and Path(media["normalized_path"]).exists()),
        "n_chunks": len(chunks),
        "n_events": len(events),
        "n_claims": len(claims),
        "n_feedback": len(fb),
    }


def _next_session_id() -> str:
    """按日期 + 现有会话数生成下一个 session id。"""
    existing = _list_sessions()
    today = datetime.datetime.now().strftime("%Y%m%d")
    seq = 0
    for s in existing:
        if s.startswith(f"S{today}"):
            try:
                seq = max(seq, int(s.split("-")[-1]))
            except (ValueError, IndexError):
                pass
    return ids.session_id(seq + 1, datetime.datetime.now())


_PATH_WIN = re.compile(r"[A-Za-z]:\\[^\s'\"]+")
_PATH_POSIX = re.compile(r"(?<![\w.])/(?:[\w.\-]+/)+[\w.\-]+")


def _sanitize_error(exc: Exception) -> str:
    """把异常变成可读的一行原因，同时抹掉绝对路径，避免泄漏目录结构。"""
    lines = (str(exc) or "").strip().splitlines()
    text = lines[0] if lines else ""
    text = _PATH_WIN.sub("<path>", text)
    text = _PATH_POSIX.sub("<path>", text)
    text = text[:200].strip()
    return text or type(exc).__name__


def _run_pipeline_task(task_id: str, video_path: Path, subject_id: str,
                       client_sample: list[float]):
    def progress(stage: str, detail: str):
        with _TASKS_LOCK:
            t = TASKS.get(task_id)
            if t:
                t["stage"] = stage
                t["detail"] = detail

    try:
        session_id = _next_session_id()
        with _TASKS_LOCK:
            TASKS[task_id]["session_id"] = session_id
            TASKS[task_id]["status"] = "running"
        summary = pipeline.run_full(video_path, client_sample, subject_id,
                                    session_id=session_id, progress=progress)
        with _TASKS_LOCK:
            TASKS[task_id]["status"] = "done"
            TASKS[task_id]["summary"] = summary
        log.info("task %s done session=%s", task_id, session_id)
    except Exception as e:  # noqa: BLE001
        log.exception("task %s failed", task_id)
        with _TASKS_LOCK:
            TASKS[task_id]["status"] = "failed"
            # 回传可读原因，但抹掉绝对路径；完整堆栈只进日志
            TASKS[task_id]["error"] = _sanitize_error(e)
            TASKS[task_id]["error_type"] = type(e).__name__


@app.post("/api/upload")
async def upload_video(
    video: UploadFile = File(...),
    subject_id: str = Form(...),
    client_t0: float = Form(...),
    client_t1: float = Form(...),
):
    """上传视频并后台跑全流程。client_t0/client_t1 = 来访者声音样本时间段（秒）。"""
    if client_t1 <= client_t0:
        raise HTTPException(400, "声音样本时间段无效：client_t1 必须大于 client_t0")

    # 路径与格式校验（subject_id 会被拼进落盘路径）
    _safe_subject(subject_id)
    ext = Path(video.filename or "").suffix.lower() or ".mp4"
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"不支持的视频格式：{ext}")

    # 保存上传文件（带体积上限，超限立即中止并删除半成品）
    task_id = uuid.uuid4().hex[:12]
    save_path = config.DATA_ROOT / "uploads" / f"{task_id}{ext}"
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with open(save_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                break
            f.write(chunk)
    if written > max_bytes:
        save_path.unlink(missing_ok=True)
        raise HTTPException(413, f"文件超过上限 {config.MAX_UPLOAD_MB} MB")

    with _TASKS_LOCK:
        TASKS[task_id] = {
            "task_id": task_id, "status": "queued", "stage": "pending",
            "detail": "已接收视频，等待处理", "session_id": None,
            "subject_id": subject_id, "error": None, "summary": None,
        }

    threading.Thread(
        target=_run_pipeline_task,
        args=(task_id, save_path, subject_id, [client_t0, client_t1]),
        daemon=True,
    ).start()
    return {"task_id": task_id}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str):
    with _TASKS_LOCK:
        t = TASKS.get(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    return t


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/overview")
def overview():
    subjects = []
    for sid in _list_subjects():
        p = s6_profile.get_profile(sid) or {}
        subjects.append({
            "subject_id": sid,
            "n_sessions": p.get("n_sessions", 0),
            "comparability": p.get("comparability", 0.0),
            "exemplars_count": p.get("exemplars_count", 0),
        })
    sessions = [_session_summary(s) for s in _list_sessions()]
    return {"subjects": subjects, "sessions": sessions}


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    p = _safe_session(session_id)
    if not p.root.exists():
        raise HTTPException(404, "session 不存在")
    media = read_json_optional(p.media_json, default={}) or {}
    quality = read_json_optional(p.quality_json, default={}) or {}
    chunks = read_json_optional(p.chunks_json, default=[]) or []
    events = read_json_optional(p.events_json, default=[]) or []
    claims = read_json_optional(p.claims_json, default=[]) or []
    feedback = read_json_optional(p.feedback_json, default=[]) or []
    transcript = read_json_optional(p.transcript_json, default=[]) or []
    text_feat = read_json_optional(p.text_feat_json, default=[]) or []
    voice = read_json_optional(p.voice_json, default=[]) or []
    evidence = read_json_optional(p.evidence_json, default=[]) or []

    # VA 时间线（供图表）
    timeline = []
    for c in chunks:
        va = c.get("va") or {}
        timeline.append({
            "t0": c["t0"], "t1": c["t1"],
            "v": va.get("v"), "a": va.get("a"),
            "is_anomaly": c.get("is_anomaly", False),
            "anomaly_type": c.get("anomaly_type"),
            "chunk_id": c.get("chunk_id"),
        })

    # 给 claim 补 event_type，便于前端直接展示
    event_by_id = {e["event_id"]: e for e in events}
    for c in claims:
        ev = event_by_id.get(c.get("event_id"))
        if ev:
            c.setdefault("event_type", ev.get("type"))

    return {
        "session_id": session_id,
        "media": media,
        "quality": quality,
        "timeline": timeline,
        "events": events,
        "claims": claims,
        "feedback": feedback,
        "transcript": transcript,
        "text_feat": text_feat,
        "voice": voice,
        "evidence": evidence,
        "has_video": bool(media.get("normalized_path") and Path(media["normalized_path"]).exists()),
    }


@app.get("/api/subjects/{subject_id}")
def subject_profile(subject_id: str):
    _safe_subject(subject_id)
    profile = s6_profile.get_profile(subject_id)
    if profile is None:
        raise HTTPException(404, "subject 不存在")
    exemplars = s6_profile.load_exemplars(subject_id)
    sessions = [_session_summary(s) for s in _list_sessions()]
    return {"subject_id": subject_id, "profile": profile,
            "exemplars": exemplars, "sessions": sessions,
            "summary_text": s6_profile.profile_summary(profile)}


class FeedbackBody(BaseModel):
    claim_id: str
    action: str
    reason: str | None = None
    revised_text: str | None = None
    subject_id: str | None = None


@app.post("/api/sessions/{session_id}/feedback")
def submit_feedback(session_id: str, body: FeedbackBody):
    try:
        fb = s5_feedback.apply_feedback(
            session_id, body.claim_id, body.action,
            reason=body.reason, revised_text=body.revised_text,
            subject_id=body.subject_id,
        )
    except (KeyError, StageError, AssertionError) as e:
        raise HTTPException(400, str(e))
    # 刷新画像 exemplars_count
    if body.subject_id:
        sp = _safe_subject(body.subject_id)
        p = s6_profile.get_profile(body.subject_id)
        if p:
            p["exemplars_count"] = len(s6_profile.load_exemplars(body.subject_id))
            write_json(sp.profile_json, p)
    stats = s5_feedback.stats(session_id)
    return {"feedback": fb, "stats": stats}


@app.get("/media/{session_id}/norm.mp4")
def video(session_id: str):
    sp = _safe_session(session_id)
    media = read_json_optional(sp.media_json, default={}) or {}
    p = media.get("normalized_path")
    if not p or not Path(p).exists():
        raise HTTPException(404, "无可用视频")
    return FileResponse(p, media_type="video/mp4")


# ---------------------------------------------------------------------------
# 静态前端
# ---------------------------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main():
    import threading
    import webbrowser

    def _open_browser():
        webbrowser.open("http://127.0.0.1:8100")

    threading.Timer(1.2, _open_browser).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")


if __name__ == "__main__":
    main()
