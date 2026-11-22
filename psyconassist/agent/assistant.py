"""咨询辅助助手：带知识库检索的 ReAct 小循环（v2）。

修复：限制工具调用次数、强制先查知识库、避免把 JSON 当答复返回。
"""
from __future__ import annotations

import json
from typing import Any

from .. import llm
from ..kb import search
from . import context as ctx_mod

SYSTEM = """你是心理咨询师的辅助助手，通晓当前这场会谈的数据（表情、语音、原话、事件、草稿）。
任务：帮咨询师理解与修改观察草稿、回答关于来访者的问题、给出疏导方向。

规则：
1. 不做诊断；用"可能""看起来"这类可被修正的措辞。
2. 只依据工具返回的证据与知识库条目；引用知识库时写编号，如 [AU-02]。
3. 判断尽量带时间窗（如 02:58-03:23）。
4. 涉及风险时，把"优先评估风险"放在最前面。
5. 第一次工具调用必须是 search_kb（检索与问题相关的条目）。
6. 最多调用 3 次工具，然后必须给出最终答复。
7. 输出严格 JSON：{"thought":"...","action":"search_kb|get_transcript|get_metrics|list_claims|get_claim|final","args":{...},"answer":"..."}
   - action 不是 final 时，answer 留空字符串；
   - action=final 时，answer 写最终答复（中文，分点，200 字以内）。
"""


def _tool(session_id: str, ctx: dict, action: str, args: dict) -> Any:
    if action == "search_kb":
        return search(args.get("query", ""), k=int(args.get("k", 4)))
    if action == "get_transcript":
        if "t0" not in args or "t1" not in args:
            return "请提供 t0 与 t1（秒）"
        t0, t1 = float(args["t0"]), float(args["t1"])
        rows = [u for u in ctx["transcript"] if u["t1"] > t0 and u["t0"] < t1]
        return [{"t0": u["t0"], "t1": u["t1"], "text": (u.get("text") or "")[:120]} for u in rows[:12]] or "该时间段没有转写"
    if action == "get_metrics":
        if "t0" not in args or "t1" not in args:
            return "请提供 t0 与 t1（秒）"
        t0, t1 = float(args["t0"]), float(args["t1"])
        seg = [c for c in ctx["chunks"] if c["t1"] > t0 and c["t0"] < t1]
        return [{"t0": c["t0"], "t1": c["t1"], "va": c.get("va"), "type": c.get("anomaly_type")} for c in seg[:10]] or "该时间段没有切片"
    if action == "list_claims":
        return [{"sid": c.get("sid"), "state": c.get("state"), "text": (c.get("text") or "")[:80]} for c in ctx["claims"][:15]]
    if action == "get_claim":
        return next((c for c in ctx["claims"] if c.get("sid") == args.get("sid")), "未找到")
    return "未知工具"


def _strip_json(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("{"):
        try:
            obj = json.loads(t)
            return obj.get("answer") or obj.get("reply") or t
        except Exception:
            pass
    return t


def chat(session_id: str, messages: list[dict], t0: float | None = None, t1: float | None = None) -> dict:
    ctx = ctx_mod.build(session_id)
    background = ctx_mod.brief(ctx)
    if t0 is not None and t1 is not None:
        background += "\n当前选中时间窗：%s-%s 秒。" % (t0, t1)

    history = "\n".join("%s：%s" % (m.get("role"), m.get("content")) for m in messages[-6:])
    trace: list[dict] = []
    kb_hits: dict[str, dict] = {}
    observation = ""
    tool_calls = 0

    for step in range(4):
        extra = "" if tool_calls == 0 else "\n（已调用工具 %d 次，最多 3 次，请尽快给出 final）" % tool_calls
        user = background + "\n\n对话：\n" + history + "\n\n已有观察：" + (observation or "无") + extra
        out = llm.chat_json(SYSTEM, user, temperature=0.3, tag="assistant")
        if not isinstance(out, dict):
            return {"reply": "助手暂时不可用（模型未返回）。", "kb": list(kb_hits.values()), "trace": trace}
        action = out.get("action")
        if action == "final":
            return {"reply": (out.get("answer") or "").strip(), "kb": list(kb_hits.values()), "trace": trace}
        if tool_calls >= 3:
            break
        args = out.get("args") or {}
        res = _tool(session_id, ctx, action, args)
        tool_calls += 1
        if action == "search_kb" and isinstance(res, list):
            for e in res:
                kb_hits[e["id"]] = e
        trace.append({"action": action, "args": args, "result": res if not isinstance(res, list) else res[:5]})
        observation += "\n[%s] %s" % (action, json.dumps(res, ensure_ascii=False)[:800])

    fallback = llm.chat_text(
        "你是心理咨询师的辅助助手。基于给定证据，用中文分点回答，200 字以内，不做诊断，引用知识库编号（如有）。",
        background + "\n\n对话：\n" + history + "\n\n观察：\n" + observation + "\n\n请直接给出最终答复正文，不要输出 JSON。",
    )
    return {"reply": _strip_json(fallback or ""), "kb": list(kb_hits.values()), "trace": trace}
