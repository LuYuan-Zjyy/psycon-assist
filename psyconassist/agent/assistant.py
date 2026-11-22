"""咨询辅助助手：带知识库检索的 ReAct 小循环。

工具（不依赖外部服务）：
  search_kb(query)          心理学知识库检索
  get_transcript(t0, t1)    取某时间段原话
  get_metrics(t0, t1)       取某时间段的指标/表情/语音
  list_claims()             列草稿
  get_claim(sid)            取单条草稿

约束：不做诊断；只基于证据与知识库；风险优先。
"""
from __future__ import annotations

import json
from typing import Any

from .. import llm
from ..kb import search
from . import context as ctx_mod

SYSTEM = """你是心理咨询师的辅助助手，通晓当前这场会谈的数据（表情、语音、原话、事件、草稿）。
你的任务是帮咨询师理解和修改观察草稿、回答关于来访者的问题、给出疏导方向。

规则：
1. 不做诊断，不给临床结论；用"可能""看起来"这类可被修正的措辞。
2. 只依据工具返回的证据与知识库条目。引用知识库时写编号，如 [AU-02]。
3. 每条判断尽量带时间窗（如 02:58-03:23）。
4. 涉及风险（自伤/自杀/绝望）时，把"优先评估风险"放在最前面。
5. 输出严格 JSON：{"thought":"...","action":"search_kb|get_transcript|get_metrics|list_claims|get_claim|final","args":{...},"answer":"..."}
   - 需要更多信息时用 action，answer 留空；
   - 信息足够时 action="final"，answer 写最终答复（中文，分点，不超过 200 字）。
"""


def _tool(session_id: str, ctx: dict, action: str, args: dict) -> Any:
    if action == "search_kb":
        return search(args.get("query", ""), k=int(args.get("k", 4)))
    if action == "get_transcript":
        t0, t1 = float(args.get("t0", 0)), float(args.get("t1", 1e9))
        rows = [u for u in ctx["transcript"] if u["t1"] > t0 and u["t0"] < t1]
        out = [{"t0": u["t0"], "t1": u["t1"], "role": u.get("role"), "text": u.get("text")} for u in rows[:40]]
        return out or "该时间段没有转写"
    if action == "get_metrics":
        t0, t1 = float(args.get("t0", 0)), float(args.get("t1", 1e9))
        seg = [c for c in ctx["chunks"] if c["t1"] > t0 and c["t0"] < t1]
        return [{"t0": c["t0"], "t1": c["t1"], "va": c.get("va"), "is_anomaly": c.get("is_anomaly"), "type": c.get("anomaly_type")} for c in seg] or "该时间段没有切片"
    if action == "list_claims":
        return [{"sid": c.get("sid"), "state": c.get("state"), "text": c.get("text")} for c in ctx["claims"][:20]]
    if action == "get_claim":
        sid = args.get("sid")
        return next((c for c in ctx["claims"] if c.get("sid") == sid), "未找到")
    return "未知工具"


def chat(session_id: str, messages: list[dict], t0: float | None = None, t1: float | None = None) -> dict:
    ctx = ctx_mod.build(session_id)
    background = ctx_mod.brief(ctx)
    if t0 is not None and t1 is not None:
        background += "\n当前选中时间窗：%s-%s 秒。" % (t0, t1)

    history = "\n".join("%s：%s" % (m.get("role"), m.get("content")) for m in messages[-6:])
    trace: list[dict] = []
    kb_hits: dict[str, dict] = {}
    observation = ""

    for _ in range(4):
        user = background + "\n\n对话：\n" + history + "\n\n已有观察：" + (observation or "无")
        out = llm.chat_json(SYSTEM, user, temperature=0.3, tag="assistant")
        if not isinstance(out, dict):
            return {"reply": "助手暂时不可用（模型未返回）。", "kb": [], "trace": trace}
        action = out.get("action")
        if action == "final":
            for e in kb_hits.values():
                action_ids = e["id"]
            return {"reply": out.get("answer") or "", "kb": list(kb_hits.values()), "trace": trace}
        args = out.get("args") or {}
        res = _tool(session_id, ctx, action, args)
        if action == "search_kb" and isinstance(res, list):
            for e in res:
                kb_hits[e["id"]] = e
        trace.append({"action": action, "args": args, "result": res if not isinstance(res, list) else res[:6]})
        observation += "\n[%s] %s" % (action, json.dumps(res, ensure_ascii=False)[:900])

    # 用完步数还没 final，就基于已有观察给一个保守回答
    fallback = llm.chat_text(SYSTEM, background + "\n\n对话：\n" + history + "\n\n观察：" + observation + "\n请直接给出最终答复。")
    return {"reply": fallback or "抱歉，这次没能得出结论。", "kb": list(kb_hits.values()), "trace": trace}
