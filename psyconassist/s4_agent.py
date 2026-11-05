"""S4 Agent 编排（LangGraph）。

12 节点状态图，硬规则 23：LLM 只能进 synth 与 risk 复核两个白名单节点。
validator 只做存在性校验（硬规则 15），校验失败删句不重写（硬规则 16）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config
from . import ids, io, llm

# ---------------------------------------------------------------------------
# 21 条 limits（全量注入，不走检索）
LIMITS = [
    "禁止出现诊断名词：抑郁、焦虑症、创伤后应激障碍、双相、精神分裂等。",
    "禁止内心状态推断：他在回避、她感到被抛弃、他不信任咨询师等。",
    "只描述可观测行为与测量值，不解释动机。",
    "每条结论必须可回溯到 evidence_id。",
    "时间用 mm:ss 表示，且与 evidence 时间一致。",
    "severity 由规则决定，生成时不得修改。",
    "不使用绝对化措辞（一定、必然、彻底）。",
    "不把相关性表述为因果。",
    "单条 claim 只表达一个事实。",
    "效价/唤醒用「相对本场基线」表述，附标准差值。",
    "基线不可得时，不表述相对变化，只描述原始测量。",
    "缺失模态用「不可得」，不臆测。",
    "声学/文本特征按角色（来访者/咨询师）分别表述。",
    "不把单次观察上升为长期特质。",
    "不评价来访者的选择或行为。",
    "不用文学化、煽情化语言。",
    "涉及风险词时，客观转述并标注为「风险词命中」，不夸大。",
    "不比较两位当事人的优劣。",
    "不引用外部诊断标准给分。",
    "措辞面向专业咨询师，简洁、可操作。",
    "若证据不足，宁可少说，不补写推测。",
]

# 变量表（retriever 按 event.type 检索 top-3，简化内置）
VARIABLES = {
    "va_shift": ["面部效价 z 分", "唤醒 z 分", "相邻段效价差值"],
    "silence": ["来访者回应时延", "来访者发声占比", "话轮内停顿占比"],
    "risk": ["风险词命中词条", "风险词上下文", "风险词发生时间"],
    "incongruence": ["文本极性", "面部效价", "言行不一致系数 D02"],
    "low_activation": ["唤醒 z 分", "AU 强度均值", "响度相对值"],
}

# 禁词（validator 硬校验）
DIAGNOSIS_WORDS = ["抑郁", "焦虑症", "焦虑障碍", "创伤后应激", "PTSD", "双相", "精神分裂", "自闭症", "强迫症"]
INFERENCE_WORDS = ["回避", "感到被抛弃", "不信任", "抗拒", "伪装", "掩饰", "讨好", "自卑"]


class State(TypedDict, total=False):
    session_id: str
    subject_id: str
    events: list
    evidence: dict          # evidence_id -> evidence 对象
    quality: dict
    degrade: list
    variables: dict         # event.type -> 变量说明（RAG）
    limits: list
    exemplars: list
    profile_summary: str
    claims: list
    rewrite_count: int
    feedback: list
    llm_ok: bool


# ---------------------------------------------------------------------------
def _fmt_mmss(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def _ingest(state: State) -> dict:
    session_id = state["session_id"]
    wdir = config.session_dir(session_id)
    events = io.read_json(wdir / "events" / "events.json")
    quality = io.read_json(wdir / "quality.json") if (wdir / "quality.json").exists() else {}
    evidence = {}
    # 从 S3 的 evidence 重建（简化：从 events 的 evidence_ids + 生成时记录）
    ev_path = wdir / "events" / "evidence.json"
    if ev_path.exists():
        evidence = {e["evidence_id"]: e for e in io.read_json(ev_path)}
    return {"events": events, "evidence": evidence, "quality": quality,
            "claims": [], "rewrite_count": 0, "feedback": []}


def _quality_gate(state: State) -> Literal["abort", "timeline"]:
    # Demo：有事件即 ok；无事件且质量报告缺模态 → 仍继续（degraded）
    return "timeline"


def _timeline(state: State) -> dict:
    # 纯规则 + 统计：构造时间线上下文（供 synth 使用）
    return {}


def _retriever(state: State) -> dict:
    # RAG：变量 top3 + limits 全量 + 画像摘要 + 历史 exemplar（简化关键词/类型检索）
    events = state["events"]
    variables = {}
    for e in events:
        variables[e["type"]] = VARIABLES.get(e["type"], [])
    return {"variables": variables, "limits": LIMITS,
            "exemplars": state.get("exemplars", []),
            "profile_summary": state.get("profile_summary", "")}


def _risk(state: State) -> dict:
    # 词典 + 规则为主；LLM 只做二次复核（有 risk 事件时）
    return {}


def _synth(state: State) -> dict:
    """LLM 输出严格 JSON claims；LLM 不可用则规则模板（标 llm_ok=False）。"""
    events = state["events"]
    evidence = state["evidence"]
    limits = state["limits"]
    llm_ok = llm.available()

    if llm_ok:
        claims = _synth_llm(state)
        if claims:
            return {"claims": claims, "llm_ok": True}
    # fallback：规则模板
    claims = []
    for i, e in enumerate(events):
        text = _template_claim(e, evidence)
        claims.append({
            "sid": ids.claim_id(state["session_id"], i + 1),
            "event_id": e["event_id"],
            "text": text,
            "evidence_ids": e.get("evidence_ids", []),
            "state": "pending",
        })
    return {"claims": claims, "llm_ok": False}


def _synth_llm(state: State) -> list[dict]:
    events = state["events"]
    evidence = state["evidence"]
    limits = state["limits"]
    variables = state["variables"]
    exemplars = state["exemplars"]
    profile = state.get("profile_summary", "")

    lines = []
    for e in events:
        ev_ids = e.get("evidence_ids", [])
        ev_desc = []
        for eid in ev_ids:
            ev = evidence.get(eid)
            if ev:
                ev_desc.append(f"- {eid}: {ev.get('metric')}={ev.get('value')} @[{_fmt_mmss(ev.get('t0', 0))}-{_fmt_mmss(ev.get('t1', 0))}]")
        vars_desc = variables.get(e["type"], [])
        lines.append(
            f"事件{e['event_id']} 类型={e['type']} 时间[{_fmt_mmss(e['t0'])}-{_fmt_mmss(e['t1'])}] 严重度={e['severity']}\n"
            f"相关变量: {', '.join(vars_desc)}\n证据:\n" + "\n".join(ev_desc)
        )

    system = (
        "你是心理咨询会话分析助手，负责把「事件+证据」写成咨询师能读、每句都能回溯的草稿句。\n"
        "必须遵守的 limits：\n" + "\n".join(f"- {x}" for x in limits) + "\n"
        "只输出 JSON，格式：{\"claims\": [{\"event_id\": \"...\", \"text\": \"...\", \"evidence_ids\": [\"...\"]}]}\n"
        "时间用 mm:ss，text 中不得出现诊断名词或内心状态推断。"
    )
    user = "\n\n".join(lines)
    if exemplars:
        ex = "\n".join(f"- 原文: {x.get('original','')} → 采纳改写: {x.get('revised','')}" for x in exemplars[:5])
        user += "\n\n历史采纳示例（参考措辞风格）：\n" + ex
    if profile:
        user += "\n\n来访者画像摘要：" + profile

    r = llm.chat_json(system, user)
    if not r or not isinstance(r, dict) or "claims" not in r:
        return []
    claims = []
    for i, c in enumerate(r["claims"]):
        if not isinstance(c, dict) or "text" not in c:
            continue
        claims.append({
            "sid": ids.claim_id(state["session_id"], i + 1),
            "event_id": c.get("event_id"),
            "text": str(c["text"]),
            "evidence_ids": c.get("evidence_ids", []),
            "state": "pending",
        })
    return claims


def _template_claim(e: dict, evidence: dict) -> str:
    t0 = _fmt_mmss(e["t0"])
    t1 = _fmt_mmss(e["t1"])
    type_map = {
        "va_shift": f"{t0}–{t1} 面部效价出现明显波动（相对本场基线）。",
        "silence": f"{t0}–{t1} 来访者回应时延较长或发声占比偏低。",
        "risk": f"{t0}–{t1} 检出风险词，原文已记录，供进一步关注。",
        "incongruence": f"{t0}–{t1} 文本极性与面部效价方向不一致。",
        "low_activation": f"{t0}–{t1} 唤醒水平相对本场基线偏低。",
    }
    return type_map.get(e["type"], f"{t0}–{t1} 检测到需要关注的时刻。")


def _validator(state: State) -> dict:
    """校验：evidence_id 真实存在 + 禁词 + 时间格式。失败 → 该句删除。"""
    evidence = state["evidence"]
    valid, dropped = [], []
    for c in state["claims"]:
        ok = True
        eids = c.get("evidence_ids", [])
        if not eids or any(eid not in evidence for eid in eids):
            ok = False
        text = c.get("text", "")
        if any(w in text for w in DIAGNOSIS_WORDS + INFERENCE_WORDS):
            ok = False
        if not re.search(r"\d{2}:\d{2}", text):
            ok = False
        if ok:
            valid.append(c)
        else:
            dropped.append(c)
    return {"claims": valid, "feedback": state.get("feedback", []) + [
        {"action": "strip", "reason": "validator_failed", "claim_id": c.get("sid")} for c in dropped
    ]}


def _policy(state: State) -> Literal["human", "rewrite", "strip"]:
    if state["claims"]:
        return "human"
    if state["rewrite_count"] < 2:
        return "rewrite"
    return "strip"


def _rewrite(state: State) -> dict:
    return {"rewrite_count": state.get("rewrite_count", 0) + 1}


def _strip(state: State) -> dict:
    return {"claims": []}


def _human(state: State) -> dict:
    """落库 claims（state=pending），交由 S5 反馈。"""
    wdir = config.session_dir(state["session_id"])
    io.write_json(wdir / "agent" / "claims.json", state["claims"])
    return {}


def _abort(state: State) -> dict:
    return {}


def build_graph():
    g = StateGraph(State)
    g.add_node("ingest", _ingest)
    g.add_node("quality_gate", _quality_gate)
    g.add_node("timeline", _timeline)
    g.add_node("retriever", _retriever)
    g.add_node("risk", _risk)
    g.add_node("synth", _synth)
    g.add_node("validator", _validator)
    g.add_node("policy", _policy)
    g.add_node("rewrite", _rewrite)
    g.add_node("strip", _strip)
    g.add_node("human", _human)
    g.add_node("abort", _abort)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", _quality_gate, {"abort": "abort", "timeline": "timeline"})
    g.add_edge("timeline", "retriever")
    g.add_edge("retriever", "risk")
    g.add_edge("risk", "synth")
    g.add_edge("synth", "validator")
    g.add_conditional_edges("validator", _policy, {"human": "human", "rewrite": "rewrite", "strip": "strip"})
    g.add_edge("rewrite", "synth")
    g.add_edge("strip", END)
    g.add_edge("human", END)
    g.add_edge("abort", END)
    return g.compile()


def run(session_id: str, subject_id: str = "", exemplars: list | None = None,
        profile_summary: str = "") -> dict:
    graph = build_graph()
    result = graph.invoke({
        "session_id": session_id,
        "subject_id": subject_id,
        "exemplars": exemplars or [],
        "profile_summary": profile_summary,
    })
    return result
