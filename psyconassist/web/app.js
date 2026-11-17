/* PsyConAssist 咨询师工作台（原生 JS + ECharts） */

const TYPE_ZH = {
  va_shift: "效价突变",
  silence: "沉默",
  risk: "风险词",
  incongruence: "言行不一致",
  low_activation: "低激活",
  text_shift: "文本偏移",
  normal: "正常",
};

/** 与 stages/s3_events.py SEVERITY 规则表保持一致（展示用，不可由 LLM 改） */
const SEVERITY_RULE = {
  va_shift: "high",
  risk: "high",
  text_shift: "medium",
  silence: "medium",
  incongruence: "medium",
  low_activation: "low",
};

const SEV_ZH = { high: "高", medium: "中", low: "低" };

const SEV_WHY = {
  va_shift: "规则表：效价/唤醒相对基线突变 → high",
  risk: "规则表：风险词命中 → high",
  text_shift: "规则表：文本特征偏移 → medium",
  silence: "规则表：沉默/低回应 → medium",
  incongruence: "规则表：文本与面部效价不一致 → medium",
  low_activation: "规则表：唤醒相对基线偏低 → low",
};

const METRIC_ZH = {
  "va.v": "效价 V",
  "va.a": "唤醒 A",
  polarity: "文本极性",
  anomaly_types: "异常类型",
  f0_med: "基频 F0",
  loudness_rel: "相对响度",
};

const REASON_ZH = {
  insufficient_evidence: "证据不足",
  over_inference: "过度推断",
  wording: "措辞不专业",
  missing_info: "漏了重要信息",
  time_misaligned: "定位不对",
  other: "其他",
};

const STAGE_ZH = {
  pending: "等待处理",
  s0: "S0 接入与归一化",
  s1: "S1 三路特征提取",
  s2: "S2 融合与异常切片",
  s3: "S3 事件与证据",
  s4: "S4 Agent 草稿",
  s5: "S5 反馈闭环",
  s6: "S6 画像更新",
};

const STAGE_ORDER = ["pending", "s0", "s1", "s2", "s3", "s4", "s5", "s6"];

const state = {
  subjects: [],
  sessions: [],
  currentSubject: null,
  currentSession: null,
  sessionDetail: null,
  profile: null,
  claimFilter: "all",
  activeEventId: null,
  faceBoxes: [],
  vaSeries: [],
  voiceBase: null,
};

let vaChart = null;
let fbChart = null;
let sigVoiceChart = null;
let sigPolChart = null;

const $ = (s) => document.querySelector(s);

function fmtTime(sec) {
  if (sec == null || Number.isNaN(sec)) return "--:--";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 2400);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e.detail || res.statusText);
  }
  return res.json();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
  ));
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtPct(v) { return v == null ? "—" : `${Math.round(v * 100)}%`; }
function fmtNum(v) { return v == null ? "—" : Number(v).toFixed(3); }

function evidenceMap(d) {
  const map = {};
  (d?.evidence || []).forEach((e) => { map[e.evidence_id] = e; });
  return map;
}

function eventById(d, id) {
  return (d?.events || []).find((e) => e.event_id === id) || null;
}

function claimForEvent(d, eventId) {
  return (d?.claims || []).find((c) => c.event_id === eventId) || null;
}

function severityWhy(type) {
  return SEV_WHY[type] || `规则表：${type} → ${SEVERITY_RULE[type] || "medium"}`;
}

/* ---------------- 初始化 ---------------- */
async function init() {
  try {
    const ov = await api("/api/overview");
    state.subjects = ov.subjects || [];
    state.sessions = ov.sessions || [];
    if (state.subjects.length) {
      state.currentSubject = state.subjects[0].subject_id;
    }
    if (state.sessions.length) {
      state.currentSession = state.sessions[state.sessions.length - 1].session_id;
    }
    renderSelects();
    await refreshAll();
  } catch (e) {
    toast("加载失败：" + e.message);
    renderEmptyBrief();
  }
}

function renderSelects() {
  const ss = $("#subject-select");
  const ses = $("#session-select");
  ss.innerHTML = state.subjects.length
    ? state.subjects.map((p) =>
      `<option value="${esc(p.subject_id)}">${esc(p.subject_id)}（${p.n_sessions} 次）</option>`
    ).join("")
    : `<option value="">无来访者</option>`;

  ses.innerHTML = state.sessions.length
    ? state.sessions.map((s) =>
      `<option value="${esc(s.session_id)}">${esc(s.session_id)} · ${fmtTime(s.duration)}</option>`
    ).join("")
    : `<option value="">无会谈</option>`;

  if (state.currentSubject) ss.value = state.currentSubject;
  if (state.currentSession) ses.value = state.currentSession;

  ss.onchange = async () => {
    state.currentSubject = ss.value || null;
    state.voiceBase = null;
    await loadProfile();
  };
  ses.onchange = async () => {
    state.currentSession = ses.value || null;
    state.voiceBase = null;
    state.activeEventId = null;
    await loadTimeline();
  };
}

async function refreshAll() {
  await Promise.all([loadTimeline(), loadProfile()]);
}

function renderEmptyBrief() {
  $("#brief-title").textContent = "暂无会谈数据";
  $("#brief-desc").textContent = "可先运行 python -m psyconassist.cli demo，或点击「上传新会谈」。";
  $("#brief-stats").innerHTML = "";
  $("#modality-strip").innerHTML = "";
}

/* ---------------- Session brief ---------------- */
function renderBrief(d) {
  if (!d) return renderEmptyBrief();
  const q = d.quality || {};
  const degrade = q.degrade || [];
  const pending = (d.claims || []).filter((c) => c.state === "pending").length;
  const nEvt = (d.events || []).length;
  const nUtt = (d.transcript || []).length;
  const duration = d.media?.duration;

  $("#brief-title").textContent = `${d.session_id}`;
  const base = d.baseline;
  const baseTxt = (base && base.available && base.v_sd != null)
    ? `本场面部基线 效价 ${base.v_mean.toFixed(2)}±${base.v_sd.toFixed(2)}、唤醒 ${base.a_mean.toFixed(2)}±${base.a_sd.toFixed(2)}（前 ${Math.round((base.window || 480) / 60)} 分钟）`
    : "本场基线不可得（激活判定无依据）";
  $("#brief-desc").textContent =
    `时长 ${fmtTime(duration)} · 待复核草稿 ${pending} 条 · ` + baseTxt + " · " +
    (degrade.length
      ? `已降级：${degrade.join("、")}`
      : "各模态可用");

  $("#brief-stats").innerHTML = `
    <div class="stat"><div class="stat-value">${nEvt}</div><div class="stat-label">异常事件</div></div>
    <div class="stat"><div class="stat-value">${(d.claims || []).length}</div><div class="stat-label">观察草稿</div></div>
    <div class="stat"><div class="stat-value">${pending}</div><div class="stat-label">待复核</div></div>
    <div class="stat"><div class="stat-value">${nUtt}</div><div class="stat-label">转写句</div></div>`;

  const mods = [
    {
      key: "visual",
      label: "视觉",
      ok: !degrade.includes("visual") && (q.face_cov == null || q.face_cov > 0),
      detail: typeof q.face_cov === "number" ? `覆盖 ${Math.round(q.face_cov * 100)}%` : (degrade.includes("visual") ? "不可得" : "可用"),
    },
    {
      key: "voice",
      label: "语音",
      ok: !degrade.includes("voice") && !degrade.includes("speaker_diarization"),
      detail: q.diarization_ok === false || degrade.includes("speaker_diarization")
        ? "分离降级"
        : (degrade.includes("voice") ? "不可得" : "可用"),
    },
    {
      key: "text",
      label: "文本",
      ok: q.asr_ok !== false && !degrade.includes("text"),
      detail: q.asr_ok === false || degrade.includes("text") ? "ASR 不可得" : "可用",
    },
  ];
  $("#modality-strip").innerHTML = mods.map((m) => {
    const cls = m.ok ? "ok" : (m.detail.includes("降级") ? "warn" : "bad");
    return `<span class="mod ${cls}"><i></i>${m.label} · ${m.detail}</span>`;
  }).join("") +
    `<span class="mod"><i></i>原则 · 证据可回跳 · 禁止静默填 0</span>`;
}

/* ---------------- Timeline ---------------- */
async function loadTimeline() {
  if (!state.currentSession) {
    renderEmptyBrief();
    return;
  }
  const d = await api(`/api/sessions/${state.currentSession}`);
  state.sessionDetail = d;
  state.voiceBase = null;
  const safe = (fn, arg) => { try { fn(arg); } catch (e) { console.error(e); } };
  safe(renderBrief, d);
  safe(renderQuality, d);
  safe(renderVideo, d);
  safe(renderEvents, d);
  safe(renderTranscript, d);
  safe(renderVAChart, d);
  safe(renderSignals, d);
  safe(renderClaims, d);
}

function renderQuality(d) {
  const el = $("#quality-banner");
  if (!el) return;
  const q = d.quality || {};
  const degrade = q.degrade || [];
  const parts = [];
  if (degrade.length) {
    parts.push(`本场质量降级：${degrade.join("、")}。对应信号在观察中标注为「不可得」，不参与伪造结论。`);
  } else {
    parts.push("本场未标记模态降级。异常事件仍相对本场基线检出，草稿需人工确认。");
  }
  if ((d.events || []).length === 0) parts.push("未检出异常事件（合法结果）。");
  el.textContent = parts.join(" ");
  el.classList.toggle("warn", degrade.length > 0);
  el.classList.remove("hidden");
}

function renderTranscript(d) {
  const el = $("#transcript-list");
  const meta = $("#transcript-meta");
  const utts = d.transcript || [];
  if (meta) meta.textContent = utts.length ? `${utts.length} 条 · 点击跳转` : "暂无";
  if (!el) return;
  if (!utts.length) {
    el.innerHTML = `<div class="empty">暂无转写（ASR 不可用或未产出）</div>`;
    return;
  }
  el.innerHTML = utts.map((u) => `
    <div class="utt" data-t="${u.t0}">
      <span class="utt-time">${fmtTime(u.t0)}</span>
      <span class="utt-role ${u.role || ""}">${u.role === "counselor" ? "咨询师" : u.role === "client" ? "来访者" : (u.role || "?")}</span>
      <span class="utt-text">${escapeHtml(u.text || "")}</span>
    </div>`).join("");
  el.querySelectorAll(".utt").forEach((row) => {
    row.addEventListener("click", () => seekTo(parseFloat(row.dataset.t) || 0));
  });
}

function renderSignals(d) {
  const voice = d.voice || [];
  const feat = d.text_feat || [];
  const client = voice.filter((s) => s.role === "client");
  const duration = d.media?.duration || 0;
  const axisColor = "#6a7885";
  const split = "#e4ebf0";

  const vEl = $("#sig-voice-chart");
  if (vEl && typeof echarts !== "undefined") {
    if (!sigVoiceChart) sigVoiceChart = echarts.init(vEl);
    if (!client.length) {
      sigVoiceChart.setOption({
        title: { text: "无来访者语音声学（不可得）", left: "center", top: "middle",
          textStyle: { color: axisColor, fontSize: 13 } },
      }, true);
    } else {
      sigVoiceChart.setOption({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", valueFormatter: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
        legend: { data: ["基频 F0 (Hz)", "相对响度 (dB)"], textStyle: { color: axisColor } },
        grid: { left: 44, right: 44, top: 34, bottom: 28 },
        xAxis: { type: "value", min: 0, max: duration,
          axisLabel: { formatter: (v) => fmtTime(v), color: axisColor },
          splitLine: { lineStyle: { color: split } } },
        yAxis: [
          { type: "value", name: "Hz", axisLabel: { color: axisColor }, splitLine: { lineStyle: { color: split } } },
          { type: "value", name: "dB", axisLabel: { color: axisColor }, splitLine: { show: false } },
        ],
        series: [
          { name: "基频 F0 (Hz)", type: "line", data: client.map((s) => [s.t0, s.f0_med ?? null]),
            connectNulls: true, symbol: "none", lineStyle: { color: "#3a5f7a", width: 1.6 } },
          { name: "相对响度 (dB)", type: "line", yAxisIndex: 1,
            data: client.map((s) => [s.t0, s.loudness_rel ?? null]),
            connectNulls: true, symbol: "none", lineStyle: { color: "#9a6b24", width: 1.4 } },
        ],
      }, true);
    }
  }

  const pEl = $("#sig-pol-chart");
  if (pEl && typeof echarts !== "undefined") {
    if (!sigPolChart) sigPolChart = echarts.init(pEl);
    if (!feat.length) {
      sigPolChart.setOption({
        title: { text: "无文本极性（不可得）", left: "center", top: "middle",
          textStyle: { color: axisColor, fontSize: 13 } },
      }, true);
    } else {
      sigPolChart.setOption({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", valueFormatter: (v) => (v == null ? "—" : Number(v).toFixed(3)) },
        legend: { data: ["极性", "风险词命中"], textStyle: { color: axisColor } },
        grid: { left: 44, right: 20, top: 34, bottom: 28 },
        xAxis: { type: "value", min: 0, max: duration,
          axisLabel: { formatter: (v) => fmtTime(v), color: axisColor },
          splitLine: { lineStyle: { color: split } } },
        yAxis: { type: "value", min: -1.2, max: 1.2,
          axisLabel: { color: axisColor }, splitLine: { lineStyle: { color: split } } },
        series: [
          { name: "极性", type: "line", data: feat.map((s) => [s.t0, s.polarity ?? null]),
            connectNulls: true, symbol: "none", step: "end", lineStyle: { color: "#2f615c", width: 1.6 } },
          { name: "风险词命中", type: "scatter",
            data: feat.filter((s) => (s.risk_hits || []).length).map((s) => [s.t0, s.polarity ?? 0]),
            symbolSize: 9, itemStyle: { color: "#a33b3b" } },
        ],
      }, true);
    }
  }
}

function renderVideo(d) {
  const video = $("#video");
  const ph = $("#video-placeholder");
  $("#video-meta").textContent = d.media?.duration
    ? `${fmtTime(d.media.duration)} · ${d.media.fps || "—"} fps · ${d.media.width || "—"}×${d.media.height || "—"}`
    : "";
  state.faceBoxes = d.face_boxes || [];
  state.vaSeries = d.va_series || [];
  state.baseline = d.baseline || null;
  renderMiniTimeline(d);
  if (!video._ovBound) {
    video._ovBound = true;
    video.addEventListener("timeupdate", () => onVideoTime(video.currentTime));
    video.addEventListener("seeked", () => onVideoTime(video.currentTime));
    window.addEventListener("resize", () => {
      drawFaceOverlay($("#video").currentTime);
      vaChart?.resize();
      sigVoiceChart?.resize();
      sigPolChart?.resize();
      fbChart?.resize();
    });
  }
  if (d.has_video) {
    video.src = `/media/${d.session_id}/norm.mp4`;
    ph.classList.add("hidden");
    video.style.display = "block";
    $("#video-hud")?.classList.remove("hidden");
  } else {
    video.removeAttribute("src");
    video.style.display = "none";
    ph.classList.remove("hidden");
    $("#video-hud")?.classList.add("hidden");
    $("#video-caption")?.classList.remove("show");
  }
}

function renderMiniTimeline(d) {
  const anoms = $("#mini-anoms");
  const dur = d.media?.duration || 0;
  if (!anoms || !dur) return;
  const segs = (d.timeline || []).filter((c) => c.is_anomaly);
  anoms.innerHTML = segs.map((c) => {
    const left = (c.t0 / dur) * 100;
    const w = Math.max(0.4, ((c.t1 - c.t0) / dur) * 100);
    return '<span class="mini-seg" style="left:' + left + '%;width:' + w + '%"></span>';
  }).join("");
  const mt = $("#mini-timeline");
  if (mt && !mt._bound) {
    mt._bound = true;
    mt.addEventListener("click", (ev) => {
      const r = mt.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
      seekTo(ratio * dur);
    });
  }
}

function emotionWord(v, a) {
  if (v == null) return "—";
  if (v < -0.25 && a != null && a > 0.25) return "低落 · 激动";
  if (v < -0.25) return "低落";
  if (v > 0.25 && a != null && a > 0.25) return "积极 · 高昂";
  if (v > 0.25) return "积极";
  return "平稳";
}

function updateHud(t) {
  const va = vaAt(t);
  const v = va ? va.v : null;
  const a = va ? va.a : null;
  const barV = $("#hud-v"), barA = $("#hud-a");
  if (barV) barV.style.width = (v == null ? 50 : ((v + 1) / 2) * 100) + "%";
  if (barA) barA.style.width = (a == null ? 50 : ((a + 1) / 2) * 100) + "%";
  const vv = $("#hud-vv"), av = $("#hud-av");
  if (vv) vv.textContent = v == null ? "—" : v.toFixed(2);
  if (av) av.textContent = a == null ? "—" : a.toFixed(2);
  const emo = $("#hud-emo");
  const chunk = chunkAt(t);
  const EXPR_ZH = { Neutral: "中性", Happiness: "高兴", Sadness: "悲伤", Surprise: "惊讶",
                    Fear: "恐惧", Disgust: "厌恶", Anger: "愤怒", Contempt: "轻蔑" };
  if (emo) {
    if (chunk && chunk.expr) {
      emo.textContent = (EXPR_ZH[chunk.expr] || chunk.expr) + (chunk.expr_p == null ? "" : " " + chunk.expr_p.toFixed(2));
    } else {
      emo.textContent = emotionWord(v, a);
    }
  }

  const ph = $("#mini-playhead");
  const dur = state.sessionDetail?.media?.duration || 0;
  if (ph && dur) ph.style.left = ((t / dur) * 100) + "%";

  const cap = $("#video-caption");
  if (cap) {
    const notes = state.sessionDetail?.llm_notes || [];
    const n = notes.find((x) => t >= x.t0 && t < x.t1);
    if (n && n.note) {
      cap.textContent = n.note.slice(0, 72) + (n.note.length > 72 ? "…" : "");
      cap.classList.add("show");
    } else {
      cap.classList.remove("show");
    }
  }
}

function renderVAChart(d) {
  const el = $("#va-chart");
  if (typeof echarts === "undefined") {
    el.innerHTML = `<div class="empty">图表组件未加载（ECharts）</div>`;
    return;
  }
  if (!vaChart) vaChart = echarts.init(el);
  const t = d.timeline || [];
  const v = t.map((x) => [x.t1, x.v]);
  const a = t.map((x) => [x.t1, x.a]);
  const areas = [];
  for (const c of t) {
    if (c.is_anomaly) {
      areas.push([{ xAxis: c.t0, itemStyle: { color: "rgba(163,59,59,0.12)" } }, { xAxis: c.t1 }]);
    }
  }
  const axisColor = "#6a7885";
  vaChart.setOption({
    backgroundColor: "transparent",
    grid: { left: 44, right: 16, top: 24, bottom: 28 },
    tooltip: {
      trigger: "axis",
      formatter: (params) => {
        const p = params[0];
        const item = t.find((x) => Math.abs(x.t1 - p.axisValue) < 1e-6);
        let s = `${fmtTime(p.axisValue)}<br/>`;
        params.forEach((pp) => {
          const yv = Array.isArray(pp.value) ? pp.value[1] : pp.value;
          if (yv != null) s += `${pp.marker} ${pp.seriesName}: ${Number(yv).toFixed(3)}<br/>`;
        });
        if (item?.is_anomaly) s += `<b style="color:#a33b3b">${TYPE_ZH[item.anomaly_type] || item.anomaly_type}</b>`;
        return s;
      },
    },
    xAxis: {
      type: "value", min: 0, max: d.media?.duration || undefined,
      axisLabel: { formatter: (v) => fmtTime(v), color: axisColor },
      splitLine: { lineStyle: { color: "#e4ebf0" } },
    },
    yAxis: {
      type: "value", min: -1.2, max: 1.2,
      axisLabel: { color: axisColor },
      splitLine: { lineStyle: { color: "#e4ebf0" } },
    },
    series: [
      {
        name: "效价", type: "line", data: v, step: "end", symbol: "none",
        lineStyle: { color: "#2f615c", width: 1.8 },
        markArea: { silent: true, data: areas },
      },
      {
        name: "唤醒", type: "line", data: a, step: "end", symbol: "none",
        lineStyle: { color: "#9a6b24", width: 1.4, opacity: 0.9 },
      },
    ],
  }, true);
  vaChart.off("click");
  vaChart.on("click", (p) => {
    if (p.value != null && p.seriesType === "line") {
      seekTo(Array.isArray(p.value) ? p.value[0] : p.value);
    }
  });
}

function evidenceChipsHtml(ids, map, clickable) {
  if (!ids?.length) return `<span class="meta">无证据</span>`;
  return ids.map((id) => {
    const e = map[id];
    if (!e) return `<span class="ev-chip meta">${esc(id)}</span>`;
    const val = typeof e.value === "number" ? Number(e.value).toFixed(3) : esc(String(e.value ?? "—"));
    const t0 = e.t0;
    const cls = clickable ? `ev-chip` : `ev-chip`;
    const attrs = clickable ? ` data-t="${t0}" title="跳转到 ${fmtTime(t0)} · ${esc(e.tool_version || "")}"` : "";
    return `<span class="${cls}"${attrs}>${METRIC_ZH[e.metric] || esc(e.metric)} = <b>${val}</b> <span class="ev-t">@${fmtTime(t0)}</span></span>`;
  }).join("");
}

function nearbyTranscript(d, t0, t1) {
  const utts = (d.transcript || []).filter((u) => u.t0 < t1 + 1 && (u.t1 ?? u.t0) > t0 - 1);
  if (!utts.length) return "";
  const snippet = utts.slice(0, 2).map((u) => {
    const role = u.role === "counselor" ? "咨询师" : u.role === "client" ? "来访者" : "?";
    return `${role}「${(u.text || "").slice(0, 36)}${(u.text || "").length > 36 ? "…" : ""}」`;
  }).join("；");
  return snippet;
}

function renderEvents(d) {
  const el = $("#event-list");
  if (!d.events?.length) {
    el.innerHTML = `<div class="empty">本场未检出异常事件（合法结果）</div>`;
    return;
  }
  const map = evidenceMap(d);
  el.innerHTML = d.events.map((e) => {
    const claim = claimForEvent(d, e.event_id);
    const claimHint = claim
      ? `关联草稿 ${claim.sid} · ${({ pending: "待复核", accept: "已采纳", edit: "已改写", reject: "已否决" }[claim.state] || claim.state)}`
      : "暂无关联草稿";
    const open = state.activeEventId === e.event_id ? " open active" : "";
    return `
    <article class="event-item${open}" data-eid="${esc(e.event_id)}" data-t="${e.t0}">
      <button type="button" class="event-main" data-act="seek">
        <span class="event-time">${fmtTime(e.t0)}–${fmtTime(e.t1)}</span>
        <div class="event-body">
          <div class="event-type">${TYPE_ZH[e.type] || esc(e.type)}</div>
          <div class="event-desc">${e.evidence_ids.length} 条证据 · ${claimHint}</div>
          <div class="event-rule">${esc(severityWhy(e.type))}</div>
        </div>
        <span class="sev sev-${e.severity}">${SEV_ZH[e.severity] || esc(e.severity)}</span>
      </button>
      <div class="event-evidence">
        <div class="ev-row">${evidenceChipsHtml(e.evidence_ids, map, true)}</div>
        ${nearbyTranscript(d, e.t0, e.t1) ? `<p class="meta" style="margin-top:8px">邻近转写：${esc(nearbyTranscript(d, e.t0, e.t1))}</p>` : ""}
        <div class="claim-actions" style="margin-top:8px">
          <button type="button" class="btn btn-ghost" data-act="toggle">展开/收起证据</button>
          ${claim ? `<button type="button" class="btn btn-ghost" data-act="to-claim" data-sid="${esc(claim.sid)}">查看草稿</button>` : ""}
        </div>
      </div>
    </article>`;
  }).join("");

  el.querySelectorAll(".event-item").forEach((it) => {
    const eid = it.dataset.eid;
    it.querySelectorAll("[data-act]").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const act = btn.dataset.act;
        if (act === "seek") {
          state.activeEventId = eid;
          seekTo(parseFloat(it.dataset.t));
          it.classList.add("open", "active");
          el.querySelectorAll(".event-item").forEach((x) => {
            if (x !== it) x.classList.remove("active");
          });
        } else if (act === "toggle") {
          it.classList.toggle("open");
        } else if (act === "to-claim") {
          switchTab("claims");
          setTimeout(() => {
            const card = document.querySelector(`.claim-card[data-sid="${btn.dataset.sid}"]`);
            card?.scrollIntoView({ behavior: "smooth", block: "center" });
          }, 60);
        }
      });
    });
    it.querySelectorAll(".ev-chip[data-t]").forEach((chip) => {
      chip.addEventListener("click", (ev) => {
        ev.stopPropagation();
        seekTo(parseFloat(chip.dataset.t) || 0);
      });
    });
  });
}

function seekTo(sec) {
  const video = $("#video");
  if (state.sessionDetail?.has_video) {
    video.currentTime = sec;
    video.play().catch(() => {});
  } else {
    toast(`定位至 ${fmtTime(sec)}（无真实视频，已同步信号）`);
  }
  onVideoTime(sec);
  highlightTranscript(sec);
}

function highlightTranscript(t) {
  const rows = document.querySelectorAll("#transcript-list .utt");
  let best = null;
  let bd = Infinity;
  rows.forEach((row) => {
    row.classList.remove("active");
    const tt = parseFloat(row.dataset.t);
    const d = Math.abs(tt - t);
    if (d < bd) { bd = d; best = row; }
  });
  if (best && bd <= 8) {
    best.classList.add("active");
    best.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

/* ---------------- Face overlay + now state ---------------- */
function nearestBy(arr, t, getT) {
  let best = null;
  let bd = Infinity;
  for (const x of arr) {
    const d = Math.abs(getT(x) - t);
    if (d < bd) { bd = d; best = x; }
  }
  return bd <= 3 ? best : null;
}

function drawFaceOverlay(t) {
  const cv = $("#face-overlay");
  const video = $("#video");
  if (!cv || !video || video.style.display === "none") return;
  const rect = video.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(rect.width * dpr);
  cv.height = Math.round(rect.height * dpr);
  cv.style.width = rect.width + "px";
  cv.style.height = rect.height + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const boxes = state.faceBoxes || [];
  if (!boxes.length) return;
  const fb = nearestBy(boxes, t, (x) => x.ts);
  if (!fb || !fb.box) return;
  const vw = video.videoWidth || 320;
  const vh = video.videoHeight || 240;
  const scale = Math.min(rect.width / vw, rect.height / vh);
  const offX = (rect.width - vw * scale) / 2;
  const offY = (rect.height - vh * scale) / 2;
  const [x0, y0, x1, y1] = fb.box;
  const bx = offX + x0 * scale;
  const by = offY + y0 * scale;
  const bw = (x1 - x0) * scale;
  const bh = (y1 - y0) * scale;
  ctx.strokeStyle = "rgba(240,210,120,.95)";
  ctx.lineWidth = 2;
  ctx.strokeRect(bx, by, bw, bh);
  ctx.fillStyle = "rgba(240,210,120,.14)";
  ctx.fillRect(bx, by, bw, bh);
  const badge = $("#now-badge");
  if (badge) {
    badge.style.display = "block";
    badge.textContent = "face " + Math.round((fb.conf || 0) * 100) + "%";
    badge.style.left = bx + "px";
    badge.style.top = Math.max(0, by - 24) + "px";
    badge.style.bottom = "auto";
  }
}

function voiceAt(t) {
  const v = (state.sessionDetail?.voice || []).filter((s) => s.role === "client" || s.role === "unknown");
  return nearestBy(v, t, (s) => s.t0);
}
function textAt(t) {
  const f = (state.sessionDetail?.text_feat || []).filter((s) => s.role === "client" || s.role === "unknown");
  return nearestBy(f, t, (s) => s.t0);
}
function vaAt(t) {
  const s = state.vaSeries || [];
  if (!s.length) return null;
  // 逐帧采样的 ts 可能不是 1Hz，按时间就近取，不能按索引
  return nearestBy(s, t, (x) => x.ts);
}
function chunkAt(t) {
  const tl = state.sessionDetail?.timeline || [];
  return tl.find((c) => t >= c.t0 && t < c.t1) || nearestBy(tl, t, (c) => (c.t0 + c.t1) / 2);
}

function updateNowState(t) {
  ensureVoiceBase();
  const s = descState(t);
  const storyEl = $("#now-story");
  const notes = state.sessionDetail?.llm_notes || [];
  const note = notes.find((n) => t >= n.t0 && t < n.t1) || notes.find((n) => t < n.t1);
  const chunk = chunkAt(t);
  if (storyEl) {
    let head = `<b class="mono">${fmtTime(t)}</b>　`;
    if (note) {
      storyEl.innerHTML = head + escapeHtml(note.note);
    } else if (chunk?.is_anomaly) {
      storyEl.innerHTML = head +
        `异常窗「${TYPE_ZH[chunk.anomaly_type] || chunk.anomaly_type}」· ${escapeHtml(severityWhy(chunk.anomaly_type))}。` +
        `${escapeHtml(s.face)}；${escapeHtml(s.voice)}；${escapeHtml(s.text)}。`;
    } else {
      storyEl.innerHTML = head + `${escapeHtml(s.face)}；${escapeHtml(s.voice)}；${escapeHtml(s.text)}。`;
    }
  }
  const sigEl = $("#now-signal");
  if (sigEl) {
    sigEl.innerHTML = `实时信号　${escapeHtml(s.face)}；${escapeHtml(s.voice)}；${escapeHtml(s.text)}。` +
      `<span class="now-note">${escapeHtml(s.note)}</span>`;
  }
  const va = vaAt(t);
  const vc = voiceAt(t);
  const tx = textAt(t);
  if ($("#now-time")) $("#now-time").textContent = fmtTime(t);

  const faceEl = $("#now-face");
  const voiceEl = $("#now-voice");
  const textEl = $("#now-text");
  const fusedEl = $("#now-fused");

  let fv = va ? va.v : null;
  let fa = va ? va.a : null;
  // 若无逐秒 VA，回退到 chunk VA
  if (fv == null && chunk?.v != null) { fv = chunk.v; fa = chunk.a; }

  if (faceEl) {
    const EXPR = { Neutral: "中性", Happiness: "高兴", Sadness: "悲伤", Surprise: "惊讶",
                   Fear: "恐惧", Disgust: "厌恶", Anger: "愤怒", Contempt: "轻蔑" };
    const eTxt = chunk && chunk.expr
      ? `<br/>表情 <b>${EXPR[chunk.expr] || chunk.expr}</b>${chunk.expr_p == null ? "" : " " + chunk.expr_p.toFixed(2)}${chunk.expr_ent == null ? "" : `（熵 ${chunk.expr_ent.toFixed(2)}）`}`
      : "";
    faceEl.innerHTML = ((fv != null || fa != null)
      ? `效价 <b>${fv == null ? "—" : fv.toFixed(3)}</b> · 唤醒 <b>${fa == null ? "—" : fa.toFixed(3)}</b>`
      : "不可得") + eTxt;
  }
  try { updateHud(t); } catch (e) { console.error(e); }

  // 激活是否有依据：以本场个体基线为参照
  const base = state.sessionDetail?.baseline;
  if (sigEl) {
    if (base && base.available && fv != null && base.v_sd) {
      const z = (fv - base.v_mean) / base.v_sd;
      sigEl.innerHTML += '　<span class="now-note">相对本场基线 ' + (z >= 0 ? "+" : "") + z.toFixed(1) + 'σ</span>';
    } else if (base && base.available === false) {
      sigEl.innerHTML += '　<span class="now-note">基线不可得，激活判定无依据</span>';
    }
  }

  let va2 = null;
  if (vc) {
    const db = vc.loudness_rel;
    va2 = db == null ? null : Math.max(-1, Math.min(1, (db + 40) / 40 * 2 - 1));
  }
  if (voiceEl) {
    voiceEl.innerHTML = vc
      ? `F0 <b>${vc.f0_med == null ? "—" : Math.round(vc.f0_med)}</b> Hz · 响度 <b>${vc.loudness_rel == null ? "—" : vc.loudness_rel.toFixed(1)}</b> dB`
      : "不可得";
  }
  if (textEl) {
    textEl.innerHTML = tx
      ? `极性 <b>${tx.polarity == null ? "—" : tx.polarity.toFixed(2)}</b><br/><span class="meta">${escapeHtml((tx.text || "").slice(0, 28))}${(tx.text || "").length > 28 ? "…" : ""}</span>`
      : "不可得";
  }
  if (fusedEl) {
    const partsV = [];
    const partsA = [];
    if (fv != null) { partsV.push([0.5, fv]); partsA.push([0.5, fa == null ? 0 : fa]); }
    if (tx && tx.polarity != null) partsV.push([0.2, tx.polarity]);
    if (va2 != null) partsA.push([0.3, va2]);
    const wsum = (arr) => arr.reduce((s, x) => s + x[0], 0);
    const wmix = (arr) => (arr.length ? arr.reduce((s, x) => s + x[0] * x[1], 0) / wsum(arr) : null);
    const V = wmix(partsV);
    const A = wmix(partsA);
    const mods = [fv != null ? "视觉" : null, va2 != null ? "语音" : null, tx && tx.polarity != null ? "文本" : null].filter(Boolean);
    fusedEl.innerHTML =
      `效价 <b>${V == null ? "—" : V.toFixed(3)}</b> · 唤醒 <b>${A == null ? "—" : A.toFixed(3)}</b>` +
      `<div class="now-rule">参与模态：${mods.join(" + ") || "无"}（缺失不参与加权）</div>`;
  }
}

function ensureVoiceBase() {
  if (state.voiceBase) return;
  const v = (state.sessionDetail?.voice || []).filter((s) => (s.role === "client" || s.role === "unknown") && s.f0_med != null);
  if (v.length) {
    const fs = v.map((s) => s.f0_med).sort((a, b) => a - b);
    const q = (p) => fs[Math.floor(p * (fs.length - 1))];
    state.voiceBase = { med: q(0.5), loF0: q(0.25), hiF0: q(0.75) };
  }
}

function descState(t) {
  const va = vaAt(t);
  const chunk = chunkAt(t);
  const vc = voiceAt(t);
  const tx = textAt(t);
  const v = va?.v ?? chunk?.v ?? null;
  const a = va?.a ?? chunk?.a ?? null;
  const pol = tx && tx.polarity != null ? tx.polarity : null;
  const q = tx && tx.text ? String(tx.text).slice(0, 34) : null;

  let face = "面部不可得";
  if (v != null) {
    if (v < -0.25 && a != null && a > 0.25) face = "面部效价偏低且唤醒偏高";
    else if (v < -0.25) face = "面部效价偏低";
    else if (v > 0.25 && a != null && a > 0.25) face = "面部效价与唤醒均偏高";
    else if (v > 0.25) face = "面部效价偏高";
    else face = "面部效价接近中性";
  }

  let voice = "声音不可得";
  if (vc && vc.f0_med != null) {
    const b = state.voiceBase;
    if (b && vc.f0_med > b.hiF0) voice = "基频相对本场偏高";
    else if (b && vc.f0_med < b.loF0) voice = "基频相对本场偏低";
    else voice = "基频接近本场中位";
  }

  let text = "本段无清晰转写";
  if (q) {
    const pw = pol == null ? "" : (pol > 0.2 ? "（极性偏正）" : pol < -0.2 ? "（极性偏负）" : "（极性中性）");
    text = "说「" + q + "…」" + pw;
  }

  let note = "";
  if (v != null && pol != null) {
    if (pol > 0.2 && v < -0.25) note = "⚠ 文本偏正 × 面部效价偏低（incongruence 候选）";
    else if (pol < -0.2 && v > 0.25) note = "⚠ 文本偏负 × 面部效价偏高（incongruence 候选）";
  }
  if (!note && v != null) note = "可见模态方向大体一致";
  if (!note) note = "可用模态不足，不做一致性判断";

  return { face, voice, text, note };
}

function moveChartCursor(t) {
  if (!vaChart) return;
  vaChart.setOption({
    series: [{
      markLine: {
        silent: true, symbol: "none", label: { show: false },
        lineStyle: { color: "#a33b3b", width: 1, type: "solid" },
        data: [{ xAxis: t }],
      },
    }],
  });
}

function onVideoTime(t) {
  try { drawFaceOverlay(t); } catch (e) { console.error(e); }
  try { updateNowState(t); } catch (e) { console.error(e); }
  try { moveChartCursor(t); } catch (e) { console.error(e); }
}

/* ---------------- Claims ---------------- */
function renderClaims(d) {
  state.sessionDetail = d;
  const claims = d.claims || [];
  const pending = claims.filter((c) => c.state === "pending").length;
  const badge = $("#claims-badge");
  if (pending) {
    badge.textContent = String(pending);
    badge.hidden = false;
  } else {
    badge.hidden = true;
  }
  applyClaimFilter();
}

function applyClaimFilter() {
  const d = state.sessionDetail;
  if (!d) return;
  let claims = d.claims || [];
  const f = state.claimFilter;
  if (f !== "all") claims = claims.filter((c) => c.state === f);

  const stats = {
    pending: d.claims.filter((c) => c.state === "pending").length,
    edit: d.claims.filter((c) => c.state === "edit").length,
    reject: d.claims.filter((c) => c.state === "reject").length,
    accept: d.claims.filter((c) => c.state === "accept").length,
  };
  $("#claim-stats").textContent =
    `待复核 ${stats.pending} · 采纳 ${stats.accept} · 改写 ${stats.edit} · 否决 ${stats.reject}`;

  const el = $("#claim-list");
  if (!claims.length) {
    el.innerHTML = `<div class="empty">无符合条件的草稿</div>`;
    return;
  }
  el.innerHTML = claims.map((c) => claimCard(c, d)).join("");
  bindClaimActions(el);
}

function claimCard(c, d) {
  const stateZh = { pending: "待复核", accept: "已采纳", edit: "已改写", reject: "已否决" }[c.state] || c.state;
  const original = c.original_text ? `<div class="claim-original">原文：${esc(c.original_text)}</div>` : "";
  const map = evidenceMap(d);
  const ev = eventById(d, c.event_id);
  const type = c.event_type || ev?.type;
  const sev = ev?.severity || SEVERITY_RULE[type] || "medium";
  const t0 = ev?.t0;
  const t1 = ev?.t1;
  const nearby = ev ? nearbyTranscript(d, ev.t0, ev.t1) : "";

  let actions = "";
  if (c.state === "pending") {
    actions = `
      <button class="btn btn-accept" data-act="accept" data-id="${c.sid}">采纳</button>
      <button class="btn btn-edit" data-act="edit-open" data-id="${c.sid}">改写</button>
      <button class="btn btn-reject" data-act="reject-open" data-id="${c.sid}">否决</button>`;
  }
  const jump = t0 != null
    ? `<button class="btn btn-ghost claim-jump" data-act="jump" data-t="${t0}">回跳 ${fmtTime(t0)}–${fmtTime(t1)}</button>`
    : "";

  return `
    <article class="claim-card" data-sid="${c.sid}">
      <div class="claim-head">
        <span class="claim-state st-${c.state}">${stateZh}</span>
        <span class="claim-type">${TYPE_ZH[type] || esc(type) || "事件"}</span>
        <span class="sev sev-${sev}">严重度 ${SEV_ZH[sev] || sev}</span>
        <span class="meta mono">${esc(c.sid)}</span>
      </div>
      ${original}
      <div class="claim-text">${esc(c.text)}</div>
      <div class="claim-rationale">
        <strong>为何出现这条草稿</strong>
        ${esc(severityWhy(type))}
        ${ev ? ` · 事件 ${esc(ev.event_id)} · 时段 ${fmtTime(ev.t0)}–${fmtTime(ev.t1)}` : ""}
        ${nearby ? `<br/>邻近转写：${esc(nearby)}` : ""}
      </div>
      <div class="claim-evidence">${evidenceChipsHtml(c.evidence_ids, map, true)}</div>
      <div class="claim-actions">${actions}${jump}</div>
      <div class="edit-box">
        <textarea placeholder="改写后的专业表述…">${esc(c.text)}</textarea>
        <div class="claim-actions" style="margin-top:8px">
          <button class="btn btn-accept" data-act="edit-submit" data-id="${c.sid}">提交改写</button>
          <button class="btn btn-ghost" data-act="cancel" data-id="${c.sid}">取消</button>
        </div>
      </div>
      <div class="reject-box">
        <select data-id="${c.sid}">
          ${Object.entries(REASON_ZH).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}
        </select>
        <button class="btn btn-reject" data-act="reject-submit" data-id="${c.sid}" style="margin-left:8px">确认否决</button>
        <button class="btn btn-ghost" data-act="cancel" data-id="${c.sid}">取消</button>
      </div>
    </article>`;
}

function bindClaimActions(el) {
  el.querySelectorAll(".ev-chip[data-t]").forEach((chip) => {
    chip.addEventListener("click", () => {
      seekTo(parseFloat(chip.dataset.t) || 0);
      switchTab("timeline");
    });
  });
  el.querySelectorAll(".btn").forEach((b) => {
    b.onclick = async () => {
      const sid = b.dataset.id;
      const card = sid ? el.querySelector(`[data-sid="${sid}"]`) : null;
      const act = b.dataset.act;
      if (act === "jump") {
        seekTo(parseFloat(b.dataset.t) || 0);
        switchTab("timeline");
        return;
      }
      if (!card) return;
      if (act === "edit-open") {
        card.querySelector(".edit-box").style.display = "block";
        card.querySelector(".reject-box").style.display = "none";
      } else if (act === "reject-open") {
        card.querySelector(".reject-box").style.display = "block";
        card.querySelector(".edit-box").style.display = "none";
      } else if (act === "cancel") {
        card.querySelector(".edit-box").style.display = "none";
        card.querySelector(".reject-box").style.display = "none";
      } else if (act === "accept") {
        await submitFeedback(sid, "accept");
      } else if (act === "edit-submit") {
        const text = card.querySelector(".edit-box textarea").value.trim();
        if (!text) return toast("请输入改写内容");
        await submitFeedback(sid, "edit", { revised_text: text });
      } else if (act === "reject-submit") {
        const reason = card.querySelector(".reject-box select").value;
        await submitFeedback(sid, "reject", { reason });
      }
    };
  });
}

async function submitFeedback(claimId, action, extra = {}) {
  try {
    await api(`/api/sessions/${state.currentSession}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claim_id: claimId,
        action,
        subject_id: state.currentSubject,
        ...extra,
      }),
    });
    toast("已保存反馈");
    await loadTimeline();
    await loadProfile();
  } catch (e) {
    toast("操作失败：" + e.message);
  }
}

/* ---------------- Profile ---------------- */
async function loadProfile() {
  if (!state.currentSubject) return;
  try {
    const d = await api(`/api/subjects/${state.currentSubject}`);
    state.profile = d;
    renderProfile(d);
    renderFeedbackChart(d);
    renderExemplars(d);
  } catch (e) {
    $("#profile-summary").textContent = "暂无画像（完成一场会谈分析后生成）";
    $("#profile-metrics").innerHTML = "";
    $("#subject-sessions").innerHTML = `<div class="empty">暂无场次</div>`;
    $("#comparability-gate").textContent = "";
    $("#comparability-gate").className = "gate-banner";
  }
}

function renderProfile(d) {
  const p = d.profile || {};
  $("#profile-subject").textContent = d.subject_id || "";
  $("#profile-summary").textContent = d.summary_text || "";
  const b = p.baseline || {};
  const comp = p.comparability;
  const gated = comp == null || comp < 0.7;

  const gate = $("#comparability-gate");
  if (gated) {
    gate.className = "gate-banner warn";
    gate.textContent =
      `可比性 ${fmtPct(comp)} < 70%：按系统硬规则，不输出跨次比较结论。当前仅展示累积基线与场次清单，供人工参考。`;
  } else {
    gate.className = "gate-banner ok";
    gate.textContent =
      `可比性 ${fmtPct(comp)} ≥ 70%：质量指标达到跨次比较门槛（仍非诊断，仅辅助观察）。`;
  }

  $("#profile-metrics").innerHTML = `
    <div class="metric"><div class="metric-value">${p.n_sessions || 0}</div><div class="metric-label">会谈次数</div></div>
    <div class="metric${gated ? " warn" : ""}"><div class="metric-value">${fmtPct(comp)}</div><div class="metric-label">可比性</div></div>
    <div class="metric"><div class="metric-value">${fmtNum(b.v_mean)}</div><div class="metric-label">效价均值</div></div>
    <div class="metric"><div class="metric-value">${fmtNum(b.a_mean)}</div><div class="metric-label">唤醒均值</div></div>
    <div class="metric"><div class="metric-value">${p.exemplars_count || 0}</div><div class="metric-label">改写示例</div></div>`;

  // 优先用画像内 sessions 列表；API 也可能已过滤
  const allowed = new Set(p.sessions || []);
  let sessions = d.sessions || [];
  if (allowed.size) sessions = sessions.filter((s) => allowed.has(s.session_id));

  $("#subject-sessions").innerHTML = sessions.length
    ? sessions.map((s) => `
        <div class="session-row${s.session_id === state.currentSession ? " current" : ""}" data-s="${esc(s.session_id)}">
          <span><b>${esc(s.session_id)}</b></span>
          <span class="meta">${fmtTime(s.duration)} · ${s.n_events} 事件 · ${s.n_claims} 草稿</span>
        </div>`).join("")
    : `<div class="empty">暂无会话</div>`;

  $("#subject-sessions").querySelectorAll(".session-row").forEach((r) => {
    r.onclick = () => {
      $("#session-select").value = r.dataset.s;
      state.currentSession = r.dataset.s;
      state.voiceBase = null;
      loadTimeline();
      switchTab("timeline");
    };
  });
}

function renderFeedbackChart(d) {
  const fbe = $("#feedback-chart");
  if (typeof echarts === "undefined") {
    if (fbe) fbe.innerHTML = `<div class="empty">图表组件未加载</div>`;
    return;
  }
  if (!fbChart) fbChart = echarts.init(fbe);
  const fbs = state.sessionDetail?.feedback || [];
  const byReason = {};
  fbs.forEach((f) => { if (f.reason) byReason[f.reason] = (byReason[f.reason] || 0) + 1; });
  const keys = Object.keys(byReason);
  if (!keys.length) {
    fbChart.setOption({
      title: {
        text: "本场暂无否决反馈",
        subtext: d.subject_id || "",
        left: "center", top: "middle",
        textStyle: { color: "#6a7885", fontSize: 14 },
        subtextStyle: { color: "#6a7885", fontSize: 12 },
      },
    }, true);
    return;
  }
  fbChart.setOption({
    backgroundColor: "transparent",
    tooltip: {},
    grid: { left: 100, right: 24, top: 16, bottom: 20 },
    xAxis: { type: "value", axisLabel: { color: "#6a7885" }, splitLine: { lineStyle: { color: "#e4ebf0" } } },
    yAxis: {
      type: "category",
      data: keys.map((k) => REASON_ZH[k] || k),
      axisLabel: { color: "#6a7885" },
    },
    series: [{
      type: "bar",
      data: keys.map((k) => byReason[k]),
      barWidth: 14,
      itemStyle: { color: "#2f615c", borderRadius: [0, 6, 6, 0] },
      label: { show: true, position: "right", color: "#6a7885" },
    }],
  }, true);
}

function renderExemplars(d) {
  const ex = d.exemplars || [];
  const el = $("#exemplar-list");
  el.innerHTML = ex.length
    ? ex.map((x) => `
        <div class="exemplar-item">
          <div class="orig">原文：${esc(x.original)}</div>
          <div class="rev">改写：${esc(x.revised)}</div>
        </div>`).join("")
    : `<div class="empty">暂无改写示例（对草稿点击「改写」后写入 RAG）</div>`;
}

/* ---------------- Tabs ---------------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  if (name === "profile") loadProfile();
  if (name === "timeline") {
    setTimeout(() => {
      vaChart?.resize();
      sigVoiceChart?.resize();
      sigPolChart?.resize();
    }, 50);
  }
  if (name === "claims" || name === "profile") {
    setTimeout(() => fbChart?.resize(), 50);
  }
}

document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
document.querySelectorAll("#claim-filters .chip").forEach((c) => {
  c.onclick = () => {
    document.querySelectorAll("#claim-filters .chip").forEach((x) => x.classList.remove("active"));
    c.classList.add("active");
    state.claimFilter = c.dataset.filter;
    applyClaimFilter();
  };
});

init();

/* ---------------- Upload ---------------- */
let upFile = null;

function openModal() {
  $("#upload-modal").classList.remove("hidden");
  if (state.currentSubject) $("#up-subject").value = state.currentSubject;
}
function closeModal() {
  $("#upload-modal").classList.add("hidden");
}

$("#upload-btn").onclick = openModal;
$("#modal-close").onclick = closeModal;
$("#modal-cancel").onclick = closeModal;

const dropzone = $("#dropzone");
dropzone.onclick = () => $("#up-file").click();
$("#up-file").onchange = (e) => {
  upFile = e.target.files[0];
  if (upFile) {
    $("#dz-text").textContent = upFile.name;
    $("#file-info").textContent = `${upFile.name} · ${(upFile.size / 1024 / 1024).toFixed(1)} MB`;
  }
};
["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) {
    upFile = e.dataTransfer.files[0];
    $("#up-file").files = e.dataTransfer.files;
    $("#dz-text").textContent = upFile.name;
    $("#file-info").textContent = `${upFile.name} · ${(upFile.size / 1024 / 1024).toFixed(1)} MB`;
  }
});

$("#modal-submit").onclick = async () => {
  const subject = $("#up-subject").value.trim();
  const t0 = parseFloat($("#up-t0").value);
  const t1 = parseFloat($("#up-t1").value);
  if (!subject) return toast("请填写来访者 ID");
  if (!upFile) return toast("请选择视频文件");
  if (isNaN(t0) || isNaN(t1) || t1 <= t0) return toast("请填写有效的声音样本时间段（结束 > 起始）");

  closeModal();
  showProgress("正在上传并分析…");

  const fd = new FormData();
  fd.append("video", upFile);
  fd.append("subject_id", subject);
  fd.append("client_t0", t0);
  fd.append("client_t1", t1);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.detail || "上传失败");
    }
    const { task_id } = await res.json();
    pollTask(task_id);
  } catch (e) {
    hideProgress();
    toast("上传失败：" + e.message);
  }
};

function showProgress(text) {
  $("#progress-text").textContent = text;
  $("#progress-stages").innerHTML = STAGE_ORDER.map((s) =>
    `<li data-stage="${s}"><span>${STAGE_ZH[s]}</span><span></span></li>`
  ).join("");
  $("#progress-bar").classList.remove("hidden");
}
function hideProgress() {
  $("#progress-bar").classList.add("hidden");
}

function paintStages(current) {
  const idx = STAGE_ORDER.indexOf(current);
  $("#progress-stages")?.querySelectorAll("li").forEach((li) => {
    const i = STAGE_ORDER.indexOf(li.dataset.stage);
    li.classList.remove("done", "current");
    if (idx >= 0 && i < idx) li.classList.add("done");
    if (li.dataset.stage === current) li.classList.add("current");
  });
}

async function pollTask(taskId) {
  try {
    const t = await api(`/api/tasks/${taskId}`);
    if (t.status === "done") {
      hideProgress();
      toast("分析完成");
      await refreshOverview();
      if (t.session_id) {
        $("#session-select").value = t.session_id;
        state.currentSession = t.session_id;
        if (t.subject_id) {
          state.currentSubject = t.subject_id;
          $("#subject-select").value = t.subject_id;
        }
        await loadTimeline();
        switchTab("timeline");
      }
      return;
    }
    if (t.status === "failed") {
      hideProgress();
      toast("分析失败：" + (t.error || "未知错误"));
      return;
    }
    paintStages(t.stage);
    $("#progress-text").textContent = `${STAGE_ZH[t.stage] || t.stage} — ${t.detail || ""}`;
    setTimeout(() => pollTask(taskId), 1500);
  } catch (e) {
    hideProgress();
    toast("查询状态失败：" + e.message);
  }
}

async function refreshOverview() {
  const ov = await api("/api/overview");
  state.subjects = ov.subjects;
  state.sessions = ov.sessions;
  renderSelects();
}
