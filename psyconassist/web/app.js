/* PsyConAssist 前端逻辑（原生 JS + ECharts） */

const TYPE_ZH = {
  va_shift: "效价突变",
  silence: "沉默",
  risk: "风险词",
  incongruence: "言行不一致",
  low_activation: "低激活",
  normal: "正常",
};

const SEV_ZH = { high: "高", medium: "中", low: "低" };

const METRIC_ZH = {
  "va.v": "效价V",
  "va.a": "唤醒A",
  polarity: "极性",
  anomaly_types: "异常类型",
  "f0_med": "基频F0",
  "loudness_rel": "相对响度",
};

const REASON_ZH = {
  insufficient_evidence: "证据不足",
  over_inference: "过度推断",
  wording: "措辞不专业",
  missing_info: "漏了重要信息",
  time_misaligned: "定位不对",
  other: "其他",
};

const state = {
  subjects: [],
  sessions: [],
  currentSubject: null,
  currentSession: null,
  sessionDetail: null,
  profile: null,
  claimFilter: "all",
};

let vaChart = null;
let fbChart = null;
let sigVoiceChart = null;
let sigPolChart = null;

const $ = (s) => document.querySelector(s);

function fmtTime(sec) {
  if (sec == null) return "--:--";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 2200);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e.detail || res.statusText);
  }
  return res.json();
}

/* ---------------- 初始化 ---------------- */
async function init() {
  const ov = await api("/api/overview");
  state.subjects = ov.subjects;
  state.sessions = ov.sessions;
  renderSelects();

  if (state.sessions.length) {
    // 默认选中最新一次会谈（排序升序，取最后一个）
    state.currentSession = state.sessions[state.sessions.length - 1].session_id;
    const sesEl = $("#session-select");
    if (sesEl) sesEl.value = state.currentSession;
  }
  const s = state.sessions.find((x) => x.session_id === state.currentSession);
  if (s) {
    const profileSubject = state.subjects[0]?.subject_id;
    if (profileSubject) state.currentSubject = profileSubject;
  }
  await refreshAll();
}

function renderSelects() {
  const ss = $("#subject-select");
  ss.innerHTML = state.subjects.length
    ? state.subjects.map((p) => `<option value="${p.subject_id}">${p.subject_id}（${p.n_sessions} 次会谈）</option>`).join("")
    : `<option>无数据</option>`;

  const ses = $("#session-select");
  ses.innerHTML = state.sessions.length
    ? state.sessions.map((s) => `<option value="${s.session_id}">${s.session_id} · ${fmtTime(s.duration)}</option>`).join("")
    : `<option>无数据</option>`;

  ss.onchange = async () => {
    state.currentSubject = ss.value;
    await loadProfile();
    loadTimeline();
  };
  ses.onchange = async () => {
    state.currentSession = ses.value;
    await loadTimeline();
  };
}

async function refreshAll() {
  await loadTimeline();
  await loadProfile();
}

/* ---------------- 时间轴视图 ---------------- */
async function loadTimeline() {
  if (!state.currentSession) {
    $(".empty")?.remove();
    return;
  }
  const d = await api(`/api/sessions/${state.currentSession}`);
  state.sessionDetail = d;
  // 每块独立渲染，任何一块失败都不影响其余（尤其图表/视频）
  const safe = (fn, arg) => { try { fn(arg); } catch (e) { console.error("render error", e); } };
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
  const nEvt = (d.events || []).length;
  const nClaim = (d.claims || []).length;
  const nUtt = (d.transcript || []).length;
  const parts = [];
  parts.push(`转写 ${nUtt} 句`);
  parts.push(`事件 ${nEvt}`);
  parts.push(`草稿 ${nClaim}`);
  if (q.asr_ok === false) parts.push("ASR 失败");
  if (q.diarization_ok === false) parts.push("说话人分离失败");
  if (typeof q.face_cov === "number") parts.push(`人脸覆盖 ${Math.round(q.face_cov * 100)}%`);
  if (degrade.length) parts.push(`降级：${degrade.join(", ")}`);
  el.textContent = parts.join(" · ");
  el.classList.toggle("warn", degrade.length > 0 || nEvt === 0);
  el.classList.remove("hidden");
}

function renderTranscript(d) {
  const el = $("#transcript-list");
  const meta = $("#transcript-meta");
  const utts = d.transcript || [];
  if (meta) meta.textContent = utts.length ? `${utts.length} 条` : "";
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

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* ---------------- 原始信号监控（S1 三路特征可视化） ---------------- */
function renderSignals(d) {
  const voice = d.voice || [];
  const feat = d.text_feat || [];
  const client = voice.filter((s) => s.role === "client");
  const duration = d.media.duration || 0;

  // 语音声学（来访者 F0 / 响度）
  const vEl = $("#sig-voice-chart");
  if (vEl && typeof echarts !== "undefined") {
    if (!sigVoiceChart) sigVoiceChart = echarts.init(vEl);
    const f0 = client.map((s) => [s.t0, s.f0_med ?? null]);
    const loud = client.map((s) => [s.t0, s.loudness_rel ?? null]);
    if (!client.length) {
      sigVoiceChart.setOption({ title: { text: "无来访者语音声学数据", left: "center", top: "middle", textStyle: { color: "#74837f", fontSize: 13 } } }, true);
    } else {
      sigVoiceChart.setOption({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", valueFormatter: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
        legend: { data: ["基频 F0 (Hz)", "相对响度 (dB)"], textStyle: { color: "#74837f" } },
        grid: { left: 44, right: 44, top: 34, bottom: 28 },
        xAxis: { type: "value", min: 0, max: duration, axisLabel: { formatter: (v) => fmtTime(v), color: "#74837f" }, splitLine: { lineStyle: { color: "#eef1f0" } } },
        yAxis: [
          { type: "value", name: "Hz", axisLabel: { color: "#74837f" }, splitLine: { lineStyle: { color: "#eef1f0" } } },
          { type: "value", name: "dB", axisLabel: { color: "#74837f" }, splitLine: { show: false } },
        ],
        series: [
          { name: "基频 F0 (Hz)", type: "line", data: f0, connectNulls: true, symbol: "none", lineStyle: { color: "#4a90c4", width: 1.6 } },
          { name: "相对响度 (dB)", type: "line", yAxisIndex: 1, data: loud, connectNulls: true, symbol: "none", lineStyle: { color: "#e0a33c", width: 1.4 } },
        ],
      }, true);
    }
  }

  // 文本极性 + 风险标记
  const pEl = $("#sig-pol-chart");
  if (pEl && typeof echarts !== "undefined") {
    if (!sigPolChart) sigPolChart = echarts.init(pEl);
    const pol = feat.map((s) => [s.t0, s.polarity ?? null]);
    const riskPts = feat.filter((s) => (s.risk_hits || []).length).map((s) => [s.t0, s.polarity ?? 0]);
    if (!feat.length) {
      sigPolChart.setOption({ title: { text: "无文本极性数据", left: "center", top: "middle", textStyle: { color: "#74837f", fontSize: 13 } } }, true);
    } else {
      sigPolChart.setOption({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", valueFormatter: (v) => (v == null ? "—" : Number(v).toFixed(3)) },
        legend: { data: ["极性", "风险词命中"], textStyle: { color: "#74837f" } },
        grid: { left: 44, right: 20, top: 34, bottom: 28 },
        xAxis: { type: "value", min: 0, max: duration, axisLabel: { formatter: (v) => fmtTime(v), color: "#74837f" }, splitLine: { lineStyle: { color: "#eef1f0" } } },
        yAxis: { type: "value", min: -1.2, max: 1.2, axisLabel: { color: "#74837f" }, splitLine: { lineStyle: { color: "#eef1f0" } } },
        series: [
          { name: "极性", type: "line", data: pol, connectNulls: true, symbol: "none", step: "end", lineStyle: { color: "#1f9e8f", width: 1.6 } },
          { name: "风险词命中", type: "scatter", data: riskPts, symbolSize: 9, itemStyle: { color: "#d64545" } },
        ],
      }, true);
    }
  }
}

function renderVideo(d) {
  const video = $("#video");
  const ph = $("#video-placeholder");
  $("#video-meta").textContent = d.media.duration
    ? `${fmtTime(d.media.duration)} · ${d.media.fps} fps · ${d.media.width}×${d.media.height}`
    : "";
  if (d.has_video) {
    video.src = `/media/${d.session_id}/norm.mp4`;
    ph.classList.add("hidden");
    video.style.display = "block";
  } else {
    video.style.display = "none";
    ph.classList.remove("hidden");
  }
}

function renderVAChart(d) {
  const el = $("#va-chart");
  if (typeof echarts === "undefined") {
    el.innerHTML = `<div class="empty">图表组件未加载（ECharts）</div>`;
    return;
  }
  if (!vaChart) vaChart = echarts.init(el);
  const t = d.timeline;
  const times = t.map((x) => x.t1);
  const v = t.map((x) => x.v);
  const a = t.map((x) => x.a);

  // 异常段 markArea
  const areas = [];
  for (const c of t) {
    if (c.is_anomaly) {
      areas.push([{ xAxis: c.t0, itemStyle: { color: "rgba(214,69,69,0.12)" } },
                  { xAxis: c.t1 }]);
    }
  }

  const option = {
    backgroundColor: "transparent",
    grid: { left: 44, right: 16, top: 24, bottom: 28 },
    tooltip: {
      trigger: "axis",
      formatter: (params) => {
        const p = params[0];
        const item = t.find((x) => Math.abs(x.t1 - p.axisValue) < 1e-6);
        let s = `${fmtTime(p.axisValue)}<br/>`;
        params.forEach((pp) => {
          if (pp.value != null) s += `${pp.marker} ${pp.seriesName}: ${Number(pp.value).toFixed(3)}<br/>`;
        });
        if (item?.is_anomaly) s += `<b style="color:#d64545">⚠ ${TYPE_ZH[item.anomaly_type]}</b>`;
        return s;
      },
    },
    xAxis: {
      type: "value", min: 0, max: d.media.duration,
      axisLabel: { formatter: (v) => fmtTime(v), color: "#74837f" },
      splitLine: { lineStyle: { color: "#eef1f0" } },
    },
    yAxis: {
      type: "value", min: -1.2, max: 1.2,
      axisLabel: { color: "#74837f" },
      splitLine: { lineStyle: { color: "#eef1f0" } },
    },
    series: [
      { name: "效价", type: "line", data: v, step: "end", symbol: "none",
        lineStyle: { color: "#1f9e8f", width: 1.8 },
        markArea: { silent: true, data: areas } },
      { name: "唤醒", type: "line", data: a, step: "end", symbol: "none",
        lineStyle: { color: "#e0a33c", width: 1.4, opacity: .85 } },
    ],
  };
  vaChart.setOption(option, true);
  vaChart.off("click");
  vaChart.on("click", (p) => {
    if (p.value != null && p.seriesType === "line") {
      seekTo(p.value);
    }
  });
  window.addEventListener("resize", () => vaChart.resize());
}

function renderEvents(d) {
  const el = $("#event-list");
  if (!d.events.length) {
    el.innerHTML = `<div class="empty">本场未检出事件</div>`;
    return;
  }
  el.innerHTML = d.events.map((e) => `
    <div class="event-item" data-t="${e.t0}">
      <span class="event-time">${fmtTime(e.t0)}–${fmtTime(e.t1)}</span>
      <div class="event-body">
        <div class="event-type">${TYPE_ZH[e.type] || e.type}</div>
        <div class="event-desc">证据 ${e.evidence_ids.length} 条</div>
      </div>
      <span class="sev sev-${e.severity}">${SEV_ZH[e.severity] || e.severity}</span>
    </div>`).join("");
  el.querySelectorAll(".event-item").forEach((it) => {
    it.onclick = () => seekTo(parseFloat(it.dataset.t));
  });
}

function seekTo(sec) {
  const video = $("#video");
  if (state.sessionDetail?.has_video) {
    video.currentTime = sec;
    video.play().catch(() => {});
  } else {
    toast(`跳转到 ${fmtTime(sec)}（无真实视频，已用轨迹定位）`);
  }
}

/* ---------------- 草稿复核视图 ---------------- */
function renderClaims(d) {
  state.sessionDetail = d;
  const claims = d.claims || [];
  const pending = claims.filter((c) => c.state === "pending").length;
  $("#claims-badge").textContent = pending || "";
  $("#claims-badge").style.display = pending ? "inline-block" : "none";

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
  el.innerHTML = claims.map((c) => claimCard(c)).join("");
  bindClaimActions(el);
}

function claimCard(c) {
  const stateZh = { pending: "待复核", accept: "已采纳", edit: "已改写", reject: "已否决" }[c.state] || c.state;
  const original = c.original_text ? `<div class="claim-original">原文：${esc(c.original_text)}</div>` : "";

  // 证据明细
  const evMap = {};
  (state.sessionDetail?.evidence || []).forEach((e) => { evMap[e.evidence_id] = e; });
  const evChips = (c.evidence_ids || []).map((id) => {
    const e = evMap[id];
    if (!e) return `<span class="ev-chip muted">${esc(id)}</span>`;
    const val = typeof e.value === "number" ? Number(e.value).toFixed(3) : esc(String(e.value ?? ""));
    return `<span class="ev-chip">${METRIC_ZH[e.metric] || e.metric} = <b>${val}</b> <span class="ev-t">@${fmtTime(e.t0)}</span></span>`;
  }).join("");

  let actions = "";
  if (c.state === "pending") {
    actions = `
      <button class="btn btn-accept" data-act="accept" data-id="${c.sid}">采纳</button>
      <button class="btn btn-edit" data-act="edit-open" data-id="${c.sid}">改写</button>
      <button class="btn btn-reject" data-act="reject-open" data-id="${c.sid}">否决</button>`;
  }
  return `
    <div class="claim-card" data-sid="${c.sid}">
      <div class="claim-head">
        <span class="claim-state st-${c.state}">${stateZh}</span>
        <span class="muted">${c.sid}</span>
        <span class="claim-type">${TYPE_ZH[c.event_type] || c.event_type || "事件"}</span>
      </div>
      ${original}
      <div class="claim-text">${esc(c.text)}</div>
      <div class="claim-evidence">${evChips || '<span class="muted">无证据</span>'}</div>
      <div class="claim-actions">${actions}</div>
      <div class="edit-box">
        <textarea placeholder="改写后的表述…">${esc(c.text)}</textarea>
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
    </div>`;
}

function bindClaimActions(el) {
  el.querySelectorAll(".btn").forEach((b) => {
    b.onclick = async () => {
      const sid = b.dataset.id;
      const card = el.querySelector(`[data-sid="${sid}"]`);
      const act = b.dataset.act;
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
        claim_id: claimId, action, subject_id: state.currentSubject,
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

/* ---------------- 画像视图 ---------------- */
async function loadProfile() {
  if (!state.currentSubject) return;
  try {
    const d = await api(`/api/subjects/${state.currentSubject}`);
    state.profile = d;
    renderProfile(d);
    renderFeedbackChart(d);
    renderExemplars(d);
  } catch (e) { /* 无画像 */ }
}

function renderProfile(d) {
  const p = d.profile;
  $("#profile-summary").textContent = d.summary_text || "";
  const b = p.baseline || {};
  $("#profile-metrics").innerHTML = `
    <div class="metric"><div class="metric-value">${p.n_sessions || 0}</div><div class="metric-label">会谈次数</div></div>
    <div class="metric"><div class="metric-value">${fmtPct(p.comparability)}</div><div class="metric-label">可比性</div></div>
    <div class="metric"><div class="metric-value">${fmtNum(b.v_mean)}</div><div class="metric-label">效价均值</div></div>
    <div class="metric"><div class="metric-value">${fmtNum(b.a_mean)}</div><div class="metric-label">唤醒均值</div></div>
    <div class="metric"><div class="metric-value">${p.exemplars_count || 0}</div><div class="metric-label">采纳示例</div></div>`;

  const sessions = d.sessions || [];
  $("#subject-sessions").innerHTML = sessions.length
    ? sessions.map((s) => `
        <div class="session-row" data-s="${s.session_id}">
          <span><b>${s.session_id}</b></span>
          <span class="muted">${fmtTime(s.duration)} · ${s.n_events} 事件 · ${s.n_claims} 草稿</span>
        </div>`).join("")
    : `<div class="empty">暂无会话</div>`;
  $("#subject-sessions").querySelectorAll(".session-row").forEach((r) => {
    r.onclick = () => {
      $("#session-select").value = r.dataset.s;
      state.currentSession = r.dataset.s;
      loadTimeline();
      switchTab("timeline");
    };
  });
}

function renderFeedbackChart(d) {
  const fbe = $("#feedback-chart");
  if (typeof echarts === "undefined") {
    if (fbe) fbe.innerHTML = `<div class="empty">图表组件未加载（ECharts）</div>`;
    return;
  }
  if (!fbChart) fbChart = echarts.init(fbe);
  // 汇总所有 session 的反馈（简化：本 subject 的 sessions 的 feedback）
  // 这里从当前 session detail 的 feedback 拿不到全部，用 profile 外层的 sessions 概要
  // 退化为展示当前 session 反馈（若有），否则空态
  const s = state.sessionDetail;
  const fbs = s?.feedback || [];
  const byReason = {};
  fbs.forEach((f) => { if (f.reason) byReason[f.reason] = (byReason[f.reason] || 0) + 1; });
  const keys = Object.keys(byReason);
  if (!keys.length) {
    fbChart.setOption({ title: { text: "暂无反馈", left: "center", top: "middle", textStyle: { color: "#74837f", fontSize: 14 } } }, true);
    return;
  }
  fbChart.setOption({
    backgroundColor: "transparent",
    tooltip: {},
    grid: { left: 90, right: 20, top: 20, bottom: 20 },
    xAxis: { type: "value", axisLabel: { color: "#74837f" }, splitLine: { lineStyle: { color: "#eef1f0" } } },
    yAxis: { type: "category", data: keys.map((k) => REASON_ZH[k] || k), axisLabel: { color: "#74837f" } },
    series: [{ type: "bar", data: keys.map((k) => byReason[k]), barWidth: 16,
      itemStyle: { color: "#1f9e8f", borderRadius: [0, 8, 8, 0] }, label: { show: true, position: "right", color: "#74837f" } }],
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
    : `<div class="empty">暂无改写示例（对草稿点击「改写」后生成）</div>`;
}

/* ---------------- Tab ---------------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  if (name === "profile") loadProfile();
  if (name === "timeline") setTimeout(() => {
    vaChart?.resize();
    sigVoiceChart?.resize();
    sigPolChart?.resize();
  }, 50);
}

/* ---------------- 工具 ---------------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function fmtPct(v) { return v == null ? "—" : `${Math.round(v * 100)}%`; }
function fmtNum(v) { return v == null ? "—" : Number(v).toFixed(3); }

/* ---------------- 事件绑定 ---------------- */
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

/* ---------------- 上传 ---------------- */
let upFile = null;

function openModal() {
  $("#upload-modal").classList.remove("hidden");
  // 默认填入当前选中的来访者
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
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
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
  showProgress("正在上传并分析…（可能需要数分钟）");

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
  $("#progress-bar").classList.remove("hidden");
}
function hideProgress() {
  $("#progress-bar").classList.add("hidden");
}

const STAGE_ZH = {
  pending: "等待处理",
  s0: "接入与归一化",
  s1: "三路特征提取",
  s2: "融合与异常切片",
  s3: "指标与事件",
  s4: "Agent 编排（LangGraph）",
  s5: "反馈闭环",
  s6: "画像更新",
};

async function pollTask(taskId) {
  try {
    const t = await api(`/api/tasks/${taskId}`);
    if (t.status === "done") {
      hideProgress();
      toast("分析完成！");
      await refreshOverview();
      // 自动切换到新 session
      if (t.session_id) {
        $("#session-select").value = t.session_id;
        state.currentSession = t.session_id;
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
    // 进行中
    $("#progress-text").textContent =
      `${STAGE_ZH[t.stage] || t.stage} — ${t.detail || ""}`;
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
