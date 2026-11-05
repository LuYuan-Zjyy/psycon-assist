# 项目流水线与待办

> 本文档回答两个问题：**这个项目的数据是怎么流的**，以及**接下来该做什么**。
> 对应版本：v0.2（`core/` + `stages/` 重构后）

---

## 一、一句话定位

把一场会谈录像拆成**可回跳、可举证**的观察草稿：**不诊断、不推断内心、缺数据就说不可得**。
系统的价值不在"写得像人"，而在**每条结论都能指回证据，且模型无权改严重程度**。

---

## 二、流水线（S0 → S6）

```
录像 ──S0(归一化/抽音轨)──▶ 原始信号
                              ├─S1A ASR──────▶ 逐句转写
                              ├─S1B OpenFace──▶ 面部动作单元(AU)
                              └─S1C 声学/分离──▶ 声学特征 + 说话人角色
                                        │
                                   S2 切块（时间对齐，算 VA）
                                        │
                                   S3 事件（规则定 severity，绑 evidence）
                                        │
                                   S4 Agent（LangGraph，23 条硬规则，LLM 仅两处）
                                        │
                                   S5 反馈（accept/edit/reject + 六原因码）
                                        │
                                   S6 画像（跨次，可比性 <0.7 不下结论）
```

### 各阶段契约

| 阶段 | 入口文件 | 输入 | 输出（落盘） | 失败时 |
|---|---|---|---|---|
| **S0** 接入 | `stages/s0_ingest.py` | 上传视频、`client_sample=[t0,t1]` | `raw/norm.mp4`、`audio/16k.wav`、`media.json` | 缺 ffmpeg → 显式报错，不静默降级 |
| **S1A** 文本 | `stages/s1_text.py` | `audio/16k.wav` | `text/transcript.json`、`text/text_feat.json` | 缺 ASR → `null` + `quality.json` 标注 |
| **S1B** 视觉 | `stages/s1_visual.py` | `raw/norm.mp4` | `face/au.parquet` | 缺 OpenFace → `degrade` |
| **S1C** 语音 | `stages/s1_voice.py` | `audio/16k.wav` + 来访者样本段 | `speech/speaker.json`、`voice.json`、`vad.json` | 缺 pyannote/librosa → `degrade` |
| **S2** 切块 | `stages/s2_chunk.py` | 上述三类信号 | `chunks/chunks.json`（每块含 v/a、异常标记） | 信号缺失 → 该维度记 `null`，不补 0 |
| **S3** 事件 | `stages/s3_events.py` | chunks | `events/events.json`、`events/evidence.json` | 无异常 → 空列表（合法结果） |
| **S4** 编排 | `stages/s4_agent.py` | events + 证据 | `agent/claims.json` | LLM 不可用 → 仅规则产出，标注 `llm_unavailable` |
| **S5** 反馈 | `stages/s5_feedback.py` | claim + 咨询师动作 | `feedback/feedback.json`、`subjects/<P>/exemplars/` | 原因码非法 → 拒绝写入 |
| **S6** 画像 | `stages/s6_profile.py` | 多次 session | `subjects/<P>/profile.json` | 可比性不足 → 只写基线，不下跨次结论 |

### 目录约定

```
data/
├── S<yyyymmdd>-<seq>/          # 一场会谈
│   ├── media.json  quality.json
│   ├── raw/ audio/ face/ speech/ text/
│   ├── chunks/ events/ agent/ feedback/
└── subjects/P-0001/
    ├── profile.json            # 跨次画像
    └── exemplars/              # 反馈回流样例
```

路径一律走 `core/paths.py`（`SessionPaths` / `SubjectPaths`），**禁止各阶段手拼字符串**；所有 ID 在 `config.session_dir()` / `config.subject_dir()` 里做格式校验，防路径穿越。

---

## 三、当前已完成（v0.2）

- [x] `core/` 基建：路径、原子读写、错误类型、日志、模型
- [x] `stages/` 拆分 + 全流程文件化落盘
- [x] 合成数据跑通全链路（`cli demo`，无需视频/重模型）
- [x] 规则引擎决定 severity；LLM 仅 `synth` / `risk`
- [x] 反馈闭环（六原因码）+ exemplar 回流
- [x] 跨次画像 + 可比性阈值
- [x] Web UI（FastAPI + 静态前端，绑定 127.0.0.1）
- [x] 安全加固：ID 校验、上传体积上限、扩展名白名单、上传仅存 `data/uploads`

---

## 四、待办（按优先级）

### P0 — 正确性与安全（先做）

- [ ] **每条 severity 规则一个 fixture**
      现在 5 类规则（`va_shift` / `risk` / `silence` / `incongruence` / `low_activation`）没有回归测试。
      验收：`tests/fixtures/` 每条规则一个最小 chunk；改规则表会让测试红。
- [ ] **上传限速/并发上限**
      当前每个任务一个线程，无并发上限；大视频并发会打满 CPU/磁盘。
      验收：超过 N 个并发任务时返回 429；`PSY_MAX_CONCURRENCY` 可配。
- [ ] **错误信息脱敏**
      `TASKS[task]["error"]` 会把异常原文（含绝对路径）返回给前端。
      验收：前端只看到"处理失败（见服务端日志）"，细节进日志。

### P1 — 能力

- [ ] **英文 prompt 集**
      `synth` / `risk` 两个节点目前只有中文模板，英文会谈质量不稳定。
      验收：同一段英文证据，英文模板的 claim 合规率不低于中文模板。
- [ ] **时间线 UI**
      前端按时间轴回放：点 event 跳到视频对应秒数，高亮对应 transcript。
      验收：任一 event 可一键定位；视频 seek 不黑屏（已用 faststart 保证）。
- [ ] **报告导出（Markdown / PDF）**
      给督导用的单场报告：事件表 + 证据 + 咨询师反馈。
      验收：`GET /api/sessions/<id>/report.md` 返回完整报告。
- [ ] **第二个 ASR 引擎适配**
      证明信号层可插拔（现在只有 faster-whisper 路径）。
      验收：换引擎不改 S2 之后任何代码。

### P2 — 工程与体验

- [ ] **`pytest` 骨架 + CI 跑测试**（当前 CI 只有 lint + demo smoke）
- [ ] **Dockerfile**（CPU-only 镜像，重模型挂载）
- [ ] **`data/` 迁移工具**（`PSY_DATA_ROOT` 变更时批量搬目录）
- [ ] **前端国际化**（中/英切换）

---

## 五、明确不做

- ❌ 不做诊断、不做风险分级以外的临床判断、不给用药或治疗建议
- ❌ 不推断内心状态（"他在回避"这类表述被规则直接拦截）
- ❌ 不做跨来访者比较
- ❌ 不上传/不内置任何真实会谈数据

---

## 六、怎么认领

1. 在 [Issues](https://github.com/LuYuan-Zjyy/psycon-assist/issues) 里找 `good first issue`；
2. 先读 [CONTRIBUTING.md](CONTRIBUTING.md)，改动前在 issue 里说一声；
3. PR 里请附：改动的规则编号（如涉及）、fixture、`python -m psyconassist.cli demo` 的输出变化。
