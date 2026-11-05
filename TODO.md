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

## 四、待办（已全部拆成 Issue）

**任务以 GitHub Issues 为准**，本文件只做索引。认领、讨论、进度都在对应的 issue 里。

| 优先级 | Issue | 内容 | 预估 |
| :-- | :-- | :-- | :-- |
| **P0** | [#1](https://github.com/LuYuan-Zjyy/psycon-assist/issues/1) | 每条 severity 规则一个 fixture（5 类规则目前无回归测试） | 3–4h · 新人友好 |
| **P0** | [#2](https://github.com/LuYuan-Zjyy/psycon-assist/issues/2) | 上传改成有界队列，`PSY_MAX_CONCURRENCY` 限制并发 | 4–6h |
| **P1** | [#3](https://github.com/LuYuan-Zjyy/psycon-assist/issues/3) | `synth` / `risk` 的英文 prompt 集 | 6–8h |
| **P1** | [#4](https://github.com/LuYuan-Zjyy/psycon-assist/issues/4) | 时间线 UI：点事件跳转到视频对应秒数 | 8–12h |
| **P1** | [#5](https://github.com/LuYuan-Zjyy/psycon-assist/issues/5) | 单场报告导出 Markdown | 3–4h · 新人友好 |
| **P1** | [#6](https://github.com/LuYuan-Zjyy/psycon-assist/issues/6) | 第二个 ASR 引擎适配（验证信号层可插拔） | 6–8h |
| **P2** | [#7](https://github.com/LuYuan-Zjyy/psycon-assist/issues/7) | Dockerfile（CPU-only 部署） | 3–4h · 新人友好 |
| **P2** | [#8](https://github.com/LuYuan-Zjyy/psycon-assist/issues/8) | `data/` 迁移工具（`PSY_DATA_ROOT` 变更） | 2–3h |
| **P2** | [#9](https://github.com/LuYuan-Zjyy/psycon-assist/issues/9) | 前端中英切换 | 4–6h |
| **P2** | [#10](https://github.com/LuYuan-Zjyy/psycon-assist/issues/10) | 扩大 pytest 覆盖（切块 / severity / 反馈） | 4–6h |

### 已完成（不再开 issue）

- [x] `core/` + `stages/` 重构，全流程文件化落盘
- [x] 规则引擎决定 severity；LLM 仅 `synth` / `risk`
- [x] 反馈闭环（六原因码）+ exemplar 回流
- [x] 跨次画像 + 可比性阈值
- [x] Web UI（绑定 127.0.0.1）
- [x] 安全加固：ID 校验、上传体积上限、扩展名白名单
- [x] 错误信息脱敏（`_sanitize_error`）
- [x] 22 个 ID 校验单测 + CI `pytest` 步骤

> 标签约定：`p0` / `p1` / `p2` 优先级，`good first issue` 新人可上手，`help wanted` 需要外部协助。

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
