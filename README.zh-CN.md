# PsyConAssist · 心理咨询会谈分析助手

**一个"可审计"的会谈分析助手**：每条观察都绑定证据，严重程度由规则决定而不是模型，模型只在两个白名单节点被允许说话。

> ⚠️ 研究原型，**不是**诊断工具，不能替代临床判断。

## 它解决什么

"AI 心理咨询"的 demo 大多在追求"听起来共情"。这个项目追求的是**不越界**：

1. **可追溯**：每个 event 都能指回原文 / 画面 / 音频片段；
2. **克制**：允许系统说"证据不足"，而不是硬猜。

## 流水线

| 阶段 | 模块 | 作用 |
| :-- | :-- | :-- |
| S0 接入 | `s0_ingest.py` | 视频归一化、抽音轨、定位来访者声音样本 |
| S1 信号 | `s1_text.py` / `s1_visual.py` / `s1_voice.py` | ASR 转写、OpenFace 面部动作单元、声学特征 + 说话人分离 |
| S2 切分 | `s2_chunk.py` | 按时间对齐切成 chunk |
| S3 事件 | `s3_events.py` | 异常 chunk → 带证据的 event，**严重程度由规则表决定** |
| S4 编排 | `s4_agent.py` | LangGraph 12 节点 + **23 条硬规则**，LLM 只能在 `synth` / `risk` 两处调用 |
| S5 反馈 | `s5_feedback.py` | 咨询师 accept / edit / reject + 六个固定原因码，回流 exemplar |
| S6 画像 | `s6_profile.py` | 跨次画像，可比性 `< 0.7` 时拒绝下跨次结论 |

## 快速开始

```bash
git clone https://github.com/LuYuan-Zjyy/psycon-assist.git
cd psycon-assist
pip install -r requirements.txt

# 合成数据跑全流程（不需要视频、不需要重模型）
python -m psyconassist.cli demo --subject P-0001 --duration 3000

# Web 界面
python -m psyconassist.webapp      # http://localhost:8000
```

真实视频（需要 ffmpeg 与可选引擎）：

```bash
python -m psyconassist.cli run --video session.mp4 --subject P-0001 --client-sample 12.4 15.4
```

## 设计原则

- **严重程度不是模型输出**：`s3_events.py` 的规则表决定 high / medium / low。
- **LLM 只有两扇门**：只有 `synth` / `risk` 节点可以调用模型，其余全确定性。
- **删句优先于重写**：校验失败直接删掉该句，而不是让模型重写（硬规则 16）。
- **反馈结构化**：六个原因码，不允许自由文本，保证 exemplar 可检索。
- **跨次比较需要可比性**：可比性低于 0.7 时拒绝下跨次结论。

## 贡献

欢迎 issue / PR，先读 [CONTRIBUTING.md](CONTRIBUTING.md)。请勿提交任何真实会谈数据或 API key。

## 许可

[MIT](LICENSE) © 2026 LuYuan-Zjyy
