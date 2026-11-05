"""统一异常体系，便于队友定位失败阶段。"""
from __future__ import annotations


class PsyConError(Exception):
    """包内所有可预期错误的基类。"""


class PipelineError(PsyConError):
    """流水线编排层错误。"""


class StageError(PsyConError):
    """某阶段失败；带 stage 名方便日志与 API 回传。"""

    def __init__(self, stage: str, message: str, *, cause: BaseException | None = None):
        self.stage = stage
        self.cause = cause
        super().__init__(f"[{stage}] {message}")


class EngineError(PsyConError):
    """外部引擎（ffmpeg / OpenFace / HF 模型等）调用失败。"""

    def __init__(self, engine: str, message: str, *, cause: BaseException | None = None):
        self.engine = engine
        self.cause = cause
        super().__init__(f"<{engine}> {message}")


class ContractError(PsyConError):
    """阶段产物违反契约（时间轴、覆盖、evidence 等）。"""
