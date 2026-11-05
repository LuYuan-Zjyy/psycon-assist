"""会话 / 来访者目录约定，禁止各阶段手写相对路径字符串。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config


@dataclass(frozen=True)
class SessionPaths:
    """一场会谈的全部落盘路径。"""

    session_id: str

    @property
    def root(self) -> Path:
        return config.session_dir(self.session_id)

    # ---- 顶层元数据 ----
    @property
    def media_json(self) -> Path:
        return self.root / "media.json"

    @property
    def quality_json(self) -> Path:
        return self.root / "quality.json"

    # ---- S0 ----
    @property
    def norm_mp4(self) -> Path:
        return self.root / "raw" / "norm.mp4"

    @property
    def audio_wav(self) -> Path:
        return self.root / "audio" / "16k.wav"

    # ---- S1 ----
    @property
    def au_parquet(self) -> Path:
        return self.root / "face" / "au.parquet"

    @property
    def speaker_json(self) -> Path:
        return self.root / "speech" / "speaker.json"

    @property
    def voice_json(self) -> Path:
        return self.root / "speech" / "voice.json"

    @property
    def vad_json(self) -> Path:
        return self.root / "speech" / "vad.json"

    @property
    def transcript_json(self) -> Path:
        return self.root / "text" / "transcript.json"

    @property
    def text_feat_json(self) -> Path:
        return self.root / "text" / "text_feat.json"

    # ---- S2 / S3 ----
    @property
    def chunks_json(self) -> Path:
        return self.root / "chunks" / "chunks.json"

    @property
    def events_json(self) -> Path:
        return self.root / "events" / "events.json"

    @property
    def evidence_json(self) -> Path:
        return self.root / "events" / "evidence.json"

    # ---- S4 / S5 ----
    @property
    def claims_json(self) -> Path:
        return self.root / "agent" / "claims.json"

    @property
    def feedback_json(self) -> Path:
        return self.root / "feedback" / "feedback.json"

    def ensure_layout(self) -> None:
        for sub in ("raw", "audio", "face", "speech", "text", "chunks", "events", "agent", "feedback"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SubjectPaths:
    subject_id: str

    @property
    def root(self) -> Path:
        return config.subject_dir(self.subject_id)

    @property
    def profile_json(self) -> Path:
        return self.root / "profile.json"

    @property
    def exemplars_dir(self) -> Path:
        return self.root / "exemplars"

    def ensure_layout(self) -> None:
        self.exemplars_dir.mkdir(parents=True, exist_ok=True)
