"""S1C 中文情感极性。用 transformers 现成中文情感模型；缺失则降级标不可得。

极性必须来自模型，禁止用关键词规则替代（硬规则 19 / 不变量）。
"""
from __future__ import annotations


def available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


_SENTIMENT_MODEL = "uer/roberta-base-finetuned-jd-binary-chinese"


class _Sentiment:
    def __init__(self):
        from transformers import pipeline

        self.pipe = pipeline("text-classification", model=_SENTIMENT_MODEL)

    def polarity(self, texts: list[str]) -> list[float]:
        """返回 [-1, 1]；label 0=负/1=正。"""
        out = []
        for r in self.pipe(texts, truncation=True, max_length=512):
            label = r.get("label", "1")
            score = r.get("score", 0.5)
            if str(label) in ("0", "LABEL_0", "negative", "NEGATIVE"):
                pol = -score
            else:
                pol = score
            out.append(round(float(pol), 4))
        return out


_instance = None


def polarity(texts: list[str]) -> list[float | None]:
    global _instance
    if not available():
        return [None] * len(texts)
    try:
        if _instance is None:
            _instance = _Sentiment()
        return _instance.polarity(texts)
    except Exception:
        return [None] * len(texts)
