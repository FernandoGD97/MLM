"""Entity discovery from raw text only."""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re

@dataclass(frozen=True)
class Span:
    start_char: int; end_char: int; text: str
    label: str | None = None; score: float | None = None

class EntityProvider(ABC):
    @abstractmethod
    def find_entities(self, text: str) -> list[Span]: ...

class NoneEntityProvider(EntityProvider):
    def find_entities(self, text): return []

class GazetteerEntityProvider(EntityProvider):
    def __init__(self, terms, case_sensitive=False):
        flags = 0 if case_sensitive else re.IGNORECASE
        escaped = sorted((re.escape(x.strip()) for x in terms if x.strip()), key=len, reverse=True)
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", flags) if escaped else None
    def find_entities(self, text):
        return [] if self.pattern is None else [Span(m.start(), m.end(), m.group()) for m in self.pattern.finditer(text)]

class HuggingFaceNERProvider(EntityProvider):
    def __init__(self, checkpoint: str, **pipeline_kwargs):
        from transformers import pipeline
        self.pipe = pipeline("token-classification", model=checkpoint, aggregation_strategy="simple", **pipeline_kwargs)
    def find_entities(self, text):
        return [Span(x["start"], x["end"], text[x["start"]:x["end"]], x.get("entity_group"), x.get("score")) for x in self.pipe(text)]

def project_spans(spans, offsets):
    """Return whole subword index groups intersecting each character span."""
    return [[i for i, (a, b) in enumerate(offsets) if b > a and a < s.end_char and b > s.start_char] for s in spans]
