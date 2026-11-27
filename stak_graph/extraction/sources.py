"""Source metadata and character-preserving chunking."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any
from urllib.parse import urlsplit


def utc_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Timestamps must be ISO-8601 strings with a timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source_url: str
    published_at: str
    known_at: str
    title: str = ""
    source_type: str = "news"
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        for key in ("document_id", "text", "source_url"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty string")
        parsed_url = urlsplit(self.source_url)
        if parsed_url.scheme not in ("https", "http") or not parsed_url.netloc:
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        if not isinstance(self.title, str) or not isinstance(self.source_type, str):
            raise ValueError("title and source_type must be strings")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be boolean")
        for key in ("published_at", "known_at"):
            object.__setattr__(self, key, utc_timestamp(getattr(self, key)))
        if datetime.fromisoformat(self.known_at.replace("Z", "+00:00")) < datetime.fromisoformat(self.published_at.replace("Z", "+00:00")):
            raise ValueError("known_at cannot precede published_at")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Document":
        if not isinstance(payload, dict):
            raise ValueError("Each input record must be a JSON object")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError(f"Invalid document fields: {exc}") from exc

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id, "source_url": self.source_url,
            "published_at": self.published_at, "known_at": self.known_at,
            "source_hash": self.source_hash, "title": self.title,
            "source_type": self.source_type, "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str


def chunk_document(document: Document, max_chars: int = 6000, overlap: int = 500) -> list[Chunk]:
    if max_chars < 100 or overlap < 0 or overlap >= max_chars // 2:
        raise ValueError("max_chars must be >=100; overlap must be nonnegative and less than half max_chars")
    chunks, start = [], 0
    while start < len(document.text):
        end = min(start + max_chars, len(document.text))
        if end < len(document.text):
            minimum = start + max_chars // 2
            boundary = document.text.rfind("\n\n", minimum, end)
            if boundary < minimum:
                boundary = document.text.rfind(" ", minimum, end)
            if boundary >= minimum:
                end = boundary
        chunks.append(Chunk(len(chunks), start, end, document.text[start:end]))
        if end == len(document.text):
            break
        next_start = end - overlap
        while overlap and next_start > start and not document.text[next_start - 1].isspace():
            next_start -= 1
        start = next_start if next_start > start else end
    return chunks
