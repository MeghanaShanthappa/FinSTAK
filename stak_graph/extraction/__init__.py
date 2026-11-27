"""Source-grounded event extraction; no price labels or embeddings are required."""

from .pipeline import ExtractionPipeline
from .sources import Document

__all__ = ["Document", "ExtractionPipeline"]
