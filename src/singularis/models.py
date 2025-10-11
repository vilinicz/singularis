from __future__ import annotations
from typing import Literal, Tuple, Optional
from pydantic import BaseModel

BBox = Tuple[float, float, float, float]
BlockKind = Literal["body", "caption", "header", "footer", "table", "figure"]

class SourceRef(BaseModel):
    """Привязка элемента к исходному месту в PDF."""
    paper_id: Optional[str] = None
    page: int
    bbox: BBox

class TextBlock(BaseModel):
    """Минимальная единица извлечённого текста."""
    page: int
    bbox: BBox
    text: str
    kind: BlockKind = "body"
