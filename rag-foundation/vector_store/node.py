from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel


class BaseNode(BaseModel):
    id_: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class TextNode(BaseNode):
    text: str | List[str]


@dataclass
class VectorStoreQueryResult:
    """Vector store query result."""

    nodes: Optional[Sequence[BaseNode]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[str]] = None
