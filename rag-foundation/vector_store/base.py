import os
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from .node import BaseNode, TextNode


class BaseVectorStore(BaseModel):
    """Simple custom Vector Store.

    Stores documents in a simple in-memory dict.
    """

    force_index: bool = False
    persist: bool = True
    node_dict: dict[str, BaseNode] = Field(default_factory=dict)
    node_list: list[BaseNode] = Field(default_factory=list)
    saved_file: str = "rag-foundation/data/sematic_vectordb_nodes.csv"
    csv_file: Path = Path(saved_file)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.csv_file = Path(self.saved_file)
        self._setup_store()

    def _setup_store(self):
        if self.persist:
            if self.force_index:
                self._reset_csv()
            self._initialize_csv()
            self._load_from_csv()

    def _initialize_csv(self):
        """Initialize the CSV file if it doesn't exist."""
        if not self.csv_file.exists():
            logger.warning(
                f"Cannot find CSV file at `{self.saved_file}`, creating a new one..."
            )
            os.makedirs(self.csv_file.parent, exist_ok=True)
            with open(self.csv_file, "w") as f:
                f.write("id,text,embedding,metadata\n")

    def _load_from_csv(self):
        """Load the node_dict from the CSV file."""
        if self.csv_file.exists():
            df = pd.read_csv(self.csv_file)
            for _, row in df.iterrows():
                node_id = row["id"]
                text = row["text"]
                try:
                    embedding = eval(row["embedding"])
                    metadata = eval(row["metadata"])
                except TypeError:
                    embedding = None
                    metadata = None
                self.node_dict[node_id] = TextNode(
                    id_=str(node_id), text=text, embedding=embedding, metadata=metadata
                )

    def _update_csv(self):
        """Update the CSV file with the current node_dict if persist is True."""
        if self.persist:
            data = {"id": [], "text": [], "embedding": [], "metadata": []}
            for key, node in self.node_dict.items():
                data["id"].append(key)
                data["text"].append(node.text)
                data["embedding"].append(node.embedding)
                data["metadata"].append(node.metadata)
            df = pd.DataFrame(data)
            df.to_csv(self.csv_file, index=False)
        else:
            logger.warning("`persist` is set to `False`, not updating CSV file.")

    def _reset_csv(self):
        """Reset the CSV file by deleting it if it exists."""
        if self.csv_file.exists():
            self.csv_file.unlink()

    def get(self):
        """Get embedding."""

    def add(self):
        """Add nodes to index."""

    def delete(self) -> None:
        """Delete nodes using with node_id."""

    def query(self):
        """Get nodes for response."""
