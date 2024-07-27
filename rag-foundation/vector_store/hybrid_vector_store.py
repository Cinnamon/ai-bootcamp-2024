# autoflake: off
# flake8: noqa: F841
from typing import Dict, List

import numpy as np
from loguru import logger
from pydantic import Field
from tqdm import tqdm

from .base import BaseVectorStore
from .node import TextNode, VectorStoreQueryResult
from .semantic_vector_store import SemanticVectorStore
from .sparse_vector_store import SparseVectorStore

# logger.add(
#     sink=sys.stdout,
#     colorize=True,
#     format="<green>{time}</green> <level>{message}</level>",
# )

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""))


class HybridVectorStore(BaseVectorStore):
    """Hybrid Vector Store using Reciprocal Rank Fusion."""

    dense_vector_store: SemanticVectorStore = Field(SemanticVectorStore, init=False)
    sparse_vector_store: SparseVectorStore = Field(SparseVectorStore, init=False)

    def __init__(self, **data):
        super().__init__(**data)
        self.dense_vector_store = SemanticVectorStore(
            saved_file="data/dense.csv", **data
        )
        self.sparse_vector_store = SparseVectorStore(
            saved_file="data/sparse.csv", **data
        )
        # super().__init__(**data)

    def add(self, nodes: List[TextNode]) -> List[str]:
        """Add nodes to index."""
        self.dense_vector_store.add(nodes)
        self.sparse_vector_store.add(nodes)

    def delete(self, node_id: str, **delete_kwargs: Dict) -> None:
        """Delete nodes using node_id."""
        self.dense_vector_store.delete(node_id)

        # the delete function in SparseVectorStore is note implemented
        # self.sparse_vector_store.delete(node_id)

    def worker(query, vector_store):
        ranks = vector_store.get_ranks(query)
        return ranks

    def query(self, query: str, top_k: int = 3, k: int = 60) -> VectorStoreQueryResult:
        """Query similar nodes."""
        ranks = {
            "dense_ranks": self.dense_vector_store.get_ranks(query),
            "sparse_ranks": self.sparse_vector_store.get_ranks(query),
        }

        # with multiprocessing.Pool(processes=2) as pool:
        #     args = [(query, self.dense_vector_store), (query, self.sparse_vector_store)]
        #     results = pool.map(self.worker, args)

        corpus_size = self.sparse_vector_store.corpus_size
        node_list = self.sparse_vector_store.node_list

        scores = np.zeros(corpus_size)

        for _, rank in ranks.items():
            scores += 1 / (k + rank)

        best_ids = np.argsort(scores)[::-1][:top_k]
        nodes = [node_list[node_id] for node_id in best_ids]
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=[scores[doc_id] for doc_id in best_ids],
            ids=[node.id_ for node in nodes],
        )

    def batch_query(self, query: List[str]) -> List[VectorStoreQueryResult]:
        """Batch query similar nodes."""
        return [self.query(q) for q in query]
