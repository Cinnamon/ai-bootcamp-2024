# autoflake: off
# flake8: noqa: F841
from typing import Dict, List, Union, cast

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base import BaseVectorStore
from .node import TextNode, VectorStoreQueryResult
from .utils import get_rank

# logger.add(
#     sink=sys.stdout,
#     colorize=True,
#     format="<green>{time}</green> <level>{message}</level>"
# )

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""))


class SemanticVectorStore(BaseVectorStore):
    """Semantic Vector Store using SentenceTransformer embeddings."""

    saved_file: str = "rag-foundation/data/test_db_00.csv"
    embed_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embed_model: SentenceTransformer = SentenceTransformer(
        embed_model_name, trust_remote_code=True
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._setup_store()

    def get(self, text_id: str) -> TextNode:
        """Get node."""
        try:
            return self.node_dict[text_id]
        except KeyError:
            logger.error(f"Node with id `{text_id}` not found.")
            return None

    def add(self, nodes: List[TextNode]) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            if node.embedding is None:
                logger.info(
                    "Found node without embedding, calculating "
                    f"embedding with model {self.embed_model_name}"
                )
                node.embedding = self._get_text_embedding(node.text, "doc")
            self.node_dict[node.id_] = node
        self._update_csv()  # Update CSV after adding nodes
        return [node.id_ for node in nodes]

    def _get_text_embedding(self, text: str, embed_type: str) -> List[float]:
        """Calculate embedding."""
        if embed_type == "doc":
            return self.embed_model.encode("search_document: " + text).tolist()

        return self.embed_model.encode("search_query: " + text).tolist()

    def delete(self, node_id: str, **delete_kwargs: Dict) -> None:
        """Delete nodes using node_id."""
        if node_id in self.node_dict:
            del self.node_dict[node_id]
            self._update_csv()  # Update CSV after deleting nodes
        else:
            logger.error(f"Node with id `{node_id}` not found.")

    def _calculate_similarity(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        doc_ids: List[str],
        similarity_top_k: int = 3,
        return_rank: bool = False,
    ) -> Union[tuple[List[float], List[str]], np.ndarray]:
        """Get top nodes by similarity to the query."""
        qembed_np = np.array(query_embedding)
        dembed_np = np.array(doc_embeddings)

        # calculate the dot product of
        # the query embedding with the document embeddings
        # HINT: np.dot
        "Your code here"
        dproduct_arr = np.dot(dembed_np, qembed_np)
        # calculate the cosine similarity
        # by dividing the dot product by the norm
        # HINT: np.linalg.norm
        "Your code here"
        norm_q = np.linalg.norm(qembed_np)
        norm_d = np.linalg.norm(dembed_np, axis=1)
        cos_sim_arr = dproduct_arr / (norm_d * norm_q)

        if return_rank:
            return get_rank(cos_sim_arr)

        # get the indices of the top k similarities
        "Your code here"
        k_indexes = np.argsort(cos_sim_arr)[::-1][:similarity_top_k]

        similarities = cos_sim_arr[k_indexes]
        node_ids = np.array(doc_ids)[k_indexes].tolist()

        return (
            similarities,
            node_ids,
        )

    def get_ranks(self, query: str) -> np.ndarray:
        """Get rank of documents base on query"""
        query_embedding = cast(List[float], self._get_text_embedding(query, "query"))
        doc_embeddings = [node.embedding for node in self.node_dict.values()]
        doc_ids = list(self.node_dict.keys())

        ranks = self._calculate_similarity(
            query_embedding, doc_embeddings, doc_ids, return_rank=True
        )

        # add 1 because rank start at 1
        return ranks

    def query(self, query: str, top_k: int = 3) -> VectorStoreQueryResult:
        """Query similar nodes."""
        query_embedding = cast(List[float], self._get_text_embedding(query, "query"))
        doc_embeddings = [node.embedding for node in self.node_dict.values()]
        doc_ids = list(self.node_dict.keys())

        if len(doc_embeddings) == 0:
            logger.error("No documents found in the index.")
            result_nodes, similarities, node_ids = [], [], []
        else:
            similarities, node_ids = self._calculate_similarity(
                query_embedding, doc_embeddings, doc_ids, top_k
            )
            result_nodes = [self.node_dict[node_id] for node_id in node_ids]

        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )

    def batch_query(
        self, query: List[str], top_k: int = 3
    ) -> List[VectorStoreQueryResult]:
        """Batch query similar nodes."""
        return [self.query(q, top_k) for q in query]
