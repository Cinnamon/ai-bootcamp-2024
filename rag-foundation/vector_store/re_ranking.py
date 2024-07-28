import numpy as np
from sentence_transformers import CrossEncoder

from .node import VectorStoreQueryResult


class ReRanking:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", **data
    ):
        self.cross_encoder = CrossEncoder(model_name)

    def __call__(self, query: str, results: VectorStoreQueryResult):
        cross_inp = [[query, node.text] for node in results.nodes]
        cross_scores = self.cross_encoder.predict(cross_inp, convert_to_numpy=True)

        best_ids = np.argsort(cross_scores)[::-1]
        nodes = [results.nodes[node_id] for node_id in best_ids]

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=[cross_scores[doc_id] for doc_id in best_ids],
            ids=[node.id_ for node in nodes],
        )
