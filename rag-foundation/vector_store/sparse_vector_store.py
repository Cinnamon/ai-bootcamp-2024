# flake8: noqa: F841
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import ClassVar, Dict, List
from time import perf_counter

import numpy as np
from loguru import logger
from pydantic import Field
from transformers import AutoTokenizer

from .base import BaseVectorStore
from .node import TextNode, VectorStoreQueryResult

logger.add(
    sink=sys.stdout,
    colorize=True,
    format="<green>{time}</green> <level>{message}</level>",
)

TOKENIZER = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased", max_length=200, truncation=True
)
BBB=0
t0=perf_counter()
t=perf_counter()
class SparseVectorStore(BaseVectorStore):
    """VectorStore2 (add/get/delete implemented)."""

    saved_file: str = "rag-foundation/data/test_db_10.csv"
    metadata_file: Path = Path("rag-foundation/data/sparse_metadata_tmp.json")
    tokenizer: ClassVar[AutoTokenizer] = TOKENIZER
    corpus_size: int = Field(default=0, init=False)
    avgdl: float = Field(default=0.0, init=False)
    doc_freqs: List[Dict[str, int]] = Field(default_factory=list, init=False)#list[document{word:frequency}]
    idf: Dict[str, float] = Field(default_factory=dict, init=False)
    doc_len: List[int] = Field(default_factory=list, init=False)
    nd: int = Field(default=0, init=False)

    # Algorithm specific parameters
    k1: float = Field(default=1.2)
    b: float = Field(default=0.75)
    delta: float = Field(default=0.25)

    def __init__(self, **data):
        super().__init__(**data)
        if len(self.node_dict) > 0:
            self.metadata_file = Path(self.metadata_file)
            if self.metadata_file.exists() and not self.force_index:
                self._load_from_json()
            else:
                self._initialize_bm25_assets()

        self.node_list = list(self.node_dict.values())

    def _initialize_bm25_assets(self):
        """Initialize BM25 assets from the node dictionary."""
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.nd = 0

        corpus = self._tokenize_text([node.text for node in self.node_list])
        self._initialize(corpus)
        content = {
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_freqs": self.doc_freqs,
            "idf": self.idf,
            "doc_len": self.doc_len,
            "nd": self.nd,
        }
        with open(self.metadata_file, "w") as f:
            json.dump(content, f)

    def _load_from_json(self):
        with open(self.metadata_file, "r") as f:
            content = json.load(f)
            self.corpus_size = content["corpus_size"]
            self.avgdl = content["avgdl"]
            self.doc_freqs = content["doc_freqs"]
            self.idf = content["idf"]
            self.doc_len = content["doc_len"]
            self.nd = content["nd"]

    def _initialize(self, corpus: List[List[str]]):
        nd = {}  # word -> number of documents with word
        num_doc = 0 # total number of sentences
        #corpus[document[word|string?]]
        for document in corpus:
            self.doc_len.append(len(document))#list[document's length]
            num_doc += len(document)#words/strings

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size 
        self.idf = {
            word: self._calculate_idf(doc_count, self.corpus_size)
            for word, doc_count in nd.items()
        }

    def _calculate_idf(self, doc_count: int, corpus_size: int) -> float:
        # Calculate the inverse document frequency for a word
        # HINT: Use the formula provided in the BM25 algorithm and np.log()
        "Your code here"
        idf_score = np.log(1+(corpus_size-doc_count+0.5)/(doc_count+0.5))
        return idf_score

    def _tokenize_text(self, corpus: List[str] | str):
        if isinstance(corpus, str):
            return self.tokenizer.tokenize(corpus)
        else:
            pool = Pool(cpu_count()) #"multiprocessing.Pool" to parallelize the tokenization process 
            tokenized_corpus = pool.map(self.tokenizer.tokenize, corpus)
            return tokenized_corpus

    def add(self, nodes: List[TextNode]) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            self.node_dict[node.id_] = node

        self.node_list=list(self.node_dict.values())

        self._update_csv()  # Update CSV after adding nodes

        # Reinitialize BM25 assets after adding new nodes
        self._initialize_bm25_assets()

        return [node.id_ for node in nodes]

    def get(self, text_id: str) -> TextNode:
        """Get node."""
        try:
            return self.node_dict[text_id]
        except KeyError:
            logger.error(f"Node with id `{text_id}` not found.")
            return None

    def get_scores(self, query: str):
        score = np.zeros(self.corpus_size)
        tokenized_query = self._tokenize_text(query)
        for q in tokenized_query:
            # calulate the score for each token in the query
            # HINT: use self.doc_freqs, self.idf, self.corpus_size, self.avgdl

            "Your code here"

            idf = float(self.idf.get(q,0))
            # tf = self.doc_freqs[i].get(q,0)
            tf_fancy = np.zeros(self.corpus_size)
            for i in range(self.corpus_size):
                tf = self.doc_freqs[i].get(q,0)
                # print(self.k1,tf*self.k1*(1-self.b+self.b*self.doc_len[i]/self.avgdl),self.b,self.doc_len[i]/self.avgdl,1-self.b+self.b*self.doc_len[i]/self.avgdl,tf)
                tf_fancy[i] = tf/(tf+self.k1*(1-self.b+self.b*self.doc_len[i]/self.avgdl))
                # tf_fancy[i] = tf/self.avgdl

            #  = np.ndarray([ for i in range(self.corpus_size)])

            cur_score = idf * tf_fancy

            score += cur_score
        return score

    def query(self, query: str, top_k: int = 3) -> VectorStoreQueryResult:
        """Query similar nodes.

        Args:
            query (str): _description_
            top_k (int, optional): _description_. Defaults to 3.

        Returns:
            List[TextNode]: _description_
        """
        scores = self.get_scores(query)
        best_ids = np.argsort(scores)[::-1][:top_k]
        global BBB
        global t
        BBB+=1
        t=perf_counter()
        print(best_ids,f"{BBB}/1451",f"avg{(t-t0)/BBB}s")
        
        
        
        nodes = [self.node_list[node_id] for node_id in best_ids]
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=[scores[doc_id] for doc_id in best_ids],
            ids=[node.id_ for node in nodes],
        )

    def batch_query(
        self, query: List[str], top_k: int = 3
    ) -> List[VectorStoreQueryResult]:
        """Batch query similar nodes.

        Args:
            query (List[str]): _description_
            top_k (int, optional): _description_. Defaults to 3.

        Returns:
            List[VectorStoreQueryResult]: _description_
        """
        return [self.query(q, top_k) for q in query]
