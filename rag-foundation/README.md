# Week 4's Assigment report
Student: Viet The Le
> **Notes:** Due to the time-consuming nature of LLM-based answer generation and the assignment's focus on Information Retrieval, all experiments were conducted without using LLM APIs for answer generation.
### Summary:
Challenge:
1. [Done - Normal] Implement scoring mechanism in vectorstore (using cosine-similarity)
2. [Done - Intermediate] Implement scoring mechanism in TF-IDF
3. [Done - Intermediate] Evaluate RAG on a public dataset and compare dense vs sparse retriever. Create a report to discuss your findings.
4. [Done - Hard] Implement advance retrieval pipeline
	* [Done - Optional] Implement hybrid search with Reciprocal Rank Fusion
	* [Done - Optional] Implement a re-ranker step with SentenceTransformer

| Methods\Dataset                             | qasper | ms_marco |
| ------------------------------------------- | ------ | -------- |
| SematicVectorStore \| all-MiniLM-L6-v2      | 0.1849 | 0.2686   |
| SematicVectorStore \| nomic-embed-text-v1.5 | 0.2011 | 0.2756   |
| SparseVectorStore                           | 0.1491 | 0.2105   |
| HybridVectorStore \| all-MiniLM-L6-v2       | 0.1885 | 0.2404   |
:F1 - Evidence Scores Table.

### Challenge Reports:
#### Challenge 1: Implement scoring mechanism in vectorstore (using cosine-similarity)
This challenge involved implementing a cosine similarity scoring mechanism within a vector store. The initial implementation, utilizing the [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) bi-encoder model, achieved an F1-Evidence score of approximately 0.1889 on the `qasper` test set.

Analyzing the Sentence Transformer documentation revealed that the all-MiniLM-L6-v2 model is relatively small (22.7M parameters). To potentially improve performance, a larger bi-encoder model, [`nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (137M parameters), was chosen for experimentation.

Further, the input text formatting was adjusted to provide greater clarity for the bi-encoder model. This was achieved by adding distinct prompts for different embedding types, as demonstrated in the code snippet below:
```python
def _get_text_embedding(self, text: str, embed_type: str) -> List[float]:
  """Calculate embedding."""
  if embed_type == "doc":
		return self.embed_model.encode("search_document: " + text).tolist()

  return self.embed_model.encode("search_query: " + text).tolist()
```
These modifications resulted in a noticeable improvement, raising the F1-Evidence score from approximately 0.1889 to 0.2011.
#### Challenge 2: Implement scoring mechanism in TF-IDF
This challenge focused on implementing a scoring mechanism within a TF-IDF vector store using the **BM25 algorithm**. The code snippet below demonstrates the implementation of the key components:

**IDF Calculation:**
```python
def _calculate_idf(self, doc_count: int, corpus_size: int) -> float:
	idf_score = np.log((corpus_size - doc_count + 0.5) / (doc_count + 0.5))
	return idf_score
```
**BM25 Scoring:**
```python
def get_scores(self, query: str):
	score = np.zeros(self.corpus_size)
	tokenized_query = self._tokenize_text(query)
	for q in tokenized_query:
		term_frequency = np.array(
			[
				self.doc_freqs[doc][q] if q in self.doc_freqs[doc].keys() else 0
				for doc in range(self.corpus_size)
			]
		)

		doc_lens = np.array([self.doc_len[doc] for doc in range(self.corpus_size)])
		numerator = term_frequency * (self.k1 + 1)
		denominator = term_frequency + self.k1 * (
			1 - self.b + self.b * (doc_lens / self.avgdl)
		)

		if q in self.idf.keys():
			cur_score = self.idf[q] * (numerator / denominator)
		else:
			cur_score = np.zeros(self.corpus_size)

		score += cur_score
	return score
```
This implementation yielded an F1-Evidence score of approximately 0.1885 on the `qasper` test set, surpassing the performance of the semantic vector store alone.
#### Challenge 3: Evaluate RAG on a public dataset and compare dense vs sparse retriever.
This challenge involved evaluating the performance of a Retrieval-Augmented Generation (RAG) system on two public datasets: qasper for question answering and ms-marco for passage retrieval. The goal was to compare the effectiveness of dense and sparse vector stores in retrieving relevant information.

| Methods\Dataset                             | qasper | ms_marco |
| ------------------------------------------- | ------ | -------- |
| SematicVectorStore \| all-MiniLM-L6-v2      | 0.1849 | 0.2686   |
| SematicVectorStore \| nomic-embed-text-v1.5 | 0.2011 | 0.2756   |
| SparseVectorStore                           | 0.1491 | 0.2105   |
The results demonstrate that dense vector stores, particularly those utilizing larger language models, can significantly improve the performance of RAG systems on both question answering and passage retrieval tasks. These findings suggest that incorporating semantic information into the retrieval process is crucial for achieving optimal results.
#### Challenge 4: Implement advance retrieval pipeline
##### Implementing Hybrid Search with Reciprocal Rank Fusion
This section explores the implementation of a hybrid search approach using Reciprocal Rank Fusion (RRF). RRF is a relatively simple yet powerful algorithm for combining results from two different vector stores, potentially leading to improved search outcomes.

The provided code snippet demonstrates the implementation of this approach:
```python
def query(self, query: str, top_k: int = 3, k: int = 60) -> VectorStoreQueryResult:
	  """Query similar nodes."""
	  results = {
			"dense_ranks": self.dense_vector_store.get_ranks(query),
			"sparse_ranks": self.sparse_vector_store.get_ranks(query),
	  }

	  corpus_size = self.sparse_vector_store.corpus_size
	  node_list = self.sparse_vector_store.node_list

	  rrf_scores = np.zeros(corpus_size)

	  for _, rank in results.items():
			rrf_scores += 1 / (k + rank)

	  best_ids = np.argsort(rrf_scores)[::-1]
	  best_ids = best_ids[:top_k]
	  nodes = [node_list[node_id] for node_id in best_ids]

	  return VectorStoreQueryResult(
			nodes=nodes,
			similarities=[rrf_scores[doc_id] for doc_id in best_ids],
			ids=[node.id_ for node in nodes],
	  )
```
This implementation resulted in an F1-Evidence score of approximately 0.1885 on the `qasper` test set, better then Semantic Vector Store result.
##### Implement a re-ranker step with SentenceTransformer
To further enhance the retrieval pipeline, a re-ranking step was implemented using the SentenceTransformer library. The [`ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) cross-encoder model was employed for this purpose.
```python
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
```
This re-ranking step leverages the power of cross-encoders to refine the initial results based on a more nuanced semantic understanding of the query and retrieved documents.
