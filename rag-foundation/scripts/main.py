import json
from pathlib import Path

import fire
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from vector_store.node import TextNode, VectorStoreQueryResult
from vector_store.semantic_vector_store import SemanticVectorStore
from vector_store.sparse_vector_store import SparseVectorStore


def prepare_data_nodes(documents: list, chunk_size: int = 200) -> list[TextNode]:
    """
    Args:
        documents: List of documents.
        chunk_size: Chunk size for splitting the documents.
    Returns:
        text_node: List of TextNode objects.
    """
    # Load data
    documents = [Document(text=t) for t in documents]

    # Split the documents into nodes
    node_parser = SentenceSplitter(chunk_size=chunk_size)

    # Get the nodes from the documents
    nodes = node_parser.get_nodes_from_documents(documents)

    # Prepare the nodes for the vector store
    text_node = [
        TextNode(id_=str(id_), text=node.text, metadata=node.metadata)
        for id_, node in enumerate(nodes)
    ]
    return text_node


def prepare_vector_store(documents: list, mode: str, force_index=False, chunk_size=200):
    """
    Prepare the vector store with the given documents.
    Args:
        documents: List of documents to be indexed.
        mode: Mode of the vector store. Choose either `sparse` or `semantic`.
        force_index: Whether to force indexing the documents.
        chunk_size: Chunk size for splitting the documents.
    Returns:
        vector_store: Vector store object.
    """
    if mode == "sparse":
        vector_store = SparseVectorStore(
            persist=True,
            saved_file="data/sparse.csv",
            metadata_file="data/sparse_metadata.json",
            force_index=force_index,
        )
    elif mode == "semantic":
        vector_store = SemanticVectorStore(
            persist=True,
            saved_file="data/dense.csv",
            force_index=force_index,
        )
    else:
        raise ValueError("Invalid mode. Choose either `sparse` or `semantic`.")

    if force_index:
        nodes = prepare_data_nodes(documents=documents, chunk_size=chunk_size)
        vector_store.add(nodes)

    return vector_store


class RAGPipeline:
    def __init__(self, vector_store: SemanticVectorStore, prompt_template: str):
        self.vector_store = vector_store
        self.prompt_template = prompt_template

        # choose your model from groq or openai/azure
        self.model = None

        # GROQ
        # from langchain_groq import ChatGroq
        # self.model = ChatGroq(model="llama3-70b-8192", temperature=0)

        # OpenAI
        # from langchain_openai import ChatOpenAI
        # self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def retrieve(self, query: str, top_k: int = 5) -> VectorStoreQueryResult:
        query_result = self.vector_store.query(query, top_k=top_k)
        return query_result

    def answer(self, query: str, top_k: int = 5) -> tuple[str, list[str]]:
        # Generate openai code to answer the query
        result = self.retrieve(query, top_k=top_k)
        context_list = [node.text for node in result.nodes]
        context = "\n\n".join(context_list)

        self.prompt_template = (
            f"""Question: {query}\n\nGiven context: {context}\n\nAnswer:"""
        )

        if not self.model:
            raise ValueError("Model not found. Please initialize the model first.")
        try:
            response = self.model.invoke(self.prompt_template)
        except Exception as e:
            raise Exception(f"Error in calling the model: {e}")
        return response.content, context_list


def main(
    data_path: Path = Path("data/qasper-test-v0.3.json"),
    output_path: Path = Path("predictions.jsonl"),
    mode: str = "sparse",
    force_index: bool = False,
    print_context: bool = False,
    chunk_size: int = 200,
    top_k: int = 5,
    retrieval_only: bool = False,
):
    # Generate doc string
    """
    Args:
        data_path: Path to the qasper data file.
        output_path: Path to save the predictions.
        mode: Mode of the vector store. Choose either `sparse` or `semantic`.
        force_index: Whether to force indexing the documents.
        print_context: Whether to print the context.
        chunk_size: Chunk size for splitting the documents.
        top_k: Number of top k documents to retrieve.
        retrieval_only: Whether to retrieve only.
    Returns:
        None
    """
    # Load the data
    raw_data = json.load(open(data_path, "r", encoding="utf-8"))

    question_ids, predicted_answers, predicted_evidences = [], [], []

    # NOTE: qasper has many papers, each paper has multiple sections
    # we will loop through each paper, gather the full text of each section
    # and prepare the documents for the vector store
    # and answer the query
    for _, values in raw_data.items():
        # for each paper in qasper
        documents = []

        for section in values["full_text"]:
            # for each section in the paper
            documents += section["paragraphs"]

        # initialize the vector store
        # and rag pipeline
        # Remember to force_index=True if you want to override the existing index
        vector_store = prepare_vector_store(
            documents, mode=mode, force_index=force_index, chunk_size=chunk_size
        )

        # NOTE: Should design your own template
        prompt_template = """Question: {}\n\nGiven context: {}\n\nAnswer:"""

        rag_pipeline = RAGPipeline(vector_store, prompt_template=prompt_template)

        for q in values["qas"]:
            # for each question in the paper
            query = q["question"]
            question_ids.append(q["question_id"])

            # NOTE: If you just want to retrieve the top_k relevant documents
            # set retrieval_only=True
            # Otherwise, it will answer the question
            if retrieval_only:
                result = rag_pipeline.retrieve(query, top_k=top_k)
                context_list = [node.text for node in result.nodes]

                if print_context:
                    for i, context in enumerate(context_list):
                        print(f"Relevent context {i + 1}:", context)
                        print("\n\n")

                predicted_evidences.append(context_list)
                predicted_answers.append("")

            else:
                predicted_answer, context_list = rag_pipeline.answer(query, top_k=top_k)

                # Just In Case. Print out the context list for each question
                # if needed.
                if print_context:
                    for i, context in enumerate(context_list):
                        print(f"Relevent context {i + 1}:", context)
                        print("\n\n")

                    print("LLM Answer")
                    print(predicted_answer)

                predicted_evidences.append(context_list)
                predicted_answers.append(predicted_answer)

    # save the results
    with open(output_path, "w") as f:
        for question_id, predicted_answer, predicted_evidence in zip(
            question_ids, predicted_answers, predicted_evidences
        ):
            f.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "predicted_answer": predicted_answer,
                        "predicted_evidence": predicted_evidence,
                    }
                )
            )
            f.write("\n")


if __name__ == "__main__":
    fire.Fire(main)
