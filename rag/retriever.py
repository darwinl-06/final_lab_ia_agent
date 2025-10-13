# rag/retriever.py
from pathlib import Path
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

PERSIST_DIR = Path("./chroma")

class RAGRetriever:
    def __init__(self, k: int = 4):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vs = Chroma(
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR)
        )
        self.retriever = self.vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

    def get_relevant_chunks(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)

    def format_context(self, docs: List[Document]) -> str:
        lines = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            lines.append(f"[{src}] {d.page_content.strip()}")
        return "\n\n".join(lines)
