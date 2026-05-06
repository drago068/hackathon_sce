from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGEngine:
    """
    Handles PDF text → chunking → FAISS vector store → retrieval.

    Chunk sizes are kept at ~300-500 words (≈ 1500-2500 chars) for
    best RAG accuracy as specified in the project requirements.
    """

    # approximate token/word ratio: 1 word ≈ 5 chars
    CHUNK_SIZE    = 1800   # ~360 words
    CHUNK_OVERLAP = 300    # ~60 words overlap for continuity

    def __init__(self):
        print("[RAGEngine] Loading local embedding model (all-MiniLM-L6-v2)…")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = None
        self.chunks: List[str] = []

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    # ── Indexing ──────────────────────────────────────────────────
    def process_text_and_index(self, text: str) -> bool:
        """
        Splits text into chunks, embeds them, and stores in FAISS.
        Returns True on success.
        """
        if not text or not text.strip():
            return False

        self.chunks = self.text_splitter.split_text(text)
        if not self.chunks:
            return False

        print(f"[RAGEngine] Indexed {len(self.chunks)} chunks.")
        self.vector_store = FAISS.from_texts(self.chunks, self.embeddings)
        return True

    # ── Retrieval ─────────────────────────────────────────────────
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """
        Returns the top-k most semantically similar chunks to the query.
        Used after a wrong/partial answer to fetch teaching material.
        """
        if not self.vector_store:
            return []
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    # ── Random chunk for question generation ──────────────────────
    def get_random_chunk(self) -> str:
        """Returns a random chunk (used for question generation)."""
        import random
        if not self.chunks:
            return ""
        return random.choice(self.chunks)

    # ── Stats ─────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "indexed":      self.vector_store is not None
        }
