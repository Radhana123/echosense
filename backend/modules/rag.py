import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGRetriever:

    def __init__(self, docs_dir="data/rag_docs"):
        self.model  = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        self.index  = None
        if os.path.exists(docs_dir):
            self._build_index(docs_dir)

    def _build_index(self, docs_dir):
        for fname in os.listdir(docs_dir):
            path = os.path.join(docs_dir, fname)
            with open(path, encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) > 20:
                        self.chunks.append({
                            "text":   line,
                            "source": fname
                        })

        if self.chunks:
            embs = self.model.encode(
                [c["text"] for c in self.chunks],
                convert_to_numpy=True
            )
            self.index = faiss.IndexFlatL2(embs.shape[1])
            self.index.add(embs.astype(np.float32))
            print(f"✓ RAG index ready — {len(self.chunks)} chunks loaded")

    def retrieve(self, query: str, domain: str = "", k: int = 3) -> list:
        if not self.index:
            return []
        qv = self.model.encode(
            [f"{domain} {query}"],
            convert_to_numpy=True
        ).astype(np.float32)
        _, ids = self.index.search(qv, k)
        return [self.chunks[i]["text"] for i in ids[0] if i < len(self.chunks)]


if __name__ == "__main__":
    retriever = RAGRetriever(docs_dir="data/rag_docs")

    print("\nQuery: delivery time in rain")
    results = retriever.retrieve("delivery time in rain", domain="logistics")
    for r in results:
        print(f"  → {r}")

    print("\nQuery: blood glucose normal range")
    results = retriever.retrieve("blood glucose normal range", domain="healthcare")
    for r in results:
        print(f"  → {r}")

    print(f"\n{'='*50}")
    print("  ✓ WEEK 5 COMPLETE!")
    print("  Next: Week 6 — GenAI Report")
    print(f"{'='*50}\n")