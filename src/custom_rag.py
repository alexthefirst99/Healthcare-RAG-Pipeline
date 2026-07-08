"""Part 1: Custom RAG implementation (hand-rolled chunking, FAISS, no framework)."""
import os

import faiss
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHAT_MODEL = "gpt-4o-mini"


class HealthcareRAG:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def add_documents(self, file_paths):
        all_chunks = []
        for path in file_paths:
            with open(path, "rb") as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            chunks = [text[i : i + 500] for i in range(0, len(text), 500) if text[i : i + 500].strip()]
            all_chunks.extend(chunks)
        self.documents = all_chunks
        embeddings = self.embedder.encode(all_chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))
        print(f"Custom: Indexed {len(all_chunks)} chunks")

    def retrieve(self, query, k=3):
        query_emb = self.embedder.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb.astype("float32"), k)
        return [self.documents[i] for i in indices[0]]

    def generate(self, query):
        chunks = self.retrieve(query)
        context = "\n---\n".join(chunks)
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the medical question using ONLY the provided context. "
                        "If the context does not contain the answer, say so."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content, chunks
