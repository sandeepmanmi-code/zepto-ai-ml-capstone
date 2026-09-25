from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    documents = []

    for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        documents.append(
            {
                "id": file_path.stem,
                "text": text,
                "metadata": {
                    "document_id": file_path.stem,
                    "source": file_path.name,
                },
            }
        )

    return documents


def main():
    documents = load_documents()

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 documents, but found {len(documents)}."
        )

    print(f"Loaded {len(documents)} policy documents.")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Creating ChromaDB...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Delete an existing collection so the index is reproducible.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("\nIndex built successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents indexed: {collection.count()}")
    print(f"Database location: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
