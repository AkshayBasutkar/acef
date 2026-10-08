"""
Document Ingestion — chunk and embed documents into Chroma's "memory" collection.

Supports:
  - .txt and .md (plain text read)
  - .pdf (text extraction via pypdf)

Uses a sliding-window chunker (~300 words with ~50 words overlap)
and writes vectors and metadata to ChromaDB.

Usage:
    from ingestion.document_ingest import ingest_document
    count = ingest_document("data/demo_documents/pricing.md", memory_col, embed_model)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _extract_text(file_path: Path) -> str:
    """
    Extract text from a file based on its extension.

    Parameters
    ----------
    file_path : Path
        Path to the target document.

    Returns
    -------
    str
        Extracted raw text content.
    """
    ext = file_path.suffix.lower()

    if ext in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="replace")

    elif ext == ".pdf":
        try:
            import pypdf
        except ImportError as e:
            raise ImportError("pypdf is required for PDF ingestion. Run `pip install pypdf`.") from e

        reader = pypdf.PdfReader(str(file_path))
        pages_text = []
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
        return "\n\n".join(pages_text)

    else:
        raise ValueError(f"Unsupported document format: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")


def _chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks based on word count.

    Parameters
    ----------
    text : str
        Raw input text.
    chunk_size : int
        Approximate number of words per chunk (default 300).
    chunk_overlap : int
        Number of overlapping words between consecutive chunks (default 50).

    Returns
    -------
    list[str]
        List of text chunks.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break

    return chunks


def ingest_document(
    file_path: str,
    chroma_collection,
    embedding_model,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> int:
    """
    Chunk and embed a document into Chroma's "memory" collection.

    Parameters
    ----------
    file_path : str
        Path to the document (.txt, .md, .pdf).
    chroma_collection : Collection
        ChromaDB collection for memory (holds document and conversation chunks).
    embedding_model : SentenceTransformer
        Shared embedding model instance.
    chunk_size : int
        Words per chunk (default 300).
    chunk_overlap : int
        Overlapping words per chunk (default 50).

    Returns
    -------
    int
        Number of chunks stored.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    filename = path.name
    raw_text = _extract_text(path)
    chunks = _chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        logger.warning("Document %s produced no text chunks.", filename)
        return 0

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"
        raw_emb = embedding_model.encode(chunk)
        emb_list = raw_emb.tolist() if hasattr(raw_emb, "tolist") else list(raw_emb)

        ids.append(chunk_id)
        embeddings.append(emb_list)
        documents.append(chunk)
        metadatas.append({
            "type": "document",
            "source": filename,
            "chunk_index": i,
        })

    # I-12: Delete any existing chunks for this document before storing new ones.
    # Guarantees idempotence: if a document got shorter, stale chunks are removed.
    try:
        if hasattr(chroma_collection, "get") and hasattr(chroma_collection, "delete"):
            existing = chroma_collection.get(
                where={
                    "$and": [
                        {"type": "document"},
                        {"source": filename},
                    ]
                }
            )
            if isinstance(existing, dict) and existing.get("ids"):
                chroma_collection.delete(ids=existing["ids"])
                logger.info(
                    "Deleted %d existing chunks for document %s before re-ingestion.",
                    len(existing["ids"]),
                    filename,
                )
    except Exception as e:
        logger.debug("Could not clear previous chunks for %s: %s", filename, e)

    # Use upsert if available to support re-ingesting documents cleanly
    if hasattr(chroma_collection, "upsert"):
        chroma_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
    else:
        chroma_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    logger.info("Ingested document %s: stored %d chunks.", filename, len(chunks))
    return len(chunks)
