"""Shared embedding configuration and atomic output for both stages."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMENSION = 1024
INDEX_VERSION = 1


def load_embedder(device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device)


def embed(model, texts: list[str], batch_size: int = 8) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.shape != (len(texts), DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError("BAAI/bge-m3 вернула некорректные эмбеддинги")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("BAAI/bge-m3 вернула ненормализованные эмбеддинги")
    return vectors


@contextmanager
def atomic_output(destination: Path, mode: str, **kwargs):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = NamedTemporaryFile(
        mode=mode, dir=destination.parent, delete=False, **kwargs
    )
    temporary = Path(handle.name)
    try:
        with handle:
            yield handle
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
