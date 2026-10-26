"""Local sentence embeddings (all-MiniLM-L6-v2 by default).

The model is loaded lazily on first use, so importing this module (for
example from unit tests) never triggers a model download. Tests can replace
the model with ``set_embedding_model`` and anything that has an
``encode(texts)`` method.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from app.config import EMBEDDING_MODEL_NAME
from app.errors import ConfigurationError

MODEL_NAME = EMBEDDING_MODEL_NAME

_model: Any = None


def _download_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [OSError]
    for module_name, attr in (("httpx", "HTTPError"), ("httpx2", "HTTPError"), ("huggingface_hub.errors", "HfHubHTTPError")):
        try:
            module = __import__(module_name, fromlist=[attr])
            types.append(getattr(module, attr))
        except (ImportError, AttributeError):
            continue
    return tuple(types)


def get_embedding_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(MODEL_NAME)
        except _download_error_types() as exc:
            # Typically: no network access to Hugging Face on first download.
            raise ConfigurationError(
                f"Could not load embedding model {MODEL_NAME!r}: {exc}. The first run "
                "downloads it from Hugging Face; check your internet connection."
            ) from exc
    return _model


def set_embedding_model(model: Any) -> None:
    """Inject an embedding model (used by tests and experiments)."""
    global _model
    _model = model


def generate_embeddings(texts: Sequence[str]) -> np.ndarray:
    return np.asarray(get_embedding_model().encode(list(texts)))
