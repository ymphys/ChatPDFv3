from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

from ..config import get_settings

logger = logging.getLogger("chatpdf")

SupportedProvider = Literal["openai", "huggingface"]


@dataclass(frozen=True)
class _HuggingFaceModel:
    name: str
    normalize: bool = True


class Embedder:
    """
    Thin wrapper around embedding backends. Defaults to OpenAI embeddings but
    can fallback to HuggingFace SentenceTransformer models (e.g., BGE).
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        *,
        provider: Optional[SupportedProvider] = None,
    ) -> None:
        self.model_name = model_name
        self.provider: SupportedProvider = provider or self._infer_provider(model_name)
        self._client = None
        self._hf_model: Optional[_HuggingFaceModel] = None
        self._hf_encoder = None

        if self.provider == "openai":
            settings = get_settings()
            from openai import OpenAI  # Lazy import to keep CLI startup fast

            self._client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._init_huggingface_model(model_name)

    def _infer_provider(self, model_name: str) -> SupportedProvider:
        lowered = model_name.lower()
        if lowered.startswith(("bge", "gte", "sentence-transformers", "m3e", "mpnet")):
            return "huggingface"
        return "openai"

    def _init_huggingface_model(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for HuggingFace embeddings. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self._hf_model = _HuggingFaceModel(name=model_name)
        self._hf_encoder = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text snippet.
        """
        if not text.strip():
            return []
        embeddings = self.embed_batch([text])
        return embeddings[0] if embeddings else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts and return embeddings as float lists.
        """
        if not texts:
            return []

        if self.provider == "openai":
            assert self._client is not None
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [record.embedding for record in response.data]

        assert self._hf_encoder is not None
        # SentenceTransformer returns numpy arrays by default; convert to Python lists.
        embeddings = self._hf_encoder.encode(
            texts,
            normalize_embeddings=self._hf_model.normalize if self._hf_model else True,
        )
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()

        return [list(map(float, emb)) for emb in embeddings]


__all__ = ["Embedder"]
