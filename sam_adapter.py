"""Segmenter adapter abstraction for the crystal annotation tool.

This module isolates all segmentation-backend concerns (model loading,
device placement, embedding computation/caching, and mask prediction)
behind a small ``SegmenterAdapter`` interface so that ``0_annotator_sam2.py``
never talks to SAM2 (or, in the future, micro-sam) internals directly.

Phase 1 scope:
- ``Sam2Adapter``: a faithful port of the current SAM2 behavior, plus a
  correctness-first embedding LRU cache and explicit GPU teardown on
  ``unload()``.
- ``MicroSamAdapter``: an interface-complete skeleton only. Real inference
  is deferred to a later phase; micro-sam is never imported at module load
  time.

Design notes (see task spec / Codex consultation for full rationale):
- The embedding cache key is a 5-tuple: ``(image_hash, model_name,
  checkpoint_hash, preprocessing_signature, tile_settings)``. Including
  ``model_name``/``checkpoint_hash`` in the key -- even though a single
  adapter instance only ever serves one model -- makes cross-model /
  cross-checkpoint embedding reuse structurally impossible, not just
  "unlikely by construction".
- ``image_hash`` is a blake2b digest of the raw image *file* bytes on disk
  (not the post-``transform_input`` array). This is cheap (a single
  streamed read, no extra array hashing) and, combined with
  ``preprocessing_signature`` (which fully determines how
  ``transform_input`` maps the raw file to the array actually handed to
  the encoder), is sufficient to distinguish every distinct encoder input.
- The cache stores references (not clones) to SAM2's internal embedding
  tensors. ``SAM2ImagePredictor.set_image()`` runs under ``torch.no_grad()``
  and ``predict()`` only reads ``_features``/``_orig_hw`` without mutating
  them in place, so reference storage is safe and avoids doubling VRAM
  usage. Restoration is validated (shape/dtype/device/batch-mode) before
  use and falls back to recomputation on any mismatch ("fail closed").
"""

from __future__ import annotations

import gc
import hashlib
import importlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEFAULT_EMBEDDING_CACHE_SIZE = 1
HASH_DIGEST_SIZE = 32
HASH_CHUNK_SIZE = 1 << 20  # 1 MiB
PREPROCESSING_PIPELINE_VERSION = "transform_input_v1"

# SAM2.1 model size -> (config file, checkpoint path), relative to the
# annotator's working directory (unchanged from the previous inline dict in
# clickLoadSAM).
SAM2_MODEL_SPECS: dict[str, "Sam2ModelSpec"] = {}


@dataclass(frozen=True)
class Sam2ModelSpec:
    """SAM2 model identity: name, Hydra config, and checkpoint path."""

    name: str
    config_file: str
    checkpoint_path: str


SAM2_MODEL_SPECS.update(
    {
        "tiny": Sam2ModelSpec(
            "tiny",
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            "external/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        ),
        "small": Sam2ModelSpec(
            "small",
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            "external/sam2/checkpoints/sam2.1_hiera_small.pt",
        ),
        "base_plus": Sam2ModelSpec(
            "base_plus",
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        ),
        "large": Sam2ModelSpec(
            "large",
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "external/sam2/checkpoints/sam2.1_hiera_large.pt",
        ),
    }
)

# Placeholder names for the future micro-sam integration (Phase 3). These are
# used only to populate the (disabled) model-switch menu; no micro-sam
# checkpoints or specs are wired up in Phase 1.
MICRO_SAM_MODEL_NAMES: tuple[str, ...] = ("micro-sam",)


class NotInstalledError(RuntimeError):
    """Raised when an optional segmenter backend dependency is unavailable."""


def hash_file_bytes(path: str) -> str:
    """Compute a blake2b digest of a file's raw bytes, streamed in chunks."""
    digest = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preprocessing_signature(keep_input_size: bool, max_size: float) -> str:
    """Build a stable signature describing how ``transform_input`` behaves.

    Any future change to the resize/color-space logic in
    ``MainWindow.transform_input`` must bump
    ``PREPROCESSING_PIPELINE_VERSION`` so that stale cache entries are never
    silently reused.
    """
    return (
        f"{PREPROCESSING_PIPELINE_VERSION}"
        f"|keep_input_size={keep_input_size}"
        f"|max_size={max_size}"
    )


@dataclass(frozen=True)
class ImageKey:
    """Caller-provided identity of an image + preprocessing combination.

    Deliberately excludes model identity: the adapter itself knows which
    model/checkpoint it is, and folds that in when building the internal
    cache key (see ``CacheKey``).
    """

    image_hash: str
    preprocessing_signature: str
    tile_settings: Hashable | None = None


def build_image_key(
    image_path: str, keep_input_size: bool, max_size: float
) -> ImageKey:
    """Build the caller-side cache key inputs from the source image path."""
    return ImageKey(
        image_hash=hash_file_bytes(image_path),
        preprocessing_signature=build_preprocessing_signature(
            keep_input_size, max_size
        ),
        tile_settings=None,
    )


@dataclass(frozen=True)
class CacheKey:
    """Full embedding cache key: image + model + checkpoint + preprocessing.

    All five components are required so that cross-model or
    cross-checkpoint embedding reuse is structurally impossible, even if a
    single adapter instance is (incorrectly) reused across model reloads.
    """

    image_hash: str
    model_name: str
    checkpoint_hash: str
    preprocessing_signature: str
    tile_settings: Hashable | None


PredictionResult = tuple[np.ndarray, np.ndarray, np.ndarray]
AutoSegmentationResult = list[dict[str, object]]


class SegmenterAdapter(ABC):
    """Backend-agnostic interface for prompt-based interactive segmentation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique model name (e.g. 'tiny', 'large')."""

    @property
    @abstractmethod
    def checkpoint_hash(self) -> str | None:
        """Return the loaded checkpoint's content hash, or None if unloaded."""

    @abstractmethod
    def load(self) -> None:
        """Load the model weights and initialize the underlying predictor."""

    @abstractmethod
    def set_image(self, image_np: np.ndarray, image_key: ImageKey) -> None:
        """Activate the embedding for ``image_np``.

        Restores a cached embedding on a cache hit, or encodes ``image_np``
        and stores the result on a miss. Callers must not need to know
        which happened.
        """

    @abstractmethod
    def predict(
        self,
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> PredictionResult:
        """Predict masks from prompts for the currently active image."""

    @abstractmethod
    def unload(self) -> None:
        """Release the predictor, model, cached embeddings, and GPU memory."""

    def generate_auto(
        self, image_np: np.ndarray, image_key: ImageKey
    ) -> AutoSegmentationResult:
        """Hook for future automatic mask generation (AMG).

        Not implemented in Phase 1; ``clickAutoSeg`` remains a stub in
        ``0_annotator_sam2.py`` and does not call this method yet.
        """
        raise NotImplementedError(
            f"{self.name} は自動セグメンテーション(AMG)にまだ対応していません。"
        )


@dataclass(frozen=True)
class _Sam2EmbeddingState:
    """Snapshot of ``SAM2ImagePredictor``'s private per-image state."""

    image_embed: torch.Tensor
    high_res_feats: tuple[torch.Tensor, ...]
    orig_hw: tuple[tuple[int, int], ...]
    is_batch: bool
    input_shape: tuple[int, ...]
    input_dtype: str


class Sam2Adapter(SegmenterAdapter):
    """SegmenterAdapter implementation backed by SAM2ImagePredictor."""

    def __init__(
        self,
        model_spec: Sam2ModelSpec,
        *,
        device: str | None = None,
        cache_maxsize: int = DEFAULT_EMBEDDING_CACHE_SIZE,
    ) -> None:
        self._model_spec = model_spec
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache_maxsize = cache_maxsize
        self._sam: torch.nn.Module | None = None
        self._predictor: SAM2ImagePredictor | None = None
        self._checkpoint_hash: str | None = None
        self._embedding_cache: "OrderedDict[CacheKey, _Sam2EmbeddingState]" = (
            OrderedDict()
        )
        # Instrumentation for tests/diagnostics: how many times set_image()
        # actually recomputed an embedding vs. restored one from cache.
        self.encode_count = 0
        self.cache_hit_count = 0

    @property
    def name(self) -> str:
        return self._model_spec.name

    @property
    def checkpoint_hash(self) -> str | None:
        return self._checkpoint_hash

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        self._checkpoint_hash = hash_file_bytes(self._model_spec.checkpoint_path)
        sam = build_sam2(
            config_file=self._model_spec.config_file,
            ckpt_path=self._model_spec.checkpoint_path,
        )
        sam.to(device=self._device)
        self._sam = sam
        self._predictor = SAM2ImagePredictor(sam)

    def set_image(self, image_np: np.ndarray, image_key: ImageKey) -> None:
        predictor = self._require_predictor()
        cache_key = self._build_cache_key(image_key)

        entry = self._embedding_cache.pop(cache_key, None)
        if entry is not None and self._can_restore(entry, image_np, predictor):
            self._embedding_cache[cache_key] = entry  # re-insert as most-recent
            self._restore_embedding(predictor, entry)
            self.cache_hit_count += 1
            return

        self._evict_to_fit()
        predictor.set_image(image_np)
        self._embedding_cache[cache_key] = self._snapshot_embedding(predictor, image_np)
        self.encode_count += 1

    def predict(
        self,
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> PredictionResult:
        predictor = self._require_predictor()
        return predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )

    def unload(self) -> None:
        self._embedding_cache.clear()
        if self._predictor is not None:
            self._predictor.reset_predictor()
        self._predictor = None
        self._sam = None
        self._checkpoint_hash = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_predictor(self) -> SAM2ImagePredictor:
        if self._predictor is None:
            raise RuntimeError(
                f"SAM2 predictor ({self.name}) is not loaded; call load() first."
            )
        return self._predictor

    def _build_cache_key(self, image_key: ImageKey) -> CacheKey:
        return CacheKey(
            image_hash=image_key.image_hash,
            model_name=self.name,
            checkpoint_hash=self.checkpoint_hash or "",
            preprocessing_signature=image_key.preprocessing_signature,
            tile_settings=image_key.tile_settings,
        )

    def _evict_to_fit(self) -> None:
        while len(self._embedding_cache) >= self._cache_maxsize:
            self._embedding_cache.popitem(last=False)

    @staticmethod
    def _snapshot_embedding(
        predictor: SAM2ImagePredictor, image_np: np.ndarray
    ) -> _Sam2EmbeddingState:
        features = predictor._features
        return _Sam2EmbeddingState(
            image_embed=features["image_embed"],
            high_res_feats=tuple(features["high_res_feats"]),
            orig_hw=tuple(predictor._orig_hw),
            is_batch=predictor._is_batch,
            input_shape=tuple(image_np.shape),
            input_dtype=str(image_np.dtype),
        )

    @staticmethod
    def _can_restore(
        entry: _Sam2EmbeddingState,
        image_np: np.ndarray,
        predictor: SAM2ImagePredictor,
    ) -> bool:
        if entry.is_batch:
            return False
        if entry.input_shape != tuple(image_np.shape):
            return False
        if entry.input_dtype != str(image_np.dtype):
            return False
        if entry.image_embed.device != predictor.device:
            return False
        return True

    @staticmethod
    def _restore_embedding(
        predictor: SAM2ImagePredictor, entry: _Sam2EmbeddingState
    ) -> None:
        predictor.reset_predictor()
        predictor._features = {
            "image_embed": entry.image_embed,
            "high_res_feats": list(entry.high_res_feats),
        }
        predictor._orig_hw = list(entry.orig_hw)
        predictor._is_image_set = True
        predictor._is_batch = False


class MicroSamAdapter(SegmenterAdapter):
    """Interface-complete skeleton for a future micro-sam backend.

    Phase 1 does not implement real inference and does not import
    micro-sam at module load time. ``load()`` performs a lazy import and
    raises a clear, Japanese-friendly error explaining that micro-sam must
    be installed in an isolated environment (Phase 3).
    """

    def __init__(self, model_name: str = "micro-sam") -> None:
        self._model_name = model_name
        self._checkpoint_hash: str | None = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def checkpoint_hash(self) -> str | None:
        return self._checkpoint_hash

    def load(self) -> None:
        try:
            importlib.import_module("micro_sam")
        except ImportError as exc:
            raise NotInstalledError(
                "micro-sam がインストールされていません。"
                "micro-sam は別環境へのインストールが必要です"
                "(Phase 3 で対応予定)。SAM2 モデルを選択してください。"
            ) from exc
        raise NotImplementedError(
            "micro-sam アダプタの推論処理は Phase 3 で実装予定です。"
        )

    def set_image(self, image_np: np.ndarray, image_key: ImageKey) -> None:
        raise NotImplementedError(
            "micro-sam アダプタの推論処理は Phase 3 で実装予定です。"
        )

    def predict(
        self,
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> PredictionResult:
        raise NotImplementedError(
            "micro-sam アダプタの推論処理は Phase 3 で実装予定です。"
        )

    def unload(self) -> None:
        return None
