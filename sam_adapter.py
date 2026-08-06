"""Segmenter adapter abstraction for the crystal annotation tool.

This module isolates all segmentation-backend concerns (model loading,
device placement, embedding computation/caching, and mask prediction)
behind a small ``SegmenterAdapter`` interface so that ``0_annotator_sam2.py``
never talks to SAM2 (or, in the future, micro-sam) internals directly.

``Sam2Adapter`` runs in the main process. ``MicroSamAdapter`` delegates to a
persistent subprocess in the isolated micro-sam environment, so micro-sam is
never imported into the annotator process.

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
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from cellpose_client import (
    CellposeClientError,
    CellposeProtocolError,
    CellposeWorkerClient,
    cellpose_interpreter_exists,
)
from matsam_client import (
    MatSamClientError,
    MatSamProtocolError,
    MatSamWorkerClient,
    matsam_interpreter_exists,
)
from micro_sam_client import (
    MicroSamClientError,
    MicroSamProtocolError,
    MicroSamWorkerClient,
    micro_sam_interpreter_exists,
)

if TYPE_CHECKING:
    from auto_annotation import AmgParams, AutoMask

DEFAULT_EMBEDDING_CACHE_SIZE = 1
HASH_DIGEST_SIZE = 32
HASH_CHUNK_SIZE = 1 << 20  # 1 MiB
PREPROCESSING_PIPELINE_VERSION = "transform_input_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
MATSAM_INTERACTIVE_UNSUPPORTED_MESSAGE = (
    "MatSAMは自動分割専用です。ポイントプロンプトによる対話的セグメンテーションには"
    "対応していません。SAM2またはmicro-samモデルを選択してください。"
)
CELLPOSE_INTERACTIVE_UNSUPPORTED_MESSAGE = (
    "Cellposeは自動分割専用です。ポイントプロンプトによる対話的セグメンテーションには"
    "対応していません。SAM2またはmicro-samモデルを選択してください。"
)

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

MICRO_SAM_MODEL_NAMES: tuple[str, ...] = (
    "vit_b_em_organelles",
    "vit_l_em_organelles",
    "vit_b",
)


@dataclass(frozen=True)
class MicroSamModelSpec:
    """Static automatic-mode capabilities for one micro-sam model."""

    name: str
    supported_auto_modes: tuple[str, ...]
    default_auto_mode: str


MICRO_SAM_MODEL_SPECS = {
    "vit_b_em_organelles": MicroSamModelSpec(
        "vit_b_em_organelles",
        ("ais", "apg", "amg"),
        "ais",
    ),
    "vit_l_em_organelles": MicroSamModelSpec(
        "vit_l_em_organelles",
        ("ais", "apg", "amg"),
        "ais",
    ),
    "vit_b": MicroSamModelSpec("vit_b", ("amg",), "amg"),
}
MICRO_SAM_CHECKPOINT_SIZE_MB: dict[str, int] = {
    "vit_b": 360,
    "vit_b_em_organelles": 360,
    "vit_l_em_organelles": 1200,
}

MATSAM_MODEL_NAMES: tuple[str, ...] = ("matsam:vit_b", "matsam:vit_h")


@dataclass(frozen=True)
class MatSamModelSpec:
    """MatSAM model identity and local SAM-v1 checkpoint path."""

    name: str
    sam_variant: str
    checkpoint_path: str


MATSAM_MODEL_SPECS = {
    "matsam:vit_b": MatSamModelSpec(
        "matsam:vit_b",
        "vit_b",
        "external/matsam/checkpoints/sam_vit_b_01ec64.pth",
    ),
    "matsam:vit_h": MatSamModelSpec(
        "matsam:vit_h",
        "vit_h",
        "external/matsam/checkpoints/sam_vit_h_4b8939.pth",
    ),
}

CELLPOSE_MODEL_NAMES: tuple[str, ...] = (
    "cellpose:cpsam_v2",
    "cellpose:cpdino",
    "cellpose:cpdino-vitb",
)


@dataclass(frozen=True)
class CellposeModelSpec:
    """Cellpose model identity and raw worker model name."""

    name: str
    cellpose_model_name: str


CELLPOSE_MODEL_SPECS: dict[str, CellposeModelSpec] = {
    "cellpose:cpsam_v2": CellposeModelSpec(
        "cellpose:cpsam_v2",
        "cpsam_v2",
    ),
    "cellpose:cpdino": CellposeModelSpec(
        "cellpose:cpdino",
        "cpdino",
    ),
    "cellpose:cpdino-vitb": CellposeModelSpec(
        "cellpose:cpdino-vitb",
        "cpdino-vitb",
    ),
}


@dataclass(frozen=True)
class MatSamParams:
    """Automatic segmentation parameters matching MatSAM's notebook defaults."""

    layers: int = 0
    scales: int = 3
    n_per_side_base: int = 32
    method_type: int = 1
    pred_iou_thresh: float = 0.90
    stability_score_thresh: float = 0.92
    box_nms_thresh: float = 0.80
    min_mask_region_area: int = 0
    max_image_size: int = 1024
    # This is SAM's automatic mask generator's dominant peak-VRAM driver;
    # lower values such as 32-64 trade inference speed for substantially less VRAM.
    points_per_batch: int = 256

    def to_worker_params(self) -> dict[str, object]:
        """Return the exact parameter schema accepted by the MatSAM worker."""
        return {
            "layers": self.layers,
            "scales": self.scales,
            "n_per_side_base": self.n_per_side_base,
            "method_type": self.method_type,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "box_nms_thresh": self.box_nms_thresh,
            "min_mask_region_area": self.min_mask_region_area,
            "max_image_size": self.max_image_size,
            "points_per_batch": self.points_per_batch,
        }


@dataclass(frozen=True)
class CellposeParams:
    """Automatic segmentation parameters for Cellpose."""

    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_mask_region_area: int = 0
    max_image_size: int = 1024

    def to_worker_params(self) -> dict[str, object]:
        """Return the exact parameter schema accepted by the Cellpose worker."""
        return {
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "min_mask_region_area": self.min_mask_region_area,
            "max_image_size": self.max_image_size,
        }


@dataclass(frozen=True)
class MicroSamParams:
    """Mode-discriminated automatic segmentation parameters for micro-sam."""

    mode: str = "ais"
    min_mask_region_area: int = 100
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    box_nms_thresh: float = 0.7
    center_distance_threshold: float = 0.5
    boundary_distance_threshold: float = 0.5
    foreground_threshold: float = 0.5
    foreground_smoothing: float = 1.0
    distance_smoothing: float = 1.6
    multimasking: bool = False
    batch_size: int = 32
    nms_threshold: float = 0.9

    def to_worker_params(self) -> dict[str, object]:
        """Return only fields accepted by the selected worker mode."""
        common: dict[str, object] = {
            "mode": self.mode,
            "min_mask_region_area": self.min_mask_region_area,
        }
        if self.mode == "amg":
            return {
                **common,
                "points_per_side": self.points_per_side,
                "pred_iou_thresh": self.pred_iou_thresh,
                "stability_score_thresh": self.stability_score_thresh,
                "box_nms_thresh": self.box_nms_thresh,
            }
        if self.mode == "ais":
            return {
                **common,
                "center_distance_threshold": self.center_distance_threshold,
                "boundary_distance_threshold": self.boundary_distance_threshold,
                "foreground_threshold": self.foreground_threshold,
                "foreground_smoothing": self.foreground_smoothing,
                "distance_smoothing": self.distance_smoothing,
            }
        if self.mode == "apg":
            return {
                **common,
                "center_distance_threshold": self.center_distance_threshold,
                "boundary_distance_threshold": self.boundary_distance_threshold,
                "foreground_threshold": self.foreground_threshold,
                "multimasking": self.multimasking,
                "batch_size": self.batch_size,
                "nms_threshold": self.nms_threshold,
            }
        raise ValueError(f"Unsupported micro-sam mode: {self.mode}")


class NotInstalledError(RuntimeError):
    """Raised when an optional segmenter backend dependency is unavailable."""


class MicroSamDownloadCancelledError(RuntimeError):
    """Raised when the user declines a first-time checkpoint download."""


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
AutoSegmentationResult = list["AutoMask"]


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
        """Load model weights and initialize the predictor.

        ``MicroSamAdapter`` additionally accepts a backend-specific
        ``confirm_download`` callback for first-time checkpoint downloads.
        """

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
        self,
        image_np: np.ndarray,
        image_key: ImageKey,
        *,
        params: "AmgParams | MicroSamParams | MatSamParams | CellposeParams | None" = None,
    ) -> AutoSegmentationResult:
        """Generate automatic masks when the backend supports AMG.

        Implementations return ``AutoMask`` objects whose ``points`` are ready
        for direct conversion to annotator polygons.
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
            device=self._device,
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

    def generate_auto(
        self,
        image_np: np.ndarray,
        image_key: ImageKey,
        *,
        params: "AmgParams | MicroSamParams | MatSamParams | None" = None,
    ) -> AutoSegmentationResult:
        """Run SAM2 AMG with the already-loaded model instance.

        ``image_key`` is accepted for adapter API consistency. AMG computes
        its own dense image embedding and does not use the prompt predictor's
        embedding cache.
        """
        if self._sam is None:
            raise RuntimeError(
                f"SAM2 model ({self.name}) is not loaded; call load() first."
            )

        from auto_annotation import AmgParams, Sam2AmgAutoAnnotator

        if isinstance(params, MicroSamParams):
            raise TypeError("Sam2Adapter requires AmgParams, not MicroSamParams")
        resolved_params = params or AmgParams()
        annotator = Sam2AmgAutoAnnotator.from_model(
            self._sam,
            resolved_params,
        )
        return annotator.generate(image_np)

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
    """Segmenter adapter backed by a persistent isolated micro-sam worker."""

    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        if model_name not in MICRO_SAM_MODEL_SPECS:
            raise ValueError(f"Unsupported micro-sam model: {model_name}")
        self._model_name = model_name
        self._device = device
        self._checkpoint_hash: str | None = None
        self._client: MicroSamWorkerClient | None = None
        self._supported_auto_modes = MICRO_SAM_MODEL_SPECS[
            model_name
        ].supported_auto_modes

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def checkpoint_hash(self) -> str | None:
        return self._checkpoint_hash

    @property
    def supported_auto_modes(self) -> tuple[str, ...]:
        """Return automatic modes supported by the loaded model."""
        return self._supported_auto_modes

    @property
    def default_auto_mode(self) -> str:
        """Return the model's default automatic mode."""
        return MICRO_SAM_MODEL_SPECS[self.name].default_auto_mode

    def load(
        self,
        *,
        confirm_download: Callable[[str], bool] | None = None,
        client: MicroSamWorkerClient | None = None,
    ) -> None:
        if not micro_sam_interpreter_exists():
            raise NotInstalledError(
                "micro-sam がインストールされていません。"
                "分離環境 envs/micro_sam が必要です。"
                "SAM2 モデルを選択してください。"
            )
        if self._client is not None:
            return
        if client is None:
            client = MicroSamWorkerClient()
        try:
            probe_metadata = client.probe(self.name)
            cached = probe_metadata.get("cached")
            if probe_metadata.get("model_type") != self.name or not isinstance(
                cached, bool
            ):
                raise MicroSamProtocolError(
                    "probe",
                    "invalid_result",
                    "worker cache metadata did not match the requested model",
                    stderr_tail=client.stderr_tail,
                    fatal=True,
                )
            if not cached and confirm_download is not None:
                proceed = confirm_download(self.name)
                if not proceed:
                    client.terminate()
                    raise MicroSamDownloadCancelledError(
                        f"{self.name} のダウンロードがキャンセルされました"
                    )
            metadata = client.init(self.name, self._device, cached=cached)
            checkpoint_hash = metadata.get("checkpoint_hash")
            supported_modes = metadata.get("supported_auto_modes")
            if (
                metadata.get("model_type") != self.name
                or not isinstance(checkpoint_hash, str)
                or len(checkpoint_hash) != 64
                or any(
                    character not in "0123456789abcdef" for character in checkpoint_hash
                )
                or supported_modes != list(self._supported_auto_modes)
            ):
                raise MicroSamProtocolError(
                    "init",
                    "invalid_result",
                    "worker metadata did not match the requested model",
                    stderr_tail=client.stderr_tail,
                    fatal=True,
                )
        except Exception:
            client.terminate()
            raise
        self._client = client
        self._checkpoint_hash = checkpoint_hash

    def set_image(self, image_np: np.ndarray, image_key: ImageKey) -> None:
        client = self._require_client()
        try:
            client.set_image(image_np, image_key)
        except MicroSamClientError as exc:
            self._handle_client_error(exc)

    def predict(
        self,
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> PredictionResult:
        client = self._require_client()
        flattened_box = box
        if box is not None:
            box_array = np.asarray(box)
            if box_array.shape == (1, 4):
                flattened_box = box_array[0]
            elif box_array.shape != (4,):
                raise ValueError("micro-sam box must have shape (4,) or (1, 4)")
        try:
            return client.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=flattened_box,
                multimask_output=multimask_output,
            )
        except MicroSamClientError as exc:
            self._handle_client_error(exc)

    def generate_auto(
        self,
        image_np: np.ndarray,
        image_key: ImageKey,
        *,
        params: "AmgParams | MicroSamParams | MatSamParams | None" = None,
    ) -> AutoSegmentationResult:
        """Generate polygon-only automatic masks through the worker."""
        if params is not None and not isinstance(params, MicroSamParams):
            raise TypeError("MicroSamAdapter requires MicroSamParams")
        resolved_params = params or MicroSamParams(mode=self.default_auto_mode)
        if resolved_params.mode not in self._supported_auto_modes:
            raise ValueError(
                f"{self.name} does not support micro-sam mode {resolved_params.mode}"
            )
        client = self._require_client()
        try:
            result = client.generate_auto(
                image_np,
                image_key,
                resolved_params.to_worker_params(),
            )
            return self._decode_auto_masks(result, client)
        except MicroSamClientError as exc:
            self._handle_client_error(exc)

    def unload(self) -> None:
        client = self._client
        self._client = None
        self._checkpoint_hash = None
        if client is None:
            return
        try:
            client.unload()
            client.shutdown()
        except MicroSamClientError:
            client.terminate()

    def _require_client(self) -> MicroSamWorkerClient:
        if self._client is None or not self._client.is_alive:
            self._client = None
            self._checkpoint_hash = None
            raise RuntimeError(
                f"micro-sam worker ({self.name}) is not loaded; call load() first."
            )
        return self._client

    def _handle_client_error(self, error: MicroSamClientError) -> None:
        if error.fatal:
            client = self._client
            self._client = None
            self._checkpoint_hash = None
            if client is not None:
                client.terminate()
        raise error

    @staticmethod
    def _decode_auto_masks(
        result: dict[str, object],
        client: MicroSamWorkerClient,
    ) -> AutoSegmentationResult:
        from auto_annotation.base import AutoMask

        if set(result) != {"mode", "image_shape", "masks"}:
            client.terminate()
            raise MicroSamProtocolError(
                "generate_auto",
                "invalid_result",
                "worker automatic result has an invalid schema",
                stderr_tail=client.stderr_tail,
                fatal=True,
            )
        records = result.get("masks")
        if not isinstance(records, list):
            client.terminate()
            raise MicroSamProtocolError(
                "generate_auto",
                "invalid_result",
                "worker masks result must be a list",
                stderr_tail=client.stderr_tail,
                fatal=True,
            )
        auto_masks: list[AutoMask] = []
        try:
            for record in records:
                if not isinstance(record, dict) or set(record) != {
                    "points",
                    "score",
                    "area",
                }:
                    raise ValueError("invalid automatic mask record")
                points_value = record["points"]
                if not isinstance(points_value, list):
                    raise ValueError("automatic mask points must be a list")
                points = [[float(x), float(y)] for x, y in points_value]
                score_value = record["score"]
                area_value = record["area"]
                auto_masks.append(
                    AutoMask(
                        segmentation=None,
                        score=(float(score_value) if score_value is not None else None),
                        area=int(area_value),
                        points=points,
                    )
                )
        except (TypeError, ValueError) as exc:
            client.terminate()
            raise MicroSamProtocolError(
                "generate_auto",
                "invalid_result",
                str(exc),
                stderr_tail=client.stderr_tail,
                fatal=True,
            ) from exc
        return auto_masks


class MatSamAdapter(SegmenterAdapter):
    """Segmenter adapter backed by a persistent isolated MatSAM worker."""

    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        if model_name not in MATSAM_MODEL_SPECS:
            raise ValueError(f"Unsupported MatSAM model: {model_name}")
        self._model_spec = MATSAM_MODEL_SPECS[model_name]
        self._device = device
        self._checkpoint_hash: str | None = None
        self._client: MatSamWorkerClient | None = None

    @property
    def name(self) -> str:
        return self._model_spec.name

    @property
    def checkpoint_hash(self) -> str | None:
        return self._checkpoint_hash

    def load(self, *, client: MatSamWorkerClient | None = None) -> None:
        if not matsam_interpreter_exists():
            raise NotInstalledError(
                "MatSAM がインストールされていません。"
                "分離環境 envs/matsam が必要です。"
                "SAM2またはmicro-samモデルを選択してください。"
            )
        checkpoint_path = self._resolve_checkpoint_path()
        if not checkpoint_path.is_file():
            raise NotInstalledError(
                f"MatSAMチェックポイントが見つかりません: {checkpoint_path}。"
                "チェックポイントを配置するか、別のSAMモデルを選択してください。"
            )
        if self._client is not None:
            return
        if client is None:
            client = MatSamWorkerClient()
        try:
            checkpoint_path_text = str(checkpoint_path)
            probe_metadata = client.probe(
                self._model_spec.sam_variant,
                checkpoint_path_text,
            )
            if probe_metadata != {
                "sam_variant": self._model_spec.sam_variant,
                "checkpoint_path": checkpoint_path_text,
                "cached": True,
            }:
                raise MatSamProtocolError(
                    "probe",
                    "invalid_result",
                    "worker checkpoint metadata did not match the requested model",
                    stderr_tail=client.stderr_tail,
                    fatal=True,
                )
            checkpoint_hash = hash_file_bytes(checkpoint_path_text)
            metadata = client.init(
                self._model_spec.sam_variant,
                checkpoint_path_text,
                self._device,
            )
            identity_hash = metadata.get("checkpoint_hash")
            if (
                set(metadata)
                != {"sam_variant", "device", "checkpoint_path", "checkpoint_hash"}
                or metadata.get("sam_variant") != self._model_spec.sam_variant
                or metadata.get("checkpoint_path") != checkpoint_path_text
                or not isinstance(metadata.get("device"), str)
                or not isinstance(identity_hash, str)
                or len(identity_hash) != 64
                or any(
                    character not in "0123456789abcdef" for character in identity_hash
                )
            ):
                raise MatSamProtocolError(
                    "init",
                    "invalid_result",
                    "worker metadata did not match the requested model",
                    stderr_tail=client.stderr_tail,
                    fatal=True,
                )
        except Exception:
            client.terminate()
            raise
        self._client = client
        self._checkpoint_hash = checkpoint_hash

    def set_image(self, image_np: np.ndarray, image_key: ImageKey) -> None:
        """Reject interactive embedding activation before a worker round-trip."""
        raise NotImplementedError(MATSAM_INTERACTIVE_UNSUPPORTED_MESSAGE)

    def predict(
        self,
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> PredictionResult:
        """Reject point- and box-prompt prediction immediately."""
        raise NotImplementedError(MATSAM_INTERACTIVE_UNSUPPORTED_MESSAGE)

    def generate_auto(
        self,
        image_np: np.ndarray,
        image_key: ImageKey,
        *,
        params: "AmgParams | MicroSamParams | MatSamParams | None" = None,
    ) -> AutoSegmentationResult:
        """Generate full-resolution polygon masks through the MatSAM worker."""
        if params is not None and not isinstance(params, MatSamParams):
            raise TypeError("MatSamAdapter requires MatSamParams")
        resolved_params = params or MatSamParams()
        client = self._require_client()
        try:
            result = client.generate_auto(
                image_np,
                image_key,
                resolved_params.to_worker_params(),
            )
            return self._decode_auto_masks(result, client, image_np.shape[:2])
        except MatSamClientError as exc:
            self._handle_client_error(exc)

    def unload(self) -> None:
        client = self._client
        self._client = None
        self._checkpoint_hash = None
        if client is None:
            return
        try:
            client.unload()
            client.shutdown()
        except MatSamClientError:
            client.terminate()

    def _resolve_checkpoint_path(self) -> Path:
        checkpoint_path = Path(self._model_spec.checkpoint_path)
        if checkpoint_path.is_absolute():
            return checkpoint_path
        return (REPO_ROOT / checkpoint_path).resolve()

    def _require_client(self) -> MatSamWorkerClient:
        if self._client is None or not self._client.is_alive:
            self._client = None
            self._checkpoint_hash = None
            raise RuntimeError(
                f"MatSAM worker ({self.name}) is not loaded; call load() first."
            )
        return self._client

    def _handle_client_error(self, error: MatSamClientError) -> None:
        if error.fatal:
            client = self._client
            self._client = None
            self._checkpoint_hash = None
            if client is not None:
                client.terminate()
        raise error

    @staticmethod
    def _decode_auto_masks(
        result: dict[str, object],
        client: MatSamWorkerClient,
        expected_image_shape: tuple[int, int],
    ) -> AutoSegmentationResult:
        from auto_annotation.base import AutoMask

        if (
            set(result) != {"mode", "image_shape", "masks"}
            or result.get("mode") != "matsam"
            or result.get("image_shape") != list(expected_image_shape)
        ):
            client.terminate()
            raise MatSamProtocolError(
                "generate_auto",
                "invalid_result",
                "worker automatic result has an invalid schema",
                stderr_tail=client.stderr_tail,
                fatal=True,
            )
        records = result.get("masks")
        if not isinstance(records, list):
            client.terminate()
            raise MatSamProtocolError(
                "generate_auto",
                "invalid_result",
                "worker masks result must be a list",
                stderr_tail=client.stderr_tail,
                fatal=True,
            )
        auto_masks: list[AutoMask] = []
        try:
            for record in records:
                if not isinstance(record, dict) or set(record) != {
                    "points",
                    "score",
                    "area",
                }:
                    raise ValueError("invalid automatic mask record")
                points_value = record["points"]
                if not isinstance(points_value, list):
                    raise ValueError("automatic mask points must be a list")
                points = [[float(x), float(y)] for x, y in points_value]
                score_value = record["score"]
                area_value = record["area"]
                auto_masks.append(
                    AutoMask(
                        segmentation=None,
                        score=(float(score_value) if score_value is not None else None),
                        area=int(area_value),
                        points=points,
                    )
                )
        except (TypeError, ValueError) as exc:
            client.terminate()
            raise MatSamProtocolError(
                "generate_auto",
                "invalid_result",
                str(exc),
                stderr_tail=client.stderr_tail,
                fatal=True,
            ) from exc
        return auto_masks


class CellposeAdapter(SegmenterAdapter):
    """Segmenter adapter backed by a persistent isolated Cellpose worker."""

    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        if model_name not in CELLPOSE_MODEL_NAMES:
            raise ValueError(f"Unsupported Cellpose model: {model_name}")
        self._model_spec = CELLPOSE_MODEL_SPECS[model_name]
        self._device = device
        self._checkpoint_hash: str | None = None
        self._client: CellposeWorkerClient | None = None

    @property
    def name(self) -> str:
        return self._model_spec.name

    @property
    def checkpoint_hash(self) -> str | None:
        return self._checkpoint_hash

    def load(self, *, client: CellposeWorkerClient | None = None) -> None:
        if not cellpose_interpreter_exists():
            raise NotInstalledError(
                "Cellpose がインストールされていません。"
                "分離環境 envs/cellpose が必要です。"
                "SAM2またはmicro-sam、MatSAMモデルを選択してください。"
            )
        if self._client is not None:
            return
        if client is None:
            client = CellposeWorkerClient()
        raw_model_name = self._model_spec.cellpose_model_name
        try:
            probe_metadata = client.probe(raw_model_name)
            cached = probe_metadata.get("cached")
            if (
                set(probe_metadata) != {"model_name", "cached"}
                or probe_metadata.get("model_name") != raw_model_name
                or not isinstance(cached, bool)
            ):
                raise CellposeProtocolError(
                    "probe",
                    "invalid_result",
                    "worker dependency metadata did not match the requested model",
                    stderr_tail=client.stderr_tail,
                    fatal=True,
                )
            if not cached:
                client.terminate()
                raise NotInstalledError(
                    f"{self.name} には dinov3 が必要ですが、Cellpose分離環境に"
                    "インストールされていません。次のコマンドで導入してください: "
                    "uv pip install --python envs/cellpose/bin/python "
                    "git+https://github.com/facebookresearch/dinov3。"
                    "代わりに cellpose:cpsam_v2 を選択できます。"
                )
            metadata = client.init(raw_model_name, self._device)
            checkpoint_hash = metadata.get("checkpoint_hash")
            if (
                set(metadata) != {"model_name", "device", "checkpoint_hash"}
                or metadata.get("model_name") != raw_model_name
                or not isinstance(metadata.get("device"), str)
                or not isinstance(checkpoint_hash, str)
                or len(checkpoint_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in checkpoint_hash
                )
            ):
                raise CellposeProtocolError(
                    "init",
                    "invalid_result",
                    "worker metadata did not match the requested model",
                    stderr_tail=client.stderr_tail,
                    fatal=True,
                )
        except Exception:
            client.terminate()
            raise
        self._client = client
        self._checkpoint_hash = checkpoint_hash

    def set_image(self, image_np: np.ndarray, image_key: ImageKey) -> None:
        """Reject interactive embedding activation before a worker round-trip."""
        raise NotImplementedError(CELLPOSE_INTERACTIVE_UNSUPPORTED_MESSAGE)

    def predict(
        self,
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> PredictionResult:
        """Reject point- and box-prompt prediction immediately."""
        raise NotImplementedError(CELLPOSE_INTERACTIVE_UNSUPPORTED_MESSAGE)

    def generate_auto(
        self,
        image_np: np.ndarray,
        image_key: ImageKey,
        *,
        params: "AmgParams | MicroSamParams | MatSamParams | CellposeParams | None" = None,
    ) -> AutoSegmentationResult:
        """Generate full-resolution polygon masks through the Cellpose worker."""
        if params is not None and not isinstance(params, CellposeParams):
            raise TypeError("CellposeAdapter requires CellposeParams")
        resolved_params = params or CellposeParams()
        client = self._require_client()
        try:
            result = client.generate_auto(
                image_np,
                image_key,
                resolved_params.to_worker_params(),
            )
            return self._decode_auto_masks(result, client, image_np.shape[:2])
        except CellposeClientError as exc:
            self._handle_client_error(exc)

    def unload(self) -> None:
        client = self._client
        self._client = None
        self._checkpoint_hash = None
        if client is None:
            return
        try:
            client.unload()
            client.shutdown()
        except CellposeClientError:
            client.terminate()

    def _require_client(self) -> CellposeWorkerClient:
        if self._client is None or not self._client.is_alive:
            self._client = None
            self._checkpoint_hash = None
            raise RuntimeError(
                f"Cellpose worker ({self.name}) is not loaded; call load() first."
            )
        return self._client

    def _handle_client_error(self, error: CellposeClientError) -> None:
        if error.fatal:
            client = self._client
            self._client = None
            self._checkpoint_hash = None
            if client is not None:
                client.terminate()
        raise error

    @staticmethod
    def _decode_auto_masks(
        result: dict[str, object],
        client: CellposeWorkerClient,
        expected_image_shape: tuple[int, int],
    ) -> AutoSegmentationResult:
        from auto_annotation.base import AutoMask

        if (
            set(result) != {"mode", "image_shape", "masks"}
            or result.get("mode") != "cellpose"
            or result.get("image_shape") != list(expected_image_shape)
        ):
            client.terminate()
            raise CellposeProtocolError(
                "generate_auto",
                "invalid_result",
                "worker automatic result has an invalid schema",
                stderr_tail=client.stderr_tail,
                fatal=True,
            )
        records = result.get("masks")
        if not isinstance(records, list):
            client.terminate()
            raise CellposeProtocolError(
                "generate_auto",
                "invalid_result",
                "worker masks result must be a list",
                stderr_tail=client.stderr_tail,
                fatal=True,
            )
        auto_masks: list[AutoMask] = []
        try:
            for record in records:
                if not isinstance(record, dict) or set(record) != {
                    "points",
                    "score",
                    "area",
                }:
                    raise ValueError("invalid automatic mask record")
                points_value = record["points"]
                if not isinstance(points_value, list):
                    raise ValueError("automatic mask points must be a list")
                points = [[float(x), float(y)] for x, y in points_value]
                score_value = record["score"]
                area_value = record["area"]
                auto_masks.append(
                    AutoMask(
                        segmentation=None,
                        score=(float(score_value) if score_value is not None else None),
                        area=int(area_value),
                        points=points,
                    )
                )
        except (TypeError, ValueError) as exc:
            client.terminate()
            raise CellposeProtocolError(
                "generate_auto",
                "invalid_result",
                str(exc),
                stderr_tail=client.stderr_tail,
                fatal=True,
            ) from exc
        return auto_masks
