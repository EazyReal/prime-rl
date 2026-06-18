from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from prime_rl.transport.types import MMRefs

if TYPE_CHECKING:
    import torch

IMAGE_MODALITY = "image"
SUPPORTED_MODALITIES = {IMAGE_MODALITY}
PROCESSED_MM_KEYS = {"pixel_values", "image_embeds", "image_features"}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Raw multimodal image refs must be file:// URIs, got {uri!r}")
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(f"file:// multimodal refs must be local paths, got {uri!r}")
    return Path(unquote(parsed.path))


def missing_file_uris(uris: Iterable[str]) -> list[str]:
    """Return missing local ``file://`` image refs; non-file refs are ignored."""
    missing: list[str] = []
    for uri in uris:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            continue
        if not Path(unquote(parsed.path)).exists():
            missing.append(uri)
    return missing


def image_file_uris_from_messages(messages: Iterable[Any]) -> list[str]:
    uris: list[str] = []
    for message in messages:
        content = _field(message, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            if _field(part, "type") != "image_url":
                continue
            image_url = _field(part, "image_url")
            url = image_url if isinstance(image_url, str) else _field(image_url, "url")
            if isinstance(url, str):
                uris.append(url)
    return uris


def _normalize_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple):
        return [_normalize_json_value(v, f"{path}[]") for v in value]
    if isinstance(value, list):
        return [_normalize_json_value(v, f"{path}[]") for v in value]
    if isinstance(value, Mapping):
        return {str(k): _normalize_json_value(v, f"{path}.{k}") for k, v in value.items()}
    raise TypeError(
        f"v1 multimodal sidecars must be JSON-safe raw image descriptors; {path} has unsupported {type(value).__name__}"
    )


def validate_raw_mm_item(item: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = PROCESSED_MM_KEYS.intersection(item)
    if forbidden:
        raise TypeError(
            "v1 multimodal sidecars must be raw image descriptors, not processed payloads "
            f"({', '.join(sorted(forbidden))})"
        )
    return {str(k): _normalize_json_value(v, str(k)) for k, v in item.items()}


def _validate_modalities(mm_items: Mapping[str, list[Any]]) -> None:
    unsupported = sorted(
        modality for modality, items in mm_items.items() if items and modality not in SUPPORTED_MODALITIES
    )
    if unsupported:
        raise NotImplementedError(
            "v1 multimodal training currently supports raw image refs only; "
            f"unsupported modalities: {', '.join(unsupported)}"
        )


def _placeholder_dict(placeholder: Any) -> dict[str, int]:
    return {
        "offset": int(_field(placeholder, "offset")),
        "length": int(_field(placeholder, "length")),
    }


def build_mm_refs(multi_modal_data: Any, messages: Iterable[Any]) -> MMRefs | None:
    mm_items = _field(multi_modal_data, "mm_items", None)
    if not mm_items:
        return None
    _validate_modalities(mm_items)

    image_items = [validate_raw_mm_item(item) for item in mm_items.get(IMAGE_MODALITY, [])]
    if not image_items:
        return None

    mm_hashes = _field(multi_modal_data, "mm_hashes", {}) or {}
    image_hashes = list(mm_hashes.get(IMAGE_MODALITY, []))
    if len(image_hashes) != len(image_items):
        raise ValueError(
            "Raw image descriptor/hash mismatch: "
            f"{len(image_items)} image descriptors but {len(image_hashes)} image hashes"
        )

    mm_placeholders = _field(multi_modal_data, "mm_placeholders", {}) or {}
    image_placeholders = [_placeholder_dict(p) for p in mm_placeholders.get(IMAGE_MODALITY, [])]
    if image_placeholders and len(image_placeholders) != len(image_items):
        raise ValueError(
            "Raw image placeholder/descriptor mismatch: "
            f"{len(image_placeholders)} placeholders but {len(image_items)} image descriptors"
        )

    uris = image_file_uris_from_messages(messages)
    if len(uris) != len(image_items):
        raise ValueError(
            "Raw image URI/descriptor mismatch: "
            f"{len(uris)} file refs in messages but {len(image_items)} image descriptors"
        )
    for uri in uris:
        file_uri_to_path(uri)

    return MMRefs(
        descriptor={
            "mm_items": {IMAGE_MODALITY: image_items},
            "mm_hashes": {IMAGE_MODALITY: image_hashes},
            "mm_placeholders": {IMAGE_MODALITY: image_placeholders},
        },
        uris=uris,
    )


def sha256_32(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _tensorize(value: Any) -> torch.Tensor:
    import torch

    if isinstance(value, torch.Tensor):
        return value.contiguous()
    return torch.as_tensor(value).contiguous()


def _expected_image_grid(item: Mapping[str, Any]) -> list[int] | None:
    grid = item.get("image_grid_thw")
    if grid is None:
        return None
    if len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    return [int(v) for v in grid]


def _patch_area(patch_size: Any) -> int:
    if isinstance(patch_size, list | tuple):
        return math.prod(int(dim) for dim in patch_size)
    size = int(patch_size)
    return size * size


def _temporal_patch_extent(temporal_patch_size: Any) -> int:
    if isinstance(temporal_patch_size, list | tuple):
        return math.prod(int(dim) for dim in temporal_patch_size)
    return int(temporal_patch_size)


class RawImageMaterializer:
    """Materialize raw image refs with the trainer model's HF image processor."""

    def __init__(self, model_name: str, *, trust_remote_code: bool):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self._image_processor = None

    @property
    def image_processor(self):
        if self._image_processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is None:
                raise ValueError(f"{self.model_name!r} does not expose an image_processor")
            self._image_processor = image_processor
        return self._image_processor

    def materialize(self, refs: MMRefs | None) -> dict[str, torch.Tensor] | None:
        if refs is None:
            return None

        image_items = refs.descriptor.get("mm_items", {}).get(IMAGE_MODALITY, [])
        image_hashes = refs.descriptor.get("mm_hashes", {}).get(IMAGE_MODALITY, [])
        if not image_items:
            return None
        if len(refs.uris) != len(image_items) or len(image_hashes) != len(image_items):
            raise ValueError(
                "Raw image refs must have matching URI, descriptor, and hash counts "
                f"(uris={len(refs.uris)}, descriptors={len(image_items)}, hashes={len(image_hashes)})"
            )

        from PIL import Image

        images = []
        for uri, item, expected_hash in zip(refs.uris, image_items, image_hashes, strict=True):
            validate_raw_mm_item(item)
            raw = file_uri_to_path(uri).read_bytes()
            actual_hash = sha256_32(raw)
            if actual_hash != expected_hash:
                raise ValueError(f"Raw image hash mismatch for {uri}: expected {expected_hash}, got {actual_hash}")
            images.append(Image.open(BytesIO(raw)).convert("RGB"))

        processed = self.image_processor(images=images, return_tensors="pt")
        tensors = {str(k): _tensorize(v) for k, v in dict(processed).items()}

        expected_grids = [_expected_image_grid(item) for item in image_items]
        if any(grid is not None for grid in expected_grids):
            if "image_grid_thw" not in tensors:
                raise ValueError("Image descriptors include image_grid_thw but the trainer processor did not return it")
            actual_grids = tensors["image_grid_thw"].tolist()
            for idx, expected in enumerate(expected_grids):
                if expected is not None and actual_grids[idx] != expected:
                    raise ValueError(
                        f"Image grid mismatch at index {idx}: expected {expected}, got {actual_grids[idx]}"
                    )

        return tensors

    def placeholder_feature_dim(self) -> int:
        image_processor = self.image_processor
        patch_size = getattr(image_processor, "patch_size", None)
        temporal_patch_size = getattr(image_processor, "temporal_patch_size", None)
        image_mean = getattr(image_processor, "image_mean", None)
        channels = len(image_mean) if image_mean is not None else getattr(image_processor, "num_channels", 3)
        if patch_size is None or temporal_patch_size is None:
            raise ValueError(
                "Cannot synthesize raw image placeholders without image processor patch_size and temporal_patch_size"
            )
        return int(channels) * _temporal_patch_extent(temporal_patch_size) * _patch_area(patch_size)

    def synthesize_placeholder(self, refs: MMRefs | None) -> dict[str, torch.Tensor] | None:
        """Build zero-valued Qwen-style image tensors from raw descriptor geometry.

        This recovery path is only used when raw image files disappear before
        trainer materialization. The original ``image_grid_thw`` is preserved so
        the model sees the same placeholder geometry used during rollout
        tokenization, while the dataloader masks the affected microbatch loss.
        """
        if refs is None:
            return None

        import torch

        image_items = refs.descriptor.get("mm_items", {}).get(IMAGE_MODALITY, [])
        image_hashes = refs.descriptor.get("mm_hashes", {}).get(IMAGE_MODALITY, [])
        if not image_items:
            return None
        if len(refs.uris) != len(image_items) or len(image_hashes) != len(image_items):
            raise ValueError(
                "Raw image refs must have matching URI, descriptor, and hash counts "
                f"(uris={len(refs.uris)}, descriptors={len(image_items)}, hashes={len(image_hashes)})"
            )

        feature_dim = self.placeholder_feature_dim()
        pixel_values: list[torch.Tensor] = []
        image_grid_thw: list[list[int]] = []
        for idx, item in enumerate(image_items):
            validate_raw_mm_item(item)
            grid = _expected_image_grid(item)
            if grid is None:
                raise ValueError(f"Cannot synthesize image placeholder {idx}: image_grid_thw is missing")
            if len(grid) != 3 or any(dim <= 0 for dim in grid):
                raise ValueError(f"Cannot synthesize image placeholder {idx}: invalid image_grid_thw={grid}")
            pixel_values.append(torch.zeros((math.prod(grid), feature_dim), dtype=torch.float32))
            image_grid_thw.append(grid)

        return {
            "pixel_values": torch.cat(pixel_values, dim=0).contiguous(),
            "image_grid_thw": torch.tensor(image_grid_thw, dtype=torch.long),
        }
