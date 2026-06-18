from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.rl.data import DataLoader
from prime_rl.transport.types import MicroBatch, MMRefs
from prime_rl.utils.mm import RawImageMaterializer, missing_file_uris


class _ImageProcessor:
    patch_size = 2
    temporal_patch_size = 2
    image_mean = [0.5, 0.5, 0.5]


class _MissingMaterializer:
    def materialize(self, refs):
        raise FileNotFoundError("missing image")

    def synthesize_placeholder(self, refs):
        return {
            "pixel_values": torch.zeros((1, 24), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


def _refs(uri: str = "file:///tmp/missing-image.png") -> MMRefs:
    return MMRefs(
        descriptor={
            "mm_items": {"image": [{"image_grid_thw": [[1, 1, 1]]}]},
            "mm_hashes": {"image": ["a" * 32]},
        },
        uris=[uri],
    )


def _loader(policy: str = "placeholder_zero_loss") -> DataLoader:
    loader = object.__new__(DataLoader)
    loader.multi_run_manager = SimpleNamespace(max_runs=1)
    loader.mm_materializer = _MissingMaterializer()
    loader.missing_mm_image_policy = policy
    loader.last_mm_materialize_time = 0.0
    loader.last_mm_images_materialized = 0
    loader.last_mm_images_placeholdered = 0
    return loader


def _micro_batch() -> MicroBatch:
    return MicroBatch(
        input_ids=[10, 11, 12],
        loss_mask=[False, True, True],
        advantages=[1.5, 1.5, 1.5],
        inference_logprobs=[0.0, -0.1, -0.2],
        position_ids=[0, 1, 2],
        temperatures=[1.0, 1.0, 1.0],
        env_names=["env", "env", "env"],
        lora_num_tokens=[3],
        mm_refs=_refs(),
        mm_token_type_ids=[0, 1, 0],
    )


def test_missing_file_uris_reports_missing_local_refs(tmp_path):
    existing = tmp_path / "image.png"
    existing.write_bytes(b"image")

    assert missing_file_uris(
        [existing.as_uri(), (tmp_path / "missing.png").as_uri(), "https://example.test/i.png"]
    ) == [(tmp_path / "missing.png").as_uri()]


def test_raw_image_materializer_synthesizes_qwen_placeholder_from_descriptor():
    materializer = RawImageMaterializer("unused", trust_remote_code=False)
    materializer._image_processor = _ImageProcessor()

    mm_kwargs = materializer.synthesize_placeholder(
        MMRefs(
            descriptor={
                "mm_items": {"image": [{"image_grid_thw": [[1, 2, 3]]}]},
                "mm_hashes": {"image": ["b" * 32]},
            },
            uris=["file:///tmp/missing-image.png"],
        )
    )

    assert mm_kwargs is not None
    assert mm_kwargs["pixel_values"].shape == (6, 24)
    assert not bool(mm_kwargs["pixel_values"].any())
    assert mm_kwargs["image_grid_thw"].tolist() == [[1, 2, 3]]


def test_dataloader_uses_zero_loss_placeholder_for_missing_raw_image():
    tensor_batch = _loader()._micro_batch_to_tensor(_micro_batch())

    assert tensor_batch["mm_kwargs"] is not None
    assert tensor_batch["mm_kwargs"]["pixel_values"].shape == (1, 24)
    assert tensor_batch["loss_mask"].tolist() == [[False, False, False]]
    assert tensor_batch["advantages"].tolist() == [[0.0, 0.0, 0.0]]
    assert tensor_batch["mm_token_type_ids"].tolist() == [[0, 1, 0]]


def test_dataloader_can_fail_fast_on_missing_raw_image():
    with pytest.raises(FileNotFoundError):
        _loader(policy="error")._micro_batch_to_tensor(_micro_batch())
