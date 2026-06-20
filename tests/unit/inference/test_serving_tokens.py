"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM 0.20's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``serialize_routed_experts`` round-trips a compact raw-byte payload.
    * The subclass attaches its overrides without monkey-patching the parent.
    * ``_client_set_max_tokens`` distinguishes raw-body shapes correctly.
"""

from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import pybase64
from vllm.entrypoints.serve.disagg.protocol import GenerateResponse, GenerateResponseChoice

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlServingTokens,
    _client_set_max_tokens,
    _GenerateRoutedExpertsCapture,
)


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


class _FakeRawRequest:
    def __init__(self, body):
        self._body = body
        self._raise = isinstance(body, Exception)

    async def json(self):
        if self._raise:
            raise self._body
        return self._body


async def _empty_request_outputs():
    if False:
        yield


def test_subclass_only_overrides_serve_tokens():
    assert PrimeRlServingTokens.serve_tokens is not PrimeRlServingTokens.__mro__[1].serve_tokens
    assert (
        PrimeRlServingTokens.serve_tokens_full_generator
        is not PrimeRlServingTokens.__mro__[1].serve_tokens_full_generator
    )


def test_serialize_routed_experts_uses_compact_raw_payload():
    routed_experts = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int64,
    )

    encoded = serialize_routed_experts(routed_experts)
    assert encoded is not None

    decoded = _decode_routed_experts(encoded)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, routed_experts)


def test_generate_response_post_process_replaces_upstream_routed_experts():
    compact_routed_experts = {"data": "AQID", "shape": [1, 1, 3], "start": 0}
    capture = _GenerateRoutedExpertsCapture(_empty_request_outputs())
    capture.routed_experts[0] = compact_routed_experts
    response = GenerateResponse(
        request_id="request-id",
        choices=[
            GenerateResponseChoice(
                index=0,
                token_ids=[1, 2, 3],
                routed_experts="upstream-npy-payload",
            )
        ],
    )

    processed = capture.post_process(response)

    assert processed.choices[0].routed_experts == compact_routed_experts


def test_client_set_max_tokens_recognizes_explicit_value():
    body = {"token_ids": [1, 2, 3], "sampling_params": {"max_tokens": 256}}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body))) is True


def test_client_set_max_tokens_detects_unset():
    body = {"token_ids": [1, 2, 3], "sampling_params": {}}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body))) is False

    body_without_sp = {"token_ids": [1, 2, 3]}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body_without_sp))) is False


def test_client_set_max_tokens_assumes_set_when_body_unreadable():
    # No raw_request → can't tell, don't override.
    assert asyncio.run(_client_set_max_tokens(None)) is True

    # body read raises → can't tell, don't override.
    err = ValueError("bad json")
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(err))) is True

    # non-dict body → can't tell, don't override.
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest([1, 2, 3]))) is True


def test_materialize_raw_image_ref_uses_generic_family_payload(tmp_path, monkeypatch):
    from PIL import Image
    from renderers.mm_store import raw_mm_ref

    from prime_rl.inference.vllm import serving_tokens

    image_dir = tmp_path / "run_serving" / "assets" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "image.png"
    Image.new("RGB", (8, 6), color=(32, 64, 128)).save(image_path)
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", str(image_dir))

    mm_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()[:32]
    fingerprint = "f" * 32
    raw_ref = raw_mm_ref(
        run_id="serving",
        family="test_family",
        fingerprint=fingerprint,
        modality="image",
        mm_hash=mm_hash,
        raw_image_id=image_path.name,
        payload={"adapter_owned": [1, 2, 3]},
    )
    processor = object()
    captured = {}

    class _Adapter:
        def materialize_for_vllm(self, image_processor, item, image, expected_placeholder_length):
            captured["image_processor"] = image_processor
            captured["item"] = item
            captured["image_size"] = image.size
            captured["expected_placeholder_length"] = expected_placeholder_length
            return {"materialized": True}

    def _get_adapter(family):
        captured["family"] = family
        return _Adapter()

    monkeypatch.setattr(serving_tokens, "_load_image_processor", lambda _model, _trust: processor)
    monkeypatch.setattr(serving_tokens, "get_multimodal_adapter", _get_adapter)

    out = serving_tokens._materialize_raw_image_ref_sync(
        raw_ref,
        expected_modality="image",
        expected_hash=mm_hash,
        expected_placeholder_length=7,
        processor_model_name="model",
        trust_remote_code=True,
    )

    assert out == {"materialized": True}
    assert captured["family"] == "test_family"
    assert captured["image_processor"] is processor
    assert captured["image_size"] == (8, 6)
    assert captured["expected_placeholder_length"] == 7
    item = captured["item"]
    assert item.family == "test_family"
    assert item.layout_fingerprint == fingerprint
    assert item.payload == {"adapter_owned": [1, 2, 3]}
