import json
import importlib.util
import sys
import types

import numpy as np


def _install_missing_import_stubs():
    if importlib.util.find_spec("jax") is None:
        jax = types.ModuleType("jax")
        jax.numpy = np
        jax.jit = lambda fn: fn
        jax.device_get = lambda value: value
        sys.modules["jax"] = jax
        sys.modules["jax.numpy"] = np

    if importlib.util.find_spec("flax") is None:
        flax = types.ModuleType("flax")
        nnx = types.ModuleType("flax.nnx")
        nnx.avg_pool = lambda *args, **kwargs: None
        nnx.max_pool = lambda *args, **kwargs: None
        flax.nnx = nnx
        sys.modules["flax"] = flax
        sys.modules["flax.nnx"] = nnx

    if importlib.util.find_spec("einops") is None:
        einops = types.ModuleType("einops")
        einops.rearrange = lambda *args, **kwargs: None
        sys.modules["einops"] = einops

    if importlib.util.find_spec("cv2") is None:
        sys.modules["cv2"] = types.ModuleType("cv2")

    if importlib.util.find_spec("beartype") is None:
        image_tools = types.ModuleType("openpi.shared.image_tools")
        image_tools.resize_with_pad = lambda *args, **kwargs: None
        sys.modules["openpi.shared.image_tools"] = image_tools


_install_missing_import_stubs()

from mme_vla_suite.shared.mem_buffer import MemoryBuffer


def _make_buffer():
    buffer = MemoryBuffer(
        num_views=1,
        img_emb_dim=2,
        pos_emb_dim=3,
        state_emb_dim=4,
        prepare_buffer=False,
    )
    for step_idx in range(6):
        buffer._history_feats[step_idx] = {
            "image_emb_4x4": np.full((1, 16, 2), step_idx, dtype=np.float32),
            "pos_emb_4x4": np.full((1, 16, 3), step_idx + 0.25, dtype=np.float32),
            "state_emb": np.arange(4, dtype=np.float32) + step_idx,
        }
    return buffer


def test_frame_sampling_trace_does_not_change_outputs(monkeypatch, tmp_path):
    step_idx = 5
    token_budget = 64
    token_per_image = 16
    buffer = _make_buffer()

    monkeypatch.delenv("MME_FRAMESAMP_TRACE_PATH", raising=False)
    outputs_without_trace = buffer.prepare_frame_sampling(
        step_idx,
        token_budget,
        token_per_image,
        buffer.default_history_feats_gather_fn,
    )

    trace_path = tmp_path / "framesamp.jsonl"
    monkeypatch.setenv("MME_FRAMESAMP_TRACE_PATH", str(trace_path))
    outputs_with_trace = buffer.prepare_frame_sampling(
        step_idx,
        token_budget,
        token_per_image,
        buffer.default_history_feats_gather_fn,
    )

    for without_trace, with_trace in zip(outputs_without_trace, outputs_with_trace, strict=True):
        np.testing.assert_array_equal(without_trace, with_trace)

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "step_idx": 5,
            "token_budget": 64,
            "token_per_image": 16,
            "num_views": 1,
            "max_size": 4,
            "indices_to_load": [0, 1, 3, 5],
        }
    ]
