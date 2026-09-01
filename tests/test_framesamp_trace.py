import json
import importlib.util
import sys
import types

import numpy as np
import pytest


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
    monkeypatch.delenv("MME_FRAMESAMP_OVERRIDE_PATH", raising=False)
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
def test_frame_sampling_override_uses_exact_indices(monkeypatch, tmp_path):
    step_idx = 5
    token_budget = 64
    token_per_image = 16
    buffer = _make_buffer()

    override_path = tmp_path / "override.json"
    override_path.write_text(
        json.dumps({
            "step_idx": 5,
            "indices_to_load": [0, 2, 4, 5],
        }),
        encoding="utf-8",
    )
    trace_path = tmp_path / "framesamp_override.jsonl"

    monkeypatch.setenv("MME_FRAMESAMP_OVERRIDE_PATH", str(override_path))
    monkeypatch.setenv("MME_FRAMESAMP_TRACE_PATH", str(trace_path))

    img_emb, _, _, _ = buffer.prepare_frame_sampling(
        step_idx,
        token_budget,
        token_per_image,
        buffer.default_history_feats_gather_fn,
    )
    sampled_steps = img_emb.reshape(4, 16, 2)[:, 0, 0]
    np.testing.assert_array_equal(
        sampled_steps,
        np.array([0, 2, 4, 5], dtype=np.float32),
    )

    record = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert record["indices_to_load"] == [0, 2, 4, 5]


def test_frame_sampling_override_fails_on_wrong_step(monkeypatch, tmp_path):
    buffer = _make_buffer()

    override_path = tmp_path / "override.json"
    override_path.write_text(
        json.dumps({
            "step_idx": 4,
            "indices_to_load": [0, 1, 3, 5],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("MME_FRAMESAMP_OVERRIDE_PATH", str(override_path))

    with pytest.raises(ValueError, match="does not match current step_idx"):
        buffer.prepare_frame_sampling(
            5,
            64,
            16,
            buffer.default_history_feats_gather_fn,
        )


def _make_temporal_override_buffer():
    buffer = MemoryBuffer(
        num_views=1,
        img_emb_dim=2,
        pos_emb_dim=6,
        state_emb_dim=4,
        prepare_buffer=False,
    )

    # PosEmb3D with dim=6 has:
    #   temporal = first 2 dims
    #   spatial  = remaining 4 dims.
    for step_idx in range(6):
        pos = np.zeros((1, 16, 6), dtype=np.float32)
        pos[..., 0] = 100 + step_idx
        pos[..., 1] = 200 + step_idx
        pos[..., 2:] = np.arange(4, dtype=np.float32) + 1000 + 10 * step_idx

        buffer._history_feats[step_idx] = {
            "image_emb_4x4": np.full(
                             (1, 16, 2), step_idx, dtype=np.float32
            ),
            "pos_emb_4x4": pos,
            "state_emb": np.arange(4, dtype=np.float32) + step_idx,
        }

    # Synthetic positional dictionary representing positional embeddings
    # for arbitrary target temporal positions. Spatial values are made
    # deliberately different so the intervention must not copy them.
    pos_dict = np.zeros((6, 16, 6), dtype=np.float32)
    for step_idx in range(6):
        pos_dict[step_idx, :, 0] = 10000 + step_idx
        pos_dict[step_idx, :, 1] = 20000 + step_idx
        pos_dict[step_idx, :, 2:] = (
            np.arange(4, dtype=np.float32)[None, :]
            + 30000
            + 100 * step_idx
        )
    buffer.pos_emb_dict = {"4x4": pos_dict}
    return buffer


def test_frame_sampling_temporal_position_override_changes_only_temporal_channels(
    monkeypatch, tmp_path
):
    buffer = _make_temporal_override_buffer()

    override_path = tmp_path / "temporal_pos_override.json"
    override_path.write_text(
        json.dumps(
            {
                "step_idx": 5,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 5},
                    {"source_index": 1, "target_temporal_position": 4},
                    {"source_index": 3, "target_temporal_position": 2},
                    {"source_index": 5, "target_temporal_position": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH", raising=False)
    baseline = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
    )
    monkeypatch.setenv(
        "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH",
        str(override_path),
    )
    intervened = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
    )

    base_img, base_pos, base_state, base_mask = baseline
    int_img, int_pos, int_state, int_mask = intervened

    # All non-positional model inputs remain bit-identical.
    np.testing.assert_array_equal(base_img, int_img)
    np.testing.assert_array_equal(base_state, int_state)
    np.testing.assert_array_equal(base_mask, int_mask)

    base_pos = base_pos.reshape(4, 16, 6)
    int_pos = int_pos.reshape(4, 16, 6)

    # Spatial positional channels remain bit-identical.
    np.testing.assert_array_equal(base_pos[..., 2:], int_pos[..., 2:])

    # Temporal channels come from the requested target temporal positions,
    # while the selected frame order remains [0, 1, 3, 5].
    target_positions = [5, 4, 2, 0]
    expected_temporal = np.stack(
               [buffer.pos_emb_dict["4x4"][idx, :, :2] for idx in target_positions],
        axis=0,
    )
    np.testing.assert_array_equal(int_pos[..., :2], expected_temporal)

    sampled_steps = int_img.reshape(4, 16, 2)[:, 0, 0]
    np.testing.assert_array_equal(
        sampled_steps,
        np.array([0, 1, 3, 5], dtype=np.float32),
    )


def test_frame_sampling_temporal_position_override_fails_on_wrong_step(
    monkeypatch, tmp_path
):
    buffer = _make_temporal_override_buffer()

    override_path = tmp_path / "temporal_pos_override.json"
    override_path.write_text(
        json.dumps(
            {
                "step_idx": 4,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 0},
                    {"source_index": 1, "target_temporal_position": 1},
                    {"source_index": 3, "target_temporal_position": 3},
                    {"source_index": 5, "target_temporal_position": 5},
                ],
            }
        ),
           encoding="utf-8",
    )
    monkeypatch.setenv(
        "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH",
        str(override_path),
    )

    with pytest.raises(ValueError, match="does not match current step_idx"):
        buffer.prepare_frame_sampling(
            5,
            64,
            16,
            buffer.default_history_feats_gather_fn,
        )


def test_frame_sampling_temporal_position_override_requires_exact_source_set(
    monkeypatch, tmp_path
):
    buffer = _make_temporal_override_buffer()

    override_path = tmp_path / "temporal_pos_override.json"
    override_path.write_text(
        json.dumps(
            {
                "step_idx": 5,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 0},
                    {"source_index": 1, "target_temporal_position": 1},
                    {"source_index": 2, "target_temporal_position": 2},
                    {"source_index": 5, "target_temporal_position": 5},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH",
        str(override_path),
    )
    with pytest.raises(ValueError, match="source indices"):
        buffer.prepare_frame_sampling(
            5,
            64,
            16,
            buffer.default_history_feats_gather_fn,
        )


def test_frame_sampling_temporal_position_override_disabled_is_exact_noop(
    monkeypatch
):
    buffer = _make_temporal_override_buffer()

    monkeypatch.delenv("MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH", raising=False)
    first = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
    )
    second = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
    )

    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_frame_sampling_temporal_position_override_composes_with_index_override(
    monkeypatch, tmp_path
):
    buffer = _make_temporal_override_buffer()

    index_override_path = tmp_path / "index_override.json"
    index_override_path.write_text(
        json.dumps(
            {
                "step_idx": 5,
                "indices_to_load": [0, 2, 4, 5],
            }
        ),
        encoding="utf-8",
    )

    temporal_override_path = tmp_path / "temporal_pos_override.json"
    temporal_override_path.write_text(
        json.dumps(
            {
                "step_idx": 5,
                                "position_map": [
                    {"source_index": 0, "target_temporal_position": 5},
                    {"source_index": 2, "target_temporal_position": 3},
                    {"source_index": 4, "target_temporal_position": 1},
                    {"source_index": 5, "target_temporal_position": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "MME_FRAMESAMP_OVERRIDE_PATH",
        str(index_override_path),
    )
    monkeypatch.setenv(
        "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH",
        str(temporal_override_path),
    )

    img_emb, pos_emb, _, _ = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
    )
    # Content/order must come from the existing index override.
    sampled_steps = img_emb.reshape(4, 16, 2)[:, 0, 0]
    np.testing.assert_array_equal(
        sampled_steps,
        np.array([0, 2, 4, 5], dtype=np.float32),
    )

    pos_emb = pos_emb.reshape(4, 16, 6)

    expected_temporal = np.stack(
        [
            buffer.pos_emb_dict["4x4"][target, :, :2]
            for target in [5, 3, 1, 0]
        ],
        axis=0,
    )
    np.testing.assert_array_equal(
        pos_emb[..., :2],
        expected_temporal,
    )
    # Spatial PE remains attached to the selected source content.
    expected_spatial = np.stack(
        [
            buffer._history_feats[source]["pos_emb_4x4"][0, :, 2:]
            for source in [0, 2, 4, 5]
        ],
        axis=0,
    )
    np.testing.assert_array_equal(
        pos_emb[..., 2:],
        expected_spatial,
    )


def test_frame_sampling_temporal_position_override_rejects_duplicate_sources(
    monkeypatch, tmp_path
):
    buffer = _make_temporal_override_buffer()

    override_path = tmp_path / "temporal_pos_override.json"
    override_path.write_text(
        json.dumps(
            {
                "step_idx": 5,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 0},
                    {"source_index": 1, "target_temporal_position": 1},
                    {"source_index": 3, "target_temporal_position": 3},
                    {"source_index": 3, "target_temporal_position": 5},
                             {"source_index": 5, "target_temporal_position": 4},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH",
        str(override_path),
    )

    with pytest.raises(ValueError, match="unique"):
        buffer.prepare_frame_sampling(
            5,
            64,
            16,
            buffer.default_history_feats_gather_fn,
        )
def test_frame_sampling_temporal_position_override_rejects_invalid_target_position(
    monkeypatch, tmp_path
):
    buffer = _make_temporal_override_buffer()

    override_path = tmp_path / "temporal_pos_override.json"
    override_path.write_text(
        json.dumps(
            {
                "step_idx": 5,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 0},
                    {"source_index": 1, "target_temporal_position": 1},
                    {"source_index": 3, "target_temporal_position": 3},
                    {"source_index": 5, "target_temporal_position": 6},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH",
        str(override_path),
    )

    with pytest.raises(ValueError, match="target temporal"):
        buffer.prepare_frame_sampling(
            5,
            64,
            16,
            buffer.default_history_feats_gather_fn,
        )


def test_frame_sampling_temporal_position_override_uses_actual_step_when_selection_omits_current(
    monkeypatch, tmp_path
):
    buffer = _make_temporal_override_buffer()

    index_override_path = tmp_path / "index_override.json"
    index_override_path.write_text(
        json.dumps(
            {
                "step_idx": 5,
                "indices_to_load": [0, 1, 2, 4],
            }
        ),
        encoding="utf-8",
    )
    temporal_override_path = tmp_path / "temporal_pos_override.json"
    temporal_override_path.write_text(
        json.dumps(
            {
                "step_idx": 5,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 5},
                    {"source_index": 1, "target_temporal_position": 4},
                    {"source_index": 2, "target_temporal_position": 3},
                    {"source_index": 4, "target_temporal_position": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MME_FRAMESAMP_OVERRIDE_PATH",
        str(index_override_path),
    )
    monkeypatch.setenv(
        "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH",
        str(temporal_override_path),
    )

    img_emb, pos_emb, state_emb, mask = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
    )

    # Selection is valid even though actual step_idx=5 is not selected.
    sampled_steps = img_emb.reshape(4, 16, 2)[:, 0, 0]
    np.testing.assert_array_equal(
        sampled_steps,
        np.array([0, 1, 2, 4], dtype=np.float32),
    )
    # Temporal PE uses the requested target positions.
    pos_emb = pos_emb.reshape(4, 16, 6)
    expected_temporal = np.stack(
        [
            buffer.pos_emb_dict["4x4"][target, :, :2]
            for target in [5, 4, 3, 2]
        ],
        axis=0,
    )
    np.testing.assert_array_equal(
        pos_emb[..., :2],
        expected_temporal,
    )
    # Spatial PE remains attached to the selected source content.
    expected_spatial = np.stack(
        [
            buffer._history_feats[source]["pos_emb_4x4"][0, :, 2:]
            for source in [0, 1, 2, 4]
        ],
        axis=0,
    )
    np.testing.assert_array_equal(
        pos_emb[..., 2:],
        expected_spatial,
    )
    expected_state = np.repeat(
        np.stack(
            [buffer._history_feats[source]["state_emb"] for source in [0, 1, 2, 4]],
            axis=0,
        ),
        16,
        axis=0,
    )
    np.testing.assert_array_equal(state_emb, expected_state)
    np.testing.assert_array_equal(mask, np.ones(64, dtype=np.bool_))
