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
        jax.random = types.SimpleNamespace(
            key=lambda seed: f"key_{seed}",
            split=lambda k: (f"{k}_next", f"{k}_sample"),
        )
        jax.tree = types.SimpleNamespace(
            map=lambda fn, tree: {k: fn(v) for k, v in tree.items()} if isinstance(tree, dict) else tree
        )
        sys.modules["jax"] = jax
        sys.modules["jax.numpy"] = np

    if importlib.util.find_spec("flax") is None:
        flax = types.ModuleType("flax")
        nnx = types.ModuleType("flax.nnx")
        nnx.avg_pool = lambda *args, **kwargs: None
        nnx.max_pool = lambda *args, **kwargs: None
        traverse_util = types.ModuleType("flax.traverse_util")
        traverse_util.flatten_dict = lambda d, *args, **kwargs: d
        traverse_util.unflatten_dict = lambda d, *args, **kwargs: d
        flax.nnx = nnx
        flax.traverse_util = traverse_util
        sys.modules["flax"] = flax
        sys.modules["flax.nnx"] = nnx
        sys.modules["flax.traverse_util"] = traverse_util

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


def test_in_memory_override_explicit_disable_matches_baseline_and_has_precedence(
    monkeypatch, tmp_path
):
    buffer = _make_temporal_override_buffer()

    baseline = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
        temporal_pos_override=None,
    )

    explicit_disabled = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
        temporal_pos_override={"enabled": False},
    )

    for b, d in zip(baseline, explicit_disabled, strict=True):
        np.testing.assert_array_equal(b, d)

    # Precedence: even if env var points to an active off-diagonal override file,
    # in-memory {"enabled": False} takes precedence and returns baseline.
    override_path = tmp_path / "env_override.json"
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
    monkeypatch.setenv("MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH", str(override_path))

    precedence_disabled = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
        temporal_pos_override={"enabled": False},
    )

    for b, p in zip(baseline, precedence_disabled, strict=True):
        np.testing.assert_array_equal(b, p)


def _make_consistent_temporal_buffer():
    buffer = _make_temporal_override_buffer()
    for step_idx in range(6):
        buffer._history_feats[step_idx]["pos_emb_4x4"] = buffer.pos_emb_dict["4x4"][
            step_idx : step_idx + 1
        ].copy()
    return buffer


def test_in_memory_override_identity_matches_baseline_exactly():
    buffer = _make_consistent_temporal_buffer()

    baseline = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
        temporal_pos_override={"enabled": False},
    )

    identity_override = {
        "enabled": True,
        "step_idx": 5,
        "position_map": [
            {"source_index": 0, "target_temporal_position": 0},
            {"source_index": 1, "target_temporal_position": 1},
            {"source_index": 3, "target_temporal_position": 3},
            {"source_index": 5, "target_temporal_position": 5},
        ],
    }

    identity_result = buffer.prepare_frame_sampling(
        5,
        64,
        16,
        buffer.default_history_feats_gather_fn,
        temporal_pos_override=identity_override,
    )

    for b, i in zip(baseline, identity_result, strict=True):
        np.testing.assert_array_equal(b, i)


def test_replication_a_sequence_invariants_and_cache_immutability():
    buffer = _make_consistent_temporal_buffer()

    # Deep snapshot of cache before any calls
    cache_snapshot = {
        step: {k: v.copy() for k, v in feats.items()}
        for step, feats in buffer._history_feats.items()
    }

    # FrameSamp selected indices at step 5 are [0, 1, 3, 5]
    identity_map = [
        {"source_index": 0, "target_temporal_position": 0},
        {"source_index": 1, "target_temporal_position": 1},
        {"source_index": 3, "target_temporal_position": 3},
        {"source_index": 5, "target_temporal_position": 5},
    ]
    offdiag_1_map = [
        {"source_index": 0, "target_temporal_position": 5},
        {"source_index": 1, "target_temporal_position": 3},
        {"source_index": 3, "target_temporal_position": 1},
        {"source_index": 5, "target_temporal_position": 0},
    ]
    offdiag_2_map = [
        {"source_index": 0, "target_temporal_position": 1},
        {"source_index": 1, "target_temporal_position": 3},
        {"source_index": 3, "target_temporal_position": 5},
        {"source_index": 5, "target_temporal_position": 0},
    ]

    # Call 1: Disabled
    call1 = buffer.prepare_frame_sampling(
        5, 64, 16, buffer.default_history_feats_gather_fn,
        temporal_pos_override={"enabled": False},
    )

    # Call 2: Enabled identity override
    call2 = buffer.prepare_frame_sampling(
        5, 64, 16, buffer.default_history_feats_gather_fn,
        temporal_pos_override={"enabled": True, "step_idx": 5, "position_map": identity_map},
    )

    # Call 3: Off-diagonal temporal mapping 1
    call3 = buffer.prepare_frame_sampling(
        5, 64, 16, buffer.default_history_feats_gather_fn,
        temporal_pos_override={"enabled": True, "step_idx": 5, "position_map": offdiag_1_map},
    )

    # Call 4: Off-diagonal temporal mapping 2
    call4 = buffer.prepare_frame_sampling(
        5, 64, 16, buffer.default_history_feats_gather_fn,
        temporal_pos_override={"enabled": True, "step_idx": 5, "position_map": offdiag_2_map},
    )

    # Call 5: Identity replay
    call5 = buffer.prepare_frame_sampling(
        5, 64, 16, buffer.default_history_feats_gather_fn,
        temporal_pos_override={"enabled": True, "step_idx": 5, "position_map": identity_map},
    )

    # Invariant 1: disabled == enabled identity exactly
    for a, b in zip(call1, call2, strict=True):
        np.testing.assert_array_equal(a, b)

    # Invariant 2: initial identity == identity replay exactly
    for a, b in zip(call2, call5, strict=True):
        np.testing.assert_array_equal(a, b)

    # Invariant 3: cached history features remain bit-identical
    assert set(buffer._history_feats.keys()) == set(cache_snapshot.keys())
    for step in cache_snapshot:
        for k in cache_snapshot[step]:
            np.testing.assert_array_equal(
                buffer._history_feats[step][k], cache_snapshot[step][k]
            )

    # Invariant 4: non-temporal components and spatial PE remain bit-identical across all 5 calls
    all_calls = [call1, call2, call3, call4, call5]
    temporal_dim = buffer.pos_emb_dim // 3  # 6 // 3 = 2 for this test buffer
    for c in all_calls:
        # img_emb
        np.testing.assert_array_equal(c[0], call1[0])
        # state_emb
        np.testing.assert_array_equal(c[2], call1[2])
        # mask
        np.testing.assert_array_equal(c[3], call1[3])
        # spatial channels of pos_emb
        pos_reshaped = c[1].reshape(4, 16, buffer.pos_emb_dim)
        call1_pos_reshaped = call1[1].reshape(4, 16, buffer.pos_emb_dim)
        np.testing.assert_array_equal(
            pos_reshaped[..., temporal_dim:], call1_pos_reshaped[..., temporal_dim:]
        )

    # Invariant 5: only intended temporal positional channels change for off-diagonal calls
    call3_pos = call3[1].reshape(4, 16, buffer.pos_emb_dim)
    expected_t3 = np.stack(
        [buffer.pos_emb_dict["4x4"][pos, :, :temporal_dim] for pos in [5, 3, 1, 0]],
        axis=0,
    )
    np.testing.assert_array_equal(call3_pos[..., :temporal_dim], expected_t3)

    call4_pos = call4[1].reshape(4, 16, buffer.pos_emb_dim)
    expected_t4 = np.stack(
        [buffer.pos_emb_dict["4x4"][pos, :, :temporal_dim] for pos in [1, 3, 5, 0]],
        axis=0,
    )
    np.testing.assert_array_equal(call4_pos[..., :temporal_dim], expected_t4)


def test_in_memory_override_validation_and_fail_closed():
    buffer = _make_temporal_override_buffer()

    # Must be a dict
    with pytest.raises(TypeError, match="must be a dict"):
        buffer.prepare_frame_sampling(
            5, 64, 16, buffer.default_history_feats_gather_fn,
            temporal_pos_override="invalid_string",
        )

    # Source indices missing an element (selected are [0, 1, 3, 5])
    with pytest.raises(ValueError, match="source indices must exactly match"):
        buffer.prepare_frame_sampling(
            5, 64, 16, buffer.default_history_feats_gather_fn,
            temporal_pos_override={
                "enabled": True,
                "step_idx": 5,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 0},
                    {"source_index": 1, "target_temporal_position": 1},
                    {"source_index": 3, "target_temporal_position": 3},
                ],
            },
        )

    # Source indices containing an extra element
    with pytest.raises(ValueError, match="source indices must exactly match"):
        buffer.prepare_frame_sampling(
            5, 64, 16, buffer.default_history_feats_gather_fn,
            temporal_pos_override={
                "enabled": True,
                "step_idx": 5,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 0},
                    {"source_index": 1, "target_temporal_position": 1},
                    {"source_index": 2, "target_temporal_position": 2},
                    {"source_index": 3, "target_temporal_position": 3},
                    {"source_index": 5, "target_temporal_position": 5},
                ],
            },
        )

    # Duplicate target positions
    with pytest.raises(ValueError, match="target temporal positions must be unique"):
        buffer.prepare_frame_sampling(
            5, 64, 16, buffer.default_history_feats_gather_fn,
            temporal_pos_override={
                "enabled": True,
                "step_idx": 5,
                "position_map": [
                    {"source_index": 0, "target_temporal_position": 0},
                    {"source_index": 1, "target_temporal_position": 1},
                    {"source_index": 3, "target_temporal_position": 1},
                    {"source_index": 5, "target_temporal_position": 5},
                ],
            },
        )

