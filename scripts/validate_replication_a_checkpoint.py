"""Real-checkpoint validation routine for Replication A on RunPod / GPU server.

This is the authoritative preflight for Replication A. The in-process route
validates all frozen invariants on the actual loaded model, including
model-boundary tensor checks on the prepared history embeddings.

Validated invariants:
  1. disabled initial 16x8 action chunk == enabled identity chunk exactly.
  2. initial identity chunk == identity replay chunk exactly after two
     intervening off-diagonal calls.
  3. History-cache features remain strictly unchanged (SHA-256).
  4. Model-boundary tensor checks:
       - image embeddings identical across disabled, identity, off-diagonal.
       - state embeddings identical across disabled, identity, off-diagonal.
       - masks identical across disabled, identity, off-diagonal.
       - spatial positional channels identical across all conditions.
       - disabled temporal channels equal identity temporal channels exactly.
       - off-diagonal differences are confined strictly to temporal channels.

The WebSocket route is intentionally disabled for scientific validation unless
the server exposes exact selected FrameSamp identities. The validator must not
guess them.
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def hash_buffer_feats(history_feats: dict) -> dict[int, dict[str, str]]:
    """Compute sha256 digests for all cached history feature arrays."""
    hashes = {}
    for step, feats in history_feats.items():
        hashes[step] = {}
        for key, val in feats.items():
            if isinstance(val, np.ndarray) or hasattr(val, "__array__"):
                arr = np.asarray(val)
                hashes[step][key] = hashlib.sha256(arr.tobytes()).hexdigest()
    return hashes


def _prepare_tensors(policy, step_idx, history_feats_gather_fn, temporal_pos_override):
    """Call prepare_frame_sampling directly to capture model-boundary tensors."""
    token_budget = policy.config.budget
    token_per_image = policy.config.token_per_image
    return policy.mem_buffer.prepare_frame_sampling(
        step_idx,
        token_budget,
        token_per_image,
        history_feats_gather_fn,
        temporal_pos_override=temporal_pos_override,
    )


def _assert_all_equal(reference, candidates, label):
    for name, candidate in candidates:
        np.testing.assert_array_equal(
            reference,
            candidate,
            err_msg=f"{label}: disabled != {name}",
        )


def _assert_spatial_equal(reference_pos, candidates, temporal_dim):
    for name, candidate_pos in candidates:
        np.testing.assert_array_equal(
            reference_pos[..., temporal_dim:],
            candidate_pos[..., temporal_dim:],
            err_msg=f"spatial pos channels changed for {name}",
        )


def _assert_temporal_confined(reference_pos, candidate_pos, temporal_dim, name):
    pos_delta = reference_pos != candidate_pos
    np.testing.assert_array_equal(
        pos_delta[..., temporal_dim:],
        np.zeros_like(pos_delta[..., temporal_dim:], dtype=np.bool_),
        err_msg=f"{name}: non-temporal positional channels changed",
    )


def _initial_action_chunk(actions):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < 16:
        raise AssertionError(
            f"Expected actions with shape (N, 8), N >= 16; got {actions.shape}"
        )
    logging.info("Raw inferred action shape: %s; validating first 16x8 chunk", actions.shape)
    return actions[:16, :8]


def run_in_process_validation(policy):
    """Run authoritative validation directly against a loaded in-process MME_VLA_Policy."""
    logging.info("Starting in-process Replication A validation (authoritative preflight)...")

    if policy.mem_buffer is None:
        raise ValueError("Replication A validation requires a perceptual FrameSamp policy")
    if getattr(policy.config, "perceptual_memory", None) is None:
        raise ValueError("Replication A validation requires a perceptual FrameSamp policy")
    if policy.config.perceptual_memory.type != "frame_sampling":
        raise ValueError("Replication A validation requires perceptual_memory.type=frame_sampling")

    token_budget = policy.config.budget
    token_per_image = policy.config.token_per_image
    num_views = policy.config.num_views
    pos_emb_dim = policy.mem_buffer.pos_emb_dim
    temporal_dim = pos_emb_dim // 3
    expected_config = {
        "budget": 512,
        "token_per_image": 16,
        "num_views": 1,
        "pos_emb_dim": 768,
        "temporal_dim": 256,
    }
    actual_config = {
        "budget": int(token_budget),
        "token_per_image": int(token_per_image),
        "num_views": int(num_views),
        "pos_emb_dim": int(pos_emb_dim),
        "temporal_dim": int(temporal_dim),
    }
    if actual_config != expected_config:
        raise ValueError(
            "Replication A validation requires exact loaded config "
            f"{expected_config}; got {actual_config}"
        )

    if os.environ.get("MME_FRAMESAMP_OVERRIDE_PATH"):
        raise RuntimeError(
            "MME_FRAMESAMP_OVERRIDE_PATH is set; this legacy override can alter "
            "FrameSamp identities behind the validator's back. Unset it before "
            "running authoritative Replication A validation."
        )

    # 1. Populate a synthetic fixed-content demonstration.
    T = 64
    images = np.zeros((T, num_views, 224, 224, 3), dtype=np.uint8)
    for t in range(T):
        images[t, ...] = (t * 3) % 256
    states = np.zeros((T, 8), dtype=np.float32)
    for t in range(T):
        states[t] = np.arange(8, dtype=np.float32) + t * 0.1

    policy.reset()
    policy.add_buffer({
        "images": images,
        "state": states,
        "exec_start_idx": T - 1,
    })
    step_idx = policy.step_idx
    assert step_idx == T - 1, f"Expected step_idx {T - 1}, got {step_idx}"

    cache_hashes_before = hash_buffer_feats(policy.mem_buffer._history_feats)

    # 2. Derive exact FrameSamp selected indices from the actual loaded buffer.
    selected_indices = policy.mem_buffer.get_frame_sampling_indices(
        step_idx,
        token_budget,
        token_per_image,
    )
    selected_indices = [int(idx) for idx in selected_indices]
    max_size = token_budget // (token_per_image * num_views)
    assert len(selected_indices) == max_size, (
        f"Expected {max_size} selected frames from budget/token config, "
        f"got {len(selected_indices)}"
    )
    logging.info(
        "FrameSamp config: budget=%d, token_per_image=%d, num_views=%d -> %d selected frames",
        token_budget,
        token_per_image,
        num_views,
        len(selected_indices),
    )
    logging.info("Selected indices at step %d: %s", step_idx, selected_indices)

    # 3. Build override maps using only those exact selected identities.
    identity_map = [
        {"source_index": idx, "target_temporal_position": idx}
        for idx in selected_indices
    ]
    reversed_indices = list(reversed(selected_indices))
    rotated_indices = selected_indices[1:] + selected_indices[:1]
    offdiag_1_map = [
        {"source_index": idx, "target_temporal_position": rev_idx}
        for idx, rev_idx in zip(selected_indices, reversed_indices, strict=True)
    ]
    offdiag_2_map = [
        {"source_index": idx, "target_temporal_position": rot_idx}
        for idx, rot_idx in zip(selected_indices, rotated_indices, strict=True)
    ]

    # 4. Prepared-history tensor checks at the model boundary.
    logging.info("Running model-boundary tensor checks...")
    history_feats_gather_fn = policy.mem_buffer.default_history_feats_gather_fn

    disabled_tensors = _prepare_tensors(
        policy,
        step_idx,
        history_feats_gather_fn,
        temporal_pos_override={"enabled": False},
    )
    identity_tensors = _prepare_tensors(
        policy,
        step_idx,
        history_feats_gather_fn,
        temporal_pos_override={
            "enabled": True,
            "step_idx": step_idx,
            "position_map": identity_map,
        },
    )
    offdiag_1_tensors = _prepare_tensors(
        policy,
        step_idx,
        history_feats_gather_fn,
        temporal_pos_override={
            "enabled": True,
            "step_idx": step_idx,
            "position_map": offdiag_1_map,
        },
    )
    offdiag_2_tensors = _prepare_tensors(
        policy,
        step_idx,
        history_feats_gather_fn,
        temporal_pos_override={
            "enabled": True,
            "step_idx": step_idx,
            "position_map": offdiag_2_map,
        },
    )

    dis_img, dis_pos, dis_state, dis_mask = disabled_tensors
    idt_img, idt_pos, idt_state, idt_mask = identity_tensors
    od1_img, od1_pos, od1_state, od1_mask = offdiag_1_tensors
    od2_img, od2_pos, od2_state, od2_mask = offdiag_2_tensors

    logging.info(
        "Using temporal positional channels [0:%d] from pos_emb_dim=%d",
        temporal_dim,
        pos_emb_dim,
    )

    _assert_all_equal(
        dis_img,
        [("identity", idt_img), ("offdiag_1", od1_img), ("offdiag_2", od2_img)],
        "sampled image embeddings",
    )
    logging.info("[PASS] Sampled image embeddings exactly identical")

    _assert_all_equal(
        dis_state,
        [("identity", idt_state), ("offdiag_1", od1_state), ("offdiag_2", od2_state)],
        "sampled state embeddings",
    )
    logging.info("[PASS] Sampled state embeddings exactly identical")

    _assert_all_equal(
        dis_mask,
        [("identity", idt_mask), ("offdiag_1", od1_mask), ("offdiag_2", od2_mask)],
        "masks",
    )
    logging.info("[PASS] Masks exactly identical")

    _assert_spatial_equal(
        dis_pos,
        [("identity", idt_pos), ("offdiag_1", od1_pos), ("offdiag_2", od2_pos)],
        temporal_dim,
    )
    logging.info("[PASS] Spatial positional channels exactly identical")

    np.testing.assert_array_equal(
        dis_pos[..., :temporal_dim],
        idt_pos[..., :temporal_dim],
        err_msg="disabled temporal positional channels != identity temporal channels",
    )
    logging.info("[PASS] Disabled temporal channels equal identity temporal channels")

    _assert_temporal_confined(dis_pos, od1_pos, temporal_dim, "offdiag_1")
    _assert_temporal_confined(dis_pos, od2_pos, temporal_dim, "offdiag_2")
    od1_temporal_delta = np.max(np.abs(
        dis_pos[..., :temporal_dim].astype(np.float64)
        - od1_pos[..., :temporal_dim].astype(np.float64)
    ))
    od2_temporal_delta = np.max(np.abs(
        dis_pos[..., :temporal_dim].astype(np.float64)
        - od2_pos[..., :temporal_dim].astype(np.float64)
    ))
    logging.info(
        "Off-diagonal temporal pos max deltas: offdiag_1=%.6f, offdiag_2=%.6f",
        od1_temporal_delta,
        od2_temporal_delta,
    )
    if od1_temporal_delta == 0.0 or od2_temporal_delta == 0.0:
        raise AssertionError(
            "Off-diagonal temporal positional intervention produced zero tensor delta: "
            f"offdiag_1={od1_temporal_delta:.6f}, offdiag_2={od2_temporal_delta:.6f}"
        )
    logging.info("[PASS] Off-diagonal positional differences confined to temporal channels")

    cache_hashes_after_tensors = hash_buffer_feats(policy.mem_buffer._history_feats)
    assert cache_hashes_before == cache_hashes_after_tensors, (
        "History cache features mutated during tensor preparation!"
    )
    logging.info("[PASS] Cache hashes unchanged after tensor checks")

    # 5. Execute the 5-call Replication A action sequence.
    query_obs = {
        "observation/image": images[-1, 0].copy(),
        "observation/wrist_image": images[-1, 0].copy(),
        "observation/state": states[-1].copy(),
        "prompt": "validation prompt",
    }

    logging.info("Executing Call 1: Disabled override")
    call1 = policy.infer({
        **query_obs,
        "reset_rng": True,
        "temporal_pos_override": {"enabled": False},
    })

    logging.info("Executing Call 2: Enabled identity override")
    call2 = policy.infer({
        **query_obs,
        "reset_rng": True,
        "temporal_pos_override": {
            "enabled": True,
            "step_idx": step_idx,
            "position_map": identity_map,
        },
    })

    logging.info("Executing Call 3: Off-diagonal mapping 1")
    call3 = policy.infer({
        **query_obs,
        "reset_rng": True,
        "temporal_pos_override": {
            "enabled": True,
            "step_idx": step_idx,
            "position_map": offdiag_1_map,
        },
    })

    logging.info("Executing Call 4: Off-diagonal mapping 2")
    call4 = policy.infer({
        **query_obs,
        "reset_rng": True,
        "temporal_pos_override": {
            "enabled": True,
            "step_idx": step_idx,
            "position_map": offdiag_2_map,
        },
    })

    logging.info("Executing Call 5: Identity replay")
    call5 = policy.infer({
        **query_obs,
        "reset_rng": True,
        "temporal_pos_override": {
            "enabled": True,
            "step_idx": step_idx,
            "position_map": identity_map,
        },
    })

    act1_chunk = _initial_action_chunk(call1["actions"])
    act2_chunk = _initial_action_chunk(call2["actions"])
    act3_chunk = _initial_action_chunk(call3["actions"])
    act4_chunk = _initial_action_chunk(call4["actions"])
    act5_chunk = _initial_action_chunk(call5["actions"])

    np.testing.assert_array_equal(
        act1_chunk,
        act2_chunk,
        err_msg="Invariant 1 violated: disabled initial 16x8 chunk != identity chunk",
    )
    logging.info("[PASS] Invariant 1: disabled chunk == enabled identity chunk")

    np.testing.assert_array_equal(
        act2_chunk,
        act5_chunk,
        err_msg="Invariant 2 violated: identity initial 16x8 chunk != identity replay chunk",
    )
    logging.info("[PASS] Invariant 2: identity chunk == identity replay chunk")

    cache_hashes_after = hash_buffer_feats(policy.mem_buffer._history_feats)
    assert cache_hashes_before == cache_hashes_after, (
        "Invariant 3 violated: History cache features mutated during inference calls!"
    )
    logging.info("[PASS] Invariant 3: history-cache SHA-256 hashes unchanged")

    diff_3 = np.max(np.abs(act1_chunk.astype(np.float64) - act3_chunk.astype(np.float64)))
    diff_4 = np.max(np.abs(act1_chunk.astype(np.float64) - act4_chunk.astype(np.float64)))
    logging.info(
        "Off-diagonal action max deltas (informational): offdiag_1=%.6f, offdiag_2=%.6f",
        diff_3,
        diff_4,
    )

    logging.info("ALL IN-PROCESS INVARIANTS SATISFIED FOR REPLICATION A.")


def run_websocket_validation(host: str, port: int):
    """Fail closed: WebSocket cannot be authoritative without exact FrameSamp identities."""
    raise RuntimeError(
        "WebSocket Replication A scientific validation is disabled. The server at "
        f"ws://{host}:{port} does not expose exact selected FrameSamp identities or "
        "prepared history tensors to this script, and this validator must not "
        "approximate them. Run with --checkpoint-dir for authoritative in-process "
        "validation."
    )


def main():
    parser = argparse.ArgumentParser(description="Replication A Checkpoint Validation Routine")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Required for authoritative in-process validation",
    )
    parser.add_argument("--config", type=str, default="mme_vla_suite", help="Policy config name")
    args = parser.parse_args()

    if args.checkpoint_dir:
        from mme_vla_suite.policies import policy_config as _policy_config
        from mme_vla_suite.training import config as _config

        logging.info("Loading checkpoint from %s...", args.checkpoint_dir)
        policy = _policy_config.create_trained_policy(
            _config.get_config(args.config),
            args.checkpoint_dir,
            seed=42,
        )
        run_in_process_validation(policy)
    else:
        logging.error("Authoritative validation requires --checkpoint-dir.")
        run_websocket_validation(args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())

