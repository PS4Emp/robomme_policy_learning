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
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOMME_EXAMPLES = REPO_ROOT / "examples" / "robomme"
ROBOMME_BENCHMARK_SRC = REPO_ROOT / "third_party" / "robomme_benchmark" / "src"

if str(ROBOMME_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROBOMME_EXAMPLES))

if str(ROBOMME_BENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOMME_BENCHMARK_SRC))
from controlled_recency import (
    CONDITION_ORDERS,
    assemble_condition,
    build_bundle,
    load_distractor_segments,
    validate_protocol_inputs,
)
from env_runner import EnvRunner
from utils import pack_buffer

PROTOCOL_ID = "ROBOMME-RETENTION-MATCHED-TEMPORAL-A-v1.0"
MODEL_ID = "Yinpei/perceptual-framesamp-modul"
CHECKPOINT_ID = "79999"
MODEL_ID = "Yinpei/perceptual-framesamp-modul"
CHECKPOINT_ID = "79999"

CONDITIONS = ("far", "middle", "recent")
FROZEN_FAMILIES = ("VideoUnmask", "InsertPeg", "VideoPlaceOrder")
FROZEN_EPISODES = (0, 5, 11, 16, 22, 27, 33, 38, 44, 49)
PROCESS_BLOCKS = ("P1", "P2", "P3")

R_COUNT = 25
DISTRACTOR_RETAINED_SOURCE_INDICES = (0, 248)
EXPECTED_RETAINED_COUNTS = {
    "R": 25,
    "D1": 2,
    "D2": 2,
    "D3": 2,
    "Q0": 1,
}
EXPECTED_RETAINED_TOTAL = 32
ASSIGNMENT_SOURCE = (
    REPO_ROOT
    / "runs"
    / "protocol_capacity_census_2026-09-11"
    / "distractor_assignment_joint_position_balanced_candidate.csv"
)

ASSIGNMENT_SOURCE_SHA256 = (
    "432e3ab1347888e014d443e82f8f4ee18f0d52876542476be8caf01f578245f7"
)

FROZEN_30_WINDOWS_MANIFEST_SHA256 = (
    "243f5d0ce03d9f95a77da3c1767f6320d3ec5ecd1434d7ce7b84b2cb282e7717"
)



def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_frozen_assignment(query_family: str, query_episode: int):
    actual_sha = sha256_file(ASSIGNMENT_SOURCE)
    if actual_sha != ASSIGNMENT_SOURCE_SHA256:
        raise RuntimeError(
            "Assignment source SHA256 mismatch: "
            f"expected {ASSIGNMENT_SOURCE_SHA256}, got {actual_sha}"
        )

    with ASSIGNMENT_SOURCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    matches = [
        row for row in rows
        if row["query_family"] == query_family
        and int(row["query_episode"]) == int(query_episode)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one assignment row for {query_family} ep{query_episode}; "
            f"got {len(matches)}"
        )

    row = matches[0]
    distractors = [
        (row["D1_family"], int(row["D1_episode"])),
        (row["D2_family"], int(row["D2_episode"])),
        (row["D3_family"], int(row["D3_episode"])),
    ]
    validate_protocol_inputs(query_family, distractors)
    return row, distractors


def select_r_source_indices(bundle) -> list[int]:
    segment = bundle["R"]
    n = len(segment.source_frame_indices)
    if n < R_COUNT:
        raise RuntimeError(
            f"Need at least {R_COUNT} R frames; got {n}"
        )

    offsets = np.linspace(0, n - 1, R_COUNT, dtype=np.int32).tolist()

    if len(offsets) != R_COUNT or len(set(offsets)) != R_COUNT:
        raise RuntimeError(
            f"R selection did not produce {R_COUNT} unique endpoint-inclusive offsets: "
            f"{offsets}"
        )

    return [int(segment.source_frame_indices[i]) for i in offsets]

def build_retained_identity_manifest(bundle) -> list[dict]:
    selected = {
        "R": select_r_source_indices(bundle),
        "D1": list(DISTRACTOR_RETAINED_SOURCE_INDICES),
        "D2": list(DISTRACTOR_RETAINED_SOURCE_INDICES),
        "D3": list(DISTRACTOR_RETAINED_SOURCE_INDICES),
        "Q0": [int(bundle["Q0"].source_frame_indices[0])],
    }

    counts = {label: len(indices) for label, indices in selected.items()}
    if counts != EXPECTED_RETAINED_COUNTS:
        raise RuntimeError(
            f"Retained-count mismatch: expected {EXPECTED_RETAINED_COUNTS}, got {counts}"
        )

    manifest = []

    for label in ("R", "D1", "D2", "D3", "Q0"):
        segment = bundle[label]
        available = {int(x) for x in segment.source_frame_indices}
        for source_frame_index in selected[label]:
            if source_frame_index not in available:
                raise RuntimeError(
                    f"{label} source frame {source_frame_index} is not available"
                )

            manifest.append(
                {
                    "segment": label,
                    "source_frame_index": int(source_frame_index),
                    "source_family": segment.source_family,
                    "source_episode": int(segment.source_episode),
                    "source_role": segment.source_role,
                }
            )

    if len(manifest) != EXPECTED_RETAINED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_RETAINED_TOTAL} retained identities; got {len(manifest)}"
        )

    return manifest


def assembly_identity_positions(assembly) -> dict[tuple[str, int], int]:
    positions = {}

    for entry in assembly.metadata["frame_index_map"]:
        key = (
            entry["segment_label"],
            int(entry["source_frame_index"]),
        )

        if key in positions:
            raise RuntimeError(
                f"Duplicate semantic identity in {assembly.condition}: {key}"
            )

        positions[key] = int(entry["assembled_index"])

    return positions


def retained_entries_for_row(assembly, retained_manifest: list[dict]) -> list[dict]:
    wanted = {
        (entry["segment"], int(entry["source_frame_index"]))
        for entry in retained_manifest
    }
    retained = []

    for entry in assembly.metadata["frame_index_map"]:
        key = (
            entry["segment_label"],
            int(entry["source_frame_index"]),
        )
        if key in wanted:
            retained.append(entry)

    if len(retained) != EXPECTED_RETAINED_TOTAL:
        raise RuntimeError(
            f"{assembly.condition}: expected {EXPECTED_RETAINED_TOTAL} retained "
            f"entries, got {len(retained)}"
        )

    counts = {label: 0 for label in EXPECTED_RETAINED_COUNTS}
    for entry in retained:
        counts[entry["segment_label"]] += 1

    if counts != EXPECTED_RETAINED_COUNTS:
        raise RuntimeError(
            f"{assembly.condition}: expected retained counts "
            f"{EXPECTED_RETAINED_COUNTS}, got {counts}"
        )
    assembled_indices = [int(entry["assembled_index"]) for entry in retained]

    if assembled_indices != sorted(assembled_indices):
        raise RuntimeError(
            f"{assembly.condition}: retained assembled indices are not sorted"
        )

    if len(set(assembled_indices)) != EXPECTED_RETAINED_TOTAL:
        raise RuntimeError(
            f"{assembly.condition}: retained assembled indices are not unique"
        )

    return retained


def make_temporal_position_map(
    retained_entries: list[dict],
    target_positions: dict[tuple[str, int], int],
) -> list[dict]:
    position_map = []

    for entry in retained_entries:
        key = (
            entry["segment_label"],
            int(entry["source_frame_index"]),
        )

        if key not in target_positions:
            raise RuntimeError(
                f"Target layout is missing retained semantic identity {key}"
            )
        position_map.append(
            {
                "source_index": int(entry["assembled_index"]),
                "target_temporal_position": int(target_positions[key]),
            }
        )

    source_indices = [x["source_index"] for x in position_map]
    target_positions_list = [
        x["target_temporal_position"] for x in position_map
    ]

    if len(set(source_indices)) != EXPECTED_RETAINED_TOTAL:
        raise RuntimeError("Temporal position map has duplicate source indices")

    if len(set(target_positions_list)) != EXPECTED_RETAINED_TOTAL:
        raise RuntimeError("Temporal position map has duplicate target positions")

    return position_map


def action_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)

    if ref.shape != (16, 8) or cand.shape != (16, 8):
        raise RuntimeError(
            f"Scientific action chunks must both be (16, 8); got "
            f"{ref.shape} and {cand.shape}"
        )

    diff = cand - ref

    l2 = float(np.linalg.norm(diff))
    ref_l2 = float(np.linalg.norm(ref))

    if ref_l2 == 0.0:
        rho = 0.0 if l2 == 0.0 else float("inf")
    else:
        rho = l2 / ref_l2
    return {
        "mae": float(np.mean(np.abs(diff))),
        "l2_frobenius": l2,
        "max_abs": float(np.max(np.abs(diff))),
        "relative_frobenius_rho": float(rho),
    }


def validate_replication_a_policy(policy) -> dict[str, int]:
    if policy.mem_buffer is None:
        raise RuntimeError("Replication A requires a perceptual FrameSamp policy")

    if getattr(policy.config, "perceptual_memory", None) is None:
        raise RuntimeError("Replication A requires perceptual memory")

    if policy.config.perceptual_memory.type != "frame_sampling":
        raise RuntimeError(
            "Replication A requires perceptual_memory.type=frame_sampling"
        )

    actual = {
        "budget": int(policy.config.budget),
        "token_per_image": int(policy.config.token_per_image),
        "num_views": int(policy.config.num_views),
        "pos_emb_dim": int(policy.mem_buffer.pos_emb_dim),
        "temporal_dim": int(policy.mem_buffer.pos_emb_dim // 3),
    }
    expected = {
        "budget": 512,
        "token_per_image": 16,
        "num_views": 1,
        "pos_emb_dim": 768,
        "temporal_dim": 256,
    }

    if actual != expected:
        raise RuntimeError(
            f"Replication A loaded-config mismatch: expected {expected}, got {actual}"
        )

    return actual


def build_row_intervention_plan(
    content_condition: str,
    assemblies: dict,
    retained_manifest: list[dict],
) -> dict:
    if content_condition not in CONDITIONS:
        raise RuntimeError(f"Unknown content condition: {content_condition}")

    assembly = assemblies[content_condition]
    retained_entries = retained_entries_for_row(
        assembly,
        retained_manifest,
    )

    selected_indices = [
        int(entry["assembled_index"])
        for entry in retained_entries
    ]
    positions_by_condition = {
        condition: assembly_identity_positions(assemblies[condition])
        for condition in CONDITIONS
    }

    temporal_maps = {
        condition: make_temporal_position_map(
            retained_entries,
            positions_by_condition[condition],
        )
        for condition in CONDITIONS
    }

    expected_identity = [
        {
            "source_index": idx,
            "target_temporal_position": idx,
        }
        for idx in selected_indices
    ]
    if temporal_maps[content_condition] != expected_identity:
        raise RuntimeError(
            f"{content_condition}: own-condition temporal map is not an exact identity"
        )

    offdiagonal_conditions = [
        condition
        for condition in CONDITIONS
        if condition != content_condition
    ]

    return {
        "content_condition": content_condition,
        "step_idx": int(assembly.exec_start_idx),
        "selected_indices": selected_indices,
        "retained_entries": retained_entries,
        "identity_map": temporal_maps[content_condition],
        "offdiagonal_conditions": offdiagonal_conditions,
        "temporal_maps": temporal_maps,
    }


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


def build_replication_a_unit_inputs(
    query_family: str,
    query_episode: int,
    artifact_dir: Path,
    max_steps: int,
) -> dict:
    if query_family not in FROZEN_FAMILIES:
        raise RuntimeError(
            f"{query_family} is not a frozen Replication A family"
        )
    if int(query_episode) not in FROZEN_EPISODES:
        raise RuntimeError(
            f"Episode {query_episode} is not in frozen Replication A cohort"
        )

    assignment_row, distractors = read_frozen_assignment(
        query_family,
        query_episode,
    )

    query_runner = EnvRunner(
        query_family,
        artifact_dir,
        max_steps=max_steps,
    )

    try:
        query_runner.make_env(query_episode)
        query_pre_traj = query_runner.get_init_obs()
    finally:
        query_runner.close_env()
    distractor_segments = load_distractor_segments(
        distractors,
        EnvRunner,
        artifact_dir,
        max_steps,
    )

    bundle = build_bundle(
        query_pre_traj,
        query_family,
        query_episode,
        distractor_segments,
    )

    retained_manifest = build_retained_identity_manifest(bundle)

    assemblies = {
        condition: assemble_condition(
            bundle,
            condition,
            query_family,
            query_episode,
            query_pre_traj["task_goal"],
        )
        for condition in CONDITIONS
    }
    step_indices = {
        condition: int(assembly.exec_start_idx)
        for condition, assembly in assemblies.items()
    }

    if len(set(step_indices.values())) != 1:
        raise RuntimeError(
            f"FAR/MIDDLE/RECENT full-history step_idx mismatch: {step_indices}"
        )

    query_obs = {
        "observation/image": np.asarray(bundle["Q0"].images[0]).copy(),
        "observation/wrist_image": np.asarray(
            bundle["Q0"].wrist_images[0]
        ).copy(),
        "observation/state": np.asarray(bundle["Q0"].states[0]).copy(),
        "prompt": query_pre_traj["task_goal"],
    }
    return {
        "assignment_row": assignment_row,
        "distractors": distractors,
        "bundle": bundle,
        "retained_manifest": retained_manifest,
        "assemblies": assemblies,
        "query_obs": query_obs,
        "task_goal": query_pre_traj["task_goal"],
        "step_idx": next(iter(step_indices.values())),
    }


def run_replication_a_content_row(
    policy,
    assembly,
    assemblies: dict,
    retained_manifest: list[dict],
    query_obs: dict,
    override_dir: Path,
) -> dict:
    plan = build_row_intervention_plan(
        assembly.condition,
        assemblies,
        retained_manifest,
    )

    policy.reset()
    policy.add_buffer(
        pack_buffer(
            assembly.images,
            assembly.states,
            assembly.exec_start_idx,
        )
    )
    if int(policy.step_idx) != int(plan["step_idx"]):
        raise RuntimeError(
            f'{assembly.condition}: policy.step_idx={policy.step_idx} '
            f'does not match planned step_idx={plan["step_idx"]}'
        )

    override_dir.mkdir(parents=True, exist_ok=True)
    index_override_path = override_dir / f'{assembly.condition}_index_override.json'

    if index_override_path.exists():
        raise FileExistsError(index_override_path)

    index_override_path.write_text(
        json.dumps(
            {
                "step_idx": int(plan["step_idx"]),
                "indices_to_load": plan["selected_indices"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    prior_index_override = os.environ.get("MME_FRAMESAMP_OVERRIDE_PATH")
    prior_temporal_override = os.environ.get(
        "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH"
    )

    if prior_index_override is not None or prior_temporal_override is not None:
        raise RuntimeError(
            "Replication A runner requires FrameSamp override environment "
            "variables to be unset before each scientific row"
        )

    os.environ["MME_FRAMESAMP_OVERRIDE_PATH"] = str(index_override_path)

    try:
        step_idx = int(plan["step_idx"])
        identity_map = plan["identity_map"]
        offdiag_1_condition, offdiag_2_condition = plan[
            "offdiagonal_conditions"
        ]
        offdiag_1_map = plan["temporal_maps"][offdiag_1_condition]
        offdiag_2_map = plan["temporal_maps"][offdiag_2_condition]

        cache_hashes_before = hash_buffer_feats(
            policy.mem_buffer._history_feats
        )
        gather_fn = policy.mem_buffer.default_history_feats_gather_fn
        temporal_dim = policy.mem_buffer.pos_emb_dim // 3

        disabled_tensors = _prepare_tensors(
            policy,
            step_idx,
            gather_fn,
            temporal_pos_override={"enabled": False},
        )
        identity_tensors = _prepare_tensors(
            policy,
            step_idx,
            gather_fn,
            temporal_pos_override={
                "enabled": True,
                "step_idx": step_idx,
                "position_map": identity_map,
            },
        )
        offdiag_1_tensors = _prepare_tensors(
            policy,
            step_idx,
            gather_fn,
            temporal_pos_override={
                "enabled": True,
                "step_idx": step_idx,
                "position_map": offdiag_1_map,
            },
        )
        offdiag_2_tensors = _prepare_tensors(
            policy,
            step_idx,
            gather_fn,
            temporal_pos_override={
                "enabled": True,
                "step_idx": step_idx,
                "position_map": offdiag_2_map,
            },
        )

        dis_img, dis_pos, dis_state, dis_mask = disabled_tensors
        id_img, id_pos, id_state, id_mask = identity_tensors
        od1_img, od1_pos, od1_state, od1_mask = offdiag_1_tensors
        od2_img, od2_pos, od2_state, od2_mask = offdiag_2_tensors

        _assert_all_equal(
            dis_img,
            [
                ("identity", id_img),
                (offdiag_1_condition, od1_img),
                (offdiag_2_condition, od2_img),
            ],
            f"{assembly.condition}: sampled image embeddings",
        )
        _assert_all_equal(
            dis_state,
            [
                ("identity", id_state),
                (offdiag_1_condition, od1_state),
                (offdiag_2_condition, od2_state),
            ],
            f"{assembly.condition}: sampled state embeddings",
        )
        _assert_all_equal(
            dis_mask,
            [
                ("identity", id_mask),
                (offdiag_1_condition, od1_mask),
                (offdiag_2_condition, od2_mask),
            ],
            f"{assembly.condition}: masks",
        )
        _assert_spatial_equal(
            dis_pos,
            [
                ("identity", id_pos),
                (offdiag_1_condition, od1_pos),
                (offdiag_2_condition, od2_pos),
            ],
            temporal_dim,
        )
        np.testing.assert_array_equal(
            dis_pos[..., :temporal_dim],
            id_pos[..., :temporal_dim],
            err_msg=(
                f"{assembly.condition}: disabled temporal channels "
                "!= identity temporal channels"
            ),
        )

        _assert_temporal_confined(
            dis_pos,
            od1_pos,
            temporal_dim,
            f"{assembly.condition}->{offdiag_1_condition}",
        )
        _assert_temporal_confined(
            dis_pos,
            od2_pos,
            temporal_dim,
            f"{assembly.condition}->{offdiag_2_condition}",
        )

        od1_temporal_delta = float(
            np.max(
                np.abs(
                    dis_pos[..., :temporal_dim].astype(np.float64)
                    - od1_pos[..., :temporal_dim].astype(np.float64)
                )
            )
        )
        od2_temporal_delta = float(
            np.max(
                np.abs(
                    dis_pos[..., :temporal_dim].astype(np.float64)
                    - od2_pos[..., :temporal_dim].astype(np.float64)
                )
            )
        )

        if od1_temporal_delta == 0.0 or od2_temporal_delta == 0.0:
            raise RuntimeError(
                f"{assembly.condition}: zero off-diagonal temporal delta"
            )

        cache_hashes_after_tensors = hash_buffer_feats(
            policy.mem_buffer._history_feats
        )
        if cache_hashes_before != cache_hashes_after_tensors:
            raise RuntimeError(
                f"{assembly.condition}: cache mutated during tensor checks"
            )
        call_specs = [
            (
                "disabled",
                {"enabled": False},
            ),
            (
                "identity",
                {
                    "enabled": True,
                    "step_idx": step_idx,
                    "position_map": identity_map,
                },
            ),
            (
                f"to_{offdiag_1_condition}",
                {
                    "enabled": True,
                    "step_idx": step_idx,
                    "position_map": offdiag_1_map,
                },
            ),
            (
                f"to_{offdiag_2_condition}",
                {
                    "enabled": True,
                    "step_idx": step_idx,
                    "position_map": offdiag_2_map,
                },
            ),
            (
                "identity_replay",
                {
                    "enabled": True,
                    "step_idx": step_idx,
                                   "position_map": identity_map,
                },
            ),
        ]

        raw_actions = {}
        chunks = {}
        infer_times_ms = {}

        for name, temporal_override in call_specs:
            response = policy.infer(
                {
                    **query_obs,
                    "reset_rng": True,
                    "temporal_pos_override": temporal_override,
                }
            )

            raw = np.asarray(response["actions"])
            chunk = _initial_action_chunk(raw)

            raw_actions[name] = raw
            chunks[name] = chunk
            infer_times_ms[name] = float(
                response.get("infer_time_ms", float("nan"))
            )
        np.testing.assert_array_equal(
            chunks["disabled"],
            chunks["identity"],
            err_msg=(
                f"{assembly.condition}: disabled chunk != identity chunk"
            ),
        )
        np.testing.assert_array_equal(
            chunks["identity"],
            chunks["identity_replay"],
            err_msg=(
                f"{assembly.condition}: identity chunk != identity replay"
            ),
        )

        cache_hashes_after_inference = hash_buffer_feats(
            policy.mem_buffer._history_feats
        )
        if cache_hashes_before != cache_hashes_after_inference:
            raise RuntimeError(
                f"{assembly.condition}: cache mutated during inference calls"
            )

        metrics = {
            offdiag_1_condition: action_metrics(
                chunks["identity"],
                chunks[f"to_{offdiag_1_condition}"],
            ),
            offdiag_2_condition: action_metrics(
                chunks["identity"],
                chunks[f"to_{offdiag_2_condition}"],
            ),
        }
        return {
            "content_condition": assembly.condition,
            "step_idx": step_idx,
            "selected_indices": plan["selected_indices"],
            "offdiagonal_conditions": plan["offdiagonal_conditions"],
            "temporal_maps": plan["temporal_maps"],
            "temporal_tensor_max_delta": {
                offdiag_1_condition: od1_temporal_delta,
                offdiag_2_condition: od2_temporal_delta,
            },
            "raw_actions": raw_actions,
            "chunks_16x8": chunks,
            "raw_action_shapes": {
                name: list(value.shape)
                for name, value in raw_actions.items()
            },
            "infer_times_ms": infer_times_ms,
            "metrics_vs_identity": metrics,
            "cache_hashes_before": cache_hashes_before,
            "cache_hashes_after_tensors": cache_hashes_after_tensors,
            "cache_hashes_after_inference": cache_hashes_after_inference,
            "index_override_path": str(index_override_path),
            "all_invariants_passed": True,
        }

    finally:
        os.environ.pop("MME_FRAMESAMP_OVERRIDE_PATH", None)
        os.environ.pop(
            "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH",
            None,
        )


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

def save_replication_a_unit_bundle(
    output_dir: Path,
    *,
    unit_inputs: dict,
    row_results: dict,
    query_family: str,
    query_episode: int,
    process_block: str,
    checkpoint_dir: Path,
    seed: int,
    loaded_config: dict,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)

    npz_arrays = {}
    rows_json = {}

    for condition in CONDITIONS:
        row = row_results[condition]

        for call_name, array in row["raw_actions"].items():
            npz_arrays[
                f"{condition}__{call_name}__raw_actions"
            ] = np.asarray(array)

        for call_name, array in row["chunks_16x8"].items():
            npz_arrays[
                f"{condition}__{call_name}__chunk16x8"
            ] = np.asarray(array)
        selected_set = set(int(x) for x in row["selected_indices"])
        assembly = unit_inputs["assemblies"][condition]

        retained_provenance = [
            entry
            for entry in assembly.metadata["frame_index_map"]
            if int(entry["assembled_index"]) in selected_set
        ]

        row_json = {
            key: value
            for key, value in row.items()
            if key not in ("raw_actions", "chunks_16x8")
        }
        row_json["retained_provenance"] = retained_provenance
        row_json["assembly_metadata"] = assembly.metadata
        rows_json[condition] = row_json

    npz_path = output_dir / "actions_and_chunks.npz"
    np.savez_compressed(npz_path, **npz_arrays)

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        gpu = f"unavailable: {type(exc).__name__}: {exc}"

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    result = {
        "protocol_id": PROTOCOL_ID,
        "scope": "open-loop initial inference only; no robot execution",
        "scientific_boundary": "initial inferred 16x8 action chunk",
        "query_family": query_family,
        "query_episode": int(query_episode),
        "process_block": process_block,
        "process_role": "technical nuisance block; not scientific N",
        "task_goal": unit_inputs["task_goal"],
        "model_id": MODEL_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_path": str(checkpoint_dir.resolve()),
        "configured_seed": int(seed),
        "loaded_config": loaded_config,
        "git_commit": git_sha,
        "assignment_source_path": str(ASSIGNMENT_SOURCE.resolve()),
        "assignment_source_sha256": sha256_file(ASSIGNMENT_SOURCE),
        "frozen_30_windows_manifest_sha256": (
            FROZEN_30_WINDOWS_MANIFEST_SHA256
        ),
        "assignment_row": unit_inputs["assignment_row"],
        "distractors": [
            {
                "label": f"D{i}",
                "family": family,
                "episode": int(episode),
            }
            for i, (family, episode) in enumerate(
                unit_inputs["distractors"],
                start=1,
            )
        ],
        "frozen_allocation": EXPECTED_RETAINED_COUNTS,
        "r_selection_rule": (
            "deterministic endpoint-inclusive uniform spacing"
        ),
        "distractor_retained_source_indices": list(
            DISTRACTOR_RETAINED_SOURCE_INDICES
        ),
        "retained_identity_manifest": unit_inputs["retained_manifest"],
        "software": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "jax": package_version("jax"),
            "jaxlib": package_version("jaxlib"),
            "flax": package_version("flax"),
        },
        "gpu": gpu,
        "actions_npz_file": npz_path.name,
        "actions_npz_sha256": sha256_file(npz_path),
        "rows": rows_json,
        "analysis_rules": {
            "causal_comparisons": "strictly within content row",
            "process_is_scientific_n": False,
            "behavioral_claim_permitted": False,
            "natural_interference_explanation_permitted": False,
        },
        "all_invariants_passed": True,
    }

    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "result_file": result_path.name,
        "result_sha256": sha256_file(result_path),
        "actions_npz_file": npz_path.name,
        "actions_npz_sha256": sha256_file(npz_path),
    }

    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return result_path, manifest_path


def save_replication_a_unit_bundle(
    output_dir: Path,
    *,
    unit_inputs: dict,
    row_results: dict,
    query_family: str,
    query_episode: int,
    process_block: str,
    checkpoint_dir: Path,
    seed: int,
    loaded_config: dict,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)

    npz_arrays = {}
    rows_json = {}

    for condition in CONDITIONS:
        row = row_results[condition]

        for call_name, array in row["raw_actions"].items():
            npz_arrays[
                f"{condition}__{call_name}__raw_actions"
            ] = np.asarray(array)

        for call_name, array in row["chunks_16x8"].items():
            npz_arrays[
                f"{condition}__{call_name}__chunk16x8"
            ] = np.asarray(array)

        selected_set = set(int(x) for x in row["selected_indices"])
        assembly = unit_inputs["assemblies"][condition]

        retained_provenance = [
            entry
            for entry in assembly.metadata["frame_index_map"]
            if int(entry["assembled_index"]) in selected_set
        ]
        row_json = {
            key: value
            for key, value in row.items()
            if key not in ("raw_actions", "chunks_16x8")
        }
        row_json["retained_provenance"] = retained_provenance
        row_json["assembly_metadata"] = assembly.metadata
        rows_json[condition] = row_json

    npz_path = output_dir / "actions_and_chunks.npz"
    np.savez_compressed(npz_path, **npz_arrays)
    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        gpu = f"unavailable: {type(exc).__name__}: {exc}"

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    result = {
        "protocol_id": PROTOCOL_ID,
        "scope": "open-loop initial inference only; no robot execution",
        "scientific_boundary": "initial inferred 16x8 action chunk",
        "query_family": query_family,
        "query_episode": int(query_episode),
        "process_block": process_block,
        "process_role": "technical nuisance block; not scientific N",
        "task_goal": unit_inputs["task_goal"],
        "model_id": MODEL_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_path": str(checkpoint_dir.resolve()),
        "configured_seed": int(seed),
        "loaded_config": loaded_config,
        "git_commit": git_sha,
        "assignment_source_path": str(ASSIGNMENT_SOURCE.resolve()),
        "assignment_source_sha256": sha256_file(ASSIGNMENT_SOURCE),
        "frozen_30_windows_manifest_sha256": (
            FROZEN_30_WINDOWS_MANIFEST_SHA256
        ),
        "assignment_row": unit_inputs["assignment_row"],
        "distractors": [
            {
                "label": f"D{i}",
                "family": family,
                "episode": int(episode),
            }
            for i, (family, episode) in enumerate(
                unit_inputs["distractors"],
                start=1,
            )
        ],
        "frozen_allocation": EXPECTED_RETAINED_COUNTS,
        "r_selection_rule": (
            "deterministic endpoint-inclusive uniform spacing"
        ),
        "distractor_retained_source_indices": list(
            DISTRACTOR_RETAINED_SOURCE_INDICES
        ),
        "retained_identity_manifest": unit_inputs["retained_manifest"],
        "software": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "jax": package_version("jax"),
            "jaxlib": package_version("jaxlib"),
            "flax": package_version("flax"),
        },
        "gpu": gpu,
        "actions_npz_file": npz_path.name,
        "actions_npz_sha256": sha256_file(npz_path),
        "rows": rows_json,
        "analysis_rules": {
            "causal_comparisons": "strictly within content row",
            "process_is_scientific_n": False,
            "behavioral_claim_permitted": False,
            "natural_interference_explanation_permitted": False,
        },
        "all_invariants_passed": True,
    }

    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "result_file": result_path.name,
        "result_sha256": sha256_file(result_path),
        "actions_npz_file": npz_path.name,
        "actions_npz_sha256": sha256_file(npz_path),
    }

    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return result_path, manifest_path


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
    parser = argparse.ArgumentParser(
        description="Run one frozen Replication A scientific unit"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--query-family",
        choices=FROZEN_FAMILIES,
        required=True,
    )
    parser.add_argument(
        "--query-episode",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--process-block",
        choices=PROCESS_BLOCKS,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "replication_a",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="mme_vla_suite",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1300,
    )
    args = parser.parse_args()
    if args.query_episode not in FROZEN_EPISODES:
        raise RuntimeError(
            f"Episode {args.query_episode} is not in frozen cohort"
        )

    if args.seed != 42:
        raise RuntimeError(
            f"Replication A v1.0 requires seed 42; got {args.seed}"
        )

    if args.checkpoint_dir.name != CHECKPOINT_ID:
        raise RuntimeError(
            f"Replication A requires checkpoint {CHECKPOINT_ID}; "
            f"got {args.checkpoint_dir}"
        )

    if os.environ.get("MME_FRAMESAMP_OVERRIDE_PATH"):
        raise RuntimeError(
            "MME_FRAMESAMP_OVERRIDE_PATH must be unset before runner start"
        )
    if os.environ.get("MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH"):
        raise RuntimeError(
            "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH must be unset "
            "before runner start"
        )

    output_dir = (
        args.output_root
        / args.process_block
        / f"{args.query_family}_ep{args.query_episode}"
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Scientific result directory already exists: {output_dir}"
        )

    artifact_dir = output_dir.parent / (
        f".{args.query_family}_ep{args.query_episode}_env_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Preparing Replication A unit: %s ep%d %s",
        args.query_family,
        args.query_episode,
        args.process_block,
    )

    unit_inputs = build_replication_a_unit_inputs(
        args.query_family,
        args.query_episode,
        artifact_dir,
        args.max_steps,
    )

    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.training import config as _config

    logging.info("Loading checkpoint from %s", args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.checkpoint_dir,
        seed=args.seed,
    )
    loaded_config = validate_replication_a_policy(policy)

    row_results = {}

    for condition in CONDITIONS:
        logging.info("Running content row: %s", condition.upper())

        row_results[condition] = run_replication_a_content_row(
            policy,
            unit_inputs["assemblies"][condition],
            unit_inputs["assemblies"],
            unit_inputs["retained_manifest"],
            unit_inputs["query_obs"],
            artifact_dir / "overrides",
        )
    result_path, manifest_path = save_replication_a_unit_bundle(
        output_dir,
        unit_inputs=unit_inputs,
        row_results=row_results,
        query_family=args.query_family,
        query_episode=args.query_episode,
        process_block=args.process_block,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        loaded_config=loaded_config,
    )

    logging.info("ALL SCIENTIFIC-UNIT INVARIANTS PASSED")
    logging.info("Result: %s", result_path)
    logging.info("Manifest: %s", manifest_path)

    print(f"REPLICATION_A_UNIT_COMPLETE={output_dir}")
    print(f"RESULT_SHA256={sha256_file(result_path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

