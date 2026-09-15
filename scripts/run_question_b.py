#!/usr/bin/env python3
"""Formal Question B: natural FrameSamp vs exact retention-matched FrameSamp."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import run_replication_a as ra
from controlled_recency import summarize_retained_frames


PROTOCOL_ID = "ROBOMME-NATURAL-VS-RETENTION-MATCHED-B-v1.0"
PAIRS = (("far", "middle"), ("far", "recent"), ("middle", "recent"))
def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def semantic_identity(entry: dict) -> tuple:
    return (
        entry["segment_label"],
        entry["source_family"],
        int(entry["source_episode"]),
        entry["source_role"],
        int(entry["source_frame_index"]),
    )


def semantic_identity_list(summary: dict) -> list[tuple]:
    return [semantic_identity(x) for x in summary["retained"]]
def overlap_stats(a: dict, b: dict) -> dict:
    sa = set(semantic_identity_list(a))
    sb = set(semantic_identity_list(b))
    inter = sa & sb
    union = sa | sb

    ra_set = {x for x in sa if x[0] == "R"}
    rb_set = {x for x in sb if x[0] == "R"}
    r_union = ra_set | rb_set

    return {
        "intersection_count": len(inter),
        "union_count": len(union),
        "jaccard": len(inter) / len(union) if union else 1.0,
        "relevant_R_intersection_count": len(ra_set & rb_set),
        "relevant_R_union_count": len(r_union),
        "relevant_R_jaccard": (
            len(ra_set & rb_set) / len(r_union) if r_union else 1.0
        ),
        "exact_semantic_identity_match": sa == sb,
    }
def run_arm(
    policy,
    assembly,
    query_obs: dict,
    *,
    arm_name: str,
    artifact_dir: Path,
    matched_indices: list[int] | None,
) -> dict:
    if os.environ.get("MME_FRAMESAMP_OVERRIDE_PATH"):
        raise RuntimeError("MME_FRAMESAMP_OVERRIDE_PATH unexpectedly set")
    if os.environ.get("MME_FRAMESAMP_TRACE_PATH"):
        raise RuntimeError("MME_FRAMESAMP_TRACE_PATH unexpectedly set")

    policy.reset()
    policy.add_buffer(
        ra.pack_buffer(
            assembly.images,
            assembly.states,
            assembly.exec_start_idx,
        )
    )
    step_idx = int(policy.step_idx)
    if step_idx != int(assembly.exec_start_idx):
        raise RuntimeError(
            f"{assembly.condition}/{arm_name}: step_idx {step_idx} "
            f"!= exec_start_idx {assembly.exec_start_idx}"
        )

    cache_before = ra.hash_buffer_feats(policy.mem_buffer._history_feats)

    artifact_dir.mkdir(parents=True, exist_ok=False)
    trace_path = artifact_dir / "framesamp_trace.jsonl"

    override_path = None
    if matched_indices is not None:
        if len(matched_indices) != 32:
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: expected 32 matched indices, "
                f"got {len(matched_indices)}"
            )
        if matched_indices != sorted(matched_indices):
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: matched indices not sorted"
            )
        if len(set(matched_indices)) != 32:
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: matched indices not unique"
            )

        override_path = artifact_dir / "index_override.json"
        override_path.write_text(
            json.dumps(
                {
                    "step_idx": step_idx,
                    "indices_to_load": [int(x) for x in matched_indices],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.environ["MME_FRAMESAMP_OVERRIDE_PATH"] = str(override_path)
    os.environ["MME_FRAMESAMP_TRACE_PATH"] = str(trace_path)

    try:
        raw_actions = []
        chunks = []
        infer_times = []

        for _ in range(2):
            response = policy.infer(
                {
                    **query_obs,
                    "reset_rng": True,
                    "temporal_pos_override": {"enabled": False},
                }
            )
            raw = np.asarray(response["actions"])
            chunk = ra._initial_action_chunk(raw)

            raw_actions.append(raw)
            chunks.append(chunk)
            infer_times.append(float(response.get("infer_time_ms", 0.0)))
        np.testing.assert_array_equal(
            chunks[0],
            chunks[1],
            err_msg=f"{assembly.condition}/{arm_name}: replay mismatch",
        )

        cache_after = ra.hash_buffer_feats(policy.mem_buffer._history_feats)
        if cache_before != cache_after:
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: cache mutated during inference"
            )

        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != 2:
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: expected 2 trace records, "
                f"got {len(records)}"
            )

        idx0 = [int(x) for x in records[0]["indices_to_load"]]
        idx1 = [int(x) for x in records[1]["indices_to_load"]]

        if idx0 != idx1:
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: replay selected different frames"
            )

        if int(records[0]["step_idx"]) != step_idx:
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: trace step_idx mismatch"
            )
        if matched_indices is not None and idx0 != matched_indices:
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: model-boundary matched indices "
                "differ from frozen override"
            )

        retention = summarize_retained_frames(
            assembly.metadata,
            records[0],
        )

        retained_total = sum(retention["counts"].values())
        if retained_total != 32:
            raise RuntimeError(
                f"{assembly.condition}/{arm_name}: retained {retained_total}, expected 32"
            )
        return {
            "step_idx": step_idx,
            "selected_indices": idx0,
            "retention": retention,
            "raw_action_shape": list(raw_actions[0].shape),
            "infer_times_ms": infer_times,
            "cache_hash_before": cache_before,
            "cache_hash_after": cache_after,
            "override_path": str(override_path) if override_path else None,
            "trace_path": str(trace_path),
            "raw_actions": raw_actions[0],
            "chunk_16x8": chunks[0],
            "all_invariants_passed": True,
        }

    finally:
        os.environ.pop("MME_FRAMESAMP_TRACE_PATH", None)
        os.environ.pop("MME_FRAMESAMP_OVERRIDE_PATH", None)


def strip_arrays(arm: dict) -> dict:
    return {
        k: v
        for k, v in arm.items()
        if k not in {"raw_actions", "chunk_16x8"}
    }
def run_unit(
    policy,
    *,
    family: str,
    episode: int,
    output_dir: Path,
    artifact_dir: Path,
    max_steps: int,
) -> tuple[Path, Path]:
    unit_inputs = ra.build_replication_a_unit_inputs(
        family,
        episode,
        artifact_dir / "environment",
        max_steps,
    )

    natural = {}
    matched = {}

    for condition in ra.CONDITIONS:
        assembly = unit_inputs["assemblies"][condition]

        logging.info(
            "Question B %s ep%d condition=%s arm=NATURAL",
            family,
            episode,
            condition.upper(),
        )
        natural[condition] = run_arm(
            policy,
            assembly,
            unit_inputs["query_obs"],
            arm_name="natural",
            artifact_dir=artifact_dir / condition / "natural",
            matched_indices=None,
        )

        plan = ra.build_row_intervention_plan(
            condition,
            unit_inputs["assemblies"],
            unit_inputs["retained_manifest"],
        )
        matched_indices = [int(x) for x in plan["selected_indices"]]

        logging.info(
            "Question B %s ep%d condition=%s arm=MATCHED",
            family,
            episode,
            condition.upper(),
        )
        matched[condition] = run_arm(
            policy,
            assembly,
            unit_inputs["query_obs"],
            arm_name="matched",
            artifact_dir=artifact_dir / condition / "matched",
            matched_indices=matched_indices,
        )

        counts = matched[condition]["retention"]["counts"]
        expected = {"R": 25, "D1": 2, "D2": 2, "D3": 2, "Q0": 1}
        for label, n in expected.items():
            if int(counts.get(label, 0)) != n:
                raise RuntimeError(
                    f"{family} ep{episode} {condition}: matched {label} count "
                    f"{counts.get(label)} != {n}"
                )

    # The matched semantic evidence set must be identical across FAR/MIDDLE/RECENT.
    matched_sets = {
        condition: set(semantic_identity_list(matched[condition]["retention"]))
        for condition in ra.CONDITIONS
    }
    if not (
        matched_sets["far"]
        == matched_sets["middle"]
        == matched_sets["recent"]
    ):
        raise RuntimeError(
            f"{family} ep{episode}: matched semantic identity sets differ by condition"
        )

    pairwise = {}
    retention_overlap = {}

    for a, b in PAIRS:
        natural_metrics = ra.action_metrics(
            natural[a]["chunk_16x8"],
            natural[b]["chunk_16x8"],
        )
        matched_metrics = ra.action_metrics(
            matched[a]["chunk_16x8"],
            matched[b]["chunk_16x8"],
        )

        natural_mae = float(natural_metrics["mae"])
        matched_mae = float(matched_metrics["mae"])

        pairwise[f"{a}__{b}"] = {
            "natural": natural_metrics,
            "matched": matched_metrics,
            "retention_attributable_mae_reduction": natural_mae - matched_mae,
            "matched_over_natural_mae_ratio": (
                matched_mae / natural_mae if natural_mae > 0 else None
            ),
        }
        retention_overlap[f"{a}__{b}"] = {
            "natural": overlap_stats(
                natural[a]["retention"],
                natural[b]["retention"],
            ),
            "matched": overlap_stats(
                matched[a]["retention"],
                matched[b]["retention"],
            ),
        }

    same_condition_natural_vs_matched = {
        condition: ra.action_metrics(
            natural[condition]["chunk_16x8"],
            matched[condition]["chunk_16x8"],
        )
        for condition in ra.CONDITIONS
    }
    output_dir.mkdir(parents=True, exist_ok=False)

    npz_payload = {}
    for condition in ra.CONDITIONS:
        npz_payload[f"{condition}__natural__raw_actions"] = natural[condition][
            "raw_actions"
        ]
        npz_payload[f"{condition}__natural__chunk16x8"] = natural[condition][
            "chunk_16x8"
        ]
        npz_payload[f"{condition}__matched__raw_actions"] = matched[condition][
            "raw_actions"
        ]
        npz_payload[f"{condition}__matched__chunk16x8"] = matched[condition][
            "chunk_16x8"
        ]

    npz_path = output_dir / "actions_and_chunks.npz"
    np.savez_compressed(npz_path, **npz_payload)

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ra.REPO_ROOT,
        text=True,
    ).strip()
    result = {
        "protocol_id": PROTOCOL_ID,
        "scope": "Question B open-loop natural-vs-retention-matched decomposition",
        "scientific_boundary": "initial inferred 16x8 action chunk",
        "query_family": family,
        "query_episode": episode,
        "task_goal": unit_inputs["task_goal"],
        "git_commit": git_commit,
        "model_id": ra.MODEL_ID,
        "checkpoint_id": ra.CHECKPOINT_ID,
        "configured_seed": 42,
        "assignment_source_sha256": ra.ASSIGNMENT_SOURCE_SHA256,
        "assignment_row": unit_inputs["assignment_row"],
        "distractors": unit_inputs["distractors"],
        "matched_retained_identity_manifest": unit_inputs["retained_manifest"],
        "arms": {
            "natural": {
                c: strip_arrays(natural[c])
                for c in ra.CONDITIONS
            },
            "matched": {
                c: strip_arrays(matched[c])
                for c in ra.CONDITIONS
            },
        },
        "natural_retention_overlap": {
            k: v["natural"] for k, v in retention_overlap.items()
        },
        "matched_retention_overlap": {
            k: v["matched"] for k, v in retention_overlap.items()
        },
        "pairwise_output_metrics": pairwise,
        "same_condition_natural_vs_matched": same_condition_natural_vs_matched,
        "analysis_rules": {
            "primary_metric": "MAE on initial inferred 16x8 action chunk",
            "retention_attributable_reduction": "natural MAE - matched MAE",
            "behavior_claim_permitted": False,
            "replication_a_interpretation": (
                "Replication A can inform interpretation of residual matched effects "
                "but does not establish temporal representation as the sole mechanism."
            ),
        },
        "actions_npz_file": npz_path.name,
        "actions_npz_sha256": ra.sha256_file(npz_path),
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
        "result_sha256": ra.sha256_file(result_path),
        "actions_npz_file": npz_path.name,
        "actions_npz_sha256": ra.sha256_file(npz_path),
    }

    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return result_path, manifest_path
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run formal Question B natural-vs-retention-matched cohort"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ra.REPO_ROOT / "runs" / "question_b_formal",
    )
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1300)
    args = parser.parse_args()

    if args.seed != 42:
        raise RuntimeError("Question B v1.0 requires seed 42")
    if args.checkpoint_dir.name != ra.CHECKPOINT_ID:
        raise RuntimeError(
            f"Question B requires checkpoint {ra.CHECKPOINT_ID}"
        )
    if args.output_root.exists():
        raise FileExistsError(
            f"Formal Question B output already exists: {args.output_root}"
        )
    if os.environ.get("MME_FRAMESAMP_OVERRIDE_PATH"):
        raise RuntimeError("MME_FRAMESAMP_OVERRIDE_PATH must be unset")
    if os.environ.get("MME_FRAMESAMP_TRACE_PATH"):
        raise RuntimeError("MME_FRAMESAMP_TRACE_PATH must be unset")
    if os.environ.get("MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH"):
        raise RuntimeError(
            "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH must be unset"
        )

    args.output_root.mkdir(parents=True)
    artifact_root = args.output_root / ".env_artifacts"
    artifact_root.mkdir()

    planned_units = [
        {"query_family": family, "query_episode": episode}
        for family in ra.FROZEN_FAMILIES
        for episode in ra.FROZEN_EPISODES
    ]

    progress_path = args.output_root / "block_progress.json"
    started = time.time()
    progress = {
        "protocol_id": PROTOCOL_ID,
        "status": "initializing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ra.REPO_ROOT,
            text=True,
        ).strip(),
        "planned_unit_count": 30,
        "planned_units": planned_units,
        "completed_units": [],
    }
    write_json_atomic(progress_path, progress)

    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.training import config as _config

    logging.info("Loading checkpoint once for formal Question B")
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.checkpoint_dir,
        seed=args.seed,
    )
    ra.validate_replication_a_policy(policy)
    progress["status"] = "running"
    write_json_atomic(progress_path, progress)

    try:
        for unit_number, unit in enumerate(planned_units, start=1):
            family = unit["query_family"]
            episode = unit["query_episode"]

            logging.info(
                "QUESTION B UNIT %d/30: %s ep%d",
                unit_number,
                family,
                episode,
            )

            result_path, manifest_path = run_unit(
                policy,
                family=family,
                episode=episode,
                output_dir=args.output_root / f"{family}_ep{episode}",
                artifact_dir=artifact_root / f"{family}_ep{episode}",
                max_steps=args.max_steps,
            )
            progress["completed_units"].append(
                {
                    "unit_number": unit_number,
                    "query_family": family,
                    "query_episode": episode,
                    "result_path": str(
                        result_path.relative_to(args.output_root)
                    ),
                    "result_sha256": ra.sha256_file(result_path),
                    "manifest_path": str(
                        manifest_path.relative_to(args.output_root)
                    ),
                }
            )
            progress["last_completed_unit"] = unit_number
            progress["elapsed_seconds"] = time.time() - started
            write_json_atomic(progress_path, progress)
        progress["status"] = "complete"
        progress["completed_unit_count"] = len(progress["completed_units"])
        progress["all_30_units_complete"] = (
            len(progress["completed_units"]) == 30
        )
        progress["finished_utc"] = datetime.now(timezone.utc).isoformat()
        progress["elapsed_seconds"] = time.time() - started

        manifest_path = args.output_root / "block_manifest.json"
        write_json_atomic(manifest_path, progress)
        progress_path.unlink()

        print(f"QUESTION_B_COMPLETE={args.output_root}")
        print(f"BLOCK_MANIFEST={manifest_path}")
        print(f"BLOCK_MANIFEST_SHA256={ra.sha256_file(manifest_path)}")
        return 0
    except Exception as exc:
        progress["status"] = "failed"
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        progress["failed_utc"] = datetime.now(timezone.utc).isoformat()
        progress["elapsed_seconds"] = time.time() - started
        write_json_atomic(progress_path, progress)
        raise


if __name__ == "__main__":
    sys.exit(main())
