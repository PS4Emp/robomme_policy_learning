#!/usr/bin/env python3
"""Run one complete frozen Replication A process block in a single Python process."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import run_replication_a as ra


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all 30 frozen Replication A units in one process block"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--process-block", choices=ra.PROCESS_BLOCKS, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ra.REPO_ROOT / "runs" / "replication_a_formal",
    )
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1300)
    args = parser.parse_args()

    if args.seed != 42:
        raise RuntimeError(f"Replication A v1.0 requires seed 42; got {args.seed}")
    if args.checkpoint_dir.name != ra.CHECKPOINT_ID:
        raise RuntimeError(
            f"Replication A requires checkpoint {ra.CHECKPOINT_ID}; "
            f"got {args.checkpoint_dir}"
        )
    if os.environ.get("MME_FRAMESAMP_OVERRIDE_PATH"):
        raise RuntimeError("MME_FRAMESAMP_OVERRIDE_PATH must be unset")
    if os.environ.get("MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH"):
        raise RuntimeError(
            "MME_FRAMESAMP_TEMPORAL_POS_OVERRIDE_PATH must be unset"
        )

    block_dir = args.output_root / args.process_block
    if block_dir.exists():
        raise FileExistsError(
            f"Formal process-block directory already exists: {block_dir}. "
            "A formal block must start from a fresh process and empty directory."
        )

    block_dir.mkdir(parents=True)
    artifact_root = block_dir / ".env_artifacts"
    artifact_root.mkdir()

    progress_path = block_dir / "block_progress.json"
    start_time = time.time()
    planned_units = [
        {"query_family": family, "query_episode": episode}
        for family in ra.FROZEN_FAMILIES
        for episode in ra.FROZEN_EPISODES
    ]
    progress = {
        "protocol_id": ra.PROTOCOL_ID,
        "process_block": args.process_block,
        "process_role": "technical nuisance block; not scientific N",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": ra.subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ra.REPO_ROOT,
            text=True,
        ).strip(),
        "checkpoint_id": ra.CHECKPOINT_ID,
        "checkpoint_path": str(args.checkpoint_dir.resolve()),
        "configured_seed": args.seed,
        "planned_unit_count": len(planned_units),
        "planned_units": planned_units,
        "completed_units": [],
        "status": "initializing",
    }
    write_json_atomic(progress_path, progress)

    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.training import config as _config
    logging.info(
        "Loading checkpoint once for formal process block %s",
        args.process_block,
    )
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.checkpoint_dir,
        seed=args.seed,
    )
    loaded_config = ra.validate_replication_a_policy(policy)

    progress["status"] = "running"
    write_json_atomic(progress_path, progress)

    try:
        for unit_number, unit in enumerate(planned_units, start=1):
            family = unit["query_family"]
            episode = unit["query_episode"]

            logging.info(
                "FORMAL BLOCK %s UNIT %d/%d: %s ep%d",
                args.process_block,
                unit_number,
                len(planned_units),
                family,
                episode,
            )
            unit_dir = block_dir / f"{family}_ep{episode}"
            artifact_dir = artifact_root / f"{family}_ep{episode}"
            artifact_dir.mkdir(parents=True, exist_ok=False)

            unit_inputs = ra.build_replication_a_unit_inputs(
                family,
                episode,
                artifact_dir,
                args.max_steps,
            )

            row_results = {}
            for condition in ra.CONDITIONS:
                logging.info(
                    "Unit %d/%d %s ep%d row=%s",
                    unit_number,
                    len(planned_units),
                    family,
                    episode,
                    condition.upper(),
                )
                row_results[condition] = ra.run_replication_a_content_row(
                    policy,
                    unit_inputs["assemblies"][condition],
                    unit_inputs["assemblies"],
                    unit_inputs["retained_manifest"],
                    unit_inputs["query_obs"],
                    artifact_dir / "overrides",
                )

            result_path, manifest_path = ra.save_replication_a_unit_bundle(
                unit_dir,
                unit_inputs=unit_inputs,
                row_results=row_results,
                query_family=family,
                query_episode=episode,
                process_block=args.process_block,
                checkpoint_dir=args.checkpoint_dir,
                seed=args.seed,
                loaded_config=loaded_config,
            )
            progress["completed_units"].append(
                {
                    "unit_number": unit_number,
                    "query_family": family,
                    "query_episode": episode,
                    "result_path": str(result_path.relative_to(block_dir)),
                    "result_sha256": ra.sha256_file(result_path),
                    "manifest_path": str(manifest_path.relative_to(block_dir)),
                }
            )
            progress["last_completed_unit"] = unit_number
            progress["elapsed_seconds"] = time.time() - start_time
            write_json_atomic(progress_path, progress)

        progress["status"] = "complete"
        progress["completed_unit_count"] = len(progress["completed_units"])
        progress["finished_utc"] = datetime.now(timezone.utc).isoformat()
        progress["elapsed_seconds"] = time.time() - start_time
        progress["all_30_units_complete"] = (
            len(progress["completed_units"]) == len(planned_units) == 30
    )
        final_manifest = block_dir / "block_manifest.json"
        write_json_atomic(final_manifest, progress)
        progress_path.unlink()

        print(f"REPLICATION_A_BLOCK_COMPLETE={block_dir}")
        print(f"BLOCK_MANIFEST={final_manifest}")
        print(f"BLOCK_MANIFEST_SHA256={ra.sha256_file(final_manifest)}")
        return 0

    except Exception as exc:
        progress["status"] = "failed"
        progress["failed_utc"] = datetime.now(timezone.utc).isoformat()
        progress["elapsed_seconds"] = time.time() - start_time
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        write_json_atomic(progress_path, progress)
        raise


if __name__ == "__main__":
    sys.exit(main())
