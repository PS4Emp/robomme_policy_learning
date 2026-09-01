import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


DISTRACTOR_RAW_FRAME_COUNT = 256
DISTRACTOR_STRIDE = 8
DISTRACTOR_SELECTED_INDICES = tuple(range(0, DISTRACTOR_RAW_FRAME_COUNT, DISTRACTOR_STRIDE))
CONTROLLED_RECENCY_QUERY_TASKS = (
    "VideoUnmask",
    "VideoUnmaskSwap",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "VideoRepick",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)

CONDITION_ORDERS = {
    "far": ("R", "D1", "D2", "D3", "Q0"),
    "middle": ("D1", "R", "D2", "D3", "Q0"),
    "recent": ("D1", "D2", "D3", "R", "Q0"),
}


@dataclasses.dataclass(frozen=True)
class HistorySegment:
    label: str
    images: tuple[np.ndarray, ...]
    wrist_images: tuple[np.ndarray, ...]
    states: tuple[np.ndarray, ...]
    source_family: str
    source_episode: int
    source_role: str
    source_frame_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ControlledRecencyAssembly:
    condition: str
    images: list[np.ndarray]
    wrist_images: list[np.ndarray]
    states: list[np.ndarray]
    exec_start_idx: int
    metadata: dict[str, Any]


def parse_distractor_spec(spec: str) -> list[tuple[str, int]]:
    if not spec:
        raise ValueError("controlled_recency_distractors must list exactly three Family:Episode distractors")

    distractors = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        family, sep, episode = item.partition(":")
        if not sep:
            raise ValueError(f"Invalid distractor spec '{item}'. Expected Family:Episode")
        try:
            distractors.append((family, int(episode)))
        except ValueError as exc:
            raise ValueError(f"Invalid distractor episode in '{item}'") from exc

    if len(distractors) != 3:
        raise ValueError(f"Expected exactly three distractors, got {len(distractors)}")

    return distractors


def validate_protocol_inputs(query_family: str, distractors: Sequence[tuple[str, int]]) -> None:
    if query_family not in CONTROLLED_RECENCY_QUERY_TASKS:
        raise ValueError(
            f"Controlled recency requires one of {CONTROLLED_RECENCY_QUERY_TASKS}; got query family {query_family}"
        )
    if len(distractors) != 3:
        raise ValueError(f"Expected exactly three distractors, got {len(distractors)}")

    seen = set()
    for family, episode in distractors:
        if family == query_family:
            raise ValueError(f"Distractor {family}:{episode} must not match query family {query_family}")
        key = (family, int(episode))
        if key in seen:
            raise ValueError(f"Duplicate distractor specification: {family}:{episode}")
        seen.add(key)


def split_query_pre_traj(pre_traj: dict[str, Any], family: str, episode: int) -> dict[str, HistorySegment]:
    images, wrist_images, states = _extract_pre_traj_arrays(pre_traj)
    if len(images) < 2:
        raise ValueError(f"Query {family} episode {episode} must provide at least one R frame and one Q0 frame")

    q0_index = len(images) - 1
    return {
        "R": HistorySegment(
            label="R",
            images=tuple(images[:-1]),
            wrist_images=tuple(wrist_images[:-1]),
            states=tuple(states[:-1]),
            source_family=family,
            source_episode=episode,
            source_role="query_relevant_history",
            source_frame_indices=tuple(range(q0_index)),
        ),
        "Q0": HistorySegment(
            label="Q0",
            images=(images[-1],),
            wrist_images=(wrist_images[-1],),
            states=(states[-1],),
            source_family=family,
            source_episode=episode,
            source_role="query_initial_frame",
            source_frame_indices=(q0_index,),
        ),
    }


def extract_distractor_segment(
    pre_traj: dict[str, Any],
    label: str,
    family: str,
    episode: int,
) -> HistorySegment:
    images, wrist_images, states = _extract_pre_traj_arrays(pre_traj)
    demo_frame_count = len(images) - 1
    if demo_frame_count < DISTRACTOR_RAW_FRAME_COUNT:
        raise ValueError(
            f"Distractor {label} ({family} episode {episode}) has {demo_frame_count} raw demonstration "
            f"frames; need at least {DISTRACTOR_RAW_FRAME_COUNT}"
        )

    indices = DISTRACTOR_SELECTED_INDICES
    return HistorySegment(
        label=label,
        images=tuple(images[idx] for idx in indices),
        wrist_images=tuple(wrist_images[idx] for idx in indices),
        states=tuple(states[idx] for idx in indices),
        source_family=family,
        source_episode=episode,
        source_role="distractor_history",
        source_frame_indices=indices,
    )


def load_distractor_segments(
    distractors: Sequence[tuple[str, int]],
    env_runner_factory: Callable[..., Any],
    video_save_dir: Path,
    max_steps: int,
) -> dict[str, HistorySegment]:
    segments = {}
    for idx, (family, episode) in enumerate(distractors, start=1):
        label = f"D{idx}"
        runner = env_runner_factory(family, video_save_dir, max_steps=max_steps)
        try:
            runner.make_env(episode)
            segments[label] = extract_distractor_segment(runner.get_init_obs(), label, family, episode)
        finally:
            close_env = getattr(runner, "close_env", None)
            if callable(close_env):
                close_env()
    return segments


def build_bundle(
    query_pre_traj: dict[str, Any],
    query_family: str,
    query_episode: int,
    distractor_segments: dict[str, HistorySegment],
) -> dict[str, HistorySegment]:
    query_segments = split_query_pre_traj(query_pre_traj, query_family, query_episode)
    expected = {"D1", "D2", "D3"}
    missing = expected.difference(distractor_segments)
    if missing:
        raise ValueError(f"Missing distractor segments: {sorted(missing)}")
    return {**query_segments, **{label: distractor_segments[label] for label in sorted(expected)}}


def assemble_condition(
    bundle: dict[str, HistorySegment],
    condition: str,
    query_family: str,
    query_episode: int,
    task_goal: str | None = None,
) -> ControlledRecencyAssembly:
    condition_key = condition.lower()
    if condition_key not in CONDITION_ORDERS:
        raise ValueError(f"Unknown controlled-recency condition '{condition}'")

    images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    segment_metadata = []
    frame_index_map = []

    for label in CONDITION_ORDERS[condition_key]:
        segment = bundle[label]
        start = len(images)
        images.extend(segment.images)
        wrist_images.extend(segment.wrist_images)
        states.extend(segment.states)
        end = len(images)
        segment_metadata.append(_segment_metadata(segment, start, end))
        for offset, source_frame_index in enumerate(segment.source_frame_indices):
            frame_index_map.append(
                {
                    "assembled_index": start + offset,
                    "segment_label": segment.label,
                    "segment_offset": offset,
                    "source_family": segment.source_family,
                    "source_episode": segment.source_episode,
                    "source_role": segment.source_role,
                    "source_frame_index": int(source_frame_index),
                }
            )

    exec_start_idx = len(images) - 1
    metadata = {
        "protocol": "controlled_recency_v1",
        "condition": condition_key,
        "segment_order": list(CONDITION_ORDERS[condition_key]),
        "query_family": query_family,
        "query_episode": int(query_episode),
        "task_goal": task_goal,
        "exec_start_idx": int(exec_start_idx),
        "total_frames": len(images),
        "distractor_raw_frame_count": DISTRACTOR_RAW_FRAME_COUNT,
        "distractor_stride": DISTRACTOR_STRIDE,
        "distractor_selected_indices": list(DISTRACTOR_SELECTED_INDICES),
        "segments": segment_metadata,
        "frame_index_map": frame_index_map,
    }

    return ControlledRecencyAssembly(
        condition=condition_key,
        images=images,
        wrist_images=wrist_images,
        states=states,
        exec_start_idx=exec_start_idx,
        metadata=metadata,
    )


def write_metadata(metadata: dict[str, Any], output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def write_or_validate_canonical_manifest(
    bundle: dict[str, HistorySegment],
    output_dir: Path,
    query_family: str,
    query_episode: int,
    distractors: Sequence[tuple[str, int]],
    task_goal: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_canonical_manifest(bundle, query_family, query_episode, distractors, task_goal)
    path = output_dir / canonical_manifest_filename(query_family, query_episode, distractors)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing != manifest:
            raise ValueError(f"Controlled-recency canonical manifest mismatch: {path}")
        return path

    with open(path, "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def build_canonical_manifest(
    bundle: dict[str, HistorySegment],
    query_family: str,
    query_episode: int,
    distractors: Sequence[tuple[str, int]],
    task_goal: str,
) -> dict[str, Any]:
    return {
        "protocol": "controlled_recency_v1",
        "query_family": query_family,
        "query_episode": int(query_episode),
        "task_goal": task_goal,
        "distractors": [
            {"family": family, "episode": int(episode)}
            for family, episode in distractors
        ],
        "segments": [
            _canonical_segment_fingerprint(bundle[label])
            for label in ("R", "D1", "D2", "D3", "Q0")
        ],
    }


def canonical_manifest_filename(
    query_family: str,
    query_episode: int,
    distractors: Sequence[tuple[str, int]],
) -> str:
    payload = json.dumps(
        {
            "query_family": query_family,
            "query_episode": int(query_episode),
            "distractors": [(family, int(episode)) for family, episode in distractors],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{query_family}_ep{query_episode}_{digest}_canonical_segments.json"


def summarize_retained_frames(
    metadata: dict[str, Any],
    framesamp_indices: Iterable[int] | dict[str, Any],
) -> dict[str, Any]:
    step_idx = None
    if isinstance(framesamp_indices, dict):
        trace_record = framesamp_indices
        step_idx = int(trace_record["step_idx"])
        framesamp_indices = trace_record["indices_to_load"]

    frame_map = {entry["assembled_index"]: entry for entry in metadata["frame_index_map"]}
    labels = ("R", "D1", "D2", "D3", "Q0", "Qlive")
    counts = {label: 0 for label in labels}
    source_frame_indices = {label: [] for label in labels}
    query_execution_offsets = []
    retained = []
    exec_start_idx = int(metadata["exec_start_idx"])

    for framesamp_index in framesamp_indices:
        assembled_index = int(framesamp_index)
        if step_idx is not None and assembled_index > step_idx:
            raise ValueError(f"FrameSamp index {assembled_index} is greater than trace step_idx {step_idx}")

        if assembled_index <= exec_start_idx:
            if assembled_index not in frame_map:
                raise ValueError(f"FrameSamp index {assembled_index} is outside assembled initial history")
            entry = frame_map[assembled_index]
        else:
            query_execution_offset = assembled_index - exec_start_idx
            entry = {
                "assembled_index": assembled_index,
                "segment_label": "Qlive",
                "segment_offset": query_execution_offset,
                "source_family": metadata["query_family"],
                "source_episode": metadata["query_episode"],
                "source_role": "live_query_execution_frame",
                "source_frame_index": query_execution_offset,
                "query_execution_offset": query_execution_offset,
            }
            query_execution_offsets.append(query_execution_offset)

        label = entry["segment_label"]
        counts[label] += 1
        source_frame_indices[label].append(entry["source_frame_index"])
        retained.append({"framesamp_index": assembled_index, **entry})

    return {
        "condition": metadata["condition"],
        "counts": counts,
        "source_frame_indices": source_frame_indices,
        "query_execution_offsets": query_execution_offsets,
        "retained": retained,
    }


def _extract_pre_traj_arrays(pre_traj: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    images = _copy_arrays(pre_traj["images"])
    wrist_images = _copy_arrays(pre_traj["wrist_images"])
    states = _copy_arrays(pre_traj["states"])
    if not (len(images) == len(wrist_images) == len(states)):
        raise ValueError(
            "pre_traj images, wrist_images, and states must have the same length; "
            f"got {len(images)}, {len(wrist_images)}, {len(states)}"
        )
    return images, wrist_images, states


def _copy_arrays(values: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(value).copy() for value in values]


def _segment_metadata(segment: HistorySegment, start: int, end: int) -> dict[str, Any]:
    return {
        "label": segment.label,
        "assembled_start": int(start),
        "assembled_end_exclusive": int(end),
        "num_frames": len(segment.images),
        "source_family": segment.source_family,
        "source_episode": int(segment.source_episode),
        "source_role": segment.source_role,
        "source_frame_indices": [int(idx) for idx in segment.source_frame_indices],
        "image_sha256": _hash_arrays(segment.images),
        "wrist_image_sha256": _hash_arrays(segment.wrist_images),
        "state_sha256": _hash_arrays(segment.states),
    }


def _canonical_segment_fingerprint(segment: HistorySegment) -> dict[str, Any]:
    return {
        "label": segment.label,
        "num_frames": len(segment.images),
        "source_family": segment.source_family,
        "source_episode": int(segment.source_episode),
        "source_role": segment.source_role,
        "source_frame_indices": [int(idx) for idx in segment.source_frame_indices],
        "image_sha256": _hash_arrays(segment.images),
        "wrist_image_sha256": _hash_arrays(segment.wrist_images),
        "state_sha256": _hash_arrays(segment.states),
    }


def _hash_arrays(arrays: Sequence[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.shape).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(contiguous.dtype.str.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(contiguous.tobytes())
        hasher.update(b"\n")
    return hasher.hexdigest()
