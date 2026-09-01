import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "robomme"
sys.path.insert(0, str(EXAMPLES_DIR))

from controlled_recency import CONDITION_ORDERS
from controlled_recency import DISTRACTOR_SELECTED_INDICES
from controlled_recency import assemble_condition
from controlled_recency import build_bundle
from controlled_recency import build_canonical_manifest
from controlled_recency import canonical_manifest_filename
from controlled_recency import extract_distractor_segment
from controlled_recency import summarize_retained_frames
from controlled_recency import validate_protocol_inputs
from controlled_recency import write_metadata
from controlled_recency import write_or_validate_canonical_manifest


def _make_pre_traj(length, base):
    return {
        "images": [
            np.full((2, 2, 3), (base + idx) % 256, dtype=np.uint8)
            for idx in range(length)
        ],
        "wrist_images": [
            np.full((2, 2, 3), (base + idx + 17) % 256, dtype=np.uint8)
            for idx in range(length)
        ],
        "states": [
            np.array([base, idx, base + idx], dtype=np.float32)
            for idx in range(length)
        ],
        "task_goal": "synthetic",
    }


def _stack(values):
    return np.stack(values, axis=0)


def _frame_state_multiset(assembly):
    keys = []
    for image, state in zip(assembly.images, assembly.states, strict=True):
        keys.append((image.tobytes(), image.dtype.str, image.shape, state.tobytes(), state.dtype.str, state.shape))
    return Counter(keys)


def _segment_slice(assembly, label):
    segment = next(item for item in assembly.metadata["segments"] if item["label"] == label)
    start = segment["assembled_start"]
    end = segment["assembled_end_exclusive"]
    return assembly.images[start:end], assembly.wrist_images[start:end], assembly.states[start:end]


def _make_bundle():
    query_pre_traj = _make_pre_traj(6, 10)
    distractors = {
        f"D{idx}": extract_distractor_segment(
            _make_pre_traj(257, idx * 40),
            f"D{idx}",
            f"Family{idx}",
            idx,
        )
        for idx in range(1, 4)
    }
    return build_bundle(query_pre_traj, "VideoRepick", 7, distractors)


def _repo_tmpdir():
    return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])


def test_controlled_recency_assemblies_reuse_identical_segments():
    bundle = _make_bundle()

    assemblies = {
        condition: assemble_condition(bundle, condition, "VideoRepick", 7)
        for condition in ("far", "middle", "recent")
    }

    lengths = {condition: len(assembly.images) for condition, assembly in assemblies.items()}
    assert len(set(lengths.values())) == 1

    multisets = {
        condition: _frame_state_multiset(assembly)
        for condition, assembly in assemblies.items()
    }
    assert multisets["far"] == multisets["middle"] == multisets["recent"]

    for condition, assembly in assemblies.items():
        assert assembly.metadata["segment_order"] == list(CONDITION_ORDERS[condition])
        assert assembly.exec_start_idx == len(assembly.images) - 1
        assert assembly.metadata["exec_start_idx"] == len(assembly.images) - 1

        for label, segment in bundle.items():
            images, wrist_images, states = _segment_slice(assembly, label)
            np.testing.assert_array_equal(_stack(images), _stack(segment.images))
            np.testing.assert_array_equal(_stack(wrist_images), _stack(segment.wrist_images))
            np.testing.assert_array_equal(_stack(states), _stack(segment.states))

    assert assemblies["far"].metadata["segment_order"] == ["R", "D1", "D2", "D3", "Q0"]
    assert assemblies["middle"].metadata["segment_order"] == ["D1", "R", "D2", "D3", "Q0"]
    assert assemblies["recent"].metadata["segment_order"] == ["D1", "D2", "D3", "R", "Q0"]

    starts_by_condition = {
        condition: {
            segment["label"]: segment["assembled_start"]
            for segment in assembly.metadata["segments"]
        }
        for condition, assembly in assemblies.items()
    }
    assert starts_by_condition["far"]["R"] == 0
    assert starts_by_condition["middle"]["R"] == 32
    assert starts_by_condition["recent"]["R"] == 96

    for assembly in assemblies.values():
        np.testing.assert_array_equal(assembly.images[-1], bundle["Q0"].images[0])
        np.testing.assert_array_equal(assembly.wrist_images[-1], bundle["Q0"].wrist_images[0])
        np.testing.assert_array_equal(assembly.states[-1], bundle["Q0"].states[0])


def test_distractor_extraction_uses_locked_stride_and_rejects_short_histories():
    segment = extract_distractor_segment(_make_pre_traj(257, 20), "D1", "OtherFamily", 3)

    assert segment.source_frame_indices == DISTRACTOR_SELECTED_INDICES
    assert len(segment.images) == 32
    np.testing.assert_array_equal(_stack(segment.images), _stack(_make_pre_traj(257, 20)["images"][0:256:8]))
    np.testing.assert_array_equal(_stack(segment.states), _stack(_make_pre_traj(257, 20)["states"][0:256:8]))

    with pytest.raises(ValueError, match="need at least 256"):
        extract_distractor_segment(_make_pre_traj(256, 20), "D1", "OtherFamily", 3)


def test_retained_frame_report_maps_framesamp_indices_to_segments():
    bundle = _make_bundle()
    assembly = assemble_condition(bundle, "middle", "VideoRepick", 9)

    report = summarize_retained_frames(
        assembly.metadata,
        {"step_idx": assembly.exec_start_idx + 3, "indices_to_load": [0, 31, 32, 36, assembly.exec_start_idx, assembly.exec_start_idx + 2]},
    )

    assert report["counts"] == {"R": 2, "D1": 2, "D2": 0, "D3": 0, "Q0": 1, "Qlive": 1}
    assert report["source_frame_indices"]["D1"] == [0, 248]
    assert report["source_frame_indices"]["R"] == [0, 4]
    assert report["source_frame_indices"]["Q0"] == [5]
    assert report["query_execution_offsets"] == [2]

    with pytest.raises(ValueError, match="greater than trace step_idx"):
        summarize_retained_frames(assembly.metadata, {"step_idx": assembly.exec_start_idx + 1, "indices_to_load": [assembly.exec_start_idx + 2]})


def test_canonical_manifest_validates_segments_across_condition_runs():
    distractors = [("Family1", 1), ("Family2", 2), ("Family3", 3)]
    bundle = _make_bundle()
    task_goal = "watch the video carefully"

    with _repo_tmpdir() as tmp_dir:
        tmp_path = Path(tmp_dir)
        path = write_or_validate_canonical_manifest(bundle, tmp_path, "VideoRepick", 7, distractors, task_goal)
        assert path.name == canonical_manifest_filename("VideoRepick", 7, distractors)

        regenerated = _make_bundle()
        assert write_or_validate_canonical_manifest(
            regenerated, tmp_path, "VideoRepick", 7, distractors, task_goal
        ) == path

        far_manifest = build_canonical_manifest(bundle, "VideoRepick", 7, distractors, task_goal)
        assembly = assemble_condition(bundle, "recent", "VideoRepick", 7, task_goal)
        recent_manifest = build_canonical_manifest(bundle, "VideoRepick", 7, distractors, task_goal)
        assert assembly.metadata["task_goal"] == task_goal
        assert assembly.metadata["segment_order"] == ["D1", "D2", "D3", "R", "Q0"]
        assert far_manifest == recent_manifest

        changed = _make_bundle()
        changed["D2"].images[0][0, 0, 0] ^= np.uint8(1)
        with pytest.raises(ValueError, match="canonical manifest mismatch"):
            write_or_validate_canonical_manifest(changed, tmp_path, "VideoRepick", 7, distractors, task_goal)

        changed_state = _make_bundle()
        changed_state["R"].states[0][0] += 1.0
        with pytest.raises(ValueError, match="canonical manifest mismatch"):
            write_or_validate_canonical_manifest(changed_state, tmp_path, "VideoRepick", 7, distractors, task_goal)

        with pytest.raises(ValueError, match="canonical manifest mismatch"):
            write_or_validate_canonical_manifest(
                regenerated, tmp_path, "VideoRepick", 7, distractors, "changed task goal"
            )


def test_protocol_input_validation_and_safe_metadata_write():
    validate_protocol_inputs("VideoRepick", [("BinFill", 0), ("StopCube", 1), ("PickXtimes", 2)])

    with pytest.raises(ValueError, match="must not match query family"):
        validate_protocol_inputs("VideoRepick", [("VideoRepick", 0), ("StopCube", 1), ("PickXtimes", 2)])

    with pytest.raises(ValueError, match="Duplicate distractor"):
        validate_protocol_inputs("VideoRepick", [("BinFill", 0), ("BinFill", 0), ("PickXtimes", 2)])

    with pytest.raises(ValueError, match="requires one of"):
        validate_protocol_inputs("BinFill", [("VideoRepick", 0), ("StopCube", 1), ("PickXtimes", 2)])

    with pytest.raises(ValueError, match="at least one R frame"):
        build_bundle(_make_pre_traj(1, 10), "VideoRepick", 7, {"D1": None, "D2": None, "D3": None})

    with _repo_tmpdir() as tmp_dir:
        tmp_path = Path(tmp_dir)
        write_metadata({"ok": True}, tmp_path, "metadata.json")
        with pytest.raises(FileExistsError):
            write_metadata({"ok": True}, tmp_path, "metadata.json")
