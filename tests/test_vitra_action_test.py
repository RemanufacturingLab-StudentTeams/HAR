"""Tests voor de VITRA-actierijen en hulpfuncties (zonder model/GPU)."""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplo.schema import VITRA_COLUMNS, vitra_action_rows  # noqa: E402
from duplo.vitra import (  # noqa: E402
    VitraPredictor,
    action_summary,
    annotate_vitra,
    load_hand_state,
    pad_action_mask,
    pad_state_human,
    parse_hands,
    resolve_model_path,
    resolve_statistics_path,
    state_from_hand_data,
)
from duplo.vitra_state import (  # noqa: E402
    check_hand_recon_assets,
    load_frame_states,
    reconstruct_video_states,
    resolve_hand_recon_paths,
    save_frame_states,
)


class VitraActionRowsTest(unittest.TestCase):
    def test_rows_for_both_hands(self):
        actions = np.zeros((1, 3, 102), dtype=np.float32)
        actions[0, 0, 0:3] = [1.0, 2.0, 3.0]
        actions[0, 1, 51:54] = [0.0, 0.0, 4.0]
        actions[0, 1, 51 + 6:51 + 51] = 0.5

        rows = vitra_action_rows("video.mp4", 30, 30.0, actions,
                                 ("left", "right"))

        self.assertEqual(len(rows), 6)
        self.assertEqual(tuple(rows[0]), tuple(VITRA_COLUMNS))
        self.assertEqual(rows[0]["hand"], "left")
        self.assertEqual(rows[0]["step"], 0)
        self.assertEqual(rows[0]["timestamp"], 1.0)
        self.assertEqual(rows[0]["transl_x"], 1.0)
        self.assertEqual(rows[0]["transl_y"], 2.0)
        self.assertEqual(rows[0]["transl_z"], 3.0)
        self.assertAlmostEqual(rows[0]["transl_norm"],
                               round(math.sqrt(14), 6))
        right_step_one = rows[4]
        self.assertEqual(right_step_one["hand"], "right")
        self.assertEqual(right_step_one["step"], 1)
        self.assertEqual(right_step_one["transl_z"], 4.0)
        self.assertEqual(right_step_one["pose_mean"], 0.5)

    def test_rows_skip_unselected_hand(self):
        actions = np.zeros((3, 102), dtype=np.float32)
        rows = vitra_action_rows("video.mp4", 0, 30.0, actions, ("right",))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["hand"] == "right" for row in rows))

    def test_rows_include_sample_index(self):
        actions = np.zeros((2, 3, 102), dtype=np.float32)
        rows = vitra_action_rows("video.mp4", 0, 30.0, actions, ("right",))
        self.assertEqual({row["sample"] for row in rows}, {0, 1})
        self.assertEqual(len(rows), 6)

    def test_rows_reject_wrong_shape(self):
        with self.assertRaises(ValueError):
            vitra_action_rows("video.mp4", 0, 30.0,
                              np.zeros((1, 3, 50), dtype=np.float32),
                              ("right",))

    def test_rows_reject_unknown_hand(self):
        with self.assertRaises(ValueError):
            vitra_action_rows("video.mp4", 0, 30.0,
                              np.zeros((3, 102), dtype=np.float32),
                              ("middle",))


class VitraStateTest(unittest.TestCase):
    def test_missing_state_is_zeros(self):
        state, fov, valid = load_hand_state(None, ("right",))
        self.assertEqual(state.shape, (122,))
        self.assertFalse(state.any())
        self.assertIsNone(fov)
        self.assertEqual(valid, set())

    def test_state_from_vitra_dict(self):
        hand_pose = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))
        data = {
            "fov_x": 55.0,
            "left": {
                0: {
                    "transl": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "global_orient": np.eye(3, dtype=np.float32),
                    "hand_pose": hand_pose,
                    "beta": np.arange(10, dtype=np.float32),
                }
            },
            "right": {
                0: {
                    "transl": np.array([-0.4, 0.5, 1.2], dtype=np.float32),
                    "global_orient": np.eye(3, dtype=np.float32),
                    "hand_pose": hand_pose,
                    "beta": np.full(10, 0.25, dtype=np.float32),
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npy"
            np.save(path, data, allow_pickle=True)
            state, fov, valid = load_hand_state(path, ("left", "right"))

        self.assertEqual(fov, 55.0)
        self.assertEqual(valid, {"left", "right"})
        np.testing.assert_allclose(state[0:3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state[51:61], np.arange(10))
        np.testing.assert_allclose(state[61:64], [-0.4, 0.5, 1.2])
        np.testing.assert_allclose(state[112:122], np.full(10, 0.25))
        np.testing.assert_allclose(state[64:112], 0.0)

    def test_state_raw_vector(self):
        vector = np.arange(122, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npy"
            np.save(path, vector)
            state, fov, valid = load_hand_state(path, ("right",))
        np.testing.assert_array_equal(state, vector)
        self.assertIsNone(fov)
        self.assertEqual(valid, {"right"})

    def test_state_invalid_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npy"
            np.save(path, np.zeros(5, dtype=np.float32))
            with self.assertRaises(SystemExit):
                load_hand_state(path, ("right",))

    def test_empty_hand_entry_is_treated_as_missing(self):
        # VITRA's examples/0002.npy heeft een lege 'left'-dict.
        hand_pose = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))
        data = {
            "fov_x": 66.9,
            "left": {},
            "right": {
                0: {
                    "transl": np.array([0.0, 0.1, 0.5], dtype=np.float32),
                    "global_orient": np.eye(3, dtype=np.float32),
                    "hand_pose": hand_pose,
                    "beta": np.zeros(10, dtype=np.float32),
                }
            },
        }
        state, fov, valid = state_from_hand_data(data, ("left", "right"))
        self.assertEqual(fov, 66.9)
        self.assertEqual(valid, {"right"})
        self.assertFalse(state[:51].any())
        np.testing.assert_allclose(state[61:64], [0.0, 0.1, 0.5])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npy"
            np.save(path, data, allow_pickle=True)
            loaded_state, loaded_fov, loaded_valid = load_hand_state(
                path, ("left", "right"))
        np.testing.assert_array_equal(loaded_state, state)
        self.assertEqual(loaded_fov, 66.9)
        self.assertEqual(loaded_valid, {"right"})

    def test_pad_state_right_only(self):
        state = np.arange(122, dtype=np.float32)
        padded, mask = pad_state_human(state, np.array([False, True]),
                                       102, 122, 212)
        self.assertEqual(tuple(padded.shape), (212,))
        self.assertEqual(tuple(mask.shape), (212,))
        self.assertEqual(int(mask.sum()), 51)
        self.assertFalse(padded[:51].any())
        np.testing.assert_allclose(padded[51:102], state[61:112])
        self.assertFalse(bool(mask[:51].any()))

    def test_pad_action_mask(self):
        action_mask = np.tile(np.array([[False, True]]), (4, 1))
        padded = pad_action_mask(action_mask, 102, 192)
        self.assertEqual(tuple(padded.shape), (4, 192))
        self.assertEqual(int(padded.sum()), 4 * 51)

    def test_parse_hands(self):
        self.assertEqual(parse_hands("right"), ("right",))
        self.assertEqual(parse_hands(["LEFT", "right", "left"]),
                         ("left", "right"))
        with self.assertRaises(SystemExit):
            parse_hands([])
        with self.assertRaises(SystemExit):
            parse_hands(["middle"])


class VitraFrameStatesTest(unittest.TestCase):
    def test_state_from_hand_data_matches_load(self):
        hand_pose = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))
        data = {
            "fov_x": 50.0,
            "right": {
                0: {
                    "transl": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "global_orient": np.eye(3, dtype=np.float32),
                    "hand_pose": hand_pose,
                    "beta": np.arange(10, dtype=np.float32),
                }
            },
        }
        state, fov, valid = state_from_hand_data(data, ("right",))
        self.assertEqual(fov, 50.0)
        self.assertEqual(valid, {"right"})
        np.testing.assert_allclose(state[61:64], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state[112:122], np.arange(10))

    def test_save_load_roundtrip(self):
        states = {
            0: (np.full(122, 1.0, np.float32), 55.0, {"right"}),
            30: (np.full(122, 2.0, np.float32), None, {"left", "right"}),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.npz"
            save_frame_states(path, states)
            loaded = load_frame_states(path)
        self.assertEqual(set(loaded), {0, 30})
        np.testing.assert_allclose(loaded[0][0], 1.0)
        self.assertEqual(loaded[0][1], 55.0)
        self.assertEqual(loaded[0][2], {"right"})
        self.assertIsNone(loaded[30][1])
        self.assertEqual(loaded[30][2], {"left", "right"})

    def test_empty_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.npz"
            save_frame_states(path, {})
            self.assertEqual(load_frame_states(path), {})

    def test_missing_states_file(self):
        with self.assertRaises(SystemExit):
            load_frame_states(Path("does-not-exist.npz"))

    def test_video_pass_with_fake_reconstructor(self):
        import cv2

        class FakeReconstructor:
            def reconstruct(self, frame):
                return (np.full(122, float(frame.mean()), np.float32),
                        60.0, {"right"})

            def close(self):
                raise AssertionError("injected reconstructor must not close")

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                (32, 24))
            for index in range(5):
                writer.write(np.full((24, 32, 3), index * 20, np.uint8))
            writer.release()
            path = reconstruct_video_states(
                {"video_path": str(video_path), "vitra_frame_stride": 2},
                Path(tmp) / "out", "gpu", reconstructor=FakeReconstructor())
            states = load_frame_states(path)
        self.assertEqual(set(states), {0, 2, 4})
        self.assertEqual(states[0][1], 60.0)
        self.assertEqual(states[0][2], {"right"})

    def test_resolve_defaults(self):
        paths = resolve_hand_recon_paths({"vitra_repo_path": "repo"})
        self.assertEqual(paths["hawor"],
                         Path("repo") / "weights/hawor/checkpoints/hawor.ckpt")
        self.assertEqual(paths["mano_dir"], Path("repo") / "weights/mano")

    def test_missing_mano_asset_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {
                "mano_dir": Path(tmp),
                "hawor": Path(tmp) / "hawor.ckpt",
                "detector": Path(tmp) / "detector.pt",
            }
            with self.assertRaises(SystemExit) as context:
                check_hand_recon_assets(paths)
        self.assertIn("MANO_RIGHT.pkl", str(context.exception))


class VitraAnnotationTest(unittest.TestCase):
    def test_annotation_preserves_frame(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        actions = np.zeros((1, 4, 102), dtype=np.float32)
        actions[0, :, 0] = 0.02
        actions[0, :, 51] = -0.01
        annotated = annotate_vitra(frame, actions, "Pick up the tool.",
                                   ("left", "right"))
        self.assertIsNot(annotated, frame)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, frame.dtype)
        self.assertFalse(np.array_equal(annotated, frame))

    def test_summary(self):
        actions = np.zeros((2, 102), dtype=np.float32)
        actions[0, 0:3] = [0.3, 0.4, 0.0]
        actions[1, 51:54] = [0.0, 0.0, 2.0]
        summary = action_summary(actions, ("left", "right"))
        self.assertAlmostEqual(summary["left"]["transl_norm"], 0.5)
        self.assertAlmostEqual(summary["right"]["path_length"], 2.0)

    def test_summary_direction(self):
        actions = np.zeros((3, 102), dtype=np.float32)
        actions[:, 51:54] = [0.1, 0.1, 0.0]      # rechts + omlaag
        actions[:, 0:3] = [-0.1, 0.0, 0.1]       # links + van je af
        summary = action_summary(actions, ("left", "right"))
        self.assertEqual(summary["right"]["direction"], "rechts+omlaag")
        self.assertAlmostEqual(summary["right"]["displacement"],
                               math.sqrt(0.18), places=5)
        self.assertEqual(summary["left"]["direction"], "links+van je af")

    def test_summary_quiet(self):
        actions = np.zeros((3, 102), dtype=np.float32)
        summary = action_summary(actions, ("right",))
        self.assertEqual(summary["right"]["direction"], "stil")

    def test_frame_counter_in_overlay(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        actions = np.zeros((4, 102), dtype=np.float32)
        with_counter = annotate_vitra(frame, actions, "test", ("right",),
                                      frame_idx=300, fps=30.0)
        self.assertFalse(np.array_equal(with_counter, frame))
        without = annotate_vitra(frame, actions, "test", ("right",))
        self.assertFalse(np.array_equal(with_counter, without))


class VitraPredictorGuardTest(unittest.TestCase):
    def test_cpu_is_rejected_before_loading_model(self):
        with self.assertRaises(SystemExit):
            VitraPredictor({}, "cpu")


class VitraRuntimeIsolationTest(unittest.TestCase):
    def test_child_process_is_not_relaunched(self):
        import run_vitra_test

        with mock.patch.dict(os.environ,
                             {run_vitra_test.ISOLATED_ENV: "1"}):
            run_vitra_test.ensure_pinned_transformers({})

    def test_transformers_4x_needs_no_isolation(self):
        import run_vitra_test

        with mock.patch.object(run_vitra_test, "_transformers_major",
                               return_value=4), \
                mock.patch("subprocess.call") as call, \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(run_vitra_test.ISOLATED_ENV, None)
            run_vitra_test.ensure_pinned_transformers({})
        call.assert_not_called()

    def test_missing_runtime_dir_explains_install(self):
        import run_vitra_test

        with mock.patch.object(run_vitra_test, "_transformers_major",
                               return_value=5), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(run_vitra_test.ISOLATED_ENV, None)
            with self.assertRaises(SystemExit) as context:
                run_vitra_test.ensure_pinned_transformers(
                    {"vitra_runtime_path": "does-not-exist"})
        self.assertIn("requirements-vitra.txt", str(context.exception))

    def test_present_runtime_relaunches_with_pythonpath(self):
        import run_vitra_test

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run_vitra_test, "_transformers_major",
                                  return_value=5), \
                mock.patch("subprocess.call", return_value=0) as call, \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(run_vitra_test.ISOLATED_ENV, None)
            with self.assertRaises(SystemExit) as context:
                run_vitra_test.ensure_pinned_transformers(
                    {"vitra_runtime_path": tmp})
            env = call.call_args.kwargs["env"]
        self.assertEqual(context.exception.code, 0)
        self.assertEqual(env[run_vitra_test.ISOLATED_ENV], "1")
        self.assertTrue(env["PYTHONPATH"].startswith(str(Path(tmp).resolve())))


class VitraPathResolutionTest(unittest.TestCase):
    def test_model_path_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "weights.pt"
            checkpoint.write_bytes(b"")
            self.assertEqual(resolve_model_path(tmp), str(checkpoint))

    def test_statistics_path_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp) / "statistics" / "dataset_statistics.json"
            stats.parent.mkdir()
            stats.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_statistics_path(tmp), str(stats))


if __name__ == "__main__":
    unittest.main()
