"""Tests voor taakbewust benchmarken, VLM-acties en temporele evaluatie."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplo.action_eval import (
    ActionInterval,
    evaluate_action_rows,
    interval_iou,
    load_action_ground_truth,
)
from duplo.actions import (
    ActionPrediction,
    build_vlm_prompt,
    normalize_action_model,
    parse_vlm_action_response,
)
from duplo.benchmark import (
    TASK_3D_HAND_PREDICTION,
    TASK_ACTION_CLASSIFICATION,
    TASK_HAND_DETECTION,
    TASK_LABELS,
    TASKS,
    InferenceTimer,
    system_info,
    write_run_summary,
)
from duplo.schema import ACTION_COLUMNS, action_row


class ActionEvaluationTest(unittest.TestCase):
    def test_interval_iou(self):
        self.assertEqual(interval_iou(0, 10, 10, 20), 0.0)
        self.assertAlmostEqual(interval_iou(0, 10, 5, 15), 1 / 3)
        self.assertEqual(interval_iou(0, 10, 2, 8), 0.6)

    def test_classification_and_temporal_metrics(self):
        ground_truth = [
            ActionInterval("unscrew", 0.0, 3.0),
            ActionInterval("idle", 3.0, 6.0),
        ]
        rows = [
            action_row("video.mp4", 0, 90, 30.0, "unscrew", 0.9,
                       [("unscrew", 0.9)], start_s=0.1, end_s=2.9),
            action_row("video.mp4", 90, 180, 30.0, "idle", 0.8,
                       [("idle", 0.8)], start_s=3.1, end_s=5.9),
        ]
        metrics = evaluate_action_rows(rows, ground_truth, fps=30.0)
        self.assertEqual(metrics["classification_accuracy"], 1.0)
        self.assertEqual(metrics["temporal_matches"], 2)
        self.assertGreater(metrics["temporal_iou_mean"], 0.8)
        self.assertEqual(metrics["temporal_recall_0.5"], 1.0)
        self.assertLess(metrics["temporal_start_mae_s"], 0.2)

    def test_wrong_label_and_missing_times(self):
        ground_truth = [ActionInterval("unscrew", 0.0, 3.0)]
        rows = [action_row("video.mp4", 0, 90, 30.0, "idle", None)]
        metrics = evaluate_action_rows(rows, ground_truth, fps=30.0)
        self.assertEqual(metrics["classification_accuracy"], 0.0)
        self.assertEqual(metrics["temporal_coverage"], 0.0)
        self.assertIsNone(metrics["temporal_start_mae_s"])

    def test_load_ground_truth_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "gt.csv"
            csv_path.write_text(
                "video,action,start_s,end_s\n"
                "clip.mp4,unscrew,0,3\nclip.mp4,idle,3,6\n",
                encoding="utf-8")
            intervals = load_action_ground_truth(csv_path, video="clip.mp4")
            self.assertEqual([item.action for item in intervals],
                             ["unscrew", "idle"])
            json_path = root / "gt.json"
            json_path.write_text(json.dumps([
                {"klasse": "unscrew", "start": 1, "end": 2}]), encoding="utf-8")
            json_intervals = load_action_ground_truth(json_path)
            self.assertEqual(json_intervals[0].action, "unscrew")
            self.assertEqual(json_intervals[0].start_s, 1.0)


class VlmResponseTest(unittest.TestCase):
    def setUp(self):
        self.classes = ["unscrewing a screw", "idle"]

    def test_json_response(self):
        prediction = parse_vlm_action_response(
            '```json\n{"action": "idle", "start_s": 1.5, "end_s": 3.0, '
            '"confidence": 0.82}\n```',
            self.classes, clip_start_s=10.0, clip_end_s=20.0)
        self.assertEqual(prediction.label, "idle")
        self.assertAlmostEqual(prediction.confidence, 0.82)
        self.assertAlmostEqual(prediction.start_s, 11.5)
        self.assertAlmostEqual(prediction.end_s, 13.0)

    def test_free_text_response(self):
        prediction = parse_vlm_action_response(
            "The action is unscrewing a screw from 2.0 to 4.0 seconds.",
            self.classes, clip_start_s=5.0, clip_end_s=15.0)
        self.assertEqual(prediction.label, "unscrewing a screw")
        self.assertAlmostEqual(prediction.start_s, 7.0)
        self.assertAlmostEqual(prediction.end_s, 9.0)

    def test_unknown_and_invalid_times(self):
        prediction = parse_vlm_action_response(
            "I am not sure.", self.classes, 0.0, 10.0)
        self.assertEqual(prediction.label, "unknown")
        self.assertIsNone(prediction.confidence)
        self.assertIsNone(prediction.start_s)

    def test_prompt_contains_classes_and_duration(self):
        prompt = build_vlm_prompt(self.classes, 12.0, 15.0)
        self.assertIn("unscrewing a screw", prompt)
        self.assertIn("3.00 seconds", prompt)
        self.assertIn('"action"', prompt)

    def test_prediction_accepts_temporal_fields(self):
        prediction = ActionPrediction(
            "idle", None, {}, start_s=1.0, end_s=2.0, raw_text="x")
        self.assertEqual(prediction.top_k(), [])
        self.assertEqual(prediction.start_s, 1.0)


class BenchmarkSummaryTest(unittest.TestCase):
    def test_task_catalogue(self):
        self.assertIn(TASK_HAND_DETECTION, TASKS)
        self.assertIn(TASK_3D_HAND_PREDICTION, TASKS)
        self.assertIn(TASK_ACTION_CLASSIFICATION, TASKS)
        self.assertEqual(TASK_LABELS[TASK_ACTION_CLASSIFICATION],
                         "Actieclassificatie")

    def test_system_info_has_environment_keys(self):
        info = system_info()
        for key in ("platform", "python", "torch", "cuda", "gpus"):
            self.assertIn(key, info)

    def test_write_summary_adds_task_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            timer = InferenceTimer()
            for _ in range(2):
                with timer:
                    pass
            summary = write_run_summary(
                out_dir, "handpose", model="mediapipe", device="cpu",
                video="v.mp4", frames_read=10, frames_processed=2,
                elapsed_s=1.0, rows=[{"hand": 0}], timer=timer,
                csv_file="hand_landmarks.csv", class_field="hand",
                task=TASK_HAND_DETECTION)
            saved = json.loads(
                (out_dir / "handpose_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["task"], TASK_HAND_DETECTION)
        self.assertEqual(saved["benchmark_version"], 2)
        self.assertEqual(saved["task_label"], "Handdetectie (landmarks)")
        self.assertIn("environment", saved)
        self.assertIn("inference_ms_min", saved)
        self.assertIn("inference_ms_max", saved)


class CompareTasksTest(unittest.TestCase):
    def test_handpose_and_vitra_summaries(self):
        from compare_models import summarize_handpose, summarize_vitra

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hand_dir = root / "handpose"
            hand_dir.mkdir()
            pd.DataFrame([
                {"video": "v", "frame": 0, "hand": 0, "landmark": 0,
                 "x": 0, "y": 0, "z": 0, "zichtbaar": 1.0},
                {"video": "v", "frame": 0, "hand": 0, "landmark": 1,
                 "x": 0, "y": 0, "z": 0, "zichtbaar": 1.0},
            ]).to_csv(hand_dir / "hand_landmarks.csv", index=False)
            (hand_dir / "handpose_summary.json").write_text(json.dumps({
                "model": "mediapipe", "frames_processed": 1,
                "inference_ms_median": 2.0, "pipeline_fps": 10.0,
            }), encoding="utf-8")
            hand = summarize_handpose(hand_dir / "hand_landmarks.csv")

            vitra_dir = root / "vitra"
            vitra_dir.mkdir()
            pd.DataFrame([
                {"video": "v", "frame": 0, "timestamp": 0.0, "sample": 0,
                 "hand": "right", "step": 0, "transl_x": 0, "transl_y": 0,
                 "transl_z": 0, "rot_x": 0, "rot_y": 0, "rot_z": 0,
                 "transl_norm": 0, "pose_mean": 0},
                {"video": "v", "frame": 0, "timestamp": 0.0, "sample": 0,
                 "hand": "right", "step": 1, "transl_x": 0, "transl_y": 0,
                 "transl_z": 0, "rot_x": 0, "rot_y": 0, "rot_z": 0,
                 "transl_norm": 0, "pose_mean": 0},
            ]).to_csv(vitra_dir / "vitra_actions.csv", index=False)
            (vitra_dir / "vitra_summary.json").write_text(json.dumps({
                "model": "VITRA", "frames_processed": 1,
                "inference_ms_median": 20.0, "pipeline_fps": 5.0,
            }), encoding="utf-8")
            vitra = summarize_vitra(vitra_dir / "vitra_actions.csv")

        self.assertEqual(hand["task"], TASK_HAND_DETECTION)
        self.assertEqual(hand["landmark_rows"], 2)
        self.assertEqual(hand["frames_with_hands"], 1)
        self.assertEqual(vitra["task"], TASK_3D_HAND_PREDICTION)
        self.assertEqual(vitra["action_rows"], 2)
        self.assertEqual(vitra["chunk_steps"], 2)
        self.assertEqual(vitra["hands"], "right")

    def test_action_summary_uses_ground_truth(self):
        from compare_models import summarize_action_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step = root / "qwen3vl"
            step.mkdir()
            rows = [
                action_row("v.mp4", 0, 90, 30.0, "unscrew", None,
                           start_s=0.0, end_s=3.0),
            ]
            pd.DataFrame(rows, columns=ACTION_COLUMNS).to_csv(
                step / "qwen3vl_actions.csv", index=False)
            (step / "qwen3vl_summary.json").write_text(json.dumps({
                "model": "qwen3vl", "clips_processed": 1, "fps": 30.0,
                "temporal": True,
            }), encoding="utf-8")
            summary = summarize_action_run(
                step / "qwen3vl_actions.csv",
                [ActionInterval("unscrew", 0.0, 3.0)])
        self.assertEqual(summary["task"], TASK_ACTION_CLASSIFICATION)
        self.assertTrue(summary["temporal"])
        self.assertEqual(summary["classification_accuracy"], 1.0)
        self.assertEqual(summary["temporal_iou_mean"], 1.0)

    def test_action_only_benchmark_main(self):
        import compare_models

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs"
            step = output / "smolvlm2"
            step.mkdir(parents=True)
            pd.DataFrame([
                action_row("v.mp4", 0, 90, 30.0, "idle", 0.5),
            ], columns=ACTION_COLUMNS).to_csv(
                step / "smolvlm2_actions.csv", index=False)
            (step / "smolvlm2_summary.json").write_text(json.dumps({
                "model": "smolvlm2", "clips_processed": 1,
                "task": "action_classification",
            }), encoding="utf-8")
            cfg_path = root / "cfg.yaml"
            cfg_path.write_text(
                f"video_path: v.mp4\noutput_dir: {output}\n"
                "tool_classes: []\naction_classes: [idle]\n",
                encoding="utf-8")
            compare_models.main(str(cfg_path), task="action_classification")
            report = (output / "comparison" / "model_comparison.md").read_text(
                encoding="utf-8")
            overview = json.loads((output / "comparison" /
                                   "benchmark_overview.json").read_text(
                                       encoding="utf-8"))
        self.assertIn("Actieclassificatie", report)
        self.assertNotIn("Objectdetectie / segmentatie", report)
        self.assertEqual(
            overview["tasks"][TASK_ACTION_CLASSIFICATION]["count"], 1)


class LlavaActionRuntimeTest(unittest.TestCase):
    def test_child_process_is_not_relaunched(self):
        import os

        import run_action_test

        with mock.patch.dict(
                os.environ, {run_action_test.LLAVACTION_ISOLATED_ENV: "1"}):
            run_action_test.ensure_llavaction_runtime({})

    def test_transformers5_needs_runtime(self):
        import run_action_test

        with mock.patch.object(
                run_action_test, "_transformers_major", return_value=5), \
                self.assertRaises(SystemExit) as context:
            run_action_test.ensure_llavaction_runtime(
                {"llavaction_runtime_path": "does-not-exist"})
        self.assertIn("requirements-llavaction.txt", str(context.exception))

    def test_present_runtime_relaunches_with_pythonpath(self):
        import os

        import run_action_test

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(
                    run_action_test, "_transformers_major", return_value=5), \
                mock.patch("subprocess.call", return_value=0) as call, \
                mock.patch.dict(os.environ, {}, clear=False):
            (Path(tmp) / "transformers").mkdir()
            (Path(tmp) / "transformers" / "__init__.py").write_text(
                "", encoding="utf-8")
            os.environ.pop(run_action_test.LLAVACTION_ISOLATED_ENV, None)
            with self.assertRaises(SystemExit) as context:
                run_action_test.ensure_llavaction_runtime(
                    {"llavaction_runtime_path": tmp})
            env = call.call_args.kwargs["env"]
        self.assertEqual(context.exception.code, 0)
        self.assertEqual(env[run_action_test.LLAVACTION_ISOLATED_ENV], "1")
        self.assertTrue(env["PYTHONPATH"].startswith(str(Path(tmp).resolve())))


class LlavaActionCompatibilityTest(unittest.TestCase):
    def test_transformers4_helpers_are_added(self):
        from transformers import modeling_utils

        from duplo.vlm_actions import _patch_llavaction_transformers

        _patch_llavaction_transformers()
        for name in ("apply_chunking_to_forward",
                     "find_pruneable_heads_and_indices",
                     "prune_linear_layer"):
            self.assertTrue(hasattr(modeling_utils, name))


class VlmSelectionTest(unittest.TestCase):
    def test_model_aliases(self):
        self.assertEqual(normalize_action_model("Qwen3-VL-8B-Instruct"),
                         "qwen3vl")
        self.assertEqual(
            normalize_action_model("MLAdaptiveIntelligence/LLaVAction-7B"),
            "llavaction")
        self.assertEqual(
            normalize_action_model("HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
            "smolvlm2")

    def test_factory_routes_vlm_without_loading(self):
        from unittest import mock

        from duplo.actions import create_action_classifier

        with mock.patch("duplo.vlm_actions.create_vlm_classifier",
                        return_value="classifier") as factory:
            result = create_action_classifier({}, "cpu", "qwen3vl")
        self.assertEqual(result, "classifier")
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args[2], "qwen3vl")


if __name__ == "__main__":
    unittest.main()
