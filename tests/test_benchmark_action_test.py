"""Tests voor taakbewust benchmarken en temporele actie-evaluatie."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplo.action_eval import (
    ActionInterval,
    evaluate_action_rows,
    interval_iou,
    load_action_ground_truth,
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
            step = root / "xclip"
            step.mkdir()
            rows = [
                action_row("v.mp4", 0, 90, 30.0, "unscrew", None,
                           start_s=0.0, end_s=3.0),
            ]
            pd.DataFrame(rows, columns=ACTION_COLUMNS).to_csv(
                step / "xclip_actions.csv", index=False)
            (step / "xclip_summary.json").write_text(json.dumps({
                "model": "xclip", "clips_processed": 1, "fps": 30.0,
                "temporal": True,
            }), encoding="utf-8")
            summary = summarize_action_run(
                step / "xclip_actions.csv",
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
            step = output / "xclip"
            step.mkdir(parents=True)
            pd.DataFrame([
                action_row("v.mp4", 0, 90, 30.0, "idle", 0.5),
            ], columns=ACTION_COLUMNS).to_csv(
                step / "xclip_actions.csv", index=False)
            (step / "xclip_summary.json").write_text(json.dumps({
                "model": "xclip", "clips_processed": 1,
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


if __name__ == "__main__":
    unittest.main()
