"""Tests voor de geselecteerde-modelbenchmark en multi-video-gemiddelden."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmark_compare as bench


class ModelSelectionTest(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(bench.PRESETS["action"],
                         ("actionclip", "internvideo2", "xclip"))
        self.assertEqual(bench.PRESETS["detection"],
                         ("grounding_dino", "sam31", "yolo_world"))

    def test_select_preset_and_explicit_models(self):
        self.assertEqual(bench.select_models("action"), [
            "actionclip", "internvideo2", "xclip"])
        self.assertEqual(
            bench.select_models(None, ["x-clip, actionclip", "xclip"]),
            ["xclip", "actionclip"])
        self.assertEqual(
            bench.select_models(None, single="sam3.1"), ["sam31"])
        self.assertEqual(
            bench.select_models(None, ["grounding dino", "sam 3.1"]),
            ["grounding_dino", "sam31"])

    def test_unknown_model_and_preset(self):
        with self.assertRaises(SystemExit):
            bench.select_models(None, ["does-not-exist"])
        with self.assertRaises(SystemExit):
            bench.select_models("does-not-exist")

    def test_video_slug_is_prefixed_and_safe(self):
        self.assertEqual(bench.video_slug(Path("my video.mp4"), 2),
                         "02_my_video")
        self.assertNotEqual(bench.video_slug(Path("a/clip.mp4"), 1),
                            bench.video_slug(Path("b/clip.mp4"), 2))


class ConfigPreparationTest(unittest.TestCase):
    def test_config_sets_video_output_and_disables_debug_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"video_path": "old.mp4", "output_dir": "old",
                   "save_yolo_video": True, "screenshot_every": 150,
                   "device": "gpu"}
            result = bench.prepare_config(
                cfg, root / "clip.mp4", root / "out",
                root / "cfg.yaml", "cpu", save_videos=False)
            saved = bench.yaml.safe_load(
                (root / "cfg.yaml").read_text(encoding="utf-8"))
        self.assertEqual(result["video_path"], str(root / "clip.mp4"))
        self.assertEqual(result["output_dir"], str(root / "out"))
        self.assertFalse(result["save_yolo_video"])
        self.assertEqual(result["screenshot_every"], 0)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(saved["screenshot_every"], 0)


class AverageAndReportTest(unittest.TestCase):
    def _records(self):
        return [
            bench.RunRecord(
                "xclip", "a.mp4", "action", bench.TASK_ACTION_CLASSIFICATION,
                "ok", metrics={
                    "clips": 10, "classification_accuracy": 0.5,
                    "pipeline_fps": 2.0, "top_label": "idle",
                    "evaluation": {"nested": True},
                }),
            bench.RunRecord(
                "xclip", "b.mp4", "action", bench.TASK_ACTION_CLASSIFICATION,
                "ok", metrics={
                    "clips": 20, "classification_accuracy": 1.0,
                    "pipeline_fps": 4.0, "top_label": "unscrew",
                }),
            bench.RunRecord(
                "xclip", "c.mp4", "action", bench.TASK_ACTION_CLASSIFICATION,
                "failed", error="model error"),
        ]

    def test_average_records_skips_failures(self):
        averages = bench.average_records(self._records())
        xclip = averages[0]
        self.assertEqual(xclip["videos"], 3)
        self.assertEqual(xclip["successful"], 2)
        self.assertEqual(xclip["failed"], 1)
        self.assertEqual(xclip["metrics"]["clips"], 15.0)
        self.assertEqual(xclip["metrics"]["classification_accuracy"], 0.75)
        self.assertEqual(xclip["metrics"]["pipeline_fps"], 3.0)
        self.assertNotIn("evaluation", xclip["metrics"])

    def test_write_reports_creates_all_files(self):
        records = self._records()
        averages = bench.average_records(records)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.write_reports(
                "test", ["xclip"], [Path("a.mp4"), Path("b.mp4")],
                records, averages, "cpu", root)
            report = (root / "benchmark_report.md").read_text(encoding="utf-8")
            metrics = (root / "benchmark_metrics.csv").read_text(
                encoding="utf-8")
            overview = (root / "benchmark_overview.json").read_text(
                encoding="utf-8")
        self.assertIn("Gemiddelde over video's", report)
        self.assertIn("Mislukte runs", report)
        self.assertIn("xclip", metrics)
        self.assertIn("averages", overview)


if __name__ == "__main__":
    unittest.main()
