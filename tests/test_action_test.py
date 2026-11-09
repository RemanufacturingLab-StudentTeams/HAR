"""Tests voor actieclassificatie en de portable configuratie (zonder model/GPU)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplo.actions import (
    ActionPrediction,
    PromptSet,
    VLM_ACTION_KINDS,
    VideoClipReader,
    annotate_action,
    build_prompt_set,
    build_vlm_prompt,
    clip_windows,
    normalize_action_model,
    parse_action_classes,
    parse_vlm_action_response,
    prediction_from_logits,
    softmax,
)
from duplo.config import (
    DEVICES,
    resolve_device,
    resolve_video_path,
)
from duplo.schema import ACTION_COLUMNS, action_row


class ActionSchemaTest(unittest.TestCase):
    def test_action_row(self):
        row = action_row(
            "video.mp4", 30, 60, 30.0, "unscrewing a screw", 0.81234,
            [("unscrewing a screw", 0.8123), ("idle", 0.1)])

        self.assertEqual(tuple(row), tuple(ACTION_COLUMNS))
        self.assertEqual(row["frame"], 30)
        self.assertEqual(row["clip_start"], 30)
        self.assertEqual(row["clip_end"], 60)
        self.assertEqual(row["timestamp"], 1.0)
        self.assertEqual(row["klasse"], "unscrewing a screw")
        self.assertEqual(row["confidence"], 0.8123)
        self.assertIn("unscrewing a screw=0.8123", row["top_k"])
        self.assertIn("idle=0.1000", row["top_k"])

    def test_action_row_without_fps(self):
        row = action_row("video.mp4", 4, 8, 0.0, "idle", 0.5)
        self.assertEqual(row["timestamp"], 0.0)
        self.assertEqual(row["top_k"], "")


class ActionPromptTest(unittest.TestCase):
    def test_parse_classes(self):
        self.assertEqual(
            parse_action_classes(["unscrew", " idle ", "UNSCREW", ""]),
            ["unscrew", "idle"])
        self.assertEqual(parse_action_classes("a, b,c"), ["a", "b", "c"])
        with self.assertRaises(SystemExit):
            parse_action_classes([])
        with self.assertRaises(SystemExit):
            parse_action_classes(42)

    def test_prompt_set_from_templates(self):
        prompt_set = build_prompt_set(
            ["unscrew", "idle"], templates=["a video of {}", "the person is {}"])

        self.assertEqual(prompt_set.classes, ("unscrew", "idle"))
        self.assertEqual(prompt_set.prompts, (
            "a video of unscrew", "the person is unscrew",
            "a video of idle", "the person is idle"))
        self.assertEqual(prompt_set.groups[0],
                         ("a video of unscrew", "the person is unscrew"))

    def test_prompt_set_rejects_template_without_placeholder(self):
        with self.assertRaises(SystemExit):
            build_prompt_set(["idle"], templates=["an action"])

    def test_custom_prompts_override_templates(self):
        prompt_set = build_prompt_set(
            ["unscrew", "idle"],
            templates=["a video of {}"],
            prompts={"unscrew": ["turning a screw"],
                     "idle": "doing nothing"})
        self.assertEqual(prompt_set.groups, (
            ("turning a screw",), ("doing nothing",)))

    def test_prompt_set_defaults_to_label(self):
        prompt_set = build_prompt_set(["idle"])
        self.assertEqual(prompt_set.prompts, ("idle",))

    def test_aggregate_logits_averages_per_class_before_softmax(self):
        prompt_set = PromptSet(
            classes=("a", "b"),
            groups=(("a1", "a2"), ("b1", "b2")))
        class_logits = prompt_set.aggregate_logits([1.0, 3.0, 5.0, 7.0])
        np.testing.assert_allclose(class_logits, [2.0, 6.0])
        probabilities = softmax(class_logits)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreater(probabilities[1], probabilities[0])

    def test_prediction_from_logits(self):
        prompt_set = build_prompt_set(["unscrew", "idle"])
        prediction = prediction_from_logits(prompt_set, [0.0, 2.0], top_k=2)
        self.assertEqual(prediction.label, "idle")
        self.assertGreater(prediction.confidence, 0.5)
        self.assertEqual(prediction.top_k(1)[0][0], "idle")
        self.assertEqual(set(prediction.scores), {"unscrew", "idle"})

    def test_prediction_rejects_wrong_score_count(self):
        prompt_set = build_prompt_set(["a", "b"])
        with self.assertRaises(ValueError):
            prediction_from_logits(prompt_set, [1.0])


class VlmActionPromptTest(unittest.TestCase):
    def test_prompt_contains_classes_and_relative_times(self):
        prompt = build_vlm_prompt(["unscrew", "idle"], 12.5, 15.5)
        self.assertIn("- unscrew", prompt)
        self.assertIn("- idle", prompt)
        self.assertIn("0 to 3.00", prompt)
        self.assertIn('"action"', prompt)

    def test_parse_json_with_relative_times(self):
        prediction = parse_vlm_action_response(
            '{"action": "unscrew", "start_s": 0.25, "end_s": 2.5, '
            '"confidence": 0.8}',
            ["unscrew", "idle"], 10.0, 13.0)
        self.assertEqual(prediction.label, "unscrew")
        self.assertAlmostEqual(prediction.confidence, 0.8)
        self.assertAlmostEqual(prediction.start_s, 10.25)
        self.assertAlmostEqual(prediction.end_s, 12.5)
        self.assertEqual(prediction.scores, {"unscrew": 0.8})
        self.assertIn("unscrew", prediction.raw_text)

    def test_parse_free_text_and_percentage_confidence(self):
        prediction = parse_vlm_action_response(
            "The action is idle. Confidence: 75%. Times: 0.5 to 2.0",
            ["unscrew", "idle"], 30.0, 33.0)
        self.assertEqual(prediction.label, "idle")
        self.assertAlmostEqual(prediction.confidence, 0.75)
        self.assertAlmostEqual(prediction.start_s, 30.5)
        self.assertAlmostEqual(prediction.end_s, 32.0)

    def test_parse_unknown_and_invalid_times(self):
        prediction = parse_vlm_action_response(
            '{"action": "dancing", "start_s": 4.0, "end_s": 1.0}',
            ["unscrew", "idle"], 5.0, 8.0)
        self.assertEqual(prediction.label, "unknown")
        self.assertIsNone(prediction.confidence)
        self.assertAlmostEqual(prediction.start_s, 6.0)
        self.assertAlmostEqual(prediction.end_s, 8.0)
        self.assertEqual(prediction.scores, {})

    def test_video_metadata_covers_clip_duration(self):
        from duplo.vlm_actions import _video_metadata

        metadata = _video_metadata([object()] * 8, 10.0, 13.0)
        self.assertEqual(metadata.total_num_frames, 8)
        self.assertAlmostEqual(metadata.timestamps[0], 0.0)
        self.assertAlmostEqual(metadata.timestamps[-1], 3.0, places=6)


class ClipWindowTest(unittest.TestCase):
    def test_windows_cover_video_end(self):
        windows = clip_windows(total_frames=100, clip_frames=8,
                               clip_span=30, clip_stride=30)
        self.assertEqual([window.start for window in windows],
                         [0, 30, 60, 70])
        self.assertEqual(windows[-1].end, 100)
        for window in windows:
            self.assertEqual(len(window.indices), 8)
            self.assertEqual(tuple(sorted(window.indices)), window.indices)
            self.assertGreaterEqual(min(window.indices), window.start)
            self.assertLess(max(window.indices), window.end)

    def test_short_video_is_clipped(self):
        windows = clip_windows(total_frames=5, clip_frames=8,
                               clip_span=30, clip_stride=30)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start, 0)
        self.assertEqual(windows[0].end, 5)
        self.assertEqual(len(windows[0].indices), 8)
        self.assertTrue(all(0 <= index < 5 for index in windows[0].indices))

    def test_invalid_parameters(self):
        self.assertEqual(clip_windows(0, 8, 30, 30), [])
        self.assertEqual(clip_windows(100, 0, 30, 30), [])
        self.assertEqual(clip_windows(100, 8, 30, 0), [])


class ActionAnnotationTest(unittest.TestCase):
    def test_annotation_changes_frame_but_keeps_shape(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        prediction = ActionPrediction(
            label="idle", confidence=0.8,
            scores={"idle": 0.8, "unscrew": 0.2})
        annotated = annotate_action(frame, prediction, 30, 30.0, 30, 60)
        self.assertIsNot(annotated, frame)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, frame.dtype)
        self.assertFalse(np.array_equal(annotated, frame))


class VideoClipReaderTest(unittest.TestCase):
    def test_reads_requested_frames_in_order(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
            for index in range(5):
                writer.write(np.full((24, 32, 3), index * 40, np.uint8))
            writer.release()

            with VideoClipReader(path) as reader:
                self.assertEqual(reader.total_frames, 5)
                frames = reader.read([0, 2, 4, 4])

        self.assertEqual([frame.shape for frame in frames],
                         [(24, 32, 3)] * 4)
        means = [float(frame.mean()) for frame in frames]
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])
        self.assertTrue(np.array_equal(frames[2], frames[3]))

    def test_missing_video_has_clear_error(self):
        with self.assertRaises(SystemExit):
            VideoClipReader("does-not-exist.mp4")


class ActionModelSelectionTest(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_action_model("X-CLIP"), "xclip")
        self.assertEqual(normalize_action_model("intern_video2"),
                         "internvideo2")
        self.assertEqual(normalize_action_model("ActionCLIP"), "actionclip")
        self.assertEqual(normalize_action_model("SmolVLM2"), "smolvlm2")
        self.assertEqual(normalize_action_model("InternVideo2.5"),
                         "internvideo25")
        self.assertEqual(normalize_action_model("internvideo3-8B"),
                         "internvideo3")
        self.assertEqual(normalize_action_model("intervideo3-8b"),
                         "internvideo3")

    def test_generative_kinds_are_registered(self):
        self.assertEqual(VLM_ACTION_KINDS,
                         ("smolvlm2", "internvideo25", "internvideo3"))

    def test_unknown_model(self):
        with self.assertRaises(SystemExit):
            normalize_action_model("slowfast")

    def test_create_unknown_classifier_does_not_load_model(self):
        from duplo.actions import create_action_classifier

        with self.assertRaises(SystemExit):
            create_action_classifier({}, "cpu", "slowfast")

    def test_vlm_factory_dispatches_without_loading_model(self):
        from duplo.actions import create_action_classifier

        with mock.patch(
                "duplo.vlm_actions.create_vlm_classifier",
                return_value="fake-vlm") as create:
            classifier = create_action_classifier({}, "cpu", "internvideo3")
        self.assertEqual(classifier, "fake-vlm")
        create.assert_called_once()
        self.assertEqual(create.call_args.args[2], "internvideo3")


class ActionClipPreprocessingTest(unittest.TestCase):
    def test_preprocess_shape_and_range(self):
        from duplo.actionclip import preprocess_actionclip_frames

        frames = [np.full((120, 200, 3), 128, np.uint8) for _ in range(2)]
        tensor = preprocess_actionclip_frames(frames)
        self.assertEqual(tuple(tensor.shape), (1, 2, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch_dtype())

    def test_infer_checkpoint_metadata(self):
        import torch

        from duplo.actionclip import _infer_adapter, _infer_clip_arch

        state = {
            "clip.visual.conv1.weight": torch.zeros(768, 3, 32, 32),
            "adapter.frame_position_embeddings.weight": torch.zeros(8, 512),
            "adapter.transformer.resblocks.0.attn.in_proj_weight": torch.zeros(1),
            "adapter.transformer.resblocks.5.attn.in_proj_weight": torch.zeros(1),
        }
        self.assertEqual(_infer_clip_arch(state), "ViT-B/32")
        self.assertEqual(_infer_adapter(state), (8, 6))


def torch_dtype():
    import torch

    return torch.float32


class PortableConfigTest(unittest.TestCase):
    def test_devices_include_auto(self):
        self.assertIn("auto", DEVICES)

    def test_auto_uses_cuda_when_available(self):
        with mock.patch("duplo.config.cuda_available", return_value=True):
            self.assertEqual(resolve_device(None, {"device": "auto"}), "gpu")

    def test_auto_falls_back_to_cpu(self):
        with mock.patch("duplo.config.cuda_available", return_value=False):
            self.assertEqual(resolve_device(None, {"device": "auto"}), "cpu")

    def test_cli_device_wins(self):
        self.assertEqual(resolve_device("cpu", {"device": "gpu"}), "cpu")

    def test_invalid_device(self):
        with self.assertRaises(SystemExit):
            resolve_device(None, {"device": "tpu"})

    def test_video_falls_back_to_single_file_in_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            video = root / "data" / "only.mp4"
            video.write_bytes(b"not-a-real-video")
            previous = os.getcwd()
            os.chdir(root)
            try:
                cfg = {"video_path": "data/missing.mp4"}
                resolved = resolve_video_path(cfg)
            finally:
                os.chdir(previous)
        self.assertEqual(resolved.name, "only.mp4")
        self.assertEqual(Path(cfg["video_path"]).name, "only.mp4")

    def test_video_missing_without_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                with self.assertRaises(SystemExit):
                    resolve_video_path({"video_path": "data/missing.mp4"})
            finally:
                os.chdir(previous)


class RemoteCodeCompatibilityTest(unittest.TestCase):
    def test_transformers5_tied_weights_compatibility(self):
        try:
            from transformers import PreTrainedModel
        except ImportError:
            self.skipTest("transformers is niet geinstalleerd")
        from duplo.actions import _patch_transformers_for_remote_code

        _patch_transformers_for_remote_code()
        self.assertTrue(hasattr(PreTrainedModel, "all_tied_weights_keys"))

        class DummyModel(PreTrainedModel):
            pass

        model = DummyModel.__new__(DummyModel)
        model.all_tied_weights_keys = {"target": "source"}
        self.assertEqual(model.all_tied_weights_keys,
                         {"target": "source"})


class InternVideo25RuntimeIsolationTest(unittest.TestCase):
    def test_child_process_is_not_relaunched(self):
        import run_action_test

        with mock.patch.dict(
                "os.environ",
                {run_action_test.INTERNVIDEO25_ISOLATED_ENV: "1"}):
            run_action_test.ensure_internvideo25_runtime({})

    def test_missing_runtime_explains_install(self):
        import run_action_test

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_action_test, "_transformers_major",
                                   return_value=5):
                with self.assertRaises(SystemExit):
                    run_action_test.ensure_internvideo25_runtime({
                        "internvideo25_runtime_path": tmp})

    def test_present_runtime_relaunches_with_pythonpath(self):
        import run_action_test

        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            (runtime / "transformers").mkdir()
            (runtime / "transformers" / "__init__.py").write_text("")
            with mock.patch.object(run_action_test, "_transformers_major",
                                   return_value=5), \
                    mock.patch.object(run_action_test.subprocess, "call",
                                      return_value=0) as call:
                with self.assertRaises(SystemExit):
                    run_action_test.ensure_internvideo25_runtime({
                        "internvideo25_runtime_path": tmp})
            env = call.call_args.kwargs["env"]
        self.assertIn(str(runtime.resolve()), env["PYTHONPATH"])
        self.assertEqual(env[run_action_test.INTERNVIDEO25_ISOLATED_ENV], "1")


class ActionPipelineTest(unittest.TestCase):
    def test_pipeline_writes_csv_and_summary_with_fake_model(self):
        import cv2
        import yaml

        import run_action_test

        class FakeClassifier:
            model_id = "fake-action"
            classes = ("idle", "unscrew")
            prompt_set = PromptSet(("idle", "unscrew"),
                                   (("idle",), ("unscrew",)))
            num_frames = 2

            def predict(self, frames, top_k=3, clip_start_s=None,
                        clip_end_s=None):
                return ActionPrediction(
                    "idle", 0.75, {"idle": 0.75, "unscrew": 0.25})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
            for index in range(12):
                writer.write(np.full((24, 32, 3), index * 10, np.uint8))
            writer.release()
            cfg_path = root / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump({
                "video_path": str(video),
                "output_dir": str(root / "out"),
                "device": "cpu",
                "action_classes": ["idle", "unscrew"],
                "action_frames": 2,
                "action_span": 4,
                "action_stride": 4,
                "action_top_k": 2,
                "save_xclip_video": False,
                "screenshot_every": 0,
            }), encoding="utf-8")
            with mock.patch.object(run_action_test,
                                   "create_action_classifier",
                                   return_value=FakeClassifier()):
                run_action_test.main(str(cfg_path), "cpu", "xclip")

            out_dir = root / "out" / "xclip"
            import pandas as pd

            data = pd.read_csv(out_dir / "xclip_actions.csv")
            summary = json.loads(
                (out_dir / "xclip_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(list(data.columns), ACTION_COLUMNS)
        self.assertEqual(len(data), 3)
        self.assertEqual(set(data["klasse"]), {"idle"})
        self.assertEqual(summary["task"], "action_classification")
        self.assertEqual(summary["clips_processed"], 3)


class ActionComparisonTest(unittest.TestCase):
    def test_summarize_action_run(self):
        import pandas as pd

        from compare_models import find_action_runs, summarize_action_run

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            step = out_dir / "xclip"
            step.mkdir()
            rows = [
                action_row("video.mp4", 0, 30, 30.0, "idle", 0.8,
                           [("idle", 0.8), ("unscrew", 0.2)]),
                action_row("video.mp4", 30, 60, 30.0, "unscrew", 0.6,
                           [("unscrew", 0.6), ("idle", 0.4)]),
            ]
            pd.DataFrame(rows, columns=ACTION_COLUMNS).to_csv(
                step / "xclip_actions.csv", index=False)
            (step / "xclip_summary.json").write_text(json.dumps({
                "model": "microsoft/xclip-test",
                "task": "action_classification",
                "clips_processed": 2,
                "inference_ms_median": 12.5,
                "pipeline_fps": 1.5,
            }), encoding="utf-8")
            (step / "vitra_actions.csv").write_text(
                "video,frame\nvideo.mp4,0\n", encoding="utf-8")

            paths = find_action_runs(out_dir)
            summary = summarize_action_run(paths[0])

        self.assertEqual(paths[0].name, "xclip_actions.csv")
        self.assertEqual(summary["clips"], 2)
        self.assertEqual(summary["labels_found"], 2)
        self.assertEqual(summary["top_label"], "idle")
        self.assertEqual(summary["top_label_share"], 0.5)
        self.assertEqual(summary["confidence_mean"], 0.7)
        self.assertEqual(summary["model"], "microsoft/xclip-test")


if __name__ == "__main__":
    unittest.main()
