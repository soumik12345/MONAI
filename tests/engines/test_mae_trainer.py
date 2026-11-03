"""CPU integration checks for MAE training orchestration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader

from monai.engines.mae_trainer import MAETrainer, MAETrainerConfig, TrackioConfig


class TinyMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, masking_ratio, generator):
        # Consume the supplied stream, just as the real model does.
        torch.rand((), generator=generator, device=x.device)
        return x * self.weight, torch.ones_like(x)


class TinyTrainer(MAETrainer):
    def make_dataloaders(self):
        samples = [{"image": torch.ones(1, 8, 8, 8), "valid": torch.ones(1, 8, 8, 8), "sequence": "T2"}]
        return DataLoader(samples), DataLoader(samples)

    def get_training_steps(self):
        return 3


class TestMAETrainer(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trackio = Mock()
        self.trackio.init.side_effect = self.new_run
        self.optional_import = patch("monai.engines.mae_trainer.optional_import", return_value=(self.trackio, True))
        self.import_tracking = self.optional_import.start()
        self.addCleanup(self.optional_import.stop)

    @staticmethod
    def new_run(**kwargs):
        run = Mock(spec=["name", "id", "log", "log_artifact", "finish"])
        run.name = kwargs["name"] or "generated-run"
        run.id = "test-run-id"
        return run

    def trainer(self, **kwargs):
        options = dict(
            output_dir=self.directory.name,
            dataloader_config={},
            trackio_config=TrackioConfig(project="test", enabled=True),
            device="cpu",
            local_batch_size=1,
            gradient_accumualtion_steps=2,
            mask_ratio=0.75,
            log_every=2,
            validate_every=2,
            save_every=2,
        )
        options.update(kwargs)
        return TinyTrainer(TinyMAE(), MAETrainerConfig(**options))

    def test_training_final_checkpoint_and_resume(self):
        trainer = self.trainer()
        trainer.train()
        state = torch.load(Path(self.directory.name) / "last.pt", weights_only=True)
        self.assertEqual(state["step"], 3)
        self.assertEqual(state["total_steps"], 3)
        self.assertTrue((Path(self.directory.name) / "best.pt").exists())
        self.assertLess(trainer.model.weight.item(), 1.0)
        self.assertGreater(trainer.model.weight.item(), 0.5)
        logs = [call.args[0] for call in trainer.run.log.call_args_list]
        self.assertEqual([log["optimizer_step"] for log in logs if "train/lr" in log], [2, 3])
        self.assertAlmostEqual(logs[0]["train/lr"], 0.001 * (1 - 1 / 3) ** 0.9)
        self.assertEqual([call.kwargs["step"] for call in trainer.run.log.call_args_list], [2, 2, 3, 3])
        for call in trainer.run.log_artifact.call_args_list:
            self.assertEqual(call.args, (Path(self.directory.name) / "best.pt",))
            self.assertEqual(call.kwargs, {"name": "run-test-run-id-best", "type": "model", "aliases": ["best"]})
        self.assertTrue(trainer.run.log_artifact.called)
        trainer.run.finish.assert_called_once_with()
        # Pre-migration checkpoints have a wandb config; tracking settings must
        # not prevent resuming their model, optimizer, and RNG state.
        del state["config"]["trackio_config"]
        state["config"]["wandb_config"] = {"project": "test", "mode": "disabled"}
        torch.save(state, Path(self.directory.name) / "last.pt")
        resumed = self.trainer(resume=True)
        with patch.object(resumed, "train_step") as step:
            resumed.train()
            step.assert_not_called()
        torch.testing.assert_close(resumed.model.weight, trainer.model.weight)
        self.assertTrue(torch.equal(resumed.mask_generator.get_state(), state["mask_rng"]))

    def test_resume_interrupted_training(self):
        trainer = self.trainer(steps=4)
        real_step = trainer.train_step
        attempts = 0

        def interrupted(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 3:
                raise RuntimeError("interrupted")
            return real_step(**kwargs)

        with patch.object(trainer, "train_step", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                trainer.train()
        trainer.run.finish.assert_called_once_with()
        resumed = self.trainer(steps=4, resume=True)
        with patch.object(resumed, "train_step", wraps=resumed.train_step) as step:
            resumed.train()
            self.assertEqual(step.call_count, 2)
            self.assertAlmostEqual(step.call_args_list[0].kwargs["lr"], 0.001 * 0.5**0.9)
        state = torch.load(Path(self.directory.name) / "last.pt", weights_only=True)
        self.assertEqual(state["step"], 4)

    def test_overflow_does_not_advance_schedule(self):
        trainer = self.trainer(steps=2)
        real_step = trainer.train_step
        calls = []

        def skip_first(**kwargs):
            calls.append(kwargs["lr"])
            if len(calls) == 1:
                return {"optimizer_step_skipped": True}
            return real_step(**kwargs)

        with patch.object(trainer, "train_step", side_effect=skip_first):
            trainer.train()
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], calls[1])
        self.assertLess(calls[2], calls[1])

    def test_validation_is_repeatable_and_preserves_training_rng(self):
        trainer = self.trainer()
        before = trainer.mask_generator.get_state().clone()
        first = trainer.validate()
        second = trainer.validate()
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["val/masked_mse"], 0.25)
        self.assertIn("val/mask_ratio_0.6/masked_ssim", first)
        self.assertIn("val/sequence_T2/foreground_masked_mse", first)
        self.assertTrue(torch.equal(before, trainer.mask_generator.get_state()))
        self.assertTrue(trainer.model.training)

    def test_existing_checkpoint_and_changed_recipe_are_rejected(self):
        self.trainer(steps=1).train()
        trainer = self.trainer(steps=1)
        with self.assertRaises(FileExistsError):
            trainer.train()
        trainer.run.finish.assert_called_once_with()
        resumed = self.trainer(steps=1, resume=True, mask_ratio=0.6)
        with self.assertRaisesRegex(ValueError, "mask_ratio"):
            resumed.train()

    def test_empty_validation_restores_model_mode(self):
        trainer = self.trainer()
        trainer.val_dataloader = []
        with self.assertRaisesRegex(ValueError, "nonempty"):
            trainer.validate()
        self.assertTrue(trainer.model.training)

    def test_tracking_disabled_does_not_import_trackio(self):
        trainer = self.trainer(steps=1, trackio_config=TrackioConfig())
        trainer.train()
        self.import_tracking.assert_not_called()
        self.assertIsNone(trainer.run)
        self.assertTrue((Path(self.directory.name) / "last.pt").exists())
        self.assertFalse((Path(self.directory.name) / "trackio_run.json").exists())

    def test_tracking_missing_dependency(self):
        self.import_tracking.return_value = (None, False)
        trainer = self.trainer(steps=1)
        with self.assertRaisesRegex(ImportError, "trackio>=0.43.0"):
            trainer.train()
        self.trackio.init.assert_not_called()

    def test_tracking_recipe_and_resume_identity(self):
        tracking = TrackioConfig(project="test", run_name="original", space_id="user/space", enabled=True)
        trainer = self.trainer(steps=1, trackio_config=tracking)
        recipe = {"config": {"steps": 1}, "manifest_sha256": "example"}
        trainer.train(tracking_config=recipe)
        self.import_tracking.assert_called_once_with("trackio", version="0.43.0")
        self.trackio.init.assert_called_once_with(
            project="test", name="original", space_id="user/space", config=recipe, resume="never", embed=False
        )
        metadata_path = Path(self.directory.name) / "trackio_run.json"
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(
            metadata, {"project": "test", "name": "original", "id": "test-run-id", "space_id": "user/space"}
        )
        resumed = self.trainer(steps=1, resume=True)
        resumed.train()
        options = self.trackio.init.call_args.kwargs
        self.assertEqual(options["name"], "original")
        self.assertEqual(options["space_id"], "user/space")
        self.assertEqual(options["resume"], "allow")

    def test_tracking_resume_different_project_does_not_reuse_identity(self):
        metadata_path = Path(self.directory.name) / "trackio_run.json"
        metadata_path.write_text(json.dumps({"project": "other", "name": "previous", "space_id": "user/previous"}))
        trainer = self.trainer(steps=1, resume=True)
        with self.assertRaises(FileNotFoundError):
            trainer.train()
        options = self.trackio.init.call_args.kwargs
        self.assertIsNone(options["name"])
        self.assertIsNone(options["space_id"])
        trainer.run.finish.assert_called_once_with()

    def test_tracking_resume_explicit_identity_overrides_saved_values(self):
        metadata_path = Path(self.directory.name) / "trackio_run.json"
        metadata_path.write_text(json.dumps({"project": "test", "name": "previous", "space_id": "user/previous"}))
        tracking = TrackioConfig(project="test", run_name="replacement", space_id="user/new", enabled=True)
        trainer = self.trainer(steps=1, resume=True, trackio_config=tracking)
        with self.assertRaises(FileNotFoundError):
            trainer.train()
        options = self.trackio.init.call_args.kwargs
        self.assertEqual(options["name"], "replacement")
        self.assertEqual(options["space_id"], "user/new")


if __name__ == "__main__":
    unittest.main()
