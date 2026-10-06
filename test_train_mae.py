"""Regression checks for the MAE adaptation; run with python -m unittest test_train_mae."""

import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import torch

import prepare_mae as preparation
import train_mae as mae

SMALL_FILTERS = (4, 8, 12, 16, 24, 24)


class MAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_wandb_optional_and_resume_identity(self):
        self.assertIsNone(mae.init_wandb(Namespace(wandb=False), None, None, None, None, None))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = Namespace(wandb=True, resume=False, wandb_project="test-project", wandb_entity=None,
                             wandb_name="test", wandb_mode="online", batch_size=1, accumulation=6)
            sdk = MagicMock(spec=["init"])
            run = sdk.init.return_value
            run.id, run.project, run.entity = "run-id", "test-project", "test-entity"
            manifest = {"revision": "commit", "sequences": ["PRE"]}
            partitions = {"train": [{"patient_id": "p1"}, {"patient_id": "p1"}]}
            with patch.object(mae, "wandb", sdk):
                self.assertIs(mae.init_wandb(args, output, manifest, "hash", partitions, "cpu"), run)
                self.assertRegex(sdk.init.call_args.kwargs["id"], r"^[0-9a-f]{32}$")
                config = sdk.init.call_args.kwargs["config"]
                self.assertEqual(config["effective_batch"], 6)
                self.assertEqual(config["split_patients"], {"train": 1})
                self.assertEqual(config["manifest_sha256"], "hash")
                args.resume = True
                mae.init_wandb(args, output, manifest, "hash", partitions, "cpu")
                self.assertEqual(sdk.init.call_args.kwargs["id"], "run-id")
                self.assertEqual(sdk.init.call_args.kwargs["resume"], "allow")
                self.assertEqual(sdk.init.call_args.kwargs["entity"], "test-entity")
                args.wandb_mode = "offline"
                mae.init_wandb(args, output, manifest, "hash", partitions, "cpu")
                self.assertIsNone(sdk.init.call_args.kwargs["resume"])
                run.define_metric.side_effect = RuntimeError("setup failed")
                with self.assertRaisesRegex(RuntimeError, "setup failed"):
                    mae.init_wandb(args, output, manifest, "hash", partitions, "cpu")
                run.finish.assert_called_with(exit_code=1)

    def test_reconstruction_metrics_use_hidden_voxels_and_foreground(self):
        target = torch.zeros(1, 1, 16, 16, 16)
        mask = torch.zeros(1, 1, 2, 2, 2)
        mask[:, :, 0] = 1
        foreground = torch.zeros_like(target)
        foreground[:, :, :4] = 1
        prediction = torch.full_like(target, 100)
        prediction[:, :, :4] = 2
        prediction[:, :, 4:8] = 4
        metrics = mae.reconstruction_totals(prediction, target, mask, foreground)
        self.assertAlmostEqual(metrics["masked_mse"][0] / metrics["masked_mse"][1], 10)
        self.assertAlmostEqual(metrics["masked_mae"][0] / metrics["masked_mae"][1], 3)
        self.assertAlmostEqual(metrics["foreground_masked_mse"][0] / metrics["foreground_masked_mse"][1], 4)
        perfect = mae.reconstruction_totals(target, target, mask, foreground, ssim_data_range=10)
        self.assertGreater(perfect["masked_ssim"][1], 0)
        self.assertAlmostEqual(perfect["masked_ssim"][0] / perfect["masked_ssim"][1], 1)
        # Altering visible voxels cannot affect SSIM on fully hidden windows.
        prediction = target.clone()
        prediction[:, :, 8:] = 100
        unchanged = mae.reconstruction_totals(prediction, target, mask, foreground, ssim_data_range=10)
        self.assertEqual(unchanged["masked_ssim"], perfect["masked_ssim"])
        empty = mae.reconstruction_totals(target, target, mask, torch.zeros_like(target))
        self.assertEqual(empty["foreground_masked_mse"][1], 0)

    def test_validation_groups_and_fixed_masks_are_reproducible(self):
        class ZeroModel(torch.nn.Module):
            def forward(self, image, mask):
                return torch.zeros_like(image)
        args = Namespace(seed=42, block=32, mask_ratio=None, mask_range=(0.6, 0.9),
                         val_mask_ratios=(0.6, 0.75, 0.9), ssim_data_range=10)
        x = torch.stack([torch.ones(1, 64, 64, 64), torch.full((1, 64, 64, 64), 2.)])
        loader = [{"image": x, "valid": torch.ones_like(x), "sequence": ["PRE", "FLAIR"]}]
        model = ZeroModel()
        metrics = mae.validate(model, loader, torch.device("cpu"), args)
        self.assertEqual(metrics, mae.validate(model, loader, torch.device("cpu"), args))
        self.assertAlmostEqual(metrics["val/masked_mse"], 2.5)
        self.assertAlmostEqual(metrics["val/masked_mae"], 1.5)
        self.assertAlmostEqual(metrics["val/sequence_PRE/masked_mse"], 1)
        self.assertAlmostEqual(metrics["val/sequence_FLAIR/masked_mse"], 4)
        for ratio in args.val_mask_ratios:
            self.assertAlmostEqual(metrics[f"val/mask_ratio_{ratio:g}/masked_mse"], 2.5)
            self.assertTrue(np.isfinite(metrics[f"val/mask_ratio_{ratio:g}/masked_ssim"]))

    def test_paper_mask_grid_and_hidden_counts(self):
        x = torch.empty(2, 1, 160, 160, 160)
        for ratio in (0.6, 0.75, 0.9):
            mask = mae.block_mask(x, ratio=ratio, coarse=True)
            self.assertEqual(mask.shape, (2, 1, 5, 5, 5))
            self.assertTrue(torch.equal(mask.sum((1, 2, 3, 4)), torch.full((2,), round(125 * ratio))))
            expanded = mae.expand_mask(mask, x.shape[2:])
            self.assertTrue(torch.equal(expanded[:, :, ::32, ::32, ::32], mask))
            self.assertTrue(torch.equal(expanded[:, :, 31::32, 31::32, 31::32], mask))
        with self.assertRaises(ValueError):
            mae.block_mask(torch.empty(1, 1, 161, 160, 160), coarse=True)

    def test_visible_only_norm_matches_gathered_instance_norm(self):
        x = torch.randn(2, 3, 2, 2, 2, requires_grad=True)
        visible = torch.zeros(2, 1, 2, 2, 2)
        visible[:, :, :, :, 0] = 1
        norm = torch.nn.InstanceNorm3d(3, affine=True)
        norm.weight.data.copy_(torch.tensor([0.5, 1.5, 2.0]))
        norm.bias.data.copy_(torch.tensor([1.0, -1.0, 0.5]))
        result = mae.masked_instance_norm(x, norm, visible)
        gathered = x[:, :, :, :, 0].reshape(2, 3, -1)
        expected = torch.nn.functional.instance_norm(gathered, weight=norm.weight, bias=norm.bias)
        torch.testing.assert_close(result[:, :, :, :, 0].reshape(2, 3, -1), expected)
        self.assertEqual(torch.count_nonzero(result[:, :, :, :, 1]).item(), 0)
        result.square().sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[:, :, :, :, 1]).item(), 0)

    def test_hidden_inputs_cannot_leak_into_any_encoder_scale(self):
        model = mae.DynUNetMAE(SMALL_FILTERS, checkpointing=False).eval()
        x = torch.randn(1, 1, 64, 64, 64)
        mask = mae.block_mask(x, ratio=0.6, coarse=True)
        changed = x + 10000 * mae.expand_mask(mask, x.shape[2:])
        with torch.no_grad():
            original_features = model.encode(x, mask)
            changed_features = model.encode(changed, mask)
        for original, altered in zip(original_features, changed_features):
            torch.testing.assert_close(original, altered, rtol=0, atol=0)
            hidden = mae.expand_mask(mask, original.shape[2:]).expand_as(original).bool()
            self.assertEqual(torch.count_nonzero(original[hidden]).item(), 0)

    def test_unmasked_encoder_preserves_monai_residual_blocks(self):
        model = mae.DynUNetMAE(SMALL_FILTERS, checkpointing=False).eval()
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            actual = model.encode(x, torch.zeros(1, 1, 2, 2, 2))
            for expected, block in zip(
                actual, [model.backbone.input_block, *model.backbone.downsamples, model.backbone.bottleneck]
            ):
                x = block(x)
                torch.testing.assert_close(expected, x, rtol=1e-4, atol=2e-5)

    def test_loss_includes_hidden_background_and_ignores_visible_voxels(self):
        target = torch.zeros(1, 1, 2, 2, 2)
        mask = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 1, 2)
        prediction = torch.ones_like(target)
        prediction[..., 0] = 100
        self.assertEqual(mae.masked_mse(prediction, target, mask).item(), 1)
        prediction[..., 1] = 2
        self.assertEqual(mae.masked_mse(prediction, target, mask).item(), 4)

    def test_batchaug_runs_on_cuda_batch_before_masking(self):
        if not torch.cuda.is_available():
            self.skipTest("BatchAug GPU path requires CUDA")
        if importlib.util.find_spec("batchaug") is None:
            self.skipTest("BatchAug is an optional dependency")
        x = torch.randn(2, 1, 64, 64, 64, device="cuda")
        augmented = mae.make_batch_augmenter()({"image": x})["image"]
        self.assertEqual(augmented.shape, x.shape)
        self.assertTrue(torch.isfinite(augmented).all())
        mask = mae.block_mask(augmented, ratio=0.75, coarse=True)
        self.assertEqual(mask.shape, (2, 1, 2, 2, 2))
        self.assertTrue(torch.isfinite(mae.masked_mse(torch.zeros_like(augmented), augmented, mask)))

    def test_checkpointing_and_accumulation_preserve_gradients(self):
        # CPU MKLDNN picks different kernels for batches of one and two. With
        # only three visible bottleneck cells, roundoff can change derivatives
        # near LeakyReLU's kink. Compare the same convolution implementation.
        with torch.backends.mkldnn.flags(enabled=False):
            self._compare_accumulated_gradients()

    def _compare_accumulated_gradients(self):
        reference = mae.DynUNetMAE(SMALL_FILTERS, checkpointing=False)
        accumulated = mae.DynUNetMAE(SMALL_FILTERS, checkpointing=True)
        accumulated.load_state_dict(reference.state_dict())
        x = torch.randn(2, 1, 64, 64, 64)
        mask = mae.block_mask(x, ratio=0.6, coarse=True)
        mae.masked_mse(reference(x, mask), x, mask).backward()
        for index in range(2):
            loss = mae.masked_mse(
                accumulated(x[index : index + 1], mask[index : index + 1]),
                x[index : index + 1],
                mask[index : index + 1],
            )
            (loss / 2).backward()
        for (name, first), (_, second) in zip(reference.named_parameters(), accumulated.named_parameters()):
            self.assertIsNotNone(first.grad, name)
            self.assertIsNotNone(second.grad, name)
            self.assertTrue(torch.isfinite(second.grad).all(), name)
            torch.testing.assert_close(first.grad, second.grad, rtol=3e-3, atol=2e-5, msg=name)

    def test_encoder_and_decoder_transfer_excludes_mae_modules_and_head(self):
        model = mae.DynUNetMAE(SMALL_FILTERS)
        segmentation = model.segmentation_model(2)
        source = model.backbone.state_dict()
        for name, weight in segmentation.state_dict().items():
            if not name.startswith("output_block."):
                torch.testing.assert_close(weight, source[name], rtol=0, atol=0)
            self.assertFalse("densify" in name or "mask_tokens" in name)
        with torch.no_grad():
            self.assertEqual(segmentation(torch.randn(1, 1, 64, 64, 64)).shape, (1, 2, 64, 64, 64))

    def test_patient_leakage_and_scouts_are_rejected(self):
        manifest = {"partitions": {key: [{"patient_id": key}] for key in ("train", "val", "test")}}
        mae.check_manifest(manifest)
        manifest["partitions"]["val"][0]["patient_id"] = "train"
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            mae.check_manifest(manifest)
        self.assertTrue(preparation.usable_geometry({"shape": [64] * 3, "voxel_spacing_mm": [1.0] * 3}))
        for shape, spacing in (([10, 64, 64], [1.0] * 3), ([64] * 3, [7.0, 1, 1]), ([64] * 3, [float("nan"), 1, 1])):
            self.assertFalse(preparation.usable_geometry({"shape": shape, "voxel_spacing_mm": spacing}))

    def test_prepare_pins_catalog_and_downloads_and_splits_patients(self):
        rows = [
            dict(
                patient_id=f"p{i}",
                study_id=f"s{i}",
                sequence=sequence,
                nifti_path=f"p{i}/{sequence}.nii.gz",
                sha256="test",
                integrity_status="ok",
                shape=[64] * 3,
                voxel_spacing_mm=[1.0] * 3,
            )
            for i in range(10)
            for sequence in ("PRE", "POST", "T2", "FLAIR")
        ]
        rows.append({**rows[0], "integrity_status": "bad"})
        rows.append({**rows[0], "shape": [5, 64, 64]})
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(preparation, "load_dataset", return_value=rows) as catalog,
            patch.object(
                preparation, "hf_hub_download", side_effect=lambda repo, path, **kw: "/fake/" + path
            ) as download,
            patch("huggingface_hub.HfApi.repo_info", return_value=Namespace(sha="pinned-sha")),
        ):
            args = Namespace(
                output=directory,
                revision=None,
                sequence=None,
                sequences=["PRE", "POST", "T2", "FLAIR"],
                seed=42,
                max_patients=0,
                workers=4,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                preparation.prepare(args)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            partitions = mae.check_manifest(manifest)
            self.assertEqual([len(partitions[key]) for key in ("train", "val", "test")], [32, 4, 4])
            self.assertEqual(catalog.call_args.kwargs["revision"], "pinned-sha")
            self.assertEqual(download.call_count, 40)
            self.assertTrue(all(call.kwargs["revision"] == "pinned-sha" for call in download.call_args_list))

    def test_nifti_preprocessing_cache_training_resume_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            image_path = output / "synthetic.nii.gz"
            image = np.zeros((40, 42, 44), dtype=np.float32)
            image[4:-4, 4:-4, 4:-4] = np.random.default_rng(7).normal(100, 15, image[4:-4, 4:-4, 4:-4].shape)
            nib.save(nib.Nifti1Image(image, np.diag([-1.5, 1.5, 1.5, 1])), image_path)
            rows = [{"image": str(image_path), "patient_id": "synthetic"}]
            for training, backend in ((False, "monai"), (True, "monai"), (True, "batchaug")):
                dataset = mae.make_dataset(rows, (64,) * 3, training, output / "cache", backend)
                for _ in range(2):
                    result = dataset[0]
                    self.assertEqual(result["image"].shape, (1, 64, 64, 64))
                    self.assertTrue(torch.isfinite(result["image"]).all())
                    if training:
                        self.assertNotIn("valid", result)
                    else:
                        self.assertEqual(result["valid"].shape, result["image"].shape)
                        self.assertEqual(set(result["valid"].unique().tolist()), {0.0, 1.0})
            manifest = {
                "revision": "synthetic-only",
                "sequences": ["PRE"],
                "partitions": {
                    key: [{"image": str(image_path), "patient_id": key}] for key in ("train", "val", "test")
                },
            }
            (output / "manifest.json").write_text(json.dumps(manifest))
            args = Namespace(
                output=directory,
                seed=42,
                device="cpu",
                filters=SMALL_FILTERS,
                roi=(64,) * 3,
                no_checkpointing=False,
                resume=False,
                steps=2,
                batch_size=1,
                accumulation=2,
                workers=0,
                cache_dir=str(output / "cache"),
                lr=0.01,
                momentum=0.99,
                weight_decay=3e-5,
                poly_power=0.9,
                grad_clip=12,
                block=32,
                mask_ratio=None,
                mask_range=(0.6, 0.9),
                log_every=1,
                save_every=1,
                validate_every=1,
                val_volumes=0,
            )
            original_save = mae.save_checkpoint
            tracking_run = MagicMock()

            def capture_first_step(path, *positional):
                original_save(path, *positional)
                if path.name == "last.pt" and positional[5] == 1:
                    shutil.copyfile(path, output / "first.pt")

            with (
                patch.object(mae, "save_checkpoint", side_effect=capture_first_step),
                patch.object(mae, "init_wandb", return_value=tracking_run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                mae.train(args)
            payloads = [call.args[0] for call in tracking_run.log.call_args_list]
            self.assertTrue(any("train/masked_mae" in payload for payload in payloads))
            validation = next(payload for payload in payloads if "val/masked_mse" in payload)
            for key in ("val/masked_mae", "val/foreground_masked_mse", "val/masked_ssim",
                        "val/mask_ratio_0.75/masked_mse", "val/sequence_unknown/masked_mse"):
                self.assertIn(key, validation)
                self.assertTrue(np.isfinite(validation[key]))
            tracking_run.finish.assert_called_once_with(exit_code=0)
            state = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(state["step"], 2)
            self.assertTrue(np.isfinite(state["val_loss"]))
            self.assertEqual(state["optimizer"]["param_groups"][0]["lr"], mae.poly_lr(0.01, 1, 2))
            with self.assertRaisesRegex(ValueError, "original patient split"):
                mae.check_resume(state, args, "different-hash")
            args.lr = 0.1
            with self.assertRaisesRegex(ValueError, "original --lr"):
                mae.check_resume(state, args, state["manifest_sha256"])
            args.lr = 0.01
            args.augmentation_backend = "batchaug"
            with self.assertRaisesRegex(ValueError, "original --augmentation-backend"):
                mae.check_resume(state, args, state["manifest_sha256"])
            args.augmentation_backend = "monai"
            shutil.copyfile(output / "first.pt", output / "last.pt")
            args.resume = True
            with contextlib.redirect_stdout(io.StringIO()):
                mae.train(args)
                mae.export(Namespace(checkpoint=None, output=directory, out_channels=2))
            resumed = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(resumed["step"], 2)
            self.assertTrue(np.isfinite(resumed["val_loss"]))
            exported = torch.load(output / "segmentation_init.pt", weights_only=True)
            dense = mae.build_model(2, SMALL_FILTERS)
            dense.load_state_dict(exported["model"], strict=True)
            self.assertEqual(exported["pretraining_step"], 2)
            if torch.cuda.is_available() and importlib.util.find_spec("batchaug") is not None:
                gpu_output = output / "batchaug_gpu"
                gpu_output.mkdir()
                (gpu_output / "manifest.json").write_text(json.dumps(manifest))
                gpu_args = Namespace(
                    **{
                        **vars(args),
                        "output": str(gpu_output),
                        "device": "cuda",
                        "steps": 1,
                        "resume": False,
                        "augmentation_backend": "batchaug",
                    }
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    mae.train(gpu_args)
                gpu_state = torch.load(gpu_output / "last.pt", weights_only=True)
                self.assertEqual(gpu_state["step"], 1)
                self.assertTrue(np.isfinite(gpu_state["val_loss"]))
                self.assertEqual(gpu_state["config"]["augmentation_backend"], "batchaug")


if __name__ == "__main__":
    unittest.main()
