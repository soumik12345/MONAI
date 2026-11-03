"""Train an image-only, nnU-Net-planned DynUNet masked autoencoder.

Install: uv pip install 'nnunetv2>=2.6.2,<3' nibabel
Optional tracking: uv pip install 'trackio>=0.43.0'
BatchAug additionally requires batchaug and working CUDA-matched CuPy.

Use the SAME scalar-MRI train/val/test manifest as train_dynmae.py:
    python train_nnunet_mae.py plan --manifest yale_mae_restored/manifest.json --output yale_nnunet_mae
    python train_nnunet_mae.py train --manifest yale_mae_restored/manifest.json \
        --output yale_nnunet_mae --device cuda --epochs 100 --trackio
Resume with the same arguments and --resume. Existing mae_plans.json is reused;
use a new output directory to change the plan. --roi overrides the planned crop
in transposed Z,Y,X voxels; --target-spacing overrides spacing in NIfTI X,Y,Z mm.
--batch-size overrides automatic batch selection. --base-features/--max-features
control the planned channel schedule; --filters supplies an explicit schedule.

Planning fingerprints TRAINING IMAGES ONLY, without segmentation labels or fake
labels. It uses nnU-Net's full-resolution spacing, normalization-mask decision,
and anisotropy-aware pooling/kernel rules. A MONAI MaskedAutoEncoderDynUnet is
built from that topology, rather than loading native nnU-Net segmentation weights.
Native nnU-Net's segmentation memory calibration is replaced by a forward,
backward, and SGD-step memory probe of the actual MAE on CUDA. CPU planning uses
a conservative estimate, not a guarantee. Reused plans are checked against the
actual MAE again before CUDA training. --gpu-memory-target is a budget in GB
(default 8); CUDA probes also reserve 20% of currently free memory.

Preprocessing uses native Z,Y,X axes, nonzero cropping, planned MRI z-score
normalization, and nnU-Net resampling. Spatial augmentation follows train_dynmae
with rotations/scaling in physical coordinates. The mask grid follows total
encoder stride; voxel/mm block sizes and visible-position counts are saved.
Mask ratio remains an explicit training hyperparameter, not an optimized choice.

MAETrainer provides masked MSE, accumulation, AMP, validation, checkpoints, and
optional Trackio logging. Epochs are complete shuffled training passes; --steps
uses replacement sampling. Checkpoints embed model configuration and a plan hash.
mae_plans.json, recipe.json, best.pt, and last.pt are written under --output.
Reconstruction quality should also be assessed through downstream transfer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from monai.engines.mae_trainer import MAETrainerConfig, TrackioConfig, _check_manifest
from monai.networks.nets import MaskedAutoEncoderDynUnet
from monai.transforms import Compose, EnsureTyped, MapTransform
from monai.utils import set_determinism
from train_dynmae import (
    DynMaeTrainer,
    NNSSLPatchd,
    NNSSLSpatiald,
    apply_spatial,
    check_cuda_cubic,
    initial_patch_size,
    make_batch_augmenter,
    mirror_sample,
    sample_spatial_parameters,
)

FORMAT = "NNUNetMAEPlans-v1"


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_ready(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_manifest(path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    partitions = _check_manifest(manifest)
    path_base = manifest.get("path_base", ".")
    base = path.parent if path_base in (".", "repository root") else path.parent / path_base
    files = {}
    for split in ("train", "val"):
        for row in partitions[split]:
            if not isinstance(row["image"], str):
                raise ValueError("Use a scalar-MRI MAE manifest, with one image path per row.")
            image = (base / row["image"]).resolve()
            stat = image.stat()
            row["image"] = str(image)
            files[str(image)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return manifest, {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": files}, str(base.resolve())


def load_crop(path):
    import nibabel as nib
    from nibabel.spatialimages import SpatialImage
    from scipy.ndimage import binary_fill_holes

    nifti = nib.load(path)
    if not isinstance(nifti, SpatialImage) or nifti.ndim != 3:
        raise ValueError(f"Expected scalar 3D MRI: {path}")
    image = nifti.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    spacing = np.asarray(nifti.header.get_zooms()[::-1], dtype=float)
    if not np.isfinite(image).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"Nonfinite MRI intensities or invalid spacing: {path}")
    foreground = binary_fill_holes(image != 0)
    locations = np.nonzero(foreground)
    if not locations[0].size:
        raise ValueError(f"Empty foreground: {path}")
    bbox = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in locations)
    relative_size = np.prod([item.stop - item.start for item in bbox]) / image.size
    return image[bbox].copy(), foreground[bbox], spacing, float(relative_size)


def fingerprint(rows, limit=0, seed=42, workers=1):
    selected = random.Random(seed).sample(rows, limit) if limit and len(rows) > limit else rows

    def inspect(row):
        image, _, spacing, relative_size = load_crop(row["image"])
        return {"image": row["image"], "shape": image.shape, "spacing": spacing, "relative_size": relative_size}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        volumes = list(pool.map(inspect, selected))
    return json_ready(
        {
            "spacings": [row["spacing"] for row in volumes],
            "shapes_after_crop": [row["shape"] for row in volumes],
            "median_relative_size_after_cropping": np.median([row["relative_size"] for row in volumes]),
            "volumes": volumes,
            "training_volume_count": len(rows),
        }
    )


def topology(spacing, patch, args):
    from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

    _, strides, kernels, roi, _ = get_pool_and_conv_props(spacing, patch, 4, args.max_stages - 1)
    # nnU-Net limits pooling per axis; delayed pooling along a coarse axis can
    # otherwise create more total stages than this MAE's requested depth cap.
    if len(strides) > args.max_stages:
        strides = strides[: args.max_stages]
        kernels = (*kernels[: args.max_stages - 1], (3, 3, 3))
        divisible = np.prod(np.asarray(strides), axis=0)
        roi = tuple((np.ceil(np.asarray(patch) / divisible) * divisible).astype(int))
    if len(strides) < 3:
        raise ValueError("The planned MAE requires at least three stages; increase the crop size.")
    filters = args.filters or [min(args.max_features, args.base_features * 2**index) for index in range(len(strides))]
    if len(filters) != len(strides):
        raise ValueError(f"--filters requires {len(strides)} entries for this planned topology.")
    model = {
        "in_channels": 1,
        "spatial_dims": 3,
        "kernel_size": kernels,
        "strides": strides,
        "filters": filters,
        "enable_activation_checkpointing": not args.no_checkpointing,
    }
    return tuple(roi), model


def estimate_memory(roi, model_config, batch_size):
    """Conservative dense-activation proxy; never discount voxels by mask ratio."""
    with torch.device("meta"):
        model = MaskedAutoEncoderDynUnet(**model_config)
    parameter_bytes = sum(parameter.numel() for parameter in model.parameters()) * 16
    shape = np.asarray(roi)
    activations = 0
    for stride, channels in zip(model_config["strides"], model_config["filters"]):
        shape = shape // stride
        activations += int(np.prod(shape)) * channels
    return int(parameter_bytes + batch_size * (activations * 4 * 16 + math.prod(roi) * 16))


def probe_cuda_memory(roi, model_config, batch_size, args):
    """Measure the actual dense MAE, including gradients and SGD momentum."""

    def probe():
        model = MaskedAutoEncoderDynUnet(**model_config).cuda().train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
        )
        image = torch.randn(batch_size, 1, *roi, device="cuda")
        # A second update measures activations while momentum buffers already
        # exist, rather than underestimating steady-state training memory.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                prediction, mask = model(image, masking_ratio=args.mask_ratio or max(args.mask_range))
                hidden = torch.nn.functional.interpolate(mask, size=roi, mode="nearest")
                loss = ((prediction.float() - image).square() * hidden).sum() / hidden.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            return probe() - baseline
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def check_cuda_plan(plan, args):
    budget = min(args.gpu_memory_target * 1024**3, torch.cuda.mem_get_info()[0] * 0.8)
    memory = probe_cuda_memory(tuple(plan["patch_size"]), plan["model_config"], plan["batch_size"], args)
    if memory is None or memory > budget:
        raise ValueError(
            "The saved plan does not fit this CUDA memory budget; create a smaller plan in a new --output."
        )
    return {"method": "cuda_probe", "bytes": memory, "budget_bytes": int(budget)}


def planning_settings(args):
    names = (
        "roi",
        "target_spacing",
        "batch_size",
        "max_batch_size",
        "base_features",
        "max_features",
        "filters",
        "max_stages",
        "gpu_memory_target",
        "fingerprint_volumes",
        "seed",
        "no_checkpointing",
        "mask_ratio",
        "mask_range",
        "weight_decay",
    )
    return {name: getattr(args, name) for name in names}


def create_plan(args, manifest, identity, manifest_base, device):
    # Only the native image-based planning methods are needed. Its normal
    # constructor expects segmentation dataset folders, which MAE does not have.
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

    data = fingerprint(manifest["partitions"]["train"], args.fingerprint_volumes, args.seed, args.plan_workers)

    class ImageOnlyPlanner(ExperimentPlanner):
        def __init__(self):
            self.dataset_fingerprint = data
            self.dataset_json = {"channel_names": {"0": "MRI"}}
            self.anisotropy_threshold = 3
            self.overwrite_target_spacing = args.target_spacing[::-1] if args.target_spacing else None
            self.suppress_transpose = False

    planner = ImageOnlyPlanner()
    target = planner.determine_fullres_target_spacing()
    transpose, _ = planner.determine_transpose()
    normalization, mask_for_norm = planner.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
    resampled_shapes = np.maximum(1, np.rint(np.asarray(data["shapes_after_crop"]) * data["spacings"] / target))
    median_shape = np.median(resampled_shapes, axis=0)[transpose]
    spacing = target[transpose]
    aspect = 1 / spacing
    initial = aspect * (256**3 / np.prod(aspect)) ** (1 / 3)
    patch = np.asarray(args.roi if args.roi else np.maximum(16, np.minimum(initial, median_shape)), dtype=int)
    budget = args.gpu_memory_target * 1024**3
    measured = device == "cuda"
    if measured:
        budget = min(budget, torch.cuda.mem_get_info()[0] * 0.8)
    for _ in range(64):
        roi, model_config = topology(spacing, patch, args)
        if args.roi and tuple(args.roi) != roi:
            raise ValueError(f"--roi must respect the planned strides; use {roi}.")

        def measure(batch):
            if measured:
                return probe_cuda_memory(roi, model_config, batch, args)
            return estimate_memory(roi, model_config, batch)

        one = measure(1)
        if one is not None and one <= budget:
            batch_size = args.batch_size or min(
                args.max_batch_size, len(manifest["partitions"]["train"]), max(1, int(budget // one))
            )
            while batch_size >= 1:
                memory = one if batch_size == 1 else measure(batch_size)
                if memory is not None and memory <= budget:
                    stride = np.prod(np.asarray(model_config["strides"]), axis=0)
                    grid = np.asarray(roi) // stride
                    positions = int(np.prod(grid))
                    ratio = args.mask_ratio or max(args.mask_range)
                    return json_ready(
                        {
                            "format": FORMAT,
                            "dataset_identity": identity,
                            "manifest_base": manifest_base,
                            "planning_settings": planning_settings(args),
                            "nnunet_version": version("nnunetv2"),
                            "fingerprint": data,
                            "transpose_forward": transpose,
                            "target_spacing": spacing,
                            "normalization": normalization[0],
                            "use_mask_for_norm": mask_for_norm[0],
                            "patch_size": roi,
                            "median_shape": median_shape,
                            "model_config": model_config,
                            "batch_size": batch_size,
                            "memory": {
                                "method": "cuda_probe" if measured else "estimate",
                                "bytes": memory,
                                "budget_bytes": int(budget),
                            },
                            "mask_grid": grid,
                            "mask_block_voxels": stride,
                            "mask_block_mm": stride * spacing,
                            "minimum_visible_positions": positions
                            - min(positions - 2, max(1, round(positions * ratio))),
                        }
                    )
                if args.batch_size:
                    break
                batch_size -= 1
        if args.roi:
            raise ValueError(
                "The requested crop/batch does not fit the MAE memory budget; reduce it or raise the budget."
            )
        reduced = np.maximum(16, np.floor(np.asarray(roi) * 0.8)).astype(int)
        if np.array_equal(reduced, patch):
            break
        patch = reduced
    raise ValueError(
        "No compatible MAE crop/batch fits the memory budget; increase --gpu-memory-target or reduce features."
    )


class PlannedPreprocessd(MapTransform):
    def __init__(self, plan):
        super().__init__(("image",))
        self.plan = plan

    def __call__(self, data):
        from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
        from nnunetv2.preprocessing.resampling.default_resampling import (
            compute_new_shape,
            resample_data_or_seg_to_shape,
        )

        d = dict(data)
        path = Path(self.plan["manifest_base"]) / d["image"]
        image, foreground, spacing, _ = load_crop(path)
        axes = self.plan["transpose_forward"]
        image, foreground, spacing = image.transpose(axes), foreground.transpose(axes), spacing[axes]
        normalizer = ZScoreNormalization(use_mask_for_norm=self.plan["use_mask_for_norm"], intensityproperties={})
        image = normalizer.run(image, np.where(foreground, 0, -1))
        target = np.asarray(self.plan["target_spacing"])
        shape = np.maximum(1, compute_new_shape(image.shape, spacing, target))
        d["image"] = resample_data_or_seg_to_shape(
            image[None], shape, spacing, target, order=3, order_z=0, force_separate_z=None
        )
        d["valid"] = resample_data_or_seg_to_shape(
            foreground[None].astype(np.float32),
            shape,
            spacing,
            target,
            is_seg=True,
            order=0,
            order_z=0,
            force_separate_z=None,
        )
        return d


class PlannedPatchd(NNSSLPatchd):
    def __init__(self, roi, training, spacing):
        super().__init__(roi, training)
        if training:
            self.size = tuple(
                np.maximum(roi, np.ceil(np.asarray(initial_patch_size(np.asarray(roi) * spacing)) / spacing)).astype(
                    int
                )
            )


class PlannedSpatiald(NNSSLSpatiald):
    def __init__(self, roi, spacing):
        super().__init__(roi)
        self.spacing = np.asarray(spacing)

    def __call__(self, data):
        d = dict(data)
        params = sample_spatial_parameters(self.R)
        params["matrix"] = params["matrix"] * self.spacing[None, :] / self.spacing[:, None]
        d["image"] = mirror_sample(apply_spatial(d["image"], self.roi, params), self.R)
        return d


class NNUNetMAETrainer(DynMaeTrainer):
    def make_dataloaders(self, manifest_path, plan, **kwargs):
        self.plan = plan
        return super().make_dataloaders(
            manifest_path, roi=tuple(plan["patch_size"]), transpose_forward=plan["transpose_forward"], **kwargs
        )

    def make_transforms(self, roi, training, augmentation_backend="monai", transpose_forward=(0, 1, 2)):
        spacing = np.asarray(self.plan["target_spacing"])
        transforms = [PlannedPreprocessd(self.plan), PlannedPatchd(roi, training, spacing)]
        if training and augmentation_backend == "monai":
            transforms.append(PlannedSpatiald(roi, spacing))
        keys = "image" if training else ("image", "valid")
        transforms.append(EnsureTyped(keys, dtype=torch.float32, track_meta=False))
        return Compose(transforms)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "plan"))
    parser.add_argument("--output", default="./yale_nnunet_mae")
    parser.add_argument("--manifest", help="Same MAE manifest as train_dynmae.py; default OUTPUT/manifest.json")
    parser.add_argument("--roi", type=int, nargs=3, help="Optional exact crop in planned/transposed axes")
    parser.add_argument("--target-spacing", type=float, nargs=3, help="Optional X,Y,Z spacing in mm")
    parser.add_argument("--gpu-memory-target", type=float, default=8)
    parser.add_argument("--batch-size", type=int, help="Default: plan a batch that fits the memory budget")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--base-features", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=320)
    parser.add_argument("--max-stages", type=int, default=6)
    parser.add_argument("--filters", type=int, nargs="+")
    parser.add_argument(
        "--fingerprint-volumes",
        type=int,
        default=0,
        help="0 fingerprints all training images; otherwise a seeded subset",
    )
    parser.add_argument("--plan-workers", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--steps", type=int)
    schedule.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=12)
    parser.add_argument("--mask-ratio", type=float)
    parser.add_argument("--mask-range", type=float, nargs=2, default=(0.6, 0.9))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-dir")
    parser.add_argument("--augmentation-backend", choices=("monai", "batchaug"), default="monai")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--val-volumes", type=int, default=25)
    parser.add_argument("--val-mask-ratios", type=float, nargs="+", default=(0.6, 0.75, 0.9))
    parser.add_argument("--ssim-data-range", type=float, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--trackio", action="store_true")
    parser.add_argument("--trackio-project", default="yale-nnunet-mae")
    parser.add_argument("--trackio-name")
    parser.add_argument("--trackio-space-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.steps is None and args.epochs is None:
        args.steps = 250_000
    for name in (
        "max_batch_size",
        "base_features",
        "max_features",
        "plan_workers",
        "accumulation",
        "batch_size",
        "steps",
        "epochs",
        "validate_every",
        "save_every",
        "log_every",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.max_stages < 3 or min(args.workers, args.val_volumes, args.fingerprint_volumes) < 0:
        parser.error("max-stages must be >=3; workers, val-volumes, and fingerprint-volumes must be nonnegative")
    if args.roi and min(args.roi) < 16:
        parser.error("ROI dimensions must be >=16")
    for values, name in (
        (args.target_spacing, "target-spacing"),
        (
            [args.gpu_memory_target, args.lr, args.grad_clip, args.poly_power, args.ssim_data_range],
            "training/planning parameters",
        ),
    ):
        if values and any(not math.isfinite(value) or value <= 0 for value in values):
            parser.error(f"{name} must be positive and finite")
    if not 0 < args.momentum < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("momentum must be in (0,1); weight-decay must be finite and nonnegative")
    if args.filters and min(args.filters) <= 0:
        parser.error("filters must be positive")
    if args.mask_ratio is not None and not 0 < args.mask_ratio < 1:
        parser.error("mask-ratio must be in (0,1)")
    if not 0 < args.mask_range[0] <= args.mask_range[1] < 1 or any(not 0 < ratio < 1 for ratio in args.val_mask_ratios):
        parser.error("mask ranges and validation ratios must lie in (0,1)")
    if args.resume and args.command == "plan":
        parser.error("--resume is only supported for training")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_determinism(seed=args.seed)
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else output / "manifest.json"
    manifest, identity, manifest_base = load_manifest(manifest_path)
    last = output / "last.pt"
    if args.command == "train":
        if last.exists() and not args.resume:
            raise FileExistsError(f"{last} exists; use --resume or a new --output.")
        if args.resume and not last.is_file():
            raise FileNotFoundError(f"Cannot resume: {last} is missing.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    # These roots only keep native imports quiet: MAE planning requires no
    # segmentation dataset directories and creates no synthetic labels.
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(name, str(output / name))
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "mae_plans.json"
    memory_check = None
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        if (
            plan.get("format") != FORMAT
            or plan["dataset_identity"] != identity
            or plan["planning_settings"] != json_ready(planning_settings(args))
        ):
            raise ValueError("The dataset or planning settings changed. Use the original arguments or a new --output.")
        if args.command == "train" and device == "cuda":
            memory_check = check_cuda_plan(plan, args)
    else:
        if args.resume:
            raise FileNotFoundError("Resume requires the original mae_plans.json.")
        plan = create_plan(args, manifest, identity, manifest_base, device)
        write_json(plan_path, plan)
    print(
        f"MAE plan: spacing={plan['target_spacing']}, crop={plan['patch_size']}, filters={plan['model_config']['filters']}, batch={plan['batch_size']}; memory={plan['memory']['method']}",
        flush=True,
    )
    if args.command == "plan":
        print(f"Saved plan: {plan_path}")
        return
    if args.augmentation_backend == "batchaug":
        if device != "cuda":
            raise ValueError("BatchAug augmentation requires CUDA.")
        check_cuda_cubic()
    config = MAETrainerConfig(
        output_dir=str(output),
        dataloader_config={
            "manifest_path": str(manifest_path),
            "plan": plan,
            "workers": args.workers,
            "cache_dir": str(Path(args.cache_dir).resolve()) if args.cache_dir else None,
            "val_volumes": args.val_volumes,
            "augmentation_backend": args.augmentation_backend,
        },
        trackio_config=TrackioConfig(
            project=args.trackio_project,
            run_name=args.trackio_name,
            space_id=args.trackio_space_id,
            enabled=args.trackio,
        ),
        model_config=plan["model_config"],
        plan_sha256=digest(plan),
        seed=args.seed,
        device=device,
        resume=args.resume,
        local_batch_size=plan["batch_size"],
        gradient_accumualtion_steps=args.accumulation,
        optimizer_config={
            "_target_": "torch.optim.SGD",
            "lr": args.lr,
            "momentum": args.momentum,
            "nesterov": True,
            "weight_decay": args.weight_decay,
        },
        steps=args.steps,
        epochs=args.epochs,
        mask_ratio=args.mask_ratio,
        mask_range=tuple(args.mask_range),
        grad_clip=args.grad_clip,
        poly_power=args.poly_power,
        log_every=args.log_every,
        validate_every=args.validate_every,
        save_every=args.save_every,
        val_mask_ratios=tuple(args.val_mask_ratios),
        ssim_data_range=args.ssim_data_range,
    )
    trainer = NNUNetMAETrainer(MaskedAutoEncoderDynUnet(**plan["model_config"]), config)
    recipe = {
        "args": vars(args),
        "config": asdict(config),
        "model": "MaskedAutoEncoderDynUnet",
        "plan_sha256": config.plan_sha256,
        "dataset_revision": manifest.get("revision"),
        "manifest_sha256": identity["manifest_sha256"],
        "effective_batch": trainer.effective_batch_size,
        "train_volumes": trainer.train_volume_count,
        "updates_per_epoch": trainer.updates_per_epoch,
        "validation_volume_paths": trainer.validation_volume_paths,
        "runtime_memory_check": memory_check or plan["memory"],
    }
    write_json(output / "recipe.json", recipe)
    augmenter = (
        make_batch_augmenter(tuple(plan["patch_size"]), args.seed, spacing=plan["target_spacing"])
        if args.augmentation_backend == "batchaug"
        else None
    )
    trainer.train(batch_augmenter=augmenter, tracking_config=json_ready(recipe))


if __name__ == "__main__":
    main()
