"""Train PROTEAS segmentation with a random or pretrained DyneMAE backbone.

Install: uv pip install huggingface_hub safetensors nibabel scipy
Optional experiment logging: uv pip install 'trackio>=0.43.0'

Train the full model from scratch (default; no pretrained model download):
    python train_dynmae_semantic_segmentation.py --output ./proteas_scratch
Freeze a pretrained encoder, train a new decoder and output head:
    python train_dynmae_semantic_segmentation.py \
        --model geekyrakshit/DyneMAE-Yale-Brain-Mets --output ./proteas_decoder
Freeze pretrained encoder AND decoder, train only the output head:
    python train_dynmae_semantic_segmentation.py \
        --model geekyrakshit/DyneMAE-Yale-Brain-Mets --train-mode head --output ./proteas_head
Fine-tune the full pretrained model: encoder, decoder, and output head:
    python train_dynmae_semantic_segmentation.py \
        --model geekyrakshit/DyneMAE-Yale-Brain-Mets --train-mode full --reuse-decoder --output ./proteas_full
With --model, decoder and full modes initialize a new decoder by default.
Add --reuse-decoder to initialize the trainable decoder with pretrained weights.
Without --model, all components are randomly initialized and trainable, using
six stages with filters 32,64,128,256,320,320 and strides 1,2,2,2,2,2.
--resume restores the saved model and optimizer instead of reinitializing them.

Loss defaults to Dice + cross-entropy. Use --loss dice-focal for foreground
soft Dice + four-class softmax focal loss (gamma=2), with equal coefficients.

Default channel order: T1, T1ce, T2, FLAIR. With a frozen expanded input stem,
the repeated/scaled input kernels retain their initial averaging of modalities.
Use --modalities t1ce for a single-channel input. Use --manifest for an existing
prepare_segmentation.py manifest/local mirror. By default, train on train and
validate on validation. Add --train-on-train-val to train on train+validation
and use test for validation and checkpoint selection.

Add --trackio to log configuration, metrics, and best-checkpoint artifacts.
Training metrics use train/ and validation metrics use val/ in all metric logs.
val/best_dice is logged only after the first validation result is available.
Trackio uses local storage by default (configure with TRACKIO_DIR). Pass
--trackio-space-id username/space to log to a Hugging Face Space instead.

New runs use mild intensity/affine augmentation and class-balanced crops.
Use --augmentation none for crops without other random augmentations, or
--augmentation legacy to resume the original tumor/background + flip/90-degree
pipeline. Validation always uses deterministic preprocessing only.

Download/restore the portable prepared dataset automatically:
    python train_dynmae_semantic_segmentation.py \
        --dataset geekyrakshit/PROTEAS-Brain-Metastases-Segmentation-Prepared
The archive preserves original NIfTIs; deterministic preprocessing still runs
locally. Add --cache-dir to cache it across epochs and runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
from nibabel.spatialimages import SpatialImage
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from scipy.ndimage import binary_fill_holes
from tqdm.auto import tqdm

from archive_proteas_segmentation import ARCHIVE_FORMAT, restore_files, safe_relative
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import MaskedAutoEncoderDynUnet
from monai.transforms import (
    Compose,
    EnsureTyped,
    MapTransform,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    SpatialPadd,
)
from monai.utils import optional_import, set_determinism

DATASET = "geekyrakshit/PROTEAS-Brain-Metastases-Segmentation"
SCRATCH_MODEL_KWARGS = {
    "spatial_dims": 3,
    "kernel_size": (3, 3, 3, 3, 3, 3),
    "strides": (1, 2, 2, 2, 2, 2),
    "filters": (32, 64, 128, 256, 320, 320),
    "masking_ratio": 0.75,
    "enable_activation_checkpointing": True,
}
MODALITIES = ("t1", "t1ce", "t2", "flair")
LABELS = {"0": "background", "1": "necrotic_core", "2": "enhancing_tumor", "3": "edema"}
FORMAT = "DynMAESegmentation-v1"
Case = dict[str, Any]
LEGACY_AUGMENTATION = {
    "version": 1,
    "profile": "legacy",
    "crop_sampling": "tumor_background",
    "tumor_background_ratio": [3, 1],
    "flip_probability": 0.5,
    "rotate90_probability": 0.2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def start_tracking(args: argparse.Namespace, output: Path, recipe: dict[str, Any]):
    """Initialize optional Trackio logging and retain the run identity for resume."""
    if not args.trackio:
        return None
    trackio, available = optional_import("trackio", version="0.43.0")
    if not available:
        raise ImportError("Trackio logging requires `uv pip install 'trackio>=0.43.0'`.")
    metadata_path = output / "trackio_run.json"
    previous = json.loads(metadata_path.read_text()) if args.resume and metadata_path.exists() else {}
    same_project = previous.get("project") == args.trackio_project
    name = args.trackio_name or (previous.get("name") if same_project else None)
    space_id = args.trackio_space_id or (previous.get("space_id") if same_project else None)
    run = trackio.init(
        project=args.trackio_project,
        name=name,
        space_id=space_id,
        config=recipe,
        resume="allow" if args.resume else "never",
        embed=False,
    )
    write_json(metadata_path, {"project": args.trackio_project, "name": run.name, "id": run.id, "space_id": space_id})
    return run


def split_sources(args: argparse.Namespace) -> dict[str, list[str]]:
    if args.train_on_train_val:
        return {"train": ["train", "validation"], "validation": ["test"]}
    return {"train": ["train"], "validation": ["validation"]}


def select_split_rows(source_partitions: dict[str, list[Case]], args: argparse.Namespace) -> dict[str, list[Case]]:
    """Assign published splits to training/evaluation roles, then apply limits."""
    result = {}
    for role, sources in split_sources(args).items():
        rows = []
        for source in sources:
            if not source_partitions.get(source):
                raise ValueError(f"Required source split '{source}' is missing or empty.")
            rows.extend({**row, "source_split": source} for row in source_partitions[source])
        limit = args.max_train_cases if role == "train" else args.max_val_cases
        result[role] = rows[:limit] if limit else rows
    return result


def prepare_dataset(args: argparse.Namespace) -> Path:
    """Pin a revision and fetch the files required by the selected split policy."""
    output = Path(args.data_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "manifest.json"
    existing = destination.exists()
    if existing:
        manifest = json.loads(destination.read_text())
        if args.modalities and args.modalities != manifest["channel_order"]:
            raise ValueError("--modalities must match the existing manifest channel_order.")
        if manifest.get("subset_for_smoke_test") and not (args.max_train_cases or args.max_val_cases):
            raise ValueError("This manifest is a smoke-test subset. Use a new --data-dir for the complete dataset.")
    else:
        revision = HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
        source = Path(hf_hub_download(args.dataset, "manifest.json", repo_type="dataset", revision=revision))
        manifest = json.loads(source.read_text())
        modalities = args.modalities or list(MODALITIES)
        indices = [manifest["channel_order"].index(modality) for modality in modalities]
        source_counts = {name: len(rows) for name, rows in manifest["partitions"].items()}
        for split, rows in manifest["partitions"].items():
            manifest["partitions"][split] = [{**row, "image": [row["image"][i] for i in indices]} for row in rows]
        manifest.update(
            repo_id=args.dataset,
            revision=revision,
            channel_order=modalities,
            path_base=".",
            source_split_counts=source_counts,
            subset_for_smoke_test=bool(args.max_train_cases or args.max_val_cases),
        )

    selected = select_split_rows(manifest["partitions"], args)
    names = {name for rows in selected.values() for row in rows for name in [*row["image"], row["label"]]}
    # A previously prepared train/validation directory may still need test files.
    # Keep its manifest bytes intact so existing checkpoint signatures stay valid.
    required = sorted(name for name in names if not existing or not (output / name).is_file())
    checksums: dict[str, str] = {}
    if required:
        revision = manifest.get("revision")
        if not revision:
            raise ValueError("The existing manifest needs missing files but has no pinned dataset revision.")
        repo = manifest.get("repo_id", args.dataset)
        checksum_file = Path(hf_hub_download(repo, "SHA256SUMS", repo_type="dataset", revision=revision))
        descriptor = manifest.get("prepared_archive")
        if descriptor and sha256(checksum_file) != descriptor["checksums_sha256"]:
            raise ValueError("Prepared dataset checksum index SHA-256 mismatch.")
        checksums = {
            name.lstrip(" *"): digest
            for digest, name in (
                line.split(maxsplit=1) for line in checksum_file.read_text().splitlines() if line.strip()
            )
        }
        downloads = {name: checksums[name] for name in required}
    else:
        downloads = {}

    if downloads and manifest.get("prepared_archive"):
        descriptor = manifest["prepared_archive"]
        if descriptor["format"] != ARCHIVE_FORMAT:
            raise ValueError("Unsupported prepared archive format.")
        archive_path = Path(
            hf_hub_download(
                manifest.get("repo_id", args.dataset),
                safe_relative(descriptor["filename"]),
                repo_type="dataset",
                revision=manifest["revision"],
            )
        )
        restore_files(archive_path, output, manifest, checksums, names=required)
        if not existing:
            write_json(destination, manifest)
        return destination

    def fetch(item: tuple[str, str]) -> None:
        name, expected = item
        local = output / name
        if local.is_file() and sha256(local) == expected:
            return
        downloaded = Path(
            hf_hub_download(
                manifest.get("repo_id", args.dataset),
                name,
                repo_type="dataset",
                revision=manifest["revision"],
                local_dir=output,
                force_download=local.exists(),
            )
        )
        if sha256(downloaded) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")

    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for _ in tqdm(pool.map(fetch, downloads.items()), total=len(downloads), desc="Download/verify NIfTIs"):
            pass
    if not existing:
        write_json(destination, manifest)
    return destination


def read_manifest(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[Case]]]:
    manifest = json.loads(path.read_text())
    channels = manifest["channel_order"]
    if not channels or len(set(channels)) != len(channels) or not set(channels).issubset(MODALITIES):
        raise ValueError("Manifest channel_order must contain unique PROTEAS modality names.")
    if args.modalities and args.modalities != channels:
        raise ValueError("--modalities must match the manifest channel_order; prepare a different data directory.")
    if manifest.get("labels") != LABELS:
        raise ValueError("Expected PROTEAS labels 0 background, 1 necrotic core, 2 enhancing tumor, 3 edema.")
    if args.dataset_revision and manifest.get("revision") != args.dataset_revision:
        resolved = HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
        if manifest.get("revision") != resolved:
            raise ValueError("The existing manifest does not match --dataset-revision.")
    if manifest.get("subset_for_smoke_test") and not (args.max_train_cases or args.max_val_cases):
        raise ValueError("This manifest is a smoke-test subset. Use a new --data-dir for the complete dataset.")
    source_partitions = manifest["partitions"]
    patients = {split: {row["patient_id"] for row in rows} for split, rows in source_partitions.items()}
    names = list(patients)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if patients[first] & patients[second]:
                raise ValueError(f"Patient leakage between {first} and {second}.")
    path_base = manifest.get("path_base", ".")
    base = path.parent if path_base in (".", "repository root") else path.parent / path_base
    partitions: dict[str, list[Case]] = {}
    for split, rows in select_split_rows(source_partitions, args).items():
        partitions[split] = []
        for row in rows:
            if len(row["image"]) != len(channels):
                raise ValueError(f"Channel count mismatch for {row['case_id']}.")
            record = {
                **row,
                "image": [str((base / p).resolve()) for p in row["image"]],
                "label": str((base / row["label"]).resolve()),
            }
            for name in [*record["image"], record["label"]]:
                if not Path(name).is_file():
                    raise FileNotFoundError(f"Missing NIfTI: {name}")
            partitions[split].append(record)
    return manifest, partitions


class LoadProteasCased(MapTransform):
    """Match MAE native Z,Y,X axes, nonzero crop, and per-channel whole-crop z-score.

    The published volumes are already registered at 1 mm. Labels receive the
    identical transpose/crop and retain integer IDs without interpolation.
    """

    def __init__(self) -> None:
        super().__init__(("image", "label"))

    def __call__(self, data: Case) -> Case:
        label_nifti = nib.load(data["label"])
        if not isinstance(label_nifti, SpatialImage):
            raise ValueError(f"Expected a spatial NIfTI label: {data['case_id']}")
        label_affine = label_nifti.affine
        if label_affine is None:
            raise ValueError(f"Missing label affine: {data['case_id']}")
        label = np.asarray(label_nifti.dataobj, dtype=np.float32)
        if label.ndim != 3 or not np.isfinite(label).all() or not np.isin(label, [0, 1, 2, 3]).all():
            raise ValueError(f"Invalid segmentation labels: {data['case_id']}")
        images = []
        for name in data["image"]:
            image = nib.load(name)
            if not isinstance(image, SpatialImage):
                raise ValueError(f"Expected a spatial NIfTI image: {name}")
            image_affine = image.affine
            if (
                image_affine is None
                or image.shape != label.shape
                or not np.allclose(image_affine, label_affine, atol=1e-4)
            ):
                raise ValueError(f"Image/label geometry mismatch: {name}")
            if not np.allclose(image.header.get_zooms()[:3], (1, 1, 1), atol=1e-4):
                raise ValueError(f"Expected the published 1 mm PROTEAS grid: {name}")
            volume = image.get_fdata(dtype=np.float32)
            if not np.isfinite(volume).all():
                raise ValueError(f"Nonfinite MRI intensities: {name}")
            images.append(volume.transpose(2, 1, 0))
        image_array = np.stack(images)
        label = label.transpose(2, 1, 0)
        foreground = np.empty(image_array.shape[1:], dtype=np.bool_)
        binary_fill_holes(np.any(image_array != 0, axis=0), output=foreground)
        coordinates = np.nonzero(foreground)
        if not coordinates[0].size:
            raise ValueError(f"Empty MRI foreground: {data['case_id']}")
        bbox = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
        cropped_label = label[bbox]
        if np.count_nonzero(cropped_label) != np.count_nonzero(label):
            raise ValueError(f"Image foreground crop would discard tumor labels: {data['case_id']}")
        image_array = image_array[(slice(None), *bbox)].copy()
        mean = image_array.mean(axis=(1, 2, 3), keepdims=True)
        std = image_array.std(axis=(1, 2, 3), keepdims=True)
        image_array = (image_array - mean) / np.maximum(std, 1e-8)
        return {
            "image": image_array,
            "label": cropped_label[None].astype(np.int64),
            "case_id": data["case_id"],
            "patient_id": data["patient_id"],
        }


class RandGaussianNoiseRanged(RandGaussianNoised):
    """Dictionary Gaussian noise with standard deviation sampled in [low, high]."""

    def __init__(self, std_range: tuple[float, float], prob: float) -> None:
        super().__init__(keys="image", prob=prob, std=std_range[1], sample_std=False)
        self.std_range = std_range

    def randomize(self, data: Any = None) -> None:
        super().randomize(data)
        if self._do_transform:
            self.rand_gaussian_noise.std = float(self.R.uniform(*self.std_range))


def augmentation_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return effective settings, excluding options unused by the chosen profile."""
    if args.augmentation == "legacy":
        return dict(LEGACY_AUGMENTATION)
    enabled = args.augmentation == "mild"
    config: dict[str, Any] = {
        "version": 1,
        "profile": args.augmentation,
        "crop_sampling": "class_balanced",
        "class_crop_weights": list(args.class_crop_weights),
        "max_samples_per_class": 100_000,
        "flip_probability": 0.5 if enabled else 0.0,
        "rotate90_probability": 0.2 if enabled else 0.0,
    }
    if enabled:
        config.update(
            affine_probability=args.affine_prob,
            rotation_degrees=args.rotation_degrees,
            scale_range=list(args.scale_range),
            noise_probability=args.noise_prob,
            noise_std_range=list(args.noise_std_range),
            gamma_probability=args.gamma_prob,
            gamma_range=list(args.gamma_range),
            gamma_retain_stats=True,
            bias_probability=args.bias_prob,
            bias_degree=3,
            bias_coeff_range=[-args.bias_coeff, args.bias_coeff],
            spatial_interpolation=["bilinear", "nearest"],
            spatial_padding="zeros",
        )
    return config


def make_transforms(roi: Sequence[int], training: bool, augmentation: dict[str, Any] | None = None) -> Compose:
    transforms: list[Any] = [LoadProteasCased()]
    if training:
        config = augmentation if augmentation is not None else augmentation_config(parse_args([]))
        crop = (
            RandCropByPosNegLabeld(
                keys=("image", "label"), label_key="label", spatial_size=roi, pos=3, neg=1, num_samples=1
            )
            if config["profile"] == "legacy"
            else RandCropByLabelClassesd(
                keys=("image", "label"),
                label_key="label",
                spatial_size=roi,
                ratios=config["class_crop_weights"],
                num_classes=4,
                num_samples=1,
                warn=False,
                max_samples_per_class=config["max_samples_per_class"],
            )
        )
        transforms.extend(
            [
                SpatialPadd(keys=("image", "label"), spatial_size=roi),
                crop,
            ]
        )
        if config["profile"] != "none":
            transforms.extend(
                [
                    *[
                        RandFlipd(keys=("image", "label"), prob=config["flip_probability"], spatial_axis=axis)
                        for axis in range(3)
                    ],
                    RandRotate90d(keys=("image", "label"), prob=config["rotate90_probability"], spatial_axes=(1, 2)),
                ]
            )
        if config["profile"] == "mild":
            angle = math.radians(config["rotation_degrees"])
            scales = tuple((config["scale_range"][0] - 1, config["scale_range"][1] - 1) for _ in range(3))
            transforms.extend(
                [
                    RandAffined(
                        keys=("image", "label"),
                        spatial_size=roi,
                        prob=config["affine_probability"],
                        rotate_range=(angle,) * 3,
                        scale_range=scales,
                        mode=("bilinear", "nearest"),
                        padding_mode="zeros",
                    ),
                    RandBiasFieldd(
                        keys="image",
                        degree=config["bias_degree"],
                        coeff_range=tuple(config["bias_coeff_range"]),
                        prob=config["bias_probability"],
                    ),
                    RandAdjustContrastd(
                        keys="image",
                        gamma=tuple(config["gamma_range"]),
                        retain_stats=True,
                        prob=config["gamma_probability"],
                    ),
                    RandGaussianNoiseRanged(
                        std_range=tuple(config["noise_std_range"]), prob=config["noise_probability"]
                    ),
                ]
            )
    transforms.extend(
        [
            EnsureTyped(keys="image", dtype=torch.float32, track_meta=False),
            EnsureTyped(keys="label", dtype=torch.long, track_meta=False),
        ]
    )
    return Compose(transforms)


def configure_trainable(model: MaskedAutoEncoderDynUnet, mode: str) -> None:
    if mode not in ("decoder", "head", "full"):
        raise ValueError("Training mode must be decoder, head, or full.")
    model.requires_grad_(mode == "full")
    model.backbone.output_block.requires_grad_(True)
    if mode == "decoder":
        model.backbone.upsamples.requires_grad_(True)


def set_training_mode(model: MaskedAutoEncoderDynUnet, mode: str) -> None:
    model.train()
    if mode != "full":
        for block in (model.backbone.input_block, model.backbone.downsamples, model.backbone.bottleneck):
            block.eval()
    if mode == "head":
        model.backbone.upsamples.eval()


def build_model(args: argparse.Namespace, channels: int, checkpoint: dict[str, Any] | None = None):
    if checkpoint is None:
        if args.model is None:
            model = MaskedAutoEncoderDynUnet(in_channels=channels, **SCRATCH_MODEL_KWARGS)
            source = {"model": None, "revision": None, "initialization": "random"}
        else:
            local = Path(args.model).expanduser().is_dir()
            revision = None if local else HfApi().model_info(args.model, revision=args.model_revision).sha
            model = MaskedAutoEncoderDynUnet.from_pretrained(args.model, revision=revision, device="cpu")
            source = {"model": args.model, "revision": revision}
        kwargs = {
            "in_channels": model.in_channels,
            "spatial_dims": model.spatial_dims,
            "kernel_size": model.kernel_size,
            "strides": model.strides,
            "filters": model.filters,
            "masking_ratio": model.masking_ratio,
            "enable_activation_checkpointing": True,
        }
    else:
        kwargs, source = checkpoint["mae_kwargs"], checkpoint["model_source"]
        model = MaskedAutoEncoderDynUnet(**kwargs)
    model.enable_activation_checkpointing = True
    model.add_segmentation_head(4, in_channels=channels, reuse_decoder=args.train_mode == "head" or args.reuse_decoder)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    configure_trainable(model, args.train_mode)
    return model, kwargs, source


class DiceSoftmaxFocalLoss(torch.nn.Module):
    """Foreground Dice plus softmax focal over all classes, including background."""

    def __init__(self) -> None:
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        # DiceFocalLoss's softmax option only affects Dice; explicitly select
        # multiclass softmax focal and retain competition with background here.
        self.focal = FocalLoss(to_onehot_y=True, use_softmax=True, include_background=True, gamma=2.0)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, labels) + self.focal(logits, labels)


def build_loss(name: str) -> torch.nn.Module:
    if name == "dice-ce":
        return DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    if name == "dice-focal":
        return DiceSoftmaxFocalLoss()
    raise ValueError(f"Unknown segmentation loss: {name}")


def train_epoch(model, loader, optimizer, scaler, loss_function, device, args) -> float:
    set_training_mode(model, args.train_mode)
    batches = min(len(loader), args.max_train_batches or len(loader))
    total, examples = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(
        tqdm(
            islice(loader, batches),
            total=batches,
            desc="Train",
            position=1,
            leave=False,
            dynamic_ncols=True,
            unit="batch",
        )
    ):
        inputs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(inputs)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("Expected segmentation logits.")
            loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite segmentation loss.")
        group_start = index // args.accumulation * args.accumulation
        group_end = min(group_start + args.accumulation, batches)
        group_samples = sum(
            min(loader.batch_size, len(loader.dataset) - n * loader.batch_size) for n in range(group_start, group_end)
        )
        scaler.scale(loss * inputs.shape[0] / group_samples).backward()
        total += float(loss.detach()) * inputs.shape[0]
        examples += inputs.shape[0]
        if (index + 1) % args.accumulation == 0 or index + 1 == batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("Frozen backbone unexpectedly received gradients.")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return total / examples


@torch.inference_mode()
def validate(model, loader, device, args) -> dict[str, float | int | None]:
    model.eval()
    metric = DiceMetric(
        include_background=False, reduction="mean_batch", get_not_nans=True, ignore_empty=True, num_classes=4
    )

    def predict(inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            prediction = model(inputs)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Expected segmentation logits.")
        return prediction.float()

    evaluation_split = "+".join(split_sources(args)["validation"])
    for batch in tqdm(
        loader, desc=f"Validate ({evaluation_split})", position=1, leave=False, dynamic_ncols=True, unit="case"
    ):
        # Full volumes and stitched logits stay on CPU; only windows go to GPU.
        logits = sliding_window_inference(
            batch["image"],
            args.roi,
            args.sw_batch_size,
            predict,
            overlap=args.overlap,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu"),
        )
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Expected sliding-window logits.")
        metric(y_pred=logits.argmax(dim=1, keepdim=True), y=batch["label"])
    aggregated = metric.aggregate()
    if not isinstance(aggregated, tuple):
        raise TypeError("Expected Dice scores and valid-case counts.")
    scores, counts = aggregated
    present = counts > 0
    if not present.any():
        raise ValueError("Validation contains no foreground labels.")
    result: dict[str, float | int | None] = {"mean_dice": float(scores[present].mean())}
    for index, name in enumerate(list(LABELS.values())[1:]):
        result[f"dice_{name}"] = float(scores[index]) if counts[index] else None
        result[f"valid_cases_{name}"] = int(counts[index])
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--train-mode",
        choices=("decoder", "head", "full"),
        help="Trainable layers: decoder freezes the encoder and trains the decoder plus output head; "
        "head freezes the pretrained encoder and decoder and trains only the output head; "
        "full trains the encoder, decoder, and output head. Default: full without --model, decoder with --model. "
        "Without --model, only full is supported",
    )
    parser.add_argument(
        "--reuse-decoder",
        action="store_true",
        help="In decoder or full mode, initialize the trainable decoder with pretrained weights instead of random weights. "
        "Requires --model. Head mode always reuses the pretrained decoder (default: disabled)",
    )
    parser.add_argument(
        "--train-on-train-val",
        action="store_true",
        help="Combine the published train and validation splits for training, and use test for validation "
        "and best-checkpoint selection. By default, train on train and validate on validation",
    )
    parser.add_argument(
        "--model",
        help="Pretrained MAE Hub model ID or local directory containing config.json and model.safetensors "
        "(default: none; randomly initialize and train the full model from scratch)",
    )
    parser.add_argument(
        "--model-revision",
        help="Hub model branch, tag, or commit to initialize from; defaults to main, resolved to an immutable commit. "
        "Used when starting a new run from a Hub model",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET,
        help="Hub dataset ID with the PROTEAS manifest and NIfTIs, or the portable -Prepared archive. "
        "Prepared archives are downloaded and restored automatically (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-revision",
        help="Dataset branch, tag, or commit; new downloads pin the resolved commit (default: main). "
        "When supplied, an existing manifest must record the matching revision",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Existing local PROTEAS manifest; image paths are resolved from its directory and path_base. "
        "If omitted, prepare or reuse DATA_DIR/manifest.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./proteas_segmentation"),
        help="Directory for the selected NIfTIs and generated manifest; missing test files are fetched when "
        "--train-on-train-val is enabled. "
        "used when --manifest is omitted (default: %(default)s)",
    )
    parser.add_argument(
        "--modalities",
        choices=MODALITIES,
        nargs="+",
        help="Unique MRI modalities in input-channel order. New preparations default to t1 t1ce t2 flair. "
        "For an existing manifest, inherit its order if omitted or specify the exact same order. "
        "Use t1ce for a single-channel input",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Number of concurrent individual NIfTI downloads; prepared archives restore sequentially "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download and verify the selected data, or validate an existing manifest, then exit before "
        "loading the model or training (default: disabled)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./proteas_dynmae_segmentation"),
        help="Directory for best.pt, last.pt, recipe.json, and metrics.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total training epochs, including completed epochs when resuming; also sets the cosine "
        "learning-rate schedule length (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training cases per microbatch, with one random crop per case; validation uses one full volume "
        "at a time (default: %(default)s)",
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=1,
        help="Microbatches accumulated per optimizer update; nominal effective batch size is batch-size times "
        "accumulation. Final partial groups use their actual sample count (default: %(default)s)",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=3,
        default=(160, 160, 160),
        metavar=("D", "H", "W"),
        help="Training crop and validation sliding-window dimensions in voxels, in native Z,Y,X order. "
        "Each dimension must be at least 64 and divisible by 32 (default: %(default)s)",
    )
    parser.add_argument(
        "--augmentation",
        choices=("mild", "legacy", "none"),
        default="mild",
        help="Training augmentation profile: mild adds class-balanced crops, affine, bias field, gamma and noise; "
        "legacy preserves the original tumor/background crops, flips and 90-degree rotations; "
        "none keeps class-balanced random crops only. Validation is never augmented (default: %(default)s)",
    )
    parser.add_argument(
        "--class-crop-weights",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 1.0, 1.0),
        metavar=("BG", "CORE", "ENHANCING", "EDEMA"),
        help="Positive crop-center weights for labels 0,1,2,3 in mild/none profiles. "
        "Classes absent from a case are skipped and remaining weights are normalized (default: %(default)s)",
    )
    parser.add_argument(
        "--affine-prob",
        type=float,
        default=0.2,
        help="Probability of a shared image/mask rotation and scaling in the mild profile (default: %(default)s)",
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=10.0,
        help="Maximum absolute affine rotation per spatial axis, in degrees, in the mild profile; "
        "must be in [0,180] (default: %(default)s)",
    )
    parser.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        default=(0.9, 1.1),
        metavar=("MIN", "MAX"),
        help="Positive ordered affine scale factors, sampled independently per axis in the mild profile "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.15,
        help="Probability of adding image-only Gaussian noise in the mild profile (default: %(default)s)",
    )
    parser.add_argument(
        "--noise-std-range",
        type=float,
        nargs=2,
        default=(0.01, 0.05),
        metavar=("MIN", "MAX"),
        help="Ordered nonnegative Gaussian-noise standard deviations, sampled uniformly after intensity "
        "normalization in the mild profile (default: %(default)s)",
    )
    parser.add_argument(
        "--gamma-prob",
        type=float,
        default=0.15,
        help="Probability of image-only gamma contrast adjustment in the mild profile (default: %(default)s)",
    )
    parser.add_argument(
        "--gamma-range",
        type=float,
        nargs=2,
        default=(0.8, 1.2),
        metavar=("MIN", "MAX"),
        help="Positive ordered gamma bounds in the mild profile; the transform retains intensity mean "
        "and standard deviation (default: %(default)s)",
    )
    parser.add_argument(
        "--bias-prob",
        type=float,
        default=0.1,
        help="Probability of a smooth image-only MRI bias field in the mild profile (default: %(default)s)",
    )
    parser.add_argument(
        "--bias-coeff",
        type=float,
        default=0.03,
        help="Nonnegative maximum absolute coefficient of the degree-3 MRI bias field in the mild profile "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--loss",
        choices=("dice-ce", "dice-focal"),
        default="dice-ce",
        help="Foreground soft Dice plus cross-entropy or softmax focal (gamma=2) over all four classes, "
        "with equal coefficients. Must match when resuming (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial AdamW learning rate, decayed with a cosine schedule over the total epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Nonnegative AdamW weight decay applied to trainable parameters (default: %(default)s)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Positive maximum global gradient norm for trainable parameters, applied after AMP unscaling "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes per split; use 0 to load data in the main process (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional directory for a persistent disk cache of deterministically preprocessed volumes; "
        "random training augmentations still run each epoch. Disabled if omitted",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Validate every N epochs and at the final epoch; best.pt is selected by mean foreground Dice "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="Validation sliding windows evaluated together on the model device; full volumes and stitched "
        "outputs remain on CPU (default: %(default)s)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Fractional overlap between validation windows, in [0,1); predictions use Gaussian blending "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA mixed precision for training and validation, with gradient scaling during training. "
        "Use --no-amp to disable; CPU execution uses full precision (default: enabled)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Model execution device; auto selects CUDA when available and CPU otherwise (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model initialization, data shuffling, and augmentation setup (default: %(default)s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume OUTPUT/last.pt, restoring model, optimizer, scheduler, AMP scaler, and shuffle state. "
        "Requires the original manifest and matching training configuration (default: disabled)",
    )
    parser.add_argument(
        "--max-train-cases",
        type=int,
        help="Use only the first N training cases in manifest order (train followed by validation when "
        "--train-on-train-val is enabled) for development/smoke tests; "
        "defaults to all training cases and marks limited runs as smoke tests",
    )
    parser.add_argument(
        "--max-val-cases",
        type=int,
        help="Use only the first N evaluation cases in manifest order for development/smoke tests; "
        "the evaluation source is test with --train-on-train-val and validation otherwise. "
        "Defaults to all evaluation cases and marks limited runs as smoke tests",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Limit training to N microbatches per epoch for development/smoke tests; "
        "defaults to the complete training loader",
    )
    parser.add_argument(
        "--trackio",
        action="store_true",
        help="Log configuration, epoch metrics, and best-checkpoint artifacts with Trackio. Uses local storage "
        "by default; TRACKIO_DIR configures its location (default: disabled)",
    )
    parser.add_argument(
        "--trackio-project",
        default="proteas-segmentation",
        help="Trackio project used when --trackio is enabled (default: %(default)s)",
    )
    parser.add_argument(
        "--trackio-space-id",
        help="Optional Hugging Face Space, such as username/space, for Trackio logging; created if missing. "
        "If omitted, uses Trackio's environment settings or local storage",
    )
    parser.add_argument(
        "--trackio-name",
        help="Trackio run name; generated if omitted on a new run. With --resume, reuses the name saved in "
        "OUTPUT/trackio_run.json when the project matches",
    )
    args = parser.parse_args(argv)
    if args.train_mode is None:
        args.train_mode = "full" if args.model is None else "decoder"
    if args.model is None:
        if args.train_mode != "full":
            parser.error("--train-mode decoder/head requires --model; scratch training uses --train-mode full")
        if args.reuse_decoder or args.model_revision is not None:
            parser.error("--reuse-decoder and --model-revision require --model")
    for name in ("epochs", "batch_size", "accumulation", "download_workers", "val_every", "sw_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train_cases", "max_val_cases", "max_train_batches"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or any(n < 64 or n % 32 for n in args.roi):
        parser.error("workers must be nonnegative; ROI dimensions must be >=64 and divisible by 32")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        parser.error("lr and grad-clip must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.overlap < 1:
        parser.error("weight-decay must be finite and nonnegative; overlap must be in [0,1)")
    if args.modalities and len(set(args.modalities)) != len(args.modalities):
        parser.error("modalities must be unique")
    for name in ("affine_prob", "noise_prob", "gamma_prob", "bias_prob"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    for name, allow_zero in (("noise_std_range", True), ("gamma_range", False), ("scale_range", False)):
        low, high = getattr(args, name)
        if not (math.isfinite(low) and math.isfinite(high) and low <= high and (low >= 0 if allow_zero else low > 0)):
            parser.error(
                f"--{name.replace('_', '-')} requires ordered finite {'nonnegative' if allow_zero else 'positive'} bounds"
            )
    if not math.isfinite(args.rotation_degrees) or not 0 <= args.rotation_degrees <= 180:
        parser.error("--rotation-degrees must be in [0,180]")
    if not math.isfinite(args.bias_coeff) or args.bias_coeff < 0:
        parser.error("--bias-coeff must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in args.class_crop_weights):
        parser.error("--class-crop-weights must contain four finite positive values")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    last = output / "last.pt"
    if not args.prepare_only and last.exists() and not args.resume:
        raise FileExistsError(f"{last} exists; use --resume or a new --output.")
    if args.resume and not last.is_file():
        raise FileNotFoundError(f"Cannot resume: {last} is missing.")
    manifest_path = args.manifest.resolve() if args.manifest else prepare_dataset(args)
    manifest, partitions = read_manifest(manifest_path, args)
    sources = split_sources(args)
    evaluation_split = "+".join(sources["validation"])
    print(
        f"Cases: { {k: len(v) for k, v in partitions.items()} }; channels: {manifest['channel_order']}; "
        f"training sources: {'+'.join(sources['train'])}; validation source: {evaluation_split}",
        flush=True,
    )
    if args.prepare_only:
        print(f"Manifest: {manifest_path}")
        return
    set_determinism(seed=args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")
    checkpoint = torch.load(last, map_location="cpu", weights_only=True) if args.resume else None
    augmentation = augmentation_config(args)
    signature = {
        "manifest_sha256": sha256(manifest_path),
        "channels": manifest["channel_order"],
        "train_mode": args.train_mode,
        "reuse_decoder": args.train_mode == "head" or args.reuse_decoder,
        "roi": list(args.roi),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "accumulation": args.accumulation,
        "seed": args.seed,
        "max_train_cases": args.max_train_cases,
        "max_val_cases": args.max_val_cases,
        "max_train_batches": args.max_train_batches,
        "split_sources": sources,
        "augmentation": augmentation,
    }
    if checkpoint is not None:
        previous_signature = dict(checkpoint.get("signature", {}))
        # Before --loss was introduced, all checkpoints used Dice + CE.
        previous_signature.setdefault("loss", "dice-ce")
        # Checkpoints predating the split flag always used train -> validation.
        previous_signature.setdefault("split_sources", {"train": ["train"], "validation": ["validation"]})
        if "augmentation" not in previous_signature:
            if args.augmentation != "legacy":
                raise ValueError("This checkpoint predates augmentation settings; resume with --augmentation legacy.")
            previous_signature["augmentation"] = dict(LEGACY_AUGMENTATION)
        if checkpoint.get("format") != FORMAT or previous_signature != signature:
            raise ValueError(
                "Resume requires a matching segmentation checkpoint, manifest, and training configuration."
            )
    model, mae_kwargs, model_source = build_model(args, len(manifest["channel_order"]), checkpoint)
    model.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    for split in ("train", "validation"):
        transforms = make_transforms(args.roi, training=split == "train", augmentation=augmentation)
        if args.cache_dir:
            cache_config = {
                "manifest": signature["manifest_sha256"],
                "roi": list(args.roi),
                "augmentation": augmentation if split == "train" else None,
            }
            cache_key = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()[:16]
            cache = args.cache_dir / f"{cache_key}-{split}"
            dataset = PersistentDataset(partitions[split], transforms, cache_dir=cache)
        else:
            dataset = Dataset(partitions[split], transforms)
        options: dict[str, Any] = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
        if args.workers:
            options.update(multiprocessing_context="spawn", persistent_workers=True)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else 1,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            **options,
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch, best_dice = 0, -1.0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        generator.set_state(checkpoint["shuffle_rng"])
        start_epoch, best_dice = checkpoint["epoch"], checkpoint["best_dice"]
    del checkpoint
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    recipe = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "signature": signature,
        "mae_kwargs": mae_kwargs,
        "model_source": model_source,
        "dataset": manifest.get("repo_id", args.dataset),
        "dataset_revision": manifest.get("revision"),
        "source_dataset": manifest.get("source_dataset"),
        "labels": LABELS,
        "cases": {k: len(v) for k, v in partitions.items()},
        "split_sources": sources,
        "augmentation": augmentation,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "preprocessing": "native_ZYX_nonzero_union_crop_per_channel_whole_crop_zscore_1mm",
        "validation": "full_cropped_volumes_sliding_window; foreground macro Dice; empty targets ignored",
        "smoke_test": bool(args.max_train_cases or args.max_val_cases or args.max_train_batches),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "recipe.json", recipe)
    print(f"Mode: {args.train_mode}; trainable: {trainable:,}; frozen: {frozen:,}; device: {device}", flush=True)
    loss_function = build_loss(args.loss).to(device)
    run = start_tracking(args, output, recipe)
    try:
        for epoch in tqdm(
            range(start_epoch, args.epochs),
            total=args.epochs,
            initial=start_epoch,
            desc="Epochs",
            position=0,
            dynamic_ncols=True,
            unit="epoch",
        ):
            metrics: dict[str, Any] = {"epoch": epoch + 1, "train/lr": optimizer.param_groups[0]["lr"]}
            metrics["train/loss"] = train_epoch(model, loaders["train"], optimizer, scaler, loss_function, device, args)
            improved = False
            if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
                metrics.update(
                    {
                        f"val/{name}": value
                        for name, value in validate(model, loaders["validation"], device, args).items()
                    }
                )
                metrics["val/split"] = evaluation_split
                improved = metrics["val/mean_dice"] > best_dice
                best_dice = max(best_dice, metrics["val/mean_dice"])
            scheduler.step()
            # Keep the pre-validation sentinel in checkpoint state only: emitting
            # it as a metric makes chart smoothing produce negative Dice scores.
            if best_dice >= 0:
                metrics["val/best_dice"] = best_dice
            state = {
                "format": FORMAT,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_dice": best_dice,
                "metrics": metrics,
                "mae_kwargs": mae_kwargs,
                "model_source": model_source,
                "signature": signature,
                "recipe": recipe,
                "shuffle_rng": generator.get_state(),
            }
            for target in [output / "best.pt", last] if improved else [last]:
                temporary = target.with_suffix(".pt.tmp")
                torch.save(state, temporary)
                temporary.replace(target)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            tqdm.write(json.dumps(metrics, allow_nan=False))
            if run is not None:
                run.log(metrics, step=epoch + 1)
                if improved:
                    run.log_artifact(output / "best.pt", name=f"run-{run.id}-best", type="model", aliases=["best"])
    finally:
        if run is not None:
            run.finish()
    print(f"Best validation Dice ({evaluation_split}): {best_dice:.6f}; checkpoints: {output}", flush=True)


if __name__ == "__main__":
    main()
