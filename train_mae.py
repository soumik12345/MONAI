"""Spark3D-style masked pretraining adapted to MONAI's residual DynUNet.

Reference: https://arxiv.org/html/2410.23132v3, Sections 2 and 4.
Prepare the dataset with prepare_mae.py before training.

    python prepare_mae.py --output ./yale_mae --max-patients 20
    python train_mae.py train --output ./yale_mae --batch-size 1 --accumulation 6
    python train_mae.py train --output ./yale_mae --batch-size 1 --accumulation 6 --resume
    python train_mae.py train --output ./yale_mae --wandb --wandb-project yale-mae
    python train_mae.py export --output ./yale_mae --out-channels 2
    python train_mae.py smoke-test
    python train_mae.py smoke-test --device cuda --augmentation-backend batchaug

Paper-aligned defaults: 1 mm spacing, 160^3 crops, bottleneck-aligned 32^3
masks, U[0.6,0.9] masking, masked-voxel MSE, SGD/Nesterov (LR .01,
momentum .99, weight decay 3e-5), polynomial decay (.9), 250,000 updates
and an effective batch size of six. Skip connections are retained.

The backbone is DynUNet, not the authors' ResEnc-L. Encoder convolutions
are re-masked, instance statistics use only visible positions, and all
feature levels receive learned tokens and densification (no convolution
at the highest resolution). Operations use dense tensors, not sparse kernels.
MONAI affine augmentation and foreground z-scoring replace nnU-Net's
preprocessing/augmentation implementation. The paper does not give exact
affine ranges: this starter uses +/-10 degrees and scales .9--1.1.
Optional --augmentation-backend batchaug moves affine and flips onto the
CUDA batch after loading and cropping, before drawing the MAE mask. Install
BatchAug from https://github.com/halleewong/batchaug to use that option.
Yale's PRE/POST/T2/FLAIR are sampled as independent single-channel volumes.
Patients remain disjoint across splits, and only train patients are pretrained.
Training logs masked MSE and MAE. Validation additionally logs foreground
masked MSE and masked SSIM, grouped by sequence and fixed mask ratios
(--val-mask-ratios, default .6 .75 .9). SSIM uses uniform 7^3 windows entirely
inside hidden blocks, clipping z-scored intensities to [-5,5] by default
(--ssim-data-range 10). Foreground is the nonzero mask before normalization,
not a brain segmentation. Additional fixed-ratio passes increase validation cost.
Existing single-sequence manifests still work; prepare a new output directory
to include all sequences. Original baseline checkpoints are incompatible.

Fine-tuning: export transfers both encoder and decoder into a regular DynUNet
with a fresh task head, excluding pretraining-only densification modules.
Use the same preprocessing, unmasked inputs, and a supervised warm-up (12,500
updates to LR 1e-3) before full-network fine-tuning; do not freeze the encoder.
The final pretraining checkpoint is the replication endpoint; reconstruction
validation alone does not establish downstream segmentation improvements.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from uuid import uuid4

import batchaug
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import RandomSampler
from monai.networks.nets import DynUNet
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.utils import set_determinism
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandSpatialCropd,
    Spacingd,
    SpatialPadd,
)
from monai.data.utils import pickle_hashing

FILTERS = (32, 64, 128, 256, 320, 320)
DOWNSAMPLE = 32
STEPS_PER_VIRTUAL_EPOCH = 250
CHECKPOINT_VERSION = 2


def build_model(out_channels=1, filters=FILTERS):
    """Build the ordinary, dense DynUNet used for downstream fine-tuning."""
    strides = [1, 2, 2, 2, 2, 2]
    if len(filters) != len(strides) or any(f <= 0 for f in filters):
        raise ValueError("Provide six positive filter widths.")
    return DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        kernel_size=[3] * len(strides),
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=filters,
        norm_name=("INSTANCE", {"affine": True}),
        res_block=True,
        deep_supervision=False,
    )


def block_mask(x, block=DOWNSAMPLE, ratio=None, generator=None, coarse=False):
    """1 denotes hidden; sample one U[.6,.9] ratio per batch, independent locations."""
    shape = x.shape[2:]
    if x.ndim != 5 or x.shape[1] != 1 or block <= 0 or any(s <= 0 or s % block for s in shape):
        raise ValueError("3D crop dimensions must be divisible by block size.")
    if ratio is None:
        ratio = 0.6 + 0.3 * torch.rand((), device=x.device, generator=generator).item()
    if not 0 < ratio < 1:
        raise ValueError("Mask ratio must be in (0,1).")
    grid = tuple(s // block for s in shape)
    n = grid[0] * grid[1] * grid[2]
    # Preserve >=2 visible bottleneck positions for meaningful instance statistics.
    if n < 3:
        raise ValueError("Need at least three bottleneck positions.")
    count = min(n - 2, max(1, round(n * ratio)))
    order = torch.rand(x.shape[0], n, device=x.device, generator=generator).argsort(dim=1)
    mask = torch.zeros(x.shape[0], n, device=x.device)
    mask.scatter_(1, order[:, :count], 1)
    mask = mask.reshape(x.shape[0], 1, *grid)
    return mask if coarse else expand_mask(mask, shape)


def expand_mask(mask, shape):
    """Nearest-neighbor expansion with exact block alignment at every level."""
    if any(s < m or s % m for s, m in zip(shape, mask.shape[2:])):
        raise ValueError("Feature dimensions must be multiples of the bottleneck grid.")
    return F.interpolate(mask.float(), size=shape, mode="nearest")


def masked_instance_norm(x, norm, visible):
    """Biased per-sample/channel variance over visible spatial positions only.

    Use FP32 reductions under AMP; apply the existing DynUNet affine parameters.
    Explicit masks avoid mutable global state and support activation checkpointing.
    """
    work = x.float()
    active = visible.float()
    count = active.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1)
    mean = (work * active).sum(dim=(2, 3, 4), keepdim=True) / count
    variance = ((work - mean).square() * active).sum(dim=(2, 3, 4), keepdim=True) / count
    out = (work - mean) * torch.rsqrt(variance + norm.eps)
    if norm.affine:
        out = out * norm.weight.float()[None, :, None, None, None]
        out = out + norm.bias.float()[None, :, None, None, None]
    return (out * active).to(x.dtype)


def masked_resblock(block, x, visible_in, visible_out):
    """The MONAI UnetResBlock operations, with masks on EVERY encoder convolution."""
    x = x * visible_in.to(x.dtype)
    out = block.conv1(x)
    out = out * visible_out.to(out.dtype)
    out = block.lrelu(masked_instance_norm(out, block.norm1, visible_out))
    out = block.conv2(out)
    out = out * visible_out.to(out.dtype)
    out = masked_instance_norm(out, block.norm2, visible_out)
    residual = x
    if hasattr(block, "conv3"):
        residual = block.conv3(residual)
        residual = residual * visible_out.to(residual.dtype)
        residual = masked_instance_norm(residual, block.norm3, visible_out)
    return block.lrelu(out + residual) * visible_out.to(out.dtype)


def densify(feature, visible, norm, token, projection):
    feature = masked_instance_norm(feature, norm, visible)
    feature = torch.where(visible.bool(), feature, token.to(feature.dtype))
    return projection(feature)


class DynUNetMAE(nn.Module):
    """Preserve the MONAI backbone weights while adapting its encoder for MAE."""

    def __init__(self, filters=FILTERS, checkpointing=True):
        super().__init__()
        self.filters = tuple(filters)
        self.checkpointing = checkpointing
        self.backbone = build_model(filters=self.filters)
        self.mask_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, c, 1, 1, 1)) for c in filters])
        self.densify_norms = nn.ModuleList([nn.InstanceNorm3d(c, affine=False) for c in filters])
        self.densify_projections = nn.ModuleList(
            [nn.Identity()] + [nn.Conv3d(c, c, kernel_size=3, padding=1) for c in filters[1:]]
        )
        for token in self.mask_tokens:
            nn.init.trunc_normal_(token, std=0.02, a=-0.02, b=0.02)

    def _call(self, function, *args):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return activation_checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def encode(self, x, mask):
        if x.ndim != 5 or x.shape[1] != 1 or any(s <= 0 or s % DOWNSAMPLE for s in x.shape[2:]):
            raise ValueError("Expected [B,1,D,H,W] with spatial dimensions divisible by 32.")
        if mask.shape != (x.shape[0], 1, *(s // DOWNSAMPLE for s in x.shape[2:])):
            raise ValueError("Pass a hidden mask on DynUNet's stride-32 bottleneck grid.")
        blocks = [self.backbone.input_block, *self.backbone.downsamples, self.backbone.bottleneck]
        visible = 1 - mask
        visible_in = expand_mask(visible, x.shape[2:])
        features = []
        for block in blocks:
            stride = block.conv1.conv.stride
            shape = tuple(s // t for s, t in zip(x.shape[2:], stride))
            visible_out = expand_mask(visible, shape)
            x = self._call(masked_resblock, block, x, visible_in, visible_out)
            features.append(x)
            visible_in = visible_out
        return features

    def forward(self, x, mask):
        features = self.encode(x, mask)
        dense = []
        for feature, norm, token, projection in zip(
            features, self.densify_norms, self.mask_tokens, self.densify_projections
        ):
            visible = expand_mask(1 - mask, feature.shape[2:])
            dense.append(self._call(densify, feature, visible, norm, token, projection))
        out = dense[-1]
        # MONAI upsamples are ordered from bottleneck to highest resolution.
        for block, skip in zip(self.backbone.upsamples, reversed(dense[:-1])):
            out = self._call(block, out, skip)
        return self.backbone.output_block(out)

    def segmentation_model(self, out_channels):
        """Transfer both encoder and decoder, retaining a fresh segmentation head."""
        model = build_model(out_channels=out_channels, filters=self.filters)
        state = {k: v for k, v in self.backbone.state_dict().items() if not k.startswith("output_block.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("output_block.") for k in missing):
            raise RuntimeError(f"Unexpected transfer mismatch: {missing}, {unexpected}")
        return model


def masked_mse(prediction, target, mask):
    """L2 on ALL hidden voxels, including background; normalize by hidden count."""
    weight = expand_mask(mask, target.shape[2:]).expand_as(target)
    return ((prediction.float() - target.float()).square() * weight).sum() / weight.sum().clamp_min(1)


def poly_lr(initial_lr, completed_steps, total_steps, power=0.9):
    return initial_lr * max(0.0, 1 - completed_steps / total_steps) ** power


class AddValidd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        image = d["image"]
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Expected a scalar 3D MRI volume.")
        if not torch.isfinite(image).all():
            raise ValueError("Nonfinite MRI intensities.")
        d["valid"] = (image != 0).float()
        if not d["valid"].any():
            raise ValueError("Empty foreground.")
        foreground = image[image != 0]
        if foreground.min() == foreground.max():
            raise ValueError("Constant foreground intensity.")
        return d


def make_transforms(roi, training, augmentation_backend="monai"):
    keys = ["image", "valid"]
    transforms = [
        LoadImaged("image", reader="NibabelReader"),
        EnsureChannelFirstd("image", channel_dim="no_channel"),
        Orientationd("image", axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacingd("image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        AddValidd("image"),
        CropForegroundd(keys, source_key="valid", margin=4, allow_smaller=True),
        NormalizeIntensityd("image", nonzero=True, channel_wise=True),
    ]
    if training:
        transforms.append(DeleteItemsd("valid"))
    transforms += [
        SpatialPadd("image" if training else keys, spatial_size=roi),
    ]
    if training:
        transforms.append(RandSpatialCropd("image", roi_size=roi, random_size=False))
        if augmentation_backend == "monai":
            transforms += [
                RandAffined(
                    "image",
                    prob=0.2,
                    spatial_size=roi,
                    mode="bilinear",
                    padding_mode="zeros",
                    rotate_range=(math.radians(10),) * 3,
                    scale_range=(0.1,) * 3,
                ),
                *[RandFlipd("image", prob=0.5, spatial_axis=axis) for axis in range(3)],
            ]
        elif augmentation_backend != "batchaug":
            raise ValueError(f"Unknown augmentation backend: {augmentation_backend}")
    else:
        transforms += [CenterSpatialCropd(keys, roi_size=roi)]
    return Compose(transforms + [EnsureTyped("image" if training else keys, dtype=torch.float32, track_meta=False)])


def make_batch_augmenter():
    """Use BatchAug only on training batches after transfer to CUDA."""
    return batchaug.Compose(
        transforms=[
            batchaug.RandAffined(
                keys=["image"],
                prob=0.2,
                rotate_range=(math.radians(10),) * 3,
                scale_range=(0.1,) * 3,
                mode="bilinear",
                padding_mode="zeros",
            ),
            *[batchaug.RandFlipd(keys=["image"], prob=0.5, spatial_axis=axis) for axis in range(3)],
        ],
        lazy=True,
        mode="bilinear",
        padding_mode="zeros",
    )


def reconstruction_totals(prediction, target, mask, foreground, ssim_data_range=None):
    """Return numerators/denominators for voxel-weighted reconstruction metrics."""
    hidden = expand_mask(mask, target.shape[2:]).bool().expand_as(target)
    error = prediction.detach().float() - target.float()
    squared = error.square()
    tissue = hidden & foreground.bool()
    totals = {
        "masked_mse": ((squared * hidden).sum().item(), hidden.sum().item()),
        "masked_mae": ((error.abs() * hidden).sum().item(), hidden.sum().item()),
        "foreground_masked_mse": ((squared * tissue).sum().item(), tissue.sum().item()),
    }
    if ssim_data_range is not None:
        # Z-scored MRI has no natural bounded range. Clip ONLY for SSIM to a
        # fixed symmetric interval (default [-5, 5]); MSE/MAE remain unclipped.
        limit = ssim_data_range / 2
        pred = prediction.detach().float().clamp(-limit, limit)
        truth = target.float().clamp(-limit, limit)

        def pool(image):
            # Separable uniform filtering avoids the cost of a dense 7^3 kernel.
            for kernel in ((7, 1, 1), (1, 7, 1), (1, 1, 7)):
                image = F.avg_pool3d(image, kernel_size=kernel, stride=1)
            return image

        mu_p, mu_t = pool(pred), pool(truth)
        var_p = (pool(pred.square()) - mu_p.square()).clamp_min(0)
        var_t = (pool(truth.square()) - mu_t.square()).clamp_min(0)
        covariance = pool(pred * truth) - mu_p * mu_t
        c1, c2 = (0.01 * ssim_data_range) ** 2, (0.03 * ssim_data_range) ** 2
        ssim = ((2 * mu_p * mu_t + c1) * (2 * covariance + c2)) / (
            (mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2)
        )
        # Include only windows wholly inside hidden blocks, preventing visible
        # voxels from inflating the reconstruction score. Uniform 7^3 windows.
        windows = pool(hidden.float()) > 1 - 1e-6
        totals["masked_ssim"] = ((ssim * windows).sum().item(), windows.sum().item())
    return totals


def add_metric_totals(accumulator, prefix, totals):
    for metric, (numerator, denominator) in totals.items():
        key = f"{prefix}/{metric}"
        old_n, old_d = accumulator.get(key, (0.0, 0.0))
        accumulator[key] = (old_n + numerator, old_d + denominator)


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    # Independent streams keep dynamic validation and each fixed ratio stable.
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    ratios = getattr(args, "val_mask_ratios", (0.6, 0.75, 0.9))
    fixed_generators = [torch.Generator(device=device).manual_seed(args.seed + 2000 + i) for i in range(len(ratios))]
    totals = {}
    for batch in loader:
        x = batch["image"].to(device)
        foreground = batch["valid"].to(device)
        evaluations = [(None, draw_mask_ratio(args, generator, device), generator)]
        evaluations += [(f"{ratio:g}", ratio, rng) for ratio, rng in zip(ratios, fixed_generators)]
        for label, ratio, rng in evaluations:
            mask = block_mask(x, args.block, ratio, rng, coarse=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(x, mask)
            prefix = "val" if label is None else f"val/mask_ratio_{label}"
            for i in range(x.shape[0]):
                metrics = reconstruction_totals(
                    prediction[i : i + 1],
                    x[i : i + 1],
                    mask[i : i + 1],
                    foreground[i : i + 1],
                    ssim_data_range=getattr(args, "ssim_data_range", 10.0),
                )
                add_metric_totals(totals, prefix, metrics)
                if label is None:
                    sequence = batch.get("sequence", ["unknown"] * x.shape[0])[i]
                    add_metric_totals(totals, f"val/sequence_{sequence}", metrics)
    # Omit undefined foreground/window metrics rather than report misleading zeros.
    return {key: numerator / denominator for key, (numerator, denominator) in totals.items() if denominator > 0}


def draw_mask_ratio(args, generator, device):
    if args.mask_ratio is not None:
        return args.mask_ratio
    low, high = args.mask_range
    return low + (high - low) * torch.rand((), generator=generator, device=device).item()


def check_manifest(manifest):
    partitions = manifest["partitions"]
    patient_sets = {key: {row["patient_id"] for row in rows} for key, rows in partitions.items()}
    for name in ("train", "val", "test"):
        if not partitions.get(name):
            raise ValueError(f"Empty {name} partition.")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if patient_sets[a] & patient_sets[b]:
            raise ValueError(f"Patient leakage between {a} and {b}.")
    return partitions


def make_dataset(rows, roi, training, cache_dir=None, augmentation_backend="monai"):
    transforms = make_transforms(roi, training, augmentation_backend)
    if cache_dir:
        return PersistentDataset(rows, transforms, cache_dir=cache_dir, hash_transform=pickle_hashing)
    return Dataset(rows, transforms)


def save_checkpoint(
    path, model, optimizer, scaler, args, manifest_hash, completed, best, val_loss, mask_generator, data_generator
):
    state = {
        "version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": completed,
        "best_val_loss": best,
        "val_loss": val_loss,
        "config": vars(args),
        "model_config": {"filters": list(model.filters)},
        "manifest_sha256": manifest_hash,
        "mask_rng": mask_generator.get_state(),
        "data_rng": data_generator.get_state(),
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def check_resume(state, args, manifest_hash):
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Checkpoint is from the old baseline; start a fresh aligned pretraining run.")
    if state["manifest_sha256"] != manifest_hash:
        raise ValueError("Resume requires the original patient split and volume manifest.")
    for key in (
        "steps",
        "roi",
        "filters",
        "block",
        "mask_ratio",
        "mask_range",
        "batch_size",
        "accumulation",
        "lr",
        "momentum",
        "weight_decay",
        "poly_power",
        "grad_clip",
        "seed",
        "val_volumes",
    ):
        if json.dumps(state["config"][key]) != json.dumps(getattr(args, key)):
            raise ValueError(f"Resume requires the original --{key.replace('_', '-')} setting.")
    if state["config"].get("augmentation_backend", "monai") != getattr(args, "augmentation_backend", "monai"):
        raise ValueError("Resume requires the original --augmentation-backend setting.")


def init_wandb(args, output, manifest, manifest_hash, partitions, device):
    """Optional tracking; keep the run identity alongside local checkpoints."""
    if not getattr(args, "wandb", False):
        return None
    identity_path = output / "wandb_run.json"
    identity = json.loads(identity_path.read_text()) if args.resume and identity_path.exists() else None
    project = args.wandb_project
    entity = args.wandb_entity
    if identity:
        if identity["project"] != project or (entity is not None and identity["entity"] != entity):
            raise ValueError("Resume W&B with the original project and entity.")
        entity = identity["entity"]
    config = {
        **vars(args),
        "dataset_revision": manifest["revision"],
        "manifest_sha256": manifest_hash,
        "sequences": manifest.get("sequences", [manifest.get("sequence", "unknown")]),
        "effective_batch": args.batch_size * args.accumulation,
        "resolved_device": str(device),
        "split_volumes": {name: len(rows) for name, rows in partitions.items()},
        "split_patients": {name: len({r["patient_id"] for r in rows}) for name, rows in partitions.items()},
    }
    run = wandb.init(
        project=project,
        entity=entity,
        name=args.wandb_name,
        id=identity["id"] if identity else uuid4().hex,
        resume="allow" if args.wandb_mode == "online" and identity else None,
        mode=args.wandb_mode,
        dir=str(output.resolve()),
        save_code=True,
        config=config,
    )
    try:
        run.define_metric("optimizer_step")
        run.define_metric("train/*", step_metric="optimizer_step")
        run.define_metric("val/*", step_metric="optimizer_step")
        identity_path.write_text(json.dumps({"id": run.id, "project": run.project, "entity": run.entity}, indent=2))
    except BaseException:
        run.finish(exit_code=1)
        raise
    return run


def train(args):
    set_determinism(seed=args.seed)
    output = Path(args.output)
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    partitions = check_manifest(manifest)
    last = output / "last.pt"
    if last.exists() and not args.resume:
        raise FileExistsError(f"{last} exists; use --resume or prepare a new output directory.")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    augmentation_backend = getattr(args, "augmentation_backend", "monai")
    if augmentation_backend == "batchaug" and device.type != "cuda":
        raise RuntimeError("BatchAug augmentation requires CUDA; use --device cuda or --augmentation-backend monai.")
    batch_augmenter = make_batch_augmenter() if augmentation_backend == "batchaug" else None
    model = DynUNetMAE(args.filters, checkpointing=not args.no_checkpointing).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mask_generator = torch.Generator(device=device).manual_seed(args.seed)
    data_generator = torch.Generator().manual_seed(args.seed)
    completed, best, val_loss = 0, float("inf"), None
    if args.resume:
        state = torch.load(last, map_location="cpu", weights_only=True)
        check_resume(state, args, manifest_hash)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        completed, best, val_loss = state["step"], state["best_val_loss"], state["val_loss"]
        mask_generator.set_state(state["mask_rng"])
        data_generator.set_state(state["data_rng"])
    if completed >= args.steps:
        print(f"Checkpoint already completed {completed} optimizer steps.")
        return
    effective_batch = args.batch_size * args.accumulation
    sequences = manifest.get("sequences", [manifest.get("sequence", "unknown")])
    print(
        f"Device {device}; sequences {sequences}; effective batch {effective_batch}; "
        f"steps {completed}/{args.steps}; dataset revision {manifest['revision']}",
        flush=True,
    )
    if effective_batch != 6:
        print("Effective batch differs from the paper's six; record this as an adaptation.", flush=True)
    # Uniform sampling over volumes, irrespective of sequence prevalence.
    dataset = make_dataset(partitions["train"], args.roi, True, args.cache_dir, augmentation_backend)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=(args.steps - completed) * effective_batch, generator=data_generator
    )
    worker_options = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers:
        worker_options.update(multiprocessing_context="spawn", persistent_workers=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, **worker_options)
    val_rows = partitions["val"]
    if args.val_volumes and len(val_rows) > args.val_volumes:
        val_rows = random.Random(args.seed).sample(val_rows, args.val_volumes)
    val_loader = DataLoader(
        make_dataset(val_rows, args.roi, False, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        **worker_options,
    )
    (output / "recipe.json").write_text(
        json.dumps(
            {
                "paper": "https://arxiv.org/html/2410.23132v3",
                "config": vars(args),
                "dataset_revision": manifest["revision"],
                "manifest_sha256": manifest_hash,
                "effective_batch": effective_batch,
                "validation_volume_paths": [row["image"] for row in val_rows],
                "adaptations": [
                    "MONAI residual DynUNet instead of ResEnc-L",
                    f"{augmentation_backend} affine augmentation and three-axis flips",
                    "MRI nonzero foreground z-scoring and linear resampling",
                    "patient-level Yale holdouts",
                    "no compressed-file size threshold",
                    "dense tensor masking; no sparse kernel speedup",
                ],
                "resume_note": (
                    "Optimizer/schedule/mask RNG resume; sampler prefetch, worker augmentation, "
                    "and BatchAug CUDA augmentation are not bitwise replayed."
                ),
            },
            indent=2,
        )
    )
    wandb_run = init_wandb(args, output, manifest, manifest_hash, partitions, device)
    exit_code = 0
    try:
        with tqdm(total=args.steps, initial=completed, desc="Training", unit="update", dynamic_ncols=True) as progress:
            batches = iter(train_loader)
            total, total_mae, logged_steps = 0.0, 0.0, 0
            while completed < args.steps:
                model.train()
                lr = poly_lr(args.lr, completed, args.steps, args.poly_power)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                # Share a ratio across the whole effective batch so accumulation uses the
                # same masked-voxel weighting as a physical batch of six.
                ratio = draw_mask_ratio(args, mask_generator, device)
                step_loss, step_mae = 0.0, 0.0
                for _ in range(args.accumulation):
                    try:
                        batch = next(batches)
                    except StopIteration:
                        # Needed only if AMP skipped optimizer updates on overflow.
                        batches = iter(train_loader)
                        batch = next(batches)
                    x = batch["image"].to(device, non_blocking=True)
                    if batch_augmenter is not None:
                        x = batch_augmenter({"image": x})["image"]
                    mask = block_mask(x, args.block, ratio, mask_generator, coarse=True)
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        prediction = model(x, mask)
                        loss = masked_mse(prediction, x, mask)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite reconstruction loss; inspect the current volumes.")
                    scaler.scale(loss / args.accumulation).backward()
                    step_loss += loss.item() / args.accumulation
                    with torch.no_grad():
                        hidden = expand_mask(mask, x.shape[2:])
                        step_mae += (
                            ((prediction.detach().float() - x.float()).abs() * hidden).sum() / hidden.sum().clamp_min(1)
                        ).item() / args.accumulation
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    tqdm.write("AMP overflow: retrying this optimizer step with a reduced scale.")
                    continue
                completed += 1
                progress.update(1)
                total += step_loss
                total_mae += step_mae
                logged_steps += 1
                if completed % args.log_every == 0 or completed == args.steps:
                    progress.set_postfix(train_loss=f"{total / logged_steps:.5f}", lr=f"{lr:.6g}")
                    tqdm.write(f"Step {completed}/{args.steps}: train={total / logged_steps:.5f}, lr={lr:.6g}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "optimizer_step": completed,
                                "train/masked_mse": total / logged_steps,
                                "train/masked_mae": total_mae / logged_steps,
                                "train/lr": lr,
                                "train/mask_ratio": ratio,
                            }
                        )
                    total, total_mae, logged_steps = 0.0, 0.0, 0
                if completed % args.validate_every == 0 or completed == args.steps:
                    val_metrics = validate(model, val_loader, device, args)
                    val_loss = val_metrics["val/masked_mse"]
                    tqdm.write(f"Step {completed}: validation masked MSE={val_loss:.5f}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "optimizer_step": completed,
                                **val_metrics,
                                "val/best_masked_mse": min(best, val_loss),
                            }
                        )
                    if val_loss < best:
                        best = val_loss
                        save_checkpoint(
                            output / "best.pt",
                            model,
                            optimizer,
                            scaler,
                            args,
                            manifest_hash,
                            completed,
                            best,
                            val_loss,
                            mask_generator,
                            data_generator,
                        )
                if completed % args.save_every == 0 or completed == args.steps:
                    save_checkpoint(
                        last,
                        model,
                        optimizer,
                        scaler,
                        args,
                        manifest_hash,
                        completed,
                        best,
                        val_loss,
                        mask_generator,
                        data_generator,
                    )
    except BaseException:
        exit_code = 1
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)


def export(args):
    path = Path(args.checkpoint) if args.checkpoint else Path(args.output) / "last.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Export requires a checkpoint from the aligned DynUNetMAE.")
    mae = DynUNetMAE(state["model_config"]["filters"], checkpointing=False)
    mae.load_state_dict(state["model"])
    segmentation = mae.segmentation_model(args.out_channels)
    target = Path(args.output) / "segmentation_init.pt"
    if target.exists():
        raise FileExistsError(f"{target} exists; choose a different --output directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": segmentation.state_dict(),
            "filters": list(mae.filters),
            "out_channels": args.out_channels,
            "pretraining_step": state["step"],
            "pretraining_config": state["config"],
            "manifest_sha256": state["manifest_sha256"],
            "recommended_finetune_lr": 1e-3,
            "recommended_warmup_steps": 12500,
        },
        target,
    )
    print(f"Transferred encoder and decoder; new task head. Saved {target}")


def smoke_test(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    augmentation_backend = getattr(args, "augmentation_backend", "monai")
    if augmentation_backend == "batchaug" and device.type != "cuda":
        raise RuntimeError("BatchAug augmentation requires CUDA; use --device cuda or --augmentation-backend monai.")
    model = DynUNetMAE(filters=(4, 8, 12, 16, 24, 24)).to(device)
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    if augmentation_backend == "batchaug":
        x = make_batch_augmenter()({"image": x})["image"]
    mask = block_mask(x, ratio=0.6, coarse=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        prediction = model(x, mask)
        loss = masked_mse(prediction, x, mask)
    assert prediction.shape == x.shape
    assert set(mask.unique().tolist()) == {0.0, 1.0}
    loss.backward()
    assert torch.isfinite(loss) and model.backbone.input_block.conv1.conv.weight.grad is not None
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.mask_tokens)
    optimizer.step()
    segmentation = model.segmentation_model(out_channels=2).to(device)
    with torch.no_grad():
        assert segmentation(x).shape == (1, 2, 64, 64, 64)
    print(
        f"Smoke test passed on {device} with {augmentation_backend} augmentation: forward, masked loss, backward, token gradients, "
        f"SGD step, encoder/decoder transfer; loss={loss.item():.5f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "export", "smoke-test"])
    parser.add_argument("--output", default="./yale_mae")
    parser.add_argument("--roi", type=int, nargs=3, default=(160, 160, 160))
    parser.add_argument("--filters", type=int, nargs=6, default=FILTERS)
    parser.add_argument("--block", type=int, choices=[DOWNSAMPLE], default=DOWNSAMPLE)
    parser.add_argument("--mask-ratio", type=float, default=None, help="Optional static-ratio ablation")
    parser.add_argument("--mask-range", type=float, nargs=2, default=(0.6, 0.9))
    parser.add_argument("--batch-size", type=int, default=6, help="Physical batch; paper effective batch is six")
    parser.add_argument("--accumulation", type=int, default=1)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--steps", type=int, default=None, help="Optimizer updates (default 250000)")
    schedule.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Compatibility: nnU-Net virtual epochs of 250 updates, NOT dataset passes",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-dir", default=None, help="Optional disk cache of deterministic preprocessing")
    parser.add_argument("--augmentation-backend", choices=["monai", "batchaug"], default="monai")
    parser.add_argument("--no-checkpointing", action="store_true", help="Disable activation checkpointing")
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--val-volumes", type=int, default=25, help="Fixed validation subset; 0 uses all")
    parser.add_argument(
        "--val-mask-ratios",
        type=float,
        nargs="+",
        default=(0.6, 0.75, 0.9),
        help="Additional fixed-ratio reconstruction validation passes",
    )
    parser.add_argument(
        "--ssim-data-range",
        type=float,
        default=10.0,
        help="SSIM clipping interval width in z-score units (default [-5,5])",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases metric logging")
    parser.add_argument("--wandb-project", default="yale-mae")
    parser.add_argument("--wandb-entity", default=None, help="W&B user or team (default: account default)")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run display name")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to export (default OUTPUT/last.pt)")
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.steps = (
        args.steps
        if args.steps is not None
        else (args.epochs * STEPS_PER_VIRTUAL_EPOCH if args.epochs is not None else 250000)
    )
    if args.mask_ratio is not None and not 0 < args.mask_ratio < 1:
        parser.error("mask-ratio must be in (0,1)")
    if not 0 < args.mask_range[0] <= args.mask_range[1] < 1:
        parser.error("mask-range must be ordered and inside (0,1)")
    if any(s < 64 or s % DOWNSAMPLE for s in args.roi):
        parser.error("ROI dimensions must be >=64 and divisible by 32")
    if (
        any(
            value <= 0
            for value in (
                args.steps,
                args.batch_size,
                args.accumulation,
                args.lr,
                args.poly_power,
                args.grad_clip,
                args.validate_every,
                args.save_every,
                args.log_every,
                args.out_channels,
                *args.filters,
            )
        )
        or not 0 < args.momentum < 1
        or args.weight_decay < 0
    ):
        parser.error("Training sizes/rates must be positive; momentum in (0,1); weight decay >=0")
    if any(not math.isfinite(r) or not 0 < r < 1 for r in args.val_mask_ratios):
        parser.error("val-mask-ratios must be finite and in (0,1)")
    if not math.isfinite(args.ssim_data_range) or args.ssim_data_range <= 0:
        parser.error("ssim-data-range must be finite and positive")
    if min(args.workers, args.val_volumes) < 0:
        parser.error("workers and val-volumes must be nonnegative")
    {"train": train, "export": export, "smoke-test": smoke_test}[args.command](args)


if __name__ == "__main__":
    main()
