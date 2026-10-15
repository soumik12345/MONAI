"""Train MaskedAutoEncoderDynUnet with nnssl-aligned preprocessing/augmentation.

    python train_dynmae.py train --manifest yale_mae_restored/manifest.json \
        --output yale_dynmae --device cuda --batch-size 3 --accumulation 2 --wandb

Epochs are complete shuffled passes through the training split, including the
final partial batch. Updates per epoch are ceil(volumes / (batch * accumulation)).
Alternatively, --steps specifies a fixed update budget with replacement sampling.
Both augmentation backends implement nnssl-aligned transforms in this module.
Preprocessing follows nnssl's default preprocessor and onemmiso planner; spatial
augmentation follows BaseMAETrainer: https://github.com/MIC-DKFZ/nnssl.
No nnssl installation is required. Both backends use cubic interpolation
(CuPy on CUDA); CPU/GPU results need not be bitwise identical. Validation's
foreground mask is an additional metric-only output.
Install requirements-mae.txt; BatchAug additionally requires CUDA-matched CuPy.
Checkpoints use MAETrainer's format and cannot resume train_mae.py checkpoints.
"""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import math
import random

import numpy as np
import torch
from torch.utils.data import RandomSampler, Sampler
from monai.data import DataLoader, Dataset, MetaTensor, PersistentDataset
from monai.networks.nets import MaskedAutoEncoderDynUnet
from monai.data.utils import pickle_hashing
from monai.engines.mae_trainer import MAETrainer, MAETrainerConfig, WandbConfig
from monai.utils import set_determinism
from monai.transforms import Compose, EnsureTyped, MapTransform, RandomizableTransform, Resample

FILTERS = (32, 64, 128, 256, 320, 320)
STRIDES = (1, 2, 2, 2, 2, 2)

ROTATION = np.pi / 6


def plan_transpose(rows):
    """Use nnssl's largest-median-spacing-first axis order in native Z,Y,X."""
    spacings = [row["voxel_spacing_mm"][::-1] for row in rows if "voxel_spacing_mm" in row]
    if len(spacings) != len(rows) or not spacings:
        raise ValueError("Planning requires voxel_spacing_mm for every training volume or explicit transpose_forward.")
    first = int(np.argmax(np.median(np.asarray(spacings), axis=0)))
    return (first, *(axis for axis in range(3) if axis != first))


def resample_volume(volume, spacing, order=3):
    """nnssl onemmiso: cubic in-plane, nearest through a uniquely coarse axis.

    Uses skimage's edge boundary, half-voxel grid, clipping, and disabled
    antialiasing. The anisotropy threshold is strictly greater than three.
    """
    from skimage.transform import resize

    volume = np.asarray(volume, dtype=np.float32)
    shape = np.asarray(volume.shape)
    spacing = np.asarray(spacing, dtype=float)
    new_shape = np.rint(shape * spacing).astype(int)
    if np.any(new_shape < 1):
        raise ValueError("Resampling would produce an empty spatial dimension.")
    if np.array_equal(shape, new_shape):
        return volume.copy()
    coarse_axes = np.flatnonzero(spacing == spacing.max())
    coarse_axis = int(coarse_axes[0]) if spacing.max() / spacing.min() > 3 and len(coarse_axes) == 1 else None
    work = volume.astype(np.float64)
    if coarse_axis is None:
        return np.asarray(
            resize(work, tuple(new_shape), order=order, mode="edge", anti_aliasing=False, preserve_range=True),
            dtype=np.float32,
        )

    slices = np.moveaxis(work, coarse_axis, 0)
    in_plane_shape = tuple(np.delete(new_shape, coarse_axis))
    resized = np.stack(
        [
            np.asarray(
                resize(plane, in_plane_shape, order=order, mode="edge", anti_aliasing=False, preserve_range=True)
            )
            for plane in slices
        ]
    )
    # Order-0 interpolation at half-voxel coordinates, with nearest boundaries.
    indices = np.floor((np.arange(new_shape[coarse_axis]) + 0.5) * len(slices) / new_shape[coarse_axis]).astype(int)
    indices = np.clip(indices, 0, len(slices) - 1)
    return np.moveaxis(resized[indices], 0, coarse_axis).astype(np.float32)


class NNSSLPreprocessd(MapTransform):
    """Native NIfTI axes -> nonzero crop -> whole-crop z-score -> 1 mm.

    NIfTI X,Y,Z is reversed to nnssl's native Z,Y,X reader convention, then
    transposed using the dataset plan. No RAS canonicalization is applied.
    """

    def __init__(self, transpose_forward=(0, 1, 2)):
        super().__init__(("image",))
        self.transpose_forward = tuple(transpose_forward)
        if sorted(self.transpose_forward) != [0, 1, 2]:
            raise ValueError("transpose_forward must be a permutation of 0, 1, 2.")

    def __call__(self, data):
        import nibabel as nib
        from nibabel.spatialimages import SpatialImage
        from scipy.ndimage import binary_fill_holes

        d = dict(data)
        nifti = nib.load(d["image"])
        if not isinstance(nifti, SpatialImage) or nifti.ndim != 3:
            raise ValueError("Expected a scalar 3D MRI volume.")
        image = nifti.get_fdata().astype(np.float32).transpose(2, 1, 0).transpose(self.transpose_forward)
        spacing = np.asarray(nifti.header.get_zooms()[::-1])[list(self.transpose_forward)]
        if not np.isfinite(image).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Nonfinite MRI intensities or invalid spacing.")
        foreground = np.empty(image.shape, dtype=np.bool_)
        binary_fill_holes(image != 0, output=foreground)
        locations = np.nonzero(foreground)
        if not locations[0].size:
            raise ValueError("Empty foreground.")
        bbox = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in locations)
        image = image[bbox].copy()
        foreground = foreground[bbox]
        mean, std = image.mean(), image.std()
        image -= mean
        image /= max(std, 1e-8)
        d["image"] = resample_volume(image, spacing)[None]
        d["valid"] = resample_volume(foreground.astype(np.float32), spacing, order=0)[None]
        return d


def initial_patch_size(roi):
    """Replicate nnssl's context-patch heuristic (including its 0.85 divisor)."""
    from batchgenerators.augmentations.utils import rotate_coords_3d

    roi = np.asarray(roi, dtype=float)
    candidates = [roi]
    for axis in range(3):
        angles = np.zeros(3)
        angles[axis] = ROTATION
        candidates.append(np.abs(rotate_coords_3d(roi, *angles)))
    return tuple((np.max(candidates, axis=0) / 0.85).astype(int))


def extract_patch(image, start, size):
    """Crop and zero-pad a C,D,H,W array at possibly negative coordinates."""
    slices = tuple(slice(max(0, lo), min(dim, lo + n)) for lo, n, dim in zip(start, size, image.shape[1:]))
    crop = image[(slice(None), *slices)]
    pad = [(0, 0), *[(max(0, -lo), max(0, lo + n - dim)) for lo, n, dim in zip(start, size, image.shape[1:])]]
    return np.pad(crop, pad, mode="constant").astype(np.float32, copy=False)


class NNSSLPatchd(RandomizableTransform, MapTransform):
    """Sample nnssl's padded context patches; use center patches for validation."""

    def __init__(self, roi, training):
        RandomizableTransform.__init__(self, prob=1)
        MapTransform.__init__(self, ("image",))
        self.roi = tuple(roi)
        self.training = training
        self.size = initial_patch_size(roi) if training else self.roi

    def __call__(self, data):
        d = dict(data)
        shape = np.asarray(d["image"].shape[1:])
        size = np.asarray(self.size)
        pad = np.maximum(size - self.roi, size - shape)
        low = -pad // 2
        high = shape + pad // 2 + pad % 2 - size
        start = (
            [self.R.randint(int(a), int(b) + 1) for a, b in zip(low, high)]
            if self.training
            else [int((a + b) / 2) for a, b in zip(low, high)]
        )
        d["image"] = extract_patch(d["image"], start, self.size)
        if self.training:
            d.pop("valid", None)
        else:
            d["valid"] = extract_patch(d["valid"], start, self.size)
        return d


def sample_spatial_parameters(rng):
    """Independent 20% rotation/scale; uniform rotation and balanced zoom in/out.

    nnssl/batchgenerators chooses each half of [0.7, 1.4] with 50% probability,
    so scale is not uniform over the entire interval. Scaling is isotropic.
    """
    from batchgenerators.augmentations.utils import rotate_coords_3d

    rotate = rng.uniform() < 0.2
    angles = np.zeros(3)
    if rotate:
        for axis in range(3):
            if rng.uniform() <= 1.0:  # reference p_rot_per_axis=1, including the RNG draw
                angles[axis] = rng.uniform(-ROTATION, ROTATION)
    scale = rng.uniform() < 0.2
    factor = 1.0
    if scale:
        factor = rng.uniform(0.7, 1.0) if rng.random() < 0.5 else rng.uniform(1.0, 1.4)
    matrix = rotate_coords_3d(np.eye(3), *angles) * factor
    return {"matrix": matrix, "modified": bool(rotate or scale)}


@lru_cache(maxsize=4)
def centered_grid(roi, device):
    axes = [torch.arange(n, dtype=torch.float64, device=device) - (n - 1) / 2 for n in roi]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))


@torch.no_grad()
def apply_spatial(image, roi, params):
    """Cubic spatial transform on CPU or CUDA; constant zero outside the image."""
    image = torch.as_tensor(image, dtype=torch.float32)
    if isinstance(image, MetaTensor):
        image = image.as_tensor()
    if any(n < r for n, r in zip(image.shape[1:], roi)):
        raise ValueError("Spatial augmentation expects a padded context patch.")
    if not params["modified"]:
        # Reference skips interpolation entirely when neither transform fires.
        start = [(n - r) // 2 for n, r in zip(image.shape[1:], roi)]
        return image[(slice(None), *(slice(s, s + r) for s, r in zip(start, roi)))].clone()
    matrix = torch.as_tensor(params["matrix"], dtype=torch.float64, device=image.device)
    coords = (matrix @ centered_grid(tuple(roi), image.device).reshape(3, -1)).reshape(3, *roi)
    coords += (
        torch.as_tensor(image.shape[1:], device=image.device, dtype=torch.float64).sub(1).div(2)[:, None, None, None]
    )
    result = Resample(mode=3, padding_mode="constant", norm_coords=False, dtype=np.float32)(image, coords)
    return result.as_tensor() if isinstance(result, MetaTensor) else result


def mirror_sample(image, rng):
    # MirrorTransform has p_per_sample=1 followed by independent 0.5 axis flips.
    rng.uniform()
    axes = [axis + 1 for axis in range(3) if rng.uniform() < 0.5]
    return torch.flip(image, axes) if axes else image


class NNSSLSpatiald(RandomizableTransform, MapTransform):
    """MONAI dictionary adapter with nnssl's spatial sampling behavior."""

    def __init__(self, roi):
        RandomizableTransform.__init__(self, prob=1)
        MapTransform.__init__(self, ("image",))
        self.roi = tuple(roi)

    def __call__(self, data):
        d = dict(data)
        d["image"] = mirror_sample(apply_spatial(d["image"], self.roi, sample_spatial_parameters(self.R)), self.R)
        return d


def make_transforms(roi, training, augmentation_backend="monai", transpose_forward=(0, 1, 2)):
    if augmentation_backend not in ("monai", "batchaug"):
        raise ValueError(f"Unknown augmentation backend: {augmentation_backend}")
    transforms = [NNSSLPreprocessd(transpose_forward), NNSSLPatchd(roi, training)]
    if training and augmentation_backend == "monai":
        transforms.append(NNSSLSpatiald(roi))
    keys = "image" if training else ("image", "valid")
    transforms.append(EnsureTyped(keys, dtype=torch.float32, track_meta=False))
    return Compose(transforms)


def make_batch_augmenter(roi=(160, 160, 160), seed=42):
    """BatchAug adapter; uses CuPy cubic splines instead of grid_sample linear.

    Lazy affine fusion is intentionally disabled: native BatchAug materializes
    fused transforms with linear grid_sample, which changes the interpolation.
    """
    import batchaug
    from batchaug import BatchDictTransform, BatchTransform

    class CubicSpatial(BatchTransform):
        def __init__(self):
            super().__init__(prob=1)
            self.rng = np.random.RandomState(seed)

        def sample_params(self, batch_size, shape, device):
            return {"samples": [sample_spatial_parameters(self.rng) for _ in range(batch_size)]}

        def apply(self, tensor, params):
            images = [apply_spatial(image, tuple(roi), p) for image, p in zip(tensor, params["samples"])]
            return torch.stack([mirror_sample(image, self.rng) for image in images])

    return batchaug.Compose([BatchDictTransform(["image"], CubicSpatial())], lazy=False)


def check_cuda_cubic():
    """Fail before starting a run if the required CUDA spline backend is absent."""
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import map_coordinates

        with cp.cuda.Device(torch.cuda.current_device()):
            map_coordinates(cp.zeros((3, 3, 3), dtype=cp.float32), cp.ones((3, 1)), order=3).get()
    except Exception as error:
        raise RuntimeError(
            "BatchAug's spatial transforms require working CuPy cubic interpolation. "
            "Install cupy-cuda13x[ctk] (CUDA 13) or cupy-cuda12x[ctk] (CUDA 12), "
            "or use --augmentation-backend monai."
        ) from error


class EpochRandomSampler(Sampler[int]):
    """One permutation per epoch, with a resumable position in that permutation."""

    def __init__(self, size: int, seed: int, effective_batch_size: int):
        self.size = size
        self.seed = seed
        self.effective_batch_size = effective_batch_size
        self.updates_per_epoch = (size + effective_batch_size - 1) // effective_batch_size
        self.epoch = 0
        self.start_index = 0

    def set_completed_updates(self, completed: int) -> None:
        self.epoch, offset = divmod(completed, self.updates_per_epoch)
        self.start_index = offset * self.effective_batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        start = self.start_index
        self.epoch += 1
        self.start_index = 0
        yield from indices[start:]

    def __len__(self):
        return self.size - self.start_index


class DynMaeTrainer(MAETrainer):
    def make_dataloaders(
        self,
        manifest_path,
        roi=(160, 160, 160),
        workers: int = 0,
        cache_dir: str | Path | None = None,
        val_volumes: int = 25,
        augmentation_backend: str = "monai",
        transpose_forward=(0, 1, 2),
    ) -> tuple[DataLoader, DataLoader]:
        manifest_bytes = Path(manifest_path).read_bytes()
        manifest = json.loads(manifest_bytes)
        self.manifest = manifest
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        partitions = self._check_manifest(manifest)

        transforms = make_transforms(
            roi, training=True, augmentation_backend=augmentation_backend, transpose_forward=transpose_forward
        )
        train_dataset = (
            PersistentDataset(partitions["train"], transforms, cache_dir=cache_dir, hash_transform=pickle_hashing)
            if cache_dir
            else Dataset(partitions["train"], transforms)
        )

        self.train_volume_count = len(train_dataset)
        self.updates_per_epoch = (self.train_volume_count + self.effective_batch_size - 1) // self.effective_batch_size
        if self.config.epochs is not None:
            steps = self.get_training_steps()
            if self.config.steps is not None and self.config.steps != steps:
                raise ValueError("steps must match the update count calculated from epochs.")
            self.config.steps = steps
            sampler = EpochRandomSampler(self.train_volume_count, self.config.seed, self.effective_batch_size)
        else:
            steps = self.get_training_steps()
            sampler = RandomSampler(
                train_dataset,
                replacement=True,
                num_samples=steps * self.effective_batch_size,
                generator=self.data_generator,
            )

        loader_options = {
            "batch_size": self.config.local_batch_size,
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
        }

        if workers:
            loader_options.update(multiprocessing_context="spawn", persistent_workers=True)

        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)

        val_rows = partitions["val"]
        if val_volumes and len(val_rows) > val_volumes:
            val_rows = random.Random(self.config.seed).sample(val_rows, val_volumes)

        self.validation_volume_paths = [row["image"] for row in val_rows]
        val_transforms = make_transforms(roi, training=False, transpose_forward=transpose_forward)
        val_dataset = (
            PersistentDataset(val_rows, val_transforms, cache_dir=cache_dir, hash_transform=pickle_hashing)
            if cache_dir
            else Dataset(val_rows, val_transforms)
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
        return train_loader, val_loader

    def get_training_steps(self) -> int:
        if self.config.epochs is not None:
            return self.config.epochs * self.updates_per_epoch
        return self.config.steps if self.config.steps is not None else 250_000

    def load_checkpoint(self, path: Path) -> tuple[int, float, float | None]:
        result = super().load_checkpoint(path)
        if self.config.epochs is not None:
            sampler = self.dataloader.sampler
            if not isinstance(sampler, EpochRandomSampler):
                raise TypeError("Epoch-based training requires an EpochRandomSampler.")
            sampler.set_completed_updates(result[0])
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--output", default="./yale_mae", help="Directory for checkpoints and run metadata")
    parser.add_argument("--manifest", help="Prepared manifest (default: OUTPUT/manifest.json)")
    parser.add_argument("--roi", type=int, nargs=3, default=(160, 160, 160))
    parser.add_argument("--filters", type=int, nargs=6, default=FILTERS)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--accumulation", type=int, default=1)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--steps", type=int, help="Optimizer updates (default: 250000)")
    schedule.add_argument("--epochs", type=int, help="Complete shuffled passes through the training split")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=12.0)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-range", type=float, nargs=2, default=(0.6, 0.9))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--augmentation-backend", choices=["monai", "batchaug"], default="monai")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--val-volumes", type=int, default=25, help="Fixed validation subset; 0 uses all")
    parser.add_argument("--val-mask-ratios", type=float, nargs="+", default=(0.6, 0.75, 0.9))
    parser.add_argument("--ssim-data-range", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="yale-mae")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.steps is None and args.epochs is None:
        args.steps = 250_000
    if any(s < 64 or s % 32 for s in args.roi):
        parser.error("ROI dimensions must be >=64 and divisible by 32")
    if any(f <= 0 for f in args.filters):
        parser.error("filters must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.momentum < 1:
        parser.error("lr must be positive and finite; momentum must be in (0,1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if min(args.workers, args.val_volumes) < 0:
        parser.error("workers and val-volumes must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_determinism(seed=args.seed)
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} is missing; unpack the prepared dataset first.")
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError(f"{output / 'last.pt'} exists; use --resume or choose a new --output.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.augmentation_backend == "batchaug" and device != "cuda":
        raise ValueError("BatchAug augmentation requires CUDA.")
    if args.augmentation_backend == "batchaug":
        check_cuda_cubic()
    batch_augmenter = make_batch_augmenter(args.roi, args.seed) if args.augmentation_backend == "batchaug" else None
    transpose_forward = plan_transpose(json.loads(manifest_path.read_text())["partitions"]["train"])
    config = MAETrainerConfig(
        output_dir=str(output),
        dataloader_config={
            "manifest_path": str(manifest_path),
            "roi": tuple(args.roi),
            "workers": args.workers,
            "cache_dir": str(Path(args.cache_dir).resolve()) if args.cache_dir else None,
            "val_volumes": args.val_volumes,
            "augmentation_backend": args.augmentation_backend,
            "transpose_forward": transpose_forward,
        },
        wandb_config=WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_name,
            mode=args.wandb_mode if args.wandb else "disabled",
        ),
        seed=args.seed,
        device=device,
        resume=args.resume,
        local_batch_size=args.batch_size,
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
    model = MaskedAutoEncoderDynUnet(
        in_channels=1,
        spatial_dims=3,
        kernel_size=(3,) * 6,
        strides=STRIDES,
        filters=tuple(args.filters),
        enable_activation_checkpointing=not args.no_checkpointing,
    )
    trainer = DynMaeTrainer(model, config)
    args.steps = trainer.get_training_steps()
    recipe = {
        "config": vars(args),
        "model": "MaskedAutoEncoderDynUnet",
        "transpose_forward": transpose_forward,
        "dataset_revision": trainer.manifest.get("revision"),
        "manifest_sha256": trainer.manifest_sha256,
        "effective_batch": trainer.effective_batch_size,
        "train_volumes": trainer.train_volume_count,
        "updates_per_epoch": trainer.updates_per_epoch if args.epochs is not None else None,
        "validation_volume_paths": trainer.validation_volume_paths,
    }
    try:
        (output / "recipe.json").write_text(json.dumps(recipe, indent=2))
        trainer.run.config.update({key: value for key, value in recipe.items() if key != "config"})
        print(
            f"Device {device}; effective batch {trainer.effective_batch_size}; "
            f"optimizer updates {args.steps}; "
            + (f"epochs {args.epochs} ({trainer.updates_per_epoch} updates each); " if args.epochs is not None else "")
            + f"manifest {manifest_path}",
            flush=True,
        )
    except BaseException:
        trainer.run.finish(exit_code=1)
        raise
    trainer.train(batch_augmenter=batch_augmenter)


if __name__ == "__main__":
    main()
