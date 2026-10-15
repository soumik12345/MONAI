"""Train MaskedAutoEncoderDynUnet with the train_mae.py pretraining recipe.

    python train_dynmae.py train --manifest yale_mae_restored/manifest.json \
        --output yale_dynmae --device cuda --batch-size 3 --accumulation 2 --wandb

Epochs are complete shuffled passes through the training split, including the
final partial batch. Updates per epoch are ceil(volumes / (batch * accumulation)).
Alternatively, --steps specifies a fixed update budget with replacement sampling.
Checkpoints use MAETrainer's format and cannot resume train_mae.py checkpoints.
"""

import argparse
import hashlib
import json
from pathlib import Path
import math
import random

import torch
from torch.utils.data import RandomSampler, Sampler
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.networks.nets import MaskedAutoEncoderDynUnet
from monai.data.utils import pickle_hashing
from monai.engines.mae_trainer import MAETrainer, MAETrainerConfig, WandbConfig
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

FILTERS = (32, 64, 128, 256, 320, 320)
STRIDES = (1, 2, 2, 2, 2, 2)


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
    transforms: list[MapTransform] = [
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
    transforms += [SpatialPadd("image" if training else keys, spatial_size=roi)]
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
    ) -> tuple[DataLoader, DataLoader]:
        manifest_bytes = Path(manifest_path).read_bytes()
        manifest = json.loads(manifest_bytes)
        self.manifest = manifest
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        partitions = self._check_manifest(manifest)

        transforms = make_transforms(roi, training=True, augmentation_backend=augmentation_backend)
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
        val_transforms = make_transforms(roi, training=False)
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
            self.dataloader.sampler.set_completed_updates(result[0])
        return result


def make_batch_augmenter():
    import batchaug

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
    batch_augmenter = make_batch_augmenter() if args.augmentation_backend == "batchaug" else None
    config = MAETrainerConfig(
        output_dir=str(output),
        dataloader_config={
            "manifest_path": str(manifest_path),
            "roi": tuple(args.roi),
            "workers": args.workers,
            "cache_dir": str(Path(args.cache_dir).resolve()) if args.cache_dir else None,
            "val_volumes": args.val_volumes,
            "augmentation_backend": args.augmentation_backend,
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
