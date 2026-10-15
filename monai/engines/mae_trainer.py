from __future__ import annotations

import math
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import TYPE_CHECKING, Any
from pathlib import Path
import torch
import wandb
import torch.nn.functional as F

from monai.bundle import ConfigComponent
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

if TYPE_CHECKING:
    from monai.networks.nets import MaskedAutoEncoderDynUnet
    from torch.utils.data import DataLoader


@dataclass
class WandbConfig:
    project: str
    entity: str | None = None
    run_name: str | None = None


@dataclass
class MAETrainerConfig:
    output_dir: str
    dataloader_config: dict[str, Any]
    wandb_config: WandbConfig
    seed: int = 42
    device: str = "auto"
    resume: bool = False
    local_batch_size: int = 4
    gradient_accumualtion_steps: int = 2
    optimizer_config: dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-3,
            "weight_decay": 3e-5,
        }
    )
    mask_ratio: float | None = None
    mask_range: tuple[float, float] = (0.6, 0.9)
    grad_clip: float = 12.0
    steps: int | None = None
    poly_power: float = 0.9
    log_every: int = 1
    validate_every: int = 50
    save_every: int = 100
    val_mask_ratios: tuple[float, ...] = (0.6, 0.75, 0.9)
    ssim_data_range: float = 10.0

    def __post_init__(self) -> None:
        if self.mask_ratio is None:
            low, high = self.mask_range
            if not 0 < low <= high < 1:
                raise ValueError("mask_range must satisfy 0 < low <= high < 1.")
        elif not 0 < self.mask_ratio < 1:
            raise ValueError("mask_ratio must be strictly between 0 and 1.")

        for name in (
            "local_batch_size",
            "gradient_accumualtion_steps",
            "log_every",
            "validate_every",
            "save_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.steps is not None and (
            isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0
        ):
            raise ValueError("steps must be a positive integer or None.")
        if any(not 0 < ratio < 1 for ratio in self.val_mask_ratios):
            raise ValueError("val_mask_ratios must be strictly between 0 and 1.")
        for name in ("grad_clip", "poly_power", "ssim_data_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


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


def _check_manifest(manifest):
    partitions = manifest["partitions"]
    patient_sets = {key: {row["patient_id"] for row in rows} for key, rows in partitions.items()}
    for name in ("train", "val", "test"):
        if not partitions.get(name):
            raise ValueError(f"Empty {name} partition.")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if patient_sets[a] & patient_sets[b]:
            raise ValueError(f"Patient leakage between {a} and {b}.")
    return partitions


def make_transforms(self, roi, training, augmentation_backend="monai"):
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


def reconstruction_totals(prediction, target, mask, foreground, ssim_data_range=None):
    """Return numerators/denominators for voxel-weighted reconstruction metrics."""
    hidden = F.interpolate(mask.float(), size=target.shape[2:], mode="nearest").bool().expand_as(target)
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


class MAETrainer(ABC):
    def __init__(self, model: MaskedAutoEncoderDynUnet, config: MAETrainerConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if self.config.device == "auto" else self.config.device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")

        self.model = self.model.to(self.device)
        optimizer = ConfigComponent(config.optimizer_config).instantiate(params=self.model.parameters())
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer_config must instantiate a torch.optim.Optimizer.")
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.mask_generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        self.data_generator = torch.Generator().manual_seed(self.config.seed)
        self.effective_batch_size = self.config.local_batch_size * self.config.gradient_accumualtion_steps

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataloader, self.val_dataloader = self.make_dataloaders(**self.config.dataloader_config)

        self.run = wandb.init(
            project=self.config.wandb_config.project,
            entity=self.config.wandb_config.entity,
            name=self.config.wandb_config.run_name,
            save_code=True,
            config=asdict(self.config),
        )
        self.run.define_metric("optimizer_step")
        self.run.define_metric("train/*", step_metric="optimizer_step")
        self.run.define_metric("val/*", step_metric="optimizer_step")

    def _check_manifest(self, manifest):
        partitions = manifest["partitions"]
        patient_sets = {key: {row["patient_id"] for row in rows} for key, rows in partitions.items()}
        for name in ("train", "val", "test"):
            if not partitions.get(name):
                raise ValueError(f"Empty {name} partition.")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if patient_sets[a] & patient_sets[b]:
                raise ValueError(f"Patient leakage between {a} and {b}.")
        return partitions

    @abstractmethod
    def make_dataloaders(self, *args, **kwargs) -> tuple[DataLoader, DataLoader]:
        raise NotImplementedError("MAETrainer.make_dataloaders should return training and validation dataloaders")

    @abstractmethod
    def get_training_steps(self) -> int:
        raise NotImplementedError("MAETrainer.get_training_steps should be implemented by a subclass")

    def train_step(
        self,
        *,
        lr: float | None = None,
        batch_augmenter: Any = None,
    ) -> dict[str, float | bool]:
        mask_ratio = self.config.mask_ratio
        if mask_ratio is None:
            low, high = self.config.mask_range
            mask_ratio = low + (high - low) * torch.rand((), generator=self.mask_generator, device=self.device).item()

        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        # Retain the iterator across calls so every update consumes new batches.
        batches = getattr(self, "_train_batches", None)
        if batches is None:
            batches = iter(self.dataloader)
            self._train_batches = batches

        step_loss, step_mae = 0.0, 0.0
        for _ in range(self.config.gradient_accumualtion_steps):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(self.dataloader)
                self._train_batches = batches
                try:
                    batch = next(batches)
                except StopIteration as error:
                    raise ValueError("The training dataloader is empty.") from error

            x = batch["image"].to(self.device, non_blocking=True)
            if batch_augmenter is not None:
                x = batch_augmenter({"image": x})["image"]
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                prediction, mask = self.model(x, masking_ratio=mask_ratio, generator=self.mask_generator)
                hidden = F.interpolate(mask.float(), size=x.shape[2:], mode="nearest").expand_as(x)
                hidden_count = hidden.sum().clamp_min(1)
                loss = ((prediction.float() - x.float()).square() * hidden).sum() / hidden_count
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reconstruction loss; inspect the current volumes.")
            self.scaler.scale(loss / self.config.gradient_accumualtion_steps).backward()
            step_loss += loss.item() / self.config.gradient_accumualtion_steps
            with torch.no_grad():
                step_mae += (
                    ((prediction.detach().float() - x.float()).abs() * hidden).sum() / hidden_count
                ).item() / self.config.gradient_accumualtion_steps

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {
            "train/masked_mse": step_loss,
            "train/masked_mae": step_mae,
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
            "train/mask_ratio": mask_ratio,
            "optimizer_step_skipped": self.scaler.get_scale() < previous_scale,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate fixed validation data with independent, repeatable mask streams.

        Validation batches require ``image`` and pre-normalization ``valid``
        foreground masks, with optional ``sequence`` labels.
        """
        was_training = self.model.training
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 1000)
        ratios = self.config.val_mask_ratios
        fixed_generators = [
            torch.Generator(device=self.device).manual_seed(self.config.seed + 2000 + i) for i in range(len(ratios))
        ]
        totals: dict[str, tuple[float, float]] = {}
        try:
            for batch in self.val_dataloader:
                x = batch["image"].to(self.device)
                foreground = batch["valid"].to(self.device)
                ratio = self.config.mask_ratio
                if ratio is None:
                    low, high = self.config.mask_range
                    ratio = low + (high - low) * torch.rand((), generator=generator, device=self.device).item()
                evaluations = [(None, ratio, generator)]
                evaluations += [(f"{r:g}", r, rng) for r, rng in zip(ratios, fixed_generators)]
                for label, ratio, rng in evaluations:
                    with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                        prediction, mask = self.model(x, masking_ratio=ratio, generator=rng)
                    prefix = "val" if label is None else f"val/mask_ratio_{label}"
                    for i in range(x.shape[0]):
                        metrics = reconstruction_totals(
                            prediction[i : i + 1],
                            x[i : i + 1],
                            mask[i : i + 1],
                            foreground[i : i + 1],
                            ssim_data_range=self.config.ssim_data_range,
                        )
                        add_metric_totals(totals, prefix, metrics)
                        if label is None:
                            sequence = batch.get("sequence", ["unknown"] * x.shape[0])[i]
                            add_metric_totals(totals, f"val/sequence_{sequence}", metrics)
            metrics = {key: n / d for key, (n, d) in totals.items() if d > 0}
            if "val/masked_mse" not in metrics:
                raise ValueError("Validation requires nonempty data with masked voxels.")
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite validation metrics.")
            return metrics
        finally:
            self.model.train(was_training)

    def save_checkpoint(self, path: Path, *, completed: int, best: float, val_loss: float | None) -> None:
        """Save atomically; worker prefetch and augmentation are not replayed on resume."""
        state = {
            "format": "MAETrainer-v1",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": completed,
            "total_steps": self._total_steps,
            "best_val_loss": best,
            "val_loss": val_loss,
            "config": asdict(self.config),
            "mask_rng": self.mask_generator.get_state(),
            "data_rng": self.data_generator.get_state(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        if self.run is not None:
            self.run.log_model(path=str(path))

    def load_checkpoint(self, path: Path) -> tuple[int, float, float | None]:
        """Restore a trainer checkpoint after checking the training recipe."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "MAETrainer-v1":
            raise ValueError("Resume requires an MAETrainer checkpoint.")
        if state["total_steps"] != self._total_steps:
            raise ValueError("Resume requires the original training step budget.")
        config = asdict(self.config)
        for key in (
            "dataloader_config",
            "seed",
            "local_batch_size",
            "gradient_accumualtion_steps",
            "optimizer_config",
            "mask_ratio",
            "mask_range",
            "grad_clip",
            "poly_power",
            "val_mask_ratios",
            "ssim_data_range",
        ):
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume requires the original {key} setting.")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.mask_generator.set_state(state["mask_rng"])
        self.data_generator.set_state(state["data_rng"])
        self._train_batches = None
        return state["step"], state["best_val_loss"], state["val_loss"]

    def train(self, *, batch_augmenter: Any = None) -> None:
        total_steps = self.config.steps
        if total_steps is None:
            total_steps = self.get_training_steps()
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("get_training_steps() must return a positive integer.")
        self._total_steps = total_steps
        last_checkpoint_path = self.output_dir / "last.pt"
        # Keep the original LR, not a checkpoint's already-decayed LR.
        initial_lr = float(self.config.optimizer_config["lr"])
        steps_completed = 0
        best_validation_loss = float("inf")
        val_loss = None
        if self.config.resume:
            steps_completed, best_validation_loss, val_loss = self.load_checkpoint(last_checkpoint_path)
        elif last_checkpoint_path.exists():
            raise FileExistsError(f"{last_checkpoint_path} exists; enable resume or choose a new output directory.")
        if steps_completed >= total_steps:
            return
        self._train_batches = None
        total_mse, total_mae, logged_steps = 0.0, 0.0, 0
        with tqdm(total=total_steps, initial=steps_completed, desc="Training", unit="update") as progress:
            while steps_completed < total_steps:
                self.model.train()
                lr = initial_lr * (1 - steps_completed / total_steps) ** self.config.poly_power
                metrics = self.train_step(lr=lr, batch_augmenter=batch_augmenter)
                if metrics["optimizer_step_skipped"]:
                    tqdm.write("AMP overflow: retrying this optimizer step with a reduced scale.")
                    continue
                steps_completed += 1
                progress.update(1)
                total_mse += float(metrics["train/masked_mse"])
                total_mae += float(metrics["train/masked_mae"])
                logged_steps += 1
                final_step = steps_completed == total_steps
                if steps_completed % self.config.log_every == 0 or final_step:
                    progress.set_postfix(train_loss=f"{total_mse / logged_steps:.5f}", lr=f"{lr:.6g}")
                    if self.run is not None:
                        self.run.log(
                            {
                                "optimizer_step": steps_completed,
                                "train/masked_mse": total_mse / logged_steps,
                                "train/masked_mae": total_mae / logged_steps,
                                "train/lr": lr,
                                "train/mask_ratio": metrics["train/mask_ratio"],
                            }
                        )
                    total_mse, total_mae, logged_steps = 0.0, 0.0, 0
                if steps_completed % self.config.validate_every == 0 or final_step:
                    val_metrics = self.validate()
                    val_loss = val_metrics["val/masked_mse"]
                    if not math.isfinite(val_loss):
                        raise FloatingPointError("Nonfinite validation loss.")
                    improved = val_loss < best_validation_loss
                    best_validation_loss = min(best_validation_loss, val_loss)
                    if self.run is not None:
                        self.run.log(
                            {
                                "optimizer_step": steps_completed,
                                **val_metrics,
                                "val/best_masked_mse": best_validation_loss,
                            }
                        )
                    if improved:
                        self.save_checkpoint(
                            self.output_dir / "best.pt",
                            completed=steps_completed,
                            best=best_validation_loss,
                            val_loss=val_loss,
                        )
                if steps_completed % self.config.save_every == 0 or final_step:
                    self.save_checkpoint(
                        last_checkpoint_path,
                        completed=steps_completed,
                        best=best_validation_loss,
                        val_loss=val_loss,
                    )
        if self.run is not None:
            self.run.finish()
