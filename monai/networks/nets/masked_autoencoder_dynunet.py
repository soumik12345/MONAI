from collections.abc import Sequence
from math import prod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .dynunet import DynUNet, UnetResBlock
from ...utils.misc import ensure_tuple_rep


def _masked_instance_norm(
    x: torch.Tensor,
    norm: nn.Module,
    visible: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(norm, (nn.InstanceNorm2d, nn.InstanceNorm3d)):
        raise TypeError("norm must be an InstanceNorm2d or InstanceNorm3d layer.")

    # Preserve double precision, but promote half/bfloat16 reductions to FP32.
    work = x if x.dtype == torch.float64 else x.float()
    active = visible.to(work.dtype)
    dims = tuple(range(2, x.ndim))

    count = active.sum(dim=dims, keepdim=True).clamp_min(1)
    mean = (work * active).sum(dim=dims, keepdim=True) / count
    variance = ((work - mean).square() * active).sum(dim=dims, keepdim=True) / count

    out = (work - mean) * torch.rsqrt(variance + norm.eps)

    if norm.affine:
        weight, bias = norm.weight, norm.bias
        if weight is None or bias is None:
            raise ValueError("Affine instance normalization requires weight and bias parameters.")
        shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
        out = out * weight.to(work.dtype).reshape(shape)
        out = out + bias.to(work.dtype).reshape(shape)

    return (out * active).to(x.dtype)


def _masked_resblock(
    block: UnetResBlock,
    x: torch.Tensor,
    visible_in: torch.Tensor,
    visible_out: torch.Tensor,
) -> torch.Tensor:
    x = x * visible_in.to(x.dtype)

    out = block.conv1(x)
    out = out * visible_out.to(out.dtype)
    out = block.lrelu(_masked_instance_norm(out, block.norm1, visible_out))

    out = block.conv2(out)
    out = out * visible_out.to(out.dtype)
    out = _masked_instance_norm(out, block.norm2, visible_out)

    residual = x
    if hasattr(block, "conv3"):
        residual = block.conv3(residual)
        residual = residual * visible_out.to(residual.dtype)
        residual = _masked_instance_norm(residual, block.norm3, visible_out)

    return block.lrelu(out + residual) * visible_out.to(out.dtype)


def _densify(
    feature: torch.Tensor,
    visible: torch.Tensor,
    norm: nn.Module,
    token: torch.Tensor,
    projection: nn.Module,
) -> torch.Tensor:
    feature = _masked_instance_norm(feature, norm, visible)
    feature = torch.where(visible.bool(), feature, token.to(feature.dtype))
    return projection(feature)


class MaskedAutoEncoderDynUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_dims: int,
        kernel_size: Sequence[Sequence[int] | int],
        strides: Sequence[Sequence[int] | int],
        filters: Sequence[int],
        masking_ratio: float = 0.75,
        enable_activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = tuple(ensure_tuple_rep(k, spatial_dims) for k in kernel_size)
        strides = tuple(ensure_tuple_rep(s, spatial_dims) for s in strides)
        filters = tuple(filters)

        self._check_parameters(in_channels, spatial_dims, masking_ratio, kernel_size, strides, filters)

        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters
        self.masking_ratio = masking_ratio
        self.enable_activation_checkpointing = enable_activation_checkpointing

        self.downsample = tuple(prod(stride[d] for stride in self.strides) for d in range(spatial_dims))

        self.backbone = DynUNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            upsample_kernel_size=self.strides[1:],
            filters=self.filters,
            norm_name=("INSTANCE", {"affine": True}),
            res_block=True,
            deep_supervision=False,
        )

        self._init_densification_layers(spatial_dims)

    def _check_parameters(
        self,
        in_channels: int,
        spatial_dims: int,
        masking_ratio: float,
        kernel_size: tuple[tuple[Any, ...], ...],
        strides: tuple[tuple[Any, ...], ...],
        filters: tuple[int, ...],
    ) -> None:
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")
        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if not 0 < masking_ratio < 1:
            raise ValueError("masking_ratio must be strictly between 0 and 1.")
        if any(c <= 0 for c in filters):
            raise ValueError("filters must contain positive integers.")
        if not (len(kernel_size) == len(strides) == len(filters) >= 3):
            raise ValueError("Provide matching kernel, stride, and filter lengths >= 3.")
        if any(k <= 0 or k % 2 == 0 for kernel in kernel_size for k in kernel):
            raise ValueError("kernel_size must contain positive odd integers.")
        if any(s not in (1, 2) for stride in strides for s in stride):
            raise ValueError("Each stride component must be 1 or 2.")
        if any(s != 1 for s in strides[0]):
            raise ValueError("The first stride must be 1 in every dimension.")

    def _init_densification_layers(self, spatial_dims: int):
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        norm = nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d

        # create one trainable token per feature level
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros((1, c) + (1,) * spatial_dims)) for c in self.filters]
        )

        # create a normalization module per level
        self.densify_norms = nn.ModuleList([norm(c, affine=False) for c in self.filters])

        # creates a convolution at each level except the highest resolution
        self.densify_projections = nn.ModuleList(
            [nn.Identity()] + [conv(c, c, kernel_size=3, padding=1) for c in self.filters[1:]]
        )

        for token in self.mask_tokens:
            nn.init.trunc_normal_(token, std=0.02, a=-0.02, b=0.02)

    def _validate_input(self, x: torch.Tensor) -> tuple[int, ...]:
        if x.ndim != self.spatial_dims + 2:
            raise ValueError(f"Expected a rank-{self.spatial_dims + 2} input.")
        if x.shape[0] <= 0 or x.shape[1] != self.in_channels:
            raise ValueError("Input batch or channel count is invalid.")
        if not x.is_floating_point():
            raise ValueError("Input must be a floating-point tensor.")
        if any(s <= 0 or s % d for s, d in zip(x.shape[2:], self.downsample)):
            raise ValueError(f"Input spatial dimensions must be positive multiples of {self.downsample}.")
        grid = tuple(s // d for s, d in zip(x.shape[2:], self.downsample))
        if prod(grid) < 3:
            raise ValueError("The bottleneck grid must contain at least three positions.")
        return grid

    def _sample_mask(
        self,
        x: torch.Tensor,
        grid: Sequence[int],
        ratio: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if not 0 < ratio < 1:
            raise ValueError("masking_ratio must be strictly between 0 and 1.")

        n = prod(grid)
        count = min(n - 2, max(1, round(n * ratio)))

        order = torch.rand(x.shape[0], n, device=x.device, generator=generator).argsort(dim=1)

        mask = torch.zeros(x.shape[0], n, device=x.device, dtype=torch.float32)
        mask.scatter_(1, order[:, :count], 1)
        return mask.reshape(x.shape[0], 1, *grid)

    def _validate_mask(
        self,
        mask: torch.Tensor,
        x: torch.Tensor,
        grid: Sequence[int],
    ) -> torch.Tensor:
        if tuple(mask.shape) != (x.shape[0], 1, *grid):
            raise ValueError(f"Expected mask shape {(x.shape[0], 1, *grid)}.")
        if mask.device != x.device:
            raise ValueError("mask and input must be on the same device.")

        if not torch.all((mask == 0) | (mask == 1)).item():
            raise ValueError("mask must contain only 0 (visible) and 1 (hidden).")
        mask = mask.detach().to(dtype=torch.float32)
        hidden = mask.flatten(1).sum(dim=1)
        if torch.any((hidden < 1) | (hidden > prod(grid) - 2)).item():
            raise ValueError("Each mask needs at least one hidden and two visible positions.")
        return mask

    def encode(self, x: torch.Tensor, mask: torch.Tensor):
        grid = self._validate_input(x)
        mask = self._validate_mask(mask, x, grid)

        blocks = [self.backbone.input_block, *self.backbone.downsamples, self.backbone.bottleneck]
        visible = 1 - mask
        visible_in = F.interpolate(mask.float(), size=tuple(x.shape[2:]), mode="nearest")

        features = []
        for block, stride in zip(blocks, self.strides):
            visible_out = F.interpolate(
                input=visible.float(), size=tuple(s // t for s, t in zip(x.shape[2:], stride)), mode="nearest"
            )
            x = (
                checkpoint(_masked_resblock, block, x, visible_in, visible_out)
                if self.enable_activation_checkpointing and self.training and torch.is_grad_enabled()
                else _masked_resblock(block, x, visible_in, visible_out)
            )
            features.append(x)
            visible_in = visible_out

        return features

    def forward(
        self,
        x: torch.Tensor,
        masking_ratio: float | None = None,
        *,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        grid = self._validate_input(x)

        if mask is None:
            ratio = self.masking_ratio if masking_ratio is None else masking_ratio
            mask = self._sample_mask(x, grid, ratio, generator)
        else:
            if masking_ratio is not None or generator is not None:
                raise ValueError("An explicit mask cannot be combined with sampling arguments.")
            mask = self._validate_mask(mask, x, grid)

        features = self.encode(x, mask)

        dense = []
        for feature, norm, token, projection in zip(
            features,
            self.densify_norms,
            self.mask_tokens,
            self.densify_projections,
        ):
            visible = F.interpolate((1 - mask).float(), size=tuple(feature.shape[2:]), mode="nearest")
            dense.append(
                checkpoint(_densify, feature, visible, norm, token, projection)
                if self.enable_activation_checkpointing and self.training and torch.is_grad_enabled()
                else _densify(feature, visible, norm, token, projection)
            )

        out = dense[-1]
        for block, skip in zip(self.backbone.upsamples, dense[:-1][::-1]):
            out = (
                checkpoint(block, out, skip)
                if self.enable_activation_checkpointing and self.training and torch.is_grad_enabled()
                else block(out, skip)
            )

        return self.backbone.output_block(out), mask
