from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import einops

from timm.models.layers import trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import conv_output_shape, same_padding

ModalitySizeT = Mapping[str, int]
PreprocessorOutputT = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorT = Callable[..., PreprocessorOutputT]
PostprocessorT = Callable[..., Any]




def space_to_depth(
        frames: torch.Tensor,
        temporal_block_size: int = 1,
        spatial_block_size: int = 1) -> torch.Tensor:
    """Reduces spatial and/or temporal dimensions by stacking features in the channel dimension."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames, "b (h dh) (w dw) c -> b h w (dh dw c)",
            dh=spatial_block_size, dw=spatial_block_size)
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames, "b (t dt) (h dh) (w dw) c -> b t h w (dt dh dw c)",
            dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)")


def reverse_space_to_depth(
        frames: torch.Tensor,
        temporal_block_size: int = 1,
        spatial_block_size: int = 1) -> torch.Tensor:
    """Reverse space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames, "b h w (dh dw c) -> b (h dh) (w dw) c",
            dh=spatial_block_size, dw=spatial_block_size)
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames, "b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c",
            dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)")


def extract_patches(images: torch.Tensor,
                    size: Sequence[int],
                    stride: Sequence[int] = 1,
                    dilation: Sequence[int] = 1,
                    padding: str = "VALID") -> torch.Tensor:
    """Extract patches from images.
  The function extracts patches of shape sizes from the input images in the same
  manner as a convolution with kernel of shape sizes, stride equal to strides,
  and the given padding scheme.
  The patches are stacked in the channel dimension.
  Args:
    images (torch.Tensor): input batch of images of shape [B, C, H, W].
    size (Sequence[int]): size of extracted patches. Must be [patch_height, patch_width].
    stride (Sequence[int]): strides, must be [stride_rows, stride_cols]. Default: 1
    dilation (Sequence[int]): as in dilated convolutions, must be [dilation_rows, dilation_cols]. Default: 1
    padding (str): padding algorithm to use. Default: VALID
  Returns:
    Tensor of shape [B, patch_rows, patch_cols, size_rows * size_cols * C]
  """
    if padding != "VALID":
        raise ValueError(f"Only valid padding is supported. Got {padding}")

    if images.ndim != 4:
        raise ValueError(
            f"Rank of images must be 4 (got tensor of shape {images.shape})")

    n, c, h, w = images.shape
    ph, pw = size

    pad = 0
    out_h, out_w = conv_output_shape((h, w), size, stride, pad, dilation)

    patches = F.unfold(images, size, dilation=dilation, padding=0, stride=stride)

    patches = einops.rearrange(patches, "n (c ph pw) (out_h out_w) -> n out_h out_w (ph pw c)",
                               c=c, ph=ph, pw=pw, out_h=out_h, out_w=out_w)
    return patches


def patches_for_flow(inputs: torch.Tensor) -> torch.Tensor:
    """Extract 3x3x2 image patches for flow inputs.
    Args:
        inputs (torch.Tensor): image inputs (N, 2, C, H, W) """

    batch_size = inputs.shape[0]

    inputs = einops.rearrange(inputs, "N T C H W -> (N T) C H W")
    padded_inputs = F.pad(inputs, [1, 1, 1, 1], mode="constant")
    outputs = extract_patches(
        padded_inputs,
        size=[3, 3],
        stride=1,
        dilation=1,
        padding="VALID")

    outputs = einops.rearrange(outputs, "(N T) H W C-> N T H W C", N=batch_size)

    return outputs


#  ------------------------------------------------------------
#  -------------------  Up/down-sampling  ---------------------
#  ------------------------------------------------------------


class Conv2DDownsample(nn.Module):
    """Downsamples 4x by applying a 2D convolution and doing max pooling."""

    def __init__(
            self,
            num_layers: int = 1,
            in_channels: int = 3,
            num_channels: int = 64,
            use_batchnorm: bool = True
    ):
        """Constructs a Conv2DDownsample model.
    Args:
      num_layers (int): The number of conv->max_pool layers. Default: 1
      in_channels (int): The number of input channels. Default: 3
      num_channels (int): The number of conv output channels. Default: 64
      use_batchnorm (bool): Whether to use batchnorm. Default: True
    """
        super().__init__()

        self._num_layers = num_layers
        self.norms = None
        if use_batchnorm:
            self.norms = nn.ModuleList()

        self.convs = nn.ModuleList()
        for _ in range(self._num_layers):
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=num_channels,
                             kernel_size=7,
                             stride=2,
                             bias=False)
            trunc_normal_(conv.weight, mean=0.0, std=0.01)
            self.convs.append(conv)
            in_channels = num_channels

            if use_batchnorm:
                batchnorm = nn.BatchNorm2d(num_features=num_channels)
                self.norms.append(batchnorm)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs
        for l, conv in enumerate(self.convs):
            pad = same_padding(out.shape[1:], conv.kernel_size, conv.stride, dims=2)
            out = F.pad(out, pad, mode="constant", value=0.0)
            out = conv(out)

            if self.norms is not None:
                out = self.norms[l](out)

            out = F.relu(out)

            pad = same_padding(out.shape[1:], 3, 2, dims=2)
            out = F.pad(out, pad, mode="constant", value=0.0)

            out = F.max_pool2d(out, kernel_size=3, stride=2)

        return out
