import os

import torch
import numpy as np
from typing import *
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.models import vgg as vgg


def kernel_views(kernel: Tensor, n: int = 2) -> List[Tensor]:
    r"""Returns the :math:`N`-dimensional views of the 1-dimensional kernel `kernel`.

    Args:
        kernel: A kernel, :math:`(C, 1, K)`.
        n: The number of dimensions :math:`N`.

    Returns:
        The list of views, each :math:`(C, 1, \underbrace{1, \dots, 1}_{i}, K, \underbrace{1, \dots, 1}_{N - i - 1})`.

    Example:
        >>> kernel = gaussian_kernel(5, sigma=1.5).repeat(3, 1, 1)
        >>> kernel.shape
        torch.Size([3, 1, 5])
        >>> views = kernel_views(kernel, n=2)
        >>> views[0].shape, views[1].shape
        (torch.Size([3, 1, 5, 1]), torch.Size([3, 1, 1, 5]))
    """

    if n == 1:
        return [kernel]
    elif n == 2:
        return [kernel.unsqueeze(-1), kernel.unsqueeze(-2)]

    # elif n > 2:
    c, _, k = kernel.shape

    shape: List[int] = [c, 1] + [1] * n
    views = []

    for i in range(2, n + 2):
        shape[i] = k
        views.append(kernel.reshape(shape))
        shape[i] = 1

    return views


def channel_conv(
    x: Tensor,
    kernel: Tensor,
    padding: int = 0,
) -> Tensor:
    r"""Returns the channel-wise convolution of :math:`x` with the kernel `kernel`.

    Args:
        x: A tensor, :math:`(N, C, *)`.
        kernel: A kernel, :math:`(C', 1, *)`.
        padding: The implicit paddings on both sides of the input dimensions.

    Example:
        >>> x = torch.arange(25).float().reshape(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernel = torch.ones((1, 1, 3, 3))
        >>> channel_conv(x, kernel)
        tensor([[[[ 54.,  63.,  72.],
                  [ 99., 108., 117.],
                  [144., 153., 162.]]]])
    """

    D = kernel.dim() - 2

    assert D <= 3, "PyTorch only supports 1D, 2D or 3D convolutions."

    if D == 3:
        return F.conv3d(x, kernel, padding=padding, groups=x.shape[-4])
    elif D == 2:
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[-3])
    elif D == 1:
        return F.conv1d(x, kernel, padding=padding, groups=x.shape[-2])
    else:
        return F.linear(x, kernel.expand(x.shape[-1]))


def channel_convs(
    x: Tensor,
    kernels: List[Tensor],
    padding: int = 0,
) -> Tensor:
    r"""Returns the channel-wise convolution of :math:`x` with the series of
    kernel `kernels`.

    Args:
        x: A tensor, :math:`(N, C, *)`.
        kernels: A list of kernels, each :math:`(C', 1, *)`.
        padding: The implicit paddings on both sides of the input dimensions.

    Example:
        >>> x = torch.arange(25).float().reshape(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernels = [torch.ones((1, 1, 3, 1)), torch.ones((1, 1, 1, 3))]
        >>> channel_convs(x, kernels)
        tensor([[[[ 54.,  63.,  72.],
                  [ 99., 108., 117.],
                  [144., 153., 162.]]]])
    """

    if padding > 0:
        pad = (padding,) * (2 * x.dim() - 4)
        x = F.pad(x, pad=pad)

    for k in kernels:
        x = channel_conv(x, k)

    return x


def gaussian_kernel(
    size: int,
    sigma: float = 1.0,
) -> Tensor:
    r"""Returns the 1-dimensional Gaussian kernel of size :math:`K`.

    .. math::
        G(x) = \gamma \exp \left(\frac{(x - \mu)^2}{2 \sigma^2}\right)

    where :math:`\gamma` is such that

    .. math:: \sum_{x = 1}^{K} G(x) = 1

    and :math:`\mu = \frac{1 + K}{2}`.

    Wikipedia:
        https://wikipedia.org/wiki/Gaussian_blur

    Note:
        An :math:`N`-dimensional Gaussian kernel is separable, meaning that
        applying it is equivalent to applying a series of :math:`N` 1-dimensional
        Gaussian kernels, which has a lower computational complexity.

    Args:
        size: The kernel size :math:`K`.
        sigma: The standard deviation :math:`\sigma` of the distribution.

    Returns:
        The kernel vector, :math:`(K,)`.

    Example:
        >>> gaussian_kernel(5, sigma=1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """

    kernel = torch.arange(size, dtype=torch.float)
    kernel -= (size - 1) / 2
    kernel = kernel ** 2 / (2 * sigma ** 2)
    kernel = torch.exp(-kernel)
    kernel /= kernel.sum()

    return kernel




@torch.jit.script_if_tracing
def ssim(
    x: Tensor,
    y: Tensor,
    kernel: Tensor,
    channel_avg: bool = True,
    padding: bool = False,
    value_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the SSIM and Contrast Sensitivity (CS) between :math:`x` and :math:`y`.

    .. math::
        \text{SSIM}(x, y) & =
            \frac{2 \mu_x \mu_y + C_1}{\mu^2_x + \mu^2_y + C_1} \text{CS}(x, y) \\
        \text{CS}(x, y) & =
            \frac{2 \sigma_{xy} + C_2}{\sigma^2_x + \sigma^2_y + C_2}

    where :math:`\mu_x`, :math:`\mu_y`, :math:`\sigma^2_x`, :math:`\sigma^2_y` and
    :math:`\sigma_{xy}` are the results of a smoothing convolution over :math:`x`,
    :math:`y`, :math:`(x - \mu_x)^2`, :math:`(y - \mu_y)^2` and :math:`(x - \mu_x)(y -
    \mu_y)`, respectively.

    In practice, SSIM and CS are averaged over the spatial dimensions. If `channel_avg`
    is :py:`True`, they are also averaged over the channels.

    Tip:
        :func:`ssim` and :class:`SSIM` can be applied to images with 1, 2 or even
        3 spatial dimensions.

    Args:
        x: An input tensor, :math:`(N, C, H, *)`.
        y: A target tensor, :math:`(N, C, H, *)`.
        kernel: A smoothing kernel, :math:`(C, 1, K)`.
        channel_avg: Whether to average over the channels or not.
        padding: Whether to pad with :math:`\frac{K}{2}` zeros the spatial
            dimensions or not.
        value_range: The value range :math:`L` of the inputs (usually 1 or 255).

    Note:
        For the remaining arguments, refer to Wang et al. (2004).

    Returns:
        The SSIM and CS tensors, both :math:`(N, C)` or :math:`(N,)`
        depending on `channel_avg`.

    Example:
        >>> x = torch.rand(5, 3, 64, 64, 64)
        >>> y = torch.rand(5, 3, 64, 64, 64)
        >>> kernel = gaussian_kernel(7).repeat(3, 1, 1)
        >>> ss, cs = ssim(x, y, kernel)
        >>> ss.shape, cs.shape
        (torch.Size([5]), torch.Size([5]))
    """

    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2

    window = kernel_views(kernel, x.dim() - 2)

    if padding:
        pad = kernel.shape[-1] // 2
    else:
        pad = 0

    # Mean (mu)
    mu_x = channel_convs(x, window, pad)
    mu_y = channel_convs(y, window, pad)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_xx = channel_convs(x ** 2, window, pad) - mu_xx
    sigma_yy = channel_convs(y ** 2, window, pad) - mu_yy
    sigma_xy = channel_convs(x * y, window, pad) - mu_xy

    # Contrast sensitivity (CS)
    cs = (2 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    # Average
    if channel_avg:
        ss, cs = ss.flatten(1), cs.flatten(1)
    else:
        ss, cs = ss.flatten(2), cs.flatten(2)

    return ss.mean(dim=-1), cs.mean(dim=-1)
