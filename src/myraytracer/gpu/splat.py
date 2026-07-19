from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from myraytracer.gpu.camera import Camera
from myraytracer.gpu.vec import cross, dot, normalize

# Added to the 2D covariance diagonal (in pixel^2) so a Gaussian never
# shrinks below a ~1px footprint -- the "low-pass filter" dilation of
# Kerbl et al. 2023 Sec. 6, needed because a too-small screen-space
# footprint would otherwise alias between pixel samples.
_COVARIANCE_DILATION = 0.3


@dataclass(frozen=True)
class Gaussians:
    """A batch of G anisotropic 3D Gaussians -- the scene representation of
    Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field
    Rendering", SIGGRAPH 2023.

    `opacity` (G,) is each Gaussian's peak alpha in [0, 1] and `color` (G, 3)
    is the RGB it deposits when fully opaque and centered on a pixel; both
    are consumed by `rasterize`'s front-to-back alpha compositing. All five
    tensors must live on the same device and share dim 0 (= G).
    """

    mean: torch.Tensor  # (G, 3) world-space centers
    scale: torch.Tensor  # (G, 3) per-axis standard deviations in the Gaussian's local frame
    quat: torch.Tensor  # (G, 4) rotation as a (w, x, y, z) quaternion, need not be unit-length
    opacity: torch.Tensor  # (G,)
    color: torch.Tensor  # (G, 3)


def _quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = normalize(quat).unbind(-1)
    row0 = torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1
    )
    row1 = torch.stack(
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1
    )
    row2 = torch.stack(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1
    )
    return torch.stack([row0, row1, row2], dim=-2)


def _camera_basis(
    camera: Camera, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin = camera.origin.to(device)
    look_at = camera.look_at.to(device)
    up = camera.up.to(device)
    forward = normalize((look_at - origin).unsqueeze(0)).squeeze(0)
    right = normalize(cross(forward, up).unsqueeze(0)).squeeze(0)
    true_up = cross(right, forward)
    return right, true_up, forward


def rasterize(
    gaussians: Gaussians,
    camera: Camera,
    width: int,
    height: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Rasterize `gaussians` into an (H, W, 3) image via EWA/3DGS splatting.

    Each Gaussian is projected to screen space and its 3D covariance is
    mapped to a 2D screen-space footprint through the local affine (Jacobian)
    approximation of the perspective projection (Zwicker et al. 2001). Every
    pixel is then shaded by alpha-compositing the Gaussians strictly
    front-to-back (nearest depth first):

        color_out = sum_i color_i * alpha_i * T_i,  T_i = prod_{j<i} (1 - alpha_j)

    where `alpha_i = opacity_i * footprint_i(pixel)` and `T_i` is the
    transmittance left over by nearer Gaussians -- `opacity` scales how much
    of a Gaussian's `color` is deposited at its own peak, and every Gaussian
    behind it is progressively occluded by `T_i`.
    """
    mean = gaussians.mean.to(device)
    scale = gaussians.scale.to(device)
    quat = gaussians.quat.to(device)
    opacity = gaussians.opacity.to(device)
    color = gaussians.color.to(device)
    dtype = mean.dtype

    origin = camera.origin.to(device)
    right, true_up, forward = _camera_basis(camera, device)

    half_height = math.tan(math.radians(camera.vfov_degrees) / 2)
    half_width = half_height * camera.aspect_ratio

    rel = mean - origin
    x_cam = dot(rel, right.expand_as(rel))
    y_cam = dot(rel, true_up.expand_as(rel))
    z_cam = dot(rel, forward.expand_as(rel))

    # Gaussians behind the camera have no valid projection. Rather than
    # branch on that per-Gaussian, `z_safe` substitutes a dummy 1.0 divisor
    # so every downstream division stays finite, and the actual exclusion
    # happens later via `torch.where(in_front, ...)` on `alpha`, which
    # replaces (not scales) the masked entries -- so no NaN/Inf from this
    # placeholder can leak into the composited image or its gradient.
    in_front = z_cam > 0.0
    z_safe = torch.where(in_front, z_cam, torch.ones_like(z_cam))

    mean_px_x = width * (x_cam / (z_safe * half_width) + 1) / 2
    mean_px_y = height * (1 - y_cam / (z_safe * half_height)) / 2
    mean_px = torch.stack([mean_px_x, mean_px_y], dim=-1)  # (G, 2)

    rotation = _quat_to_rotation_matrix(quat)  # (G, 3, 3)
    cov3d = rotation @ torch.diag_embed(scale * scale) @ rotation.transpose(-1, -2)

    world_to_camera = torch.stack([right, true_up, forward], dim=0).unsqueeze(0)  # (1, 3, 3)
    cov_cam = world_to_camera @ cov3d @ world_to_camera.transpose(-1, -2)  # (G, 3, 3)

    # Jacobian of camera-space (x, y, z) -> pixel-space (px, py), the local
    # affine approximation EWA splatting uses in place of the true (nonlinear)
    # perspective projection.
    zero = torch.zeros_like(z_safe)
    j_row0 = torch.stack(
        [
            width / (2 * half_width) / z_safe,
            zero,
            -width / (2 * half_width) * x_cam / z_safe**2,
        ],
        dim=-1,
    )
    j_row1 = torch.stack(
        [
            zero,
            -height / (2 * half_height) / z_safe,
            height / (2 * half_height) * y_cam / z_safe**2,
        ],
        dim=-1,
    )
    jacobian = torch.stack([j_row0, j_row1], dim=-2)  # (G, 2, 3)

    cov2d = jacobian @ cov_cam @ jacobian.transpose(-1, -2)  # (G, 2, 2)
    eye2 = torch.eye(2, dtype=dtype, device=device)
    cov2d = cov2d + _COVARIANCE_DILATION * eye2

    a = cov2d[..., 0, 0]
    b = cov2d[..., 0, 1]
    c = cov2d[..., 1, 1]
    det = a * c - b * b

    py_idx, px_idx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack([px_idx.reshape(-1), py_idx.reshape(-1)], dim=-1)  # (H*W, 2)

    diff = pixels.unsqueeze(1) - mean_px.unsqueeze(0)  # (H*W, G, 2)
    dx = diff[..., 0]
    dy = diff[..., 1]

    # Quadratic form dx^T Sigma^-1 dx using the closed-form 2x2 inverse
    # (Sigma^-1 = [[c, -b], [-b, a]] / det), avoiding a batched matrix solve.
    mahalanobis = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
    footprint = torch.exp(-0.5 * mahalanobis)  # (H*W, G)

    alpha = opacity.unsqueeze(0) * footprint
    alpha = torch.where(in_front.unsqueeze(0).expand_as(alpha), alpha, torch.zeros_like(alpha))
    alpha = torch.clamp(alpha, 0.0, 1.0)

    depth_order = torch.argsort(z_cam)
    alpha_sorted = alpha[:, depth_order]
    color_sorted = color[depth_order]

    # Exclusive cumulative product: transmittance[..., i] = prod_{j<i} (1 - alpha_j).
    transmittance = torch.cumprod(1 - alpha_sorted, dim=-1)
    leading_ones = torch.ones_like(transmittance[:, :1])
    transmittance = torch.cat([leading_ones, transmittance[:, :-1]], dim=-1)

    weight = alpha_sorted * transmittance
    image = (weight @ color_sorted).reshape(height, width, 3)
    return image
