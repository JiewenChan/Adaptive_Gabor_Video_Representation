from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

UINT16_MAX = 65535


class SceneNormDict(TypedDict):
    scale: float
    transfm: torch.Tensor


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [to_device(v, device) for v in batch]
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def normalize_coords(coords, h, w):
    assert coords.shape[-1] == 2
    return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [-inf, inf], np.float32
      expected_dist:, [-inf, inf], np.float32

    Returns:
      visibles: bool
    """

    def sigmoid(x):
        if x.dtype == np.ndarray:
            return 1 / (1 + np.exp(-x))
        else:
            return torch.sigmoid(x)

    visibles = (1 - sigmoid(occlusions)) * (1 - sigmoid(expected_dist)) > 0.5
    return visibles


def parse_tapir_track_info(occlusions, expected_dist, threshold=0.3):
    """
    Use minimal filtering to preserve basic quality.
    
    Args:
        occlusions: occlusion values; higher means more occluded
        expected_dist: expected distance/confidence
    
    Returns:
        visibles: visible mask (occlusions < 0.5)
        invisibles: invisible mask (occlusions >= 0.5)
        confidences: confidence scores
    """
    # Larger occlusions mean more occluded; (1 - occlusions) indicates visibility.
    confidence = expected_dist
    occlusion_score = occlusions
    valid_invisible = occlusion_score * confidence > threshold
    valid_visible = (1 - occlusion_score) * confidence > threshold
    # set all confidence < 0.5 to 0
    confidence = confidence * (valid_visible | valid_invisible).float()
    return valid_visible, valid_invisible, confidence
    

def spatial_grid_sampling(tracks_3d, target_num_points, grid_size=32, image_size=(480, 854)):
    """
    Spatial grid sampling to ensure even scene coverage.
    
    Args:
        tracks_3d: [N, T, 3] 3D trajectories
        target_num_points: target number of points
        grid_size: grid size
        image_size: image size (H, W)
    
    Returns:
        selected_indices: selected point indices
    """
    if tracks_3d.shape[0] <= target_num_points:
        return torch.arange(tracks_3d.shape[0])
    
    H, W = image_size
    query_points = tracks_3d[:, 0, :]  # Use 3D points from the query frame.
    
    # Project 3D points to a 2D grid.
    x_coords = (query_points[:, 0] + 1) / 2  # Map [-1,1] to [0,1].
    y_coords = (query_points[:, 1] + 1) / 2  # Map [-1,1] to [0,1].
    
    # Create 2D grid indices.
    grid_x = (x_coords * (grid_size - 1)).long().clamp(0, grid_size - 1)
    grid_y = (y_coords * (grid_size - 1)).long().clamp(0, grid_size - 1)
    grid_indices = grid_x * grid_size + grid_y
    
    # Count points per grid.
    unique_grids, counts = torch.unique(grid_indices, return_counts=True)
    
    selected_indices = []
    points_per_grid = max(1, target_num_points // len(unique_grids))
    
    # Select points per grid.
    for grid_idx in unique_grids:
        grid_points = torch.where(grid_indices == grid_idx)[0]
        n_select = min(points_per_grid, len(grid_points))
        
        if n_select > 0:
            # Use depth as a quality metric.
            grid_depths = query_points[grid_points, 2]
            _, top_indices = torch.topk(grid_depths, n_select)
            selected = grid_points[top_indices]
            selected_indices.append(selected)
    
    if selected_indices:
        result = torch.cat(selected_indices)
        return result[:target_num_points]  # Ensure we do not exceed target count.
    else:
        return torch.arange(min(target_num_points, tracks_3d.shape[0]))


def get_tracks_3d_for_query_frame(
    query_index: int,
    query_img: torch.Tensor,
    tracks_2d: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    extract_fg: bool = True,
    min_points: int = 1000,
    max_points: int = 8000,
    use_spatial_sampling: bool = True,
):
    """
    Improved 3D track extraction with looser filtering and better spatial coverage.
    
    Args:
        query_index: query frame index
        query_img: query frame image [H, W, 3]
        tracks_2d: 2D tracks [N, T, 4] (x, y, occlusion, confidence)
        depths: depth maps [T, H, W]
        masks: masks [T, H, W]
        extract_fg: whether to extract foreground
        min_points: minimum points
        max_points: maximum points
        use_spatial_sampling: whether to use spatial sampling
    
    Returns:
        tracks_3d: [N, T, 3] 3D tracks
        track_colors: [N, 3] track colors
        visibles: [N, T] visibility
        invisibles: [N, T] invisibility
        confidences: [N, T] confidences
    """
    T, H, W = depths.shape
    query_img = query_img[None].permute(0, 3, 1, 2)  # (1, 3, H, W)
    tracks_2d = tracks_2d.swapaxes(0, 1)  # (T, N, 4)
    tracks_2d, occs, dists = (
        tracks_2d[..., :2],
        tracks_2d[..., 2],
        tracks_2d[..., 3],
    )
    # Parse track info.
    visibles, invisibles, confidences = parse_tapir_track_info(occs, dists)
    # Unproject 2D tracks to 3D.
    # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
    track_depths = F.grid_sample(
        depths[:, None],
        normalize_coords(tracks_2d[:, None], H, W),
        align_corners=True,
        padding_mode="border",
    )[:, 0, 0]
    
    # Handle occluded frames by referencing visible depth from other time points.
    if invisibles.any():
        invisible_mask = invisibles.bool()  # [T, N]
        visible_mask = visibles.bool()

        reference_depths = track_depths.clone()
        last_visible = torch.full(
            (track_depths.shape[1],),
            float("nan"),
            device=track_depths.device,
            dtype=track_depths.dtype,
        )
        for t in range(track_depths.shape[0]):
            cur_visible = visible_mask[t]
            last_visible = torch.where(cur_visible, track_depths[t], last_visible)
            reference_depths[t] = torch.where(cur_visible, track_depths[t], last_visible)

        next_visible = torch.full(
            (track_depths.shape[1],),
            float("nan"),
            device=track_depths.device,
            dtype=track_depths.dtype,
        )
        for t in range(track_depths.shape[0] - 1, -1, -1):
            cur_visible = visible_mask[t]
            next_visible = torch.where(cur_visible, track_depths[t], next_visible)
            need_fill = torch.isnan(reference_depths[t])
            reference_depths[t] = torch.where(need_fill, next_visible, reference_depths[t])

        reference_depths = torch.where(
            torch.isnan(reference_depths), track_depths, reference_depths
        )
        track_depths = torch.where(invisible_mask, reference_depths, track_depths)

    ############ TODO add interface. we use a orthographic camera here ############
    image_size_wh = torch.tensor([W, H], device=tracks_2d.device)[None]
    tracks_2d_norm = (tracks_2d - image_size_wh / 2) / (image_size_wh / 2)
    tracks_3d = torch.cat([tracks_2d_norm, track_depths[..., None]], dim=-1)
    # Filter out out-of-mask tracks.
    # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
    is_in_masks = (
        F.grid_sample(
            masks[:, None],
            normalize_coords(tracks_2d[:, None], H, W),
            align_corners=True,
        )[:, 0, 0]
        == 1
    )
    visibles *= is_in_masks
    invisibles *= is_in_masks
    confidences *= is_in_masks.float()

    # valid if in the fg mask at least 40% of the time
    # in_mask_counts = is_in_masks.sum(0)
    # t = 0.25
    # thresh = min(t * T, in_mask_counts.float().quantile(t).item())
    # valid = in_mask_counts > thresh
    valid = is_in_masks[query_index]

    # Minimal filtering: only basic mask checks.
    if extract_fg:
        # Foreground: only needs to be inside the query-frame mask.
        valid = is_in_masks[query_index]
    else:
        # Background: inside the mask at any time.
        valid = is_in_masks.any(0)
    
    # If too many points, use spatial sampling.
    if use_spatial_sampling and valid.sum().item() > max_points:
        all_tracks_3d = tracks_3d[:, valid]
        selected_indices = spatial_grid_sampling(
            all_tracks_3d, max_points, grid_size=32, image_size=(H, W)
        )
        valid_indices = torch.where(valid)[0]
        valid_indices = valid_indices[selected_indices]
        valid = torch.zeros_like(valid)
        valid[valid_indices] = True

    # Get track's color from the query frame.
    # (1, 3, H, W), (1, 1, N, 2) -> (1, 3, 1, N) -> (N, 3)
    track_colors = F.grid_sample(
        query_img,
        normalize_coords(tracks_2d[query_index : query_index + 1, None], H, W),
        align_corners=True,
        padding_mode="border",
    )[0, :, 0].T
    return (
        tracks_3d[:, valid].swapdims(0, 1),
        track_colors[valid],
        visibles[:, valid].swapdims(0, 1),
        invisibles[:, valid].swapdims(0, 1),
        confidences[:, valid].swapdims(0, 1),
    )


def _get_padding(x, k, stride, padding, same: bool):
    if same:
        ih, iw = x.size()[2:]
        if ih % stride[0] == 0:
            ph = max(k[0] - stride[0], 0)
        else:
            ph = max(k[0] - (ih % stride[0]), 0)
        if iw % stride[1] == 0:
            pw = max(k[1] - stride[1], 0)
        else:
            pw = max(k[1] - (iw % stride[1]), 0)
        pl = pw // 2
        pr = pw - pl
        pt = ph // 2
        pb = ph - pt
        padding = (pl, pr, pt, pb)
    else:
        padding = padding
    return padding


def median_filter_2d(x, kernel_size=3, stride=1, padding=1, same: bool = True):
    """
    :param x [B, C, H, W]
    """
    k = _pair(kernel_size)
    stride = _pair(stride)  # convert to tuple
    padding = _quadruple(padding)  # convert to l, r, t, b
    # using existing pytorch functions and tensor ops so that we get autograd,
    # would likely be more efficient to implement from scratch at C/Cuda level
    x = F.pad(x, _get_padding(x, k, stride, padding, same), mode="reflect")
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


def masked_median_blur(image, mask, kernel_size=11):
    """
    Args:
        image: [B, C, H, W]
        mask: [B, C, H, W]
        kernel_size: int
    """
    assert image.shape == mask.shape
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {image.shape}")

    padding: Tuple[int, int] = _compute_zero_padding((kernel_size, kernel_size))

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d((kernel_size, kernel_size)).to(image)
    b, c, h, w = image.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(
        image.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1
    )
    masks: torch.Tensor = F.conv2d(
        mask.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1
    )
    features = features.view(b, c, -1, h, w).permute(
        0, 1, 3, 4, 2
    )  # BxCxxHxWx(K_h * K_w)
    min_value, max_value = features.min(), features.max()
    masks = masks.view(b, c, -1, h, w).permute(0, 1, 3, 4, 2)  # BxCxHxWx(K_h * K_w)
    index_invalid = (1 - masks).nonzero(as_tuple=True)
    index_b, index_c, index_h, index_w, index_k = index_invalid
    features[(index_b[::2], index_c[::2], index_h[::2], index_w[::2], index_k[::2])] = (
        min_value
    )
    features[
        (index_b[1::2], index_c[1::2], index_h[1::2], index_w[1::2], index_k[1::2])
    ] = max_value
    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=-1)[0]

    return median


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def get_binary_kernel2d(
    window_size: Union[Tuple[int, int], int],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    from kornia
    Create a binary kernel to extract the patches.
    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = torch.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def _unpack_2d_ks(kernel_size: Union[Tuple[int, int], int]) -> Tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, "2D Kernel size should have a length of 2."
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)


## Functions from GaussianShader.
def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (
        W - 1
    )
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (
        H - 1
    )
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x, indexing="ij")
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(
        B, N, C, H, W, 3
    )  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)  # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz


def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(
        depth_image[None, None, None, ...], intrinsic_matrix[None, ...]
    )
    xyz_cam = xyz_cam.reshape(-1, 3)
    xyz_world = torch.cat(
        [xyz_cam, torch.ones_like(xyz_cam[..., 0:1])], dim=-1
    ) @ torch.inverse(extrinsic_matrix).transpose(0, 1)
    xyz_world = xyz_world[..., :3]

    return xyz_world


def depth_pcd2normal(xyz):
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
    top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = xyz[..., 1 : hd - 1, 2:wd, :]
    left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)
    return xyz_normal


def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix)  # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world)

    return xyz_normal
