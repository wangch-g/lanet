import os
import cv2
import random
import collections
import torch
import numpy as np

from functools import lru_cache

@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    """Create mesh-grid given batch size, height and width dimensions.

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: torch.dtype
        Tensor dtype
    device: str
        Tensor device
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    xs: torch.Tensor
        Batched mesh-grid x-coordinates (BHW).
    ys: torch.Tensor
        Batched mesh-grid y-coordinates (BHW).
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, ones=True, normalized=False):
    """Create an image mesh grid with shape B3HW given image shape BHW

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: str
        Tensor dtype
    device: str
        Tensor device
    ones : bool
        Use (x, y, 1) coordinates
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    grid: torch.Tensor
        Mesh-grid for the corresponding image shape (B3HW)
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    coords = [xs, ys]
    if ones:
        coords.append(torch.ones_like(xs))  # BHW
    grid = torch.stack(coords, dim=1)  # B3HW
    return grid

def warp_keypoints(keypoints, H):
    """Warp keypoints given a homography

    Parameters
    ----------
    keypoints: numpy.ndarray (N,2)
        Keypoint vector.
    H: numpy.ndarray (3,3)
        Homography.

    Returns
    -------
    warped_keypoints: numpy.ndarray (N,2)
        Warped keypoints vector.
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]

def draw_keypoints(img_l, top_uvz, color=(255, 0, 0), idx=0):
    """Draw keypoints on an image"""
    vis_xyd = top_uvz.permute(0, 2, 1)[idx].detach().cpu().clone().numpy()
    vis = img_l.copy()
    cnt = 0
    for pt in vis_xyd[:,:2].astype(np.int32):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x,y), 1, color, -1)
    return vis

def prepare_dirs(config):
    for path in [config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

