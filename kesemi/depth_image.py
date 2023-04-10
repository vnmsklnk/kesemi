import numpy as np

from pathlib import Path
from PIL import Image


def __parse_intrinsic_matrix(intrinsic_matrix):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    return fx, fy, cx, cy


def get_3d_point(kpt, d, intrinsics):
    fx, fy, cx, cy = __parse_intrinsic_matrix(intrinsics)
    v, u = kpt[0], kpt[1]
    z = d
    x = (v - cx) / fx * z
    y = (u - cy) / fy * z
    return [x, y, z]


def mask_points_with_depth(depth, kpts):
    N_kpts = kpts.shape[0]
    mask = np.full(N_kpts, False)
    for i in range(N_kpts):
        if depth.getpixel((kpts[i][0], kpts[i][1])) != 0:
            mask[i] = True
    return mask


class DepthImage:
    def __init__(
        self,
        path_to_depth: Path,
        kpts_2d,
        intrinsics,
        depth_scale,
    ):
        self.depth_image = Image.open(path_to_depth)
        self.intrinsics = intrinsics

        self.mask_depth = mask_points_with_depth(self.depth_image, kpts_2d)
        self.kpts_3d = np.full((len(kpts_2d), 3), np.nan)
        for i, is_valid in enumerate(self.mask_depth):
            if not is_valid:
                continue
            self.kpts_3d[i] = get_3d_point(
                kpts_2d[i],
                self.depth_image.getpixel((kpts_2d[i][0], kpts_2d[i][1])) / depth_scale,
                intrinsics,
            )

    def filter_matches_with_depth_mask(self, matches):
        mask_depth_set_indices = set(self.mask_depth.nonzero()[0])
        return list(filter(lambda match: match[0] in mask_depth_set_indices, matches))
