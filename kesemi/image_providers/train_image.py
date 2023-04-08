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


def get_3d_keypoints(depth, kpts, intrinsics, depth_scale):
    N_kpts = kpts.shape[0]
    points = []
    for i in range(N_kpts):
        points.append(
            get_3d_point(
                kpts[i],
                depth.getpixel((kpts[i][0], kpts[i][1])) / depth_scale,
                intrinsics,
            )
        )

    return points


def mask_points_with_depth(depth, kpts):
    N_kpts = kpts.shape[0]
    mask = np.full(N_kpts, False)
    for i in range(N_kpts):
        if depth.getpixel((kpts[i][0], kpts[i][1])) != 0:
            mask[i] = True
    return mask


class TrainImage:
    def __init__(
        self,
        path_to_depth: Path,
        path_to_color: Path,
        intrinsics,
        local_extractor,
        depth_scale,
    ):
        self.depth_image = Image.open(path_to_depth)
        self.color_image = Image.open(path_to_color)
        self.intrinsics = intrinsics

        kpts, descs = local_extractor.extract_keypoints_descriptors(self.color_image)
        mask_depth = mask_points_with_depth(self.depth_image, kpts)
        kpts = kpts[mask_depth]
        self.descs = descs[mask_depth]
        self.kpts_3d = get_3d_keypoints(self.depth_image, kpts, intrinsics, depth_scale)
