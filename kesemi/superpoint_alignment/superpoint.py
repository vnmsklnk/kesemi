import numpy as np
import torch
import torchvision

from PIL import Image

from kesemi.superpoint_alignment.model import SuperPointModel
from kesemi.superpoint_alignment.decoder import SuperPointDecoder


class SuperPoint:
    _transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
        ]
    )

    def __init__(self, network_weights_path, nn_thresh, device="cpu"):
        self.model = SuperPointModel()
        model_state_dict = torch.load(
            network_weights_path, map_location=torch.device(device)
        )
        self.model.load_state_dict(model_state_dict)
        self.model.eval()  # Makes no difference for SuperPoint but is still recommended
        self.decoder = SuperPointDecoder()

        # Distance cannot be negative.
        assert nn_thresh >= 0
        # Transform the threshold for more efficient comparison later.
        self._nn_thresh_inversed = 1 - (nn_thresh**2) / 2

    def extract_keypoints_descriptors(self, image_rgb):
        image_tensor = SuperPoint._transform(image_rgb)
        H, W = image_tensor.shape[1], image_tensor.shape[2]

        inp = image_tensor.unsqueeze(0)
        with torch.no_grad():
            semi_keypts, coarse_descrs = self.model.forward(inp)
            kpts, descr = self.decoder.forward(semi_keypts, coarse_descrs, H, W)

        return kpts.detach().numpy().T.astype(int), descr.detach().numpy().T

    def match(self, train_path, query_path):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.
        Inputs:
          descrs1 - NxD numpy matrix of N unit-normalized D-dimensional descriptors.
          descrs2 - MxD numpy matrix of M unit-normalized D-dimensional descriptors.
        Returns:
          matches - list of (i_1, i_2) tuples, where i_1 and i_2 are the indices of
                    matched descriptors in desc1 and desc2 correspondingly.
        """
        train_kpts, train_descs = self.extract_keypoints_descriptors(
            Image.open(train_path)
        )
        query_kpts, query_descs = self.extract_keypoints_descriptors(
            Image.open(query_path)
        )

        # Descriptors must have the same dimensionality.
        assert train_descs.shape[1] == query_descs.shape[1]

        if train_descs.shape[0] == 0 or query_descs.shape[0] == 0:
            return []

        # Compute semi-L2-distance: vectors are unit-normalized, and to compute the
        # true L2 distance we would need to do sqrt(2 - 2 * (descrs1 @ descrs2.T)).
        semi_dist_mat = train_descs @ query_descs.T

        # Get IDs of minimal distances along both directions: lower distances
        # correspond to higher semi-distances.
        row_wise_min_dist_ids = np.argmax(semi_dist_mat, axis=1)
        col_wise_min_dist_ids = np.argmax(semi_dist_mat, axis=0)

        # Filter results that are symmetrical and satisfy nn_thresh.
        matches = []
        for row_i, col_i in enumerate(row_wise_min_dist_ids):
            min_dist = semi_dist_mat[row_i, col_i]
            if (
                col_wise_min_dist_ids[col_i] == row_i
                and min_dist >= self._nn_thresh_inversed
            ):
                matches.append((row_i, col_i))

        return matches, query_kpts, train_kpts
