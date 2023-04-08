import numpy as np


class SuperPointMatcher:
    def __init__(self, nn_thresh):
        # Distance cannot be negative.
        assert nn_thresh >= 0
        # Transform the threshold for more efficient comparison later.
        self._nn_thresh_inversed = 1 - (nn_thresh**2) / 2

    def match(self, descrs1, descrs2):
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
        # Descriptors must have the same dimensionality.
        assert descrs1.shape[1] == descrs2.shape[1]

        if descrs1.shape[0] == 0 or descrs2.shape[0] == 0:
            return []

        # Compute semi-L2-distance: vectors are unit-normalized, and to compute the
        # true L2 distance we would need to do sqrt(2 - 2 * (descrs1 @ descrs2.T)).
        semi_dist_mat = descrs1 @ descrs2.T

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

        return matches
