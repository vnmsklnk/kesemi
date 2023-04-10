import cv2
import numpy as np

from PIL import Image


class ORB:
    def __init__(self, nfeatures, ratio_threshold):
        self.orb = cv2.ORB_create(nfeatures)
        self.ratio_threshold = ratio_threshold

    def extract_keypoints_descriptors(self, image_rgb):
        opencv_image = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2GRAY)
        kpts, descs = self.orb.detectAndCompute(opencv_image, None)
        kpts = cv2.KeyPoint.convert(kpts)
        return kpts, descs

    def match(self, train_path, query_path):
        train_kpts, train_descs = self.extract_keypoints_descriptors(
            Image.open(train_path)
        )
        query_kpts, query_descs = self.extract_keypoints_descriptors(
            Image.open(query_path)
        )
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(query_descs, train_descs, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append((m.trainIdx, m.queryIdx))
        return good_matches, query_kpts, train_kpts
