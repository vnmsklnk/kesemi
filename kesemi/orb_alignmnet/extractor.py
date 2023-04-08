import cv2
import numpy as np


class ORBExtractor:
    def __init__(self, nfeatures):
        self.orb = cv2.ORB_create(nfeatures)

    def extract_keypoints_descriptors(self, image_rgb):
        opencv_image = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2GRAY)
        kpts, descs = self.orb.detectAndCompute(opencv_image, None)
        kpts = cv2.KeyPoint.convert(kpts)
        return kpts, descs
