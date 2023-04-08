import cv2


class ORBMatcher:
    def __init__(self, ratio_threshold):
        assert ratio_threshold > 0
        self.ratio_threshold = ratio_threshold

    def match(self, train_descs, query_descs):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(query_descs, train_descs, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append((m.trainIdx, m.queryIdx))
        return good_matches
