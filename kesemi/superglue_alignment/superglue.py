import cv2
import numpy as np
import torch

from PIL import Image

from kesemi.superglue_alignment.sg_model import SuperGlueModel
from kesemi.superglue_alignment.sp_model import SuperPointModel


class SuperGlue:
    def __init__(
        self,
        path_to_sp_weights,
        path_to_sg_weights,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Running inference on device "{}"'.format(self.device))

        self.super_point = SuperPointModel(path_to_sp_weights).eval().to(self.device)
        self.super_glue_matcher = (
            SuperGlueModel(path_to_sg_weights).eval().to(self.device)
        )

    def extract_features(self, image_rgb):
        frame = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2GRAY)
        inp = torch.from_numpy(frame / 255.0).float()[None, None]
        with torch.no_grad():
            features = self.super_point({"image": inp})
        return features

    def match(self, train_path, query_path):
        query_features = self.extract_features(Image.open(query_path))
        train_features = self.extract_features(Image.open(train_path))
        pred = {k + "0": v for k, v in query_features.items()}
        pred = {**pred, **{k + "1": v for k, v in train_features.items()}}
        with torch.no_grad():
            pred = self.super_glue_matcher(pred)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        matches = pred["matches1"]
        query_kpts = query_features["keypoints"][0].numpy().astype(int)
        train_kpts = train_features["keypoints"][0].numpy().astype(int)
        matches = list(filter(lambda match: match[1] > -1, enumerate(matches)))
        return matches, query_kpts, train_kpts
