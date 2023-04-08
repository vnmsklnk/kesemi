from pathlib import Path
from PIL import Image


class QueryImage:
    def __init__(self, path_to_color: Path, local_extractor):
        self.color_image = Image.open(path_to_color)
        self.kpts, self.descs = local_extractor.extract_keypoints_descriptors(
            self.color_image
        )
