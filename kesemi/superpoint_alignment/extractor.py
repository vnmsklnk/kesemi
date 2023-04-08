import torch
import torchvision

from kesemi.superpoint_alignment.model import SuperPointModel
from kesemi.superpoint_alignment.decoder import SuperPointDecoder


class SuperPointExtractor:
    _transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
        ]
    )

    def __init__(self, network_weights_path, device="cpu"):
        self.model = SuperPointModel()
        model_state_dict = torch.load(
            network_weights_path, map_location=torch.device(device)
        )
        self.model.load_state_dict(model_state_dict)
        self.model.eval()  # Makes no difference for SuperPoint but is still recommended
        self.decoder = SuperPointDecoder()

    def extract_keypoints_descriptors(self, image_rgb):
        image_tensor = SuperPointExtractor._transform(image_rgb)
        H, W = image_tensor.shape[1], image_tensor.shape[2]

        inp = image_tensor.unsqueeze(0)
        with torch.no_grad():
            semi_keypts, coarse_descrs = self.model.forward(inp)
            kpts, descr = self.decoder.forward(semi_keypts, coarse_descrs, H, W)

        return kpts.detach().numpy().T.astype(int), descr.detach().numpy().T
