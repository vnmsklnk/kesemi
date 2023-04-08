import torch


class SuperPointModel(torch.nn.Module):
    """Pytorch definition of SuperPoint Network."""

    def __init__(self):
        super(SuperPointModel, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.relu1a = torch.nn.ReLU(inplace=True)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.relu1b = torch.nn.ReLU(inplace=True)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.relu2a = torch.nn.ReLU(inplace=True)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.relu2b = torch.nn.ReLU(inplace=True)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.relu3a = torch.nn.ReLU(inplace=True)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.relu3b = torch.nn.ReLU(inplace=True)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.relu4a = torch.nn.ReLU(inplace=True)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.relu4b = torch.nn.ReLU(inplace=True)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.reluPa = torch.nn.ReLU(inplace=True)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.reluDa = torch.nn.ReLU(inplace=True)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu1a(self.conv1a(x))
        x = self.relu1b(self.conv1b(x))
        x = self.pool(x)
        x = self.relu2a(self.conv2a(x))
        x = self.relu2b(self.conv2b(x))
        x = self.pool(x)
        x = self.relu3a(self.conv3a(x))
        x = self.relu3b(self.conv3b(x))
        x = self.pool(x)
        x = self.relu4a(self.conv4a(x))
        x = self.relu4b(self.conv4b(x))
        # Detector Head.
        cPa = self.reluPa(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.reluDa(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc
