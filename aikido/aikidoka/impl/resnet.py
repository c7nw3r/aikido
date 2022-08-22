import torch
from torch.nn import Identity

from aikido.__api__.aikidoka import Aikidoka


class ResNet(Aikidoka):

    def __init__(self, headless: bool = False):
        super().__init__()
        self.embedding_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        if headless:
            self.embedding_net.fc = Identity()

    def forward(self, x):
        return self.embedding_net(x)