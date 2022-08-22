from torch.nn import Identity

from aikido.__api__.aikidoka import Aikidoka


class EfficientNet(Aikidoka):

    def __init__(self, headless: bool = False):
        super().__init__()

        try:
            import torchvision.models as models
        except ImportError:
            raise ValueError("no torchvision library found")

        self.model = models.efficientnet_b0(pretrained=True)

        if headless:
            self.model.classifier = Identity()

    def forward(self, x):
        return self.model(x)
