from torchvision.models import EfficientNet

from aikido.__api__.dojo_kun import DojoKun
from aikido.__util__.image_preprocessors import ResNetPreprocessor
from aikido.aikidoka.impl.resnet import ResNet
from aikido.dojo import BaseDojo
from aikido.dojo.listener import EvaluationListener
from aikido.dojo.listener.seed_listener import set_all_seeds
from aikido.kata.duet_dataset import DuetKata
from aikido.modeling.nn.head.duet_head import DuetHead

set_all_seeds(123)
EfficientNet
dataset_train = DuetKata("/home/christian/Pictures/topshot/topshot-dataset", ResNetPreprocessor())
dataset_train, dataset_test = dataset_train.split(0.8)

aikidoka = DuetHead(ResNet(headless=True))

dojo = BaseDojo(DojoKun(dans=10, batch_size=24), EvaluationListener(dataset_test, [], evaluate_every=5))
dojo.train(aikidoka, dataset_train)

aikidoka.save("./model-duet.pt")
