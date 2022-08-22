from aikido.__api__.dojo_kun import DojoKun
from aikido.__util__.image_preprocessors import ResNetPreprocessor
from aikido.aikidoka.impl.efficientnet import EfficientNet
from aikido.aikidoka.impl.resnet import ResNet
from aikido.dojo import BaseDojo
from aikido.dojo.listener import EvaluationListener
from aikido.dojo.listener.seed_listener import set_all_seeds
from aikido.kata.triplet_kata import AugmentationAwareTripletKata
from aikido.modeling.nn.head.triplet_net_head import TripletNetHead

set_all_seeds(123)

# dataset_train = TripletDataset("/home/christian/Pictures/topshot/topshot-dataset", ResNetPreprocessor(), cache=True)
dataset_train = AugmentationAwareTripletKata("/home/christian/Pictures/topshot/topshot-dataset", ResNetPreprocessor())
dataset_train, dataset_test = dataset_train.split(0.8)

print(len(dataset_train))
print(len(dataset_test))

aikidoka = TripletNetHead(EfficientNet(headless=True))

dojo = BaseDojo(DojoKun(dans=10, batch_size=12), EvaluationListener(dataset_test, [], evaluate_every=5))
dojo.train(aikidoka, dataset_train)

aikidoka.save("./model.pt")