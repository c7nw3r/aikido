from PIL import Image

from aikido.__util__.image_preprocessors import ResNetPreprocessor
from aikido.__util__.tensors import cosine_similarity, unsqueeze4d
from aikido.aikidoka.impl.efficientnet import EfficientNet
from aikido.modeling.nn.head.triplet_net_head import TripletNetHead

aikidoka = TripletNetHead(EfficientNet(headless=True))
aikidoka.load("./model.pt")

image1 = Image.open("/home/christian/Pictures/topshot/topshot-dataset/Banksy/banksy_soup_can.jpg")
image2 = Image.open("/home/christian/Pictures/topshot/topshot-dataset/Blob/EP21-Flowarh�_II-A-36x36_P.jpg")
image3 = Image.open("/home/christian/Pictures/topshot/topshot-dataset/Banksy/banksy_love_rat.jpg")
image4 = Image.open("/home/christian/Pictures/topshot/topshot-dataset/Kalkhof/al4oakbue00033e5bwyne3zgv-original.jpeg")
image5 = Image.open("/home/christian/Pictures/topshot/topshot-dataset/Kalkhof/al4oamjg400043e5b2vluhe5y-original.jpeg")

transform = ResNetPreprocessor()

embedding1 = aikidoka.get_embedding(transform(image1)).detach().numpy()
embedding2 = aikidoka.get_embedding(transform(image2)).detach().numpy()
embedding3 = aikidoka.get_embedding(transform(image3)).detach().numpy()
embedding4 = aikidoka.get_embedding(transform(image4)).detach().numpy()
embedding5 = aikidoka.get_embedding(transform(image5)).detach().numpy()

print(cosine_similarity(embedding1, embedding2))
print(cosine_similarity(embedding1, embedding3))
print(cosine_similarity(embedding4, embedding5))

# aikidoka.to_onnx()

import torch
torch.onnx.export(
    aikidoka.embedding_net,
    unsqueeze4d(torch.tensor(transform(image1))),
    f="./model.onnx",
    input_names=["x"],
    output_names=["y"],
    dynamic_axes={
        "x": {0: "batch", 1: "sequence"}
    },
    do_constant_folding=False,
    opset_version=14
)
