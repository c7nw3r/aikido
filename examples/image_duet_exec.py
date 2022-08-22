from PIL import Image

from aikido.__util__.image_preprocessors import ResNetPreprocessor
from aikido.__util__.tensors import cosine_similarity
from aikido.aikidoka.impl.resnet import ResNet
from aikido.modeling.nn.head.duet_head import DuetHead
from aikido.modeling.nn.head.triplet_net_head import TripletNetHead

aikidoka = DuetHead(ResNet(headless=True))
aikidoka.load("./model2-duet.pt")

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
