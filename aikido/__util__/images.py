from PIL import Image

from aikido.__api__.kata import Preprocessor


def load_image(path: str, preprocessor: Preprocessor):
    with Image.open(path) as img:
        return preprocessor(img)
