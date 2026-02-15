import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from utils.config import Config


def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = Config.TRANSFORMS["train"](img).unsqueeze(0)

    model.eval()
    output = model(img)
    _, pred = output.max(dim=1)

    return pred