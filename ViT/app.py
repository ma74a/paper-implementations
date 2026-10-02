import json
import os
from torchvision.transforms import v2
import streamlit as st
import torch
from omegaconf import OmegaConf
from PIL import Image

from src import VisionTransformer
# from train import val_transform

val_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])


# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "configs",
    "config.yaml"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pt"
)

CLASS_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "2026-10-02",
    "last_one",
    "mapping_saved_file.json"
)


# Device

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Load configuration

cfg = OmegaConf.load(CONFIG_PATH)


# Load class mapping

@st.cache_data
def load_class_mapping():
    with open(CLASS_MAPPING_PATH, "r") as f:
        return json.load(f)


# Load model

@st.cache_resource
def load_model():

    model = VisionTransformer(
        num_classes=cfg.model.num_classes,
        input_channel=cfg.model.in_channels,
        image_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embedding_dim=cfg.model.embed_dim,
        input_dropout_rate=cfg.model.input_dropout_rate,
        num_encoder_blocks=cfg.model.num_of_encoders,
        num_heads=cfg.model.num_heads,
        dff_scale=cfg.model.dff_scale_factor,
        attention_dropout_rate=cfg.model.attention_dropout_rate,
        ff_dropout_rate=cfg.model.ff_dropout_rate,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    return model


# Prediction

def predict(image: Image.Image, model, class_to_index):

    image = image.convert("RGB")

    tensor_image = val_transform(image)

    tensor_image = tensor_image.unsqueeze(0)

    tensor_image = tensor_image.to(device)

    with torch.inference_mode():

        logits = model(tensor_image)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

    # JSON is usually index -> class
    if str(predicted_index) in class_to_index:
        predicted_class = class_to_index[str(predicted_index)]

    # In case your JSON uses integer keys
    elif predicted_index in class_to_index:
        predicted_class = class_to_index[predicted_index]

    else:
        predicted_class = str(predicted_index)

    confidence = probabilities[0, predicted_index].item()

    return predicted_class, confidence, probabilities


# Streamlit UI

st.set_page_config(
    page_title="ViT Image Classifier",
    page_icon="🧠",
    layout="centered"
)


st.title("🧠 Vision Transformer Image Classifier")

st.write(
    "Upload an image and let the trained Vision Transformer "
    "predict its class."
)


# Sidebar

st.sidebar.header("Model Information")

st.sidebar.write(
    f"**Device:** `{device}`"
)

st.sidebar.write(
    f"**Image Size:** `{cfg.model.img_size} × {cfg.model.img_size}`"
)

st.sidebar.write(
    f"**Patch Size:** `{cfg.model.patch_size} × {cfg.model.patch_size}`"
)

st.sidebar.write(
    f"**Embedding Dimension:** `{cfg.model.embed_dim}`"
)

st.sidebar.write(
    f"**Encoder Blocks:** `{cfg.model.num_of_encoders}`"
)

st.sidebar.write(
    f"**Attention Heads:** `{cfg.model.num_heads}`"
)


# Load model and classes

try:

    model = load_model()

    class_to_index = load_class_mapping()

except FileNotFoundError as e:

    st.error(
        f"Required file was not found:\n\n`{e.filename}`"
    )

    st.stop()


# Upload image

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        with st.spinner("Running inference..."):

            predicted_class, confidence, probabilities = predict(
                image,
                model,
                class_to_index
            )

        # ================================================
        # Prediction result
        # ================================================
        print(predicted_class)

        if predicted_class == "tom_jerry_1":
            predicted_class = "Tom and Jerry"
        elif predicted_class == "tom_jerry_0":
            predicted_class = "NO tom any Jerry"
        st.success(
            f"Prediction: **{predicted_class}**"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        # ================================================
        # All class probabilities
        # ================================================

        st.subheader("Class Probabilities")

        probability_dict = {}

        for index, probability in enumerate(
            probabilities[0].cpu().tolist()
        ):

            if str(index) in class_to_index:
                class_name = class_to_index[str(index)]

            elif index in class_to_index:
                class_name = class_to_index[index]

            else:
                class_name = str(index)

            probability_dict[class_name] = probability

        st.bar_chart(probability_dict)