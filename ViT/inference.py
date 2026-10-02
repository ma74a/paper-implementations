import torch
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from PIL import Image
import json

from src import VisionTransformer
from train import val_transform


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
    device = torch.device("cpu")
    
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    vit_model = VisionTransformer(
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
            ff_dropout_rate=cfg.model.ff_dropout_rate
    )
    
    vit_model.load_state_dict(torch.load("./models/best_model.pt", map_location=device))
    vit_model = vit_model.to(device)
    vit_model.eval()
    
    img_path = input("enter image path: ")
    try:
        img = Image.open(img_path).convert("RGB")
        tensor_img = val_transform(img).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            pred_logits = vit_model(tensor_img)
            probs = torch.softmax(pred_logits, dim=1)
            pred_label_index = torch.argmax(probs, dim=1).item()
            
        with open("./outputs/2026-10-02/last_one/mapping_saved_file.json", 'r') as f:
            class_to_index = json.load(f)
            
        print(pred_logits)
        print(probs)
        print(pred_label_index)
        print(class_to_index)
        
    except FileNotFoundError:
        print("Image is not found")
    
    
if __name__ == "__main__":
    main()