import torch
from PIL import Image
import torchvision.transforms as transforms

from vision_transformer.vit import ViT
from language_model.llm import GPTModel
from language_model.tokenizer import CharacterTokenizer
from multimodal_model.connector import Connector
from multimodal_model.mllm import MLLM


def load_rl_mllm(config, checkpoint_path="./checkpoints/mllm_rl.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 1. Load tokenizer
    # -----------------------------
    tokenizer_path = config["paths"]["tokenizer_save_path"]
    tokenizer = CharacterTokenizer()
    tokenizer.load_vocab(tokenizer_path)

    # -----------------------------
    # 2. Build model (must match RL config!!)
    # -----------------------------
    model_cfg = config["model"]

    vision_encoder = ViT(
        img_size=model_cfg["vision_encoder"]["image_size"],
        patch_size=model_cfg["vision_encoder"]["patch_size"],
        in_channels=model_cfg["vision_encoder"]["in_channels"],
        d_model=model_cfg["vision_encoder"]["vision_dim"],
        num_layers=model_cfg["vision_encoder"]["n_layers"],
        n_heads=model_cfg["vision_encoder"]["n_heads"],
        d_ff=model_cfg["vision_encoder"]["d_ff"],
        dropout=model_cfg["vision_encoder"]["dropout"],
        num_classes=None,
    ).to(device)

    language_model = GPTModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg["language_model"]["language_dim"],
        num_layers=model_cfg["language_model"]["n_layers"],
        n_heads=model_cfg["language_model"]["n_heads"],
        d_ff=model_cfg["language_model"]["d_ff"],
        max_len=model_cfg["language_model"]["max_len"],
        dropout=model_cfg["language_model"]["dropout"],
    ).to(device)

    connector = Connector(
        vision_dim=model_cfg["connector"]["vision_dim"],
        language_dim=model_cfg["connector"]["language_dim"],
        connector_type=model_cfg["connector"]["type"],
        hidden_dim=model_cfg["connector"]["hidden_dim"]
    ).to(device)

    # -----------------------------
    # 3. Combine into MLLM
    # -----------------------------
    mllm = MLLM(vision_encoder, language_model, connector, tokenizer).to(device)

    # -----------------------------
    # 4. Load RL checkpoint
    # -----------------------------
    print(f"Loading RL checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    mllm.load_state_dict(state)
    mllm.eval()

    return mllm, tokenizer, device


# -------------------------------------------------------------
#                 🔥 Run Inference
# -------------------------------------------------------------
def generate_caption(config, image_path):
    mllm, tokenizer, device = load_rl_mllm(config)

    # Image transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((config["model"]["vision_encoder"]["image_size"], 
                           config["model"]["vision_encoder"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Run inference
    print("Generating caption...")
    output_text = mllm.generate(
        image=image,
        prompt="",
        max_new_tokens=50,
        temperature=0.7,
        top_k=50,
    )

    return output_text


if __name__ == "__main__":
    import yaml

    # load RL config
    config = yaml.safe_load(open("configs/rl_mllm_config.yaml", "r"))

    test_img = "/home/u1120230266/VLM-R1/Zhanghengrui/mllm_from_scratch/MLLM_from_scratch/data/test_images/cat.jpg"
    caption = generate_caption(config, test_img)
    print("Generated Caption:\n", caption)
