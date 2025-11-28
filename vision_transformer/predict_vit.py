# vision_transformer/predict_vit.py
#* OK
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os

from vision_transformer.vit import ViT

def predict(config: dict):
    """Loads a trained ViT model and predicts the class of an image."""
    data_cfg = config['data_params']
    model_cfg = config['model_params']
    train_cfg = config['training_params']
    pred_cfg = config['prediction_params']

    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else 'cpu')
    model_path = train_cfg['model_save_path']
    image_source = pred_cfg['image_source']
    
    try: 
        model = ViT(img_size=data_cfg['img_size'], patch_size=model_cfg['patch_size'], in_channels=data_cfg['in_channels'], num_classes=data_cfg['num_classes'], d_model=model_cfg['d_model'], num_layers=model_cfg['num_layers'], n_heads=model_cfg['n_heads'], d_ff=model_cfg['d_ff'], dropout=model_cfg['dropout']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from '{model_path}'.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Please run training first.")
        return

    transform = transforms.Compose([
        transforms.Resize((data_cfg['img_size'], data_cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_cfg['mean'], std=data_cfg['std'])
    ])
    
    try:
        if image_source.startswith('http'):
            response = requests.get(image_source)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image_source).convert("RGB")
            
        # --- START OF STUDENT MODIFICATION (PREDICTION) ---
        
        # TODO: 1. 对加载的 PIL 图像进行预处理
        #    a. 应用上面定义的 transform。
        x = transform(img)
        #    b. 使用 .unsqueeze(0) 添加一个批次维度 (batch dimension)，因为模型期望的输入是 [B, C, H, W]。
        x = x.unsqueeze(0)
        #    c. 将处理后的张量移动到正确的设备 (device)。
        img_tensor = x.to(device)
        #* 这样写是否正确？是否可以把 x 换成 img_tensor？ 有什么区别？
        # 都可以，没什么影响。最简洁的写法：img_tensor = transform(img).unsqueeze(0).to(device)
        

    except Exception as e:
        print(f"Failed to load or process image: {e}")
        return

    with torch.no_grad():
        # TODO: 2. 使用模型进行预测并解析结果
        #    a. 将预处理后的图像张量 img_tensor 输入模型，得到 logits。
        logits = model(img_tensor)
        
        #    b. 使用 softmax 将 logits 转换为概率。
        probabilities = torch.softmax(logits, dim=1)
        
        #    c. 使用 torch.max() 找到置信度最高（概率最大）的类别索引。
        confidence, predicted_idx = torch.max(probabilities, 1)

    # TODO: 3. 将预测的类别索引映射回类别名称
    #    - 使用 data_cfg['class_names'] 列表和 predicted_idx.item() 来获取名称。
    predicted_class = data_cfg['class_names'][predicted_idx.item()] #* 不加 item 得到的是张量，加了得到数字
    
    # --- END OF STUDENT MODIFICATION (PREDICTION) ---
    
    print(f"\n--- Prediction Result ---")
    print(f"  -> Predicted Class: '{predicted_class}'")
    print(f"  -> Confidence: {confidence.item():.4f}")
    print("-------------------------")