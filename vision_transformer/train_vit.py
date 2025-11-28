# vision_transformer/train_vit.py
#* OK
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from datasets.cifar10 import CIFAR10Dataset
from vision_transformer.vit import ViT

def train(config: dict):
    """Main function to train and evaluate the ViT model."""
    data_cfg = config['data_params']
    model_cfg = config['model_params']
    train_cfg = config['training_params']
    DEVICE = train_cfg['device'] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    os.makedirs(data_cfg['root_dir'], exist_ok=True)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(data_cfg['img_size'], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_cfg['mean'], std=data_cfg['std'])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(data_cfg['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_cfg['mean'], std=data_cfg['std'])
    ])
    train_dataset = CIFAR10Dataset(root=data_cfg['root_dir'], train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10Dataset(root=data_cfg['root_dir'], train=False, transform=test_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'], pin_memory=True)

    model = ViT(
        img_size=data_cfg['img_size'],
        in_channels=data_cfg['in_channels'],
        num_classes=data_cfg['num_classes'],
        **model_cfg
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['scheduler_T_max'])

    best_accuracy = 0.0
    
    #* --- Add: lists to store training curves ---
    train_losses = []
    test_losses = []
    accuracies = []
    
    for epoch in range(train_cfg['num_epochs']):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']} [Train]")
        
        # --- START OF STUDENT MODIFICATION (TRAINING LOOP) ---
        for images, labels in train_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # TODO: 实现标准的训练步骤
            # 1. 清空之前的梯度
            optimizer.zero_grad() #* 否则梯度会一直累积
            
            # 2. 模型前向传播，获取输出
            outputs = model(images)
            
            # 3. 计算损失
            loss = criterion(outputs, labels)
            
            # 4. 反向传播，计算梯度
            loss.backward()
            
            # 5. 更新模型参数
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        # --- END OF STUDENT MODIFICATION (TRAINING LOOP) ---

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            eval_loop = tqdm(test_loader, leave=True, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']} [Eval]")
            # --- START OF STUDENT MODIFICATION (EVALUATION LOOP) ---
            for images, labels in eval_loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # TODO: 计算预测准确率
                # 1. 从模型的输出 logits 中获取预测结果 (类别索引)
                #    - 使用 torch.max() 函数，它会返回最大值和对应的索引。我们只需要索引。
                _, predicted = torch.max(outputs, dim=1) #* 省略了先给 output 先做 softmax 一步，结果一样
                
                # 2. 累加样本总数
                total += labels.size(0)
                
                # 3. 累加预测正确的样本数
                #    - 比较 predicted 和 labels，并使用 .sum().item() 得到正确的数量。
                correct += (predicted == labels).sum().item()
            # --- END OF STUDENT MODIFICATION (EVALUATION LOOP) ---
                
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{train_cfg['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%")
        scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(os.path.dirname(train_cfg['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), train_cfg['model_save_path'])
            print(f"New best model saved with accuracy: {accuracy:.2f}%")

        #* 保存曲线数据
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        accuracies.append(accuracy)

    print(f"Training complete. Best accuracy: {best_accuracy:.2f}%")
    
    # --- 绘制训练曲线 ---
    import matplotlib.pyplot as plt

    # Loss 曲线
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ViT Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("vit_loss_curve.png")
    plt.close()

    # Accuracy 曲线
    plt.figure(figsize=(8,5))
    plt.plot(accuracies, label="Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ViT Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("vit_accuracy_curve.png")
    plt.close()

    print("Training curves saved as vit_loss_curve.png and vit_accuracy_curve.png")