# 从零开始构建多模态大模型并进行自由扩展

本项目系统性展示了一个多模态大模型（Multimodal Large Language Model, MLLM）从零实现的完整流程，包括 Transformer 基础组件、Vision Transformer、GPT-style 语言模型、多模态融合机制，以及进一步的强化学习扩展（SCST）。

项目主要基于 PyTorch 手写核心模块，尽量不依赖 `transformers`、`timm` 等高封装库，适合用来理解多模态模型背后的数据链条、训练逻辑和工程组织方式。

*当前版本是补完版本；如果需要教学用的填空版本，可以搜索代码中的 `TODO` 注释并手动处理。*


---

# 📁 项目结构（Project Structure）

```
MLLM-from-scratch/
├── configs/
│   ├── vit_config.yaml
│   ├── llm_config.yaml
│   └── mllm_config.yaml
│
├── datasets/
│   ├── __init__.py
│   ├── data_utils.py       # 数据集基类、下载、预处理函数
│   ├── cifar10.py          # CIFAR-10 数据集处理
│   ├── tinyshakespeare.py  # Tiny Shakespeare 数据集处理
│   └── Flickr8k.py         # Flickr8k 数据集处理
│
├── transformer_from_scratch/
│   ├── __init__.py
│   ├── attention.py        # ScaledDotProductAttention, MultiHeadAttention
│   ├── layers.py           # FFN、PositionalEncoding、LayerNorm 等
│   ├── blocks.py           # EncoderBlock, DecoderBlock
│   └── model.py            # TransformerEncoder, Decoder, Full Transformer
│
├── vision_transformer/
│   ├── __init__.py
│   ├── vit.py              # ViT 实现
│   ├── train_vit.py        # ViT 训练脚本
│   └── predict_vit.py      # ViT 推理脚本
│
├── language_model/
│   ├── __init__.py
│   ├── tokenizer.py        # 字符级 tokenizer
│   ├── llm.py              # GPT-style 自回归语言模型
│   ├── train_llm.py        # 训练脚本
│   └── generate_text.py    # 文本生成脚本
│
├── multimodal_model/
│   ├── __init__.py
│   ├── connector.py        # 视觉特征 → 语言 embedding 的映射模块
│   ├── mllm.py             # MLLM 组装（ViT + Connector + LLM）
│   ├── train_mllm.py       # 多模态训练
│   └── inference_mllm.py   # “看图说话”推理脚本
│
├── tests/
│   ├── test_attention.py
│   ├── test_blocks.py
│   └── test_transformer.py
│
├── utils/
│   ├── __init__.py
│   ├── training_utils.py   # 训练循环、日志、checkpoint
│   └── config_parser.py    # Yaml 配置解析
│
├── main.py                 # 脚本入口
├── requirements.txt
└── README.md
```
---
## 模型结构示意图
<img width="1936" height="1160" alt="图片" src="https://github.com/user-attachments/assets/5fb3b91c-a00f-4084-8344-16ca752ea738" />

---

## 快速入口

建议在项目根目录执行：

```bash
cd ..../MLLM_from_scratch
```

安装依赖：

```bash
pip install -r requirements.txt
```

统一入口是 `main.py`，通过 `--task` 选择任务，通过 `--config` 指定配置文件：

```bash
python main.py --task train_vit --config configs/vit_config.yaml
python main.py --task predict_vit --config configs/vit_config.yaml

python main.py --task train_llm --config configs/llm_config.yaml
python main.py --task generate_text --config configs/llm_config.yaml

python main.py --task train_mllm --config configs/mllm_config.yaml
python main.py --task inference_mllm --config configs/mllm_config.yaml

python main.py --task train_rl_mllm --config configs/rl_mllm_config.yaml
```

`script/*.sh` 中也提供了调用脚本，但目前部分路径是服务器绝对路径。本地运行时更推荐直接使用上面的相对路径命令，或先修改脚本中的 `--config` 路径。



## 数据集与训练链条

### 1. ViT 图像分类链条

目标：在 CIFAR-10 上训练一个从零实现的 Vision Transformer 分类器。

相关文件：

- 配置：`configs/vit_config.yaml`
- 数据集：`datasets/cifar10.py`
- 模型：`vision_transformer/vit.py`
- 训练：`vision_transformer/train_vit.py`
- 推理：`vision_transformer/predict_vit.py`

数据流：

```text
CIFAR-10 image
-> torchvision transforms
-> ViT
-> class logits
-> CrossEntropyLoss
```

训练命令：

```bash
python main.py --task train_vit --config configs/vit_config.yaml
```

推理命令：

```bash
python main.py --task predict_vit --config configs/vit_config.yaml
```

主要输出：

- `checkpoints/vit_cifar10.pth`
- `vit_loss_curve.png`
- `vit_accuracy_curve.png`

### 2. GPT 字符级语言模型链条

目标：在 Tiny Shakespeare 上训练一个 GPT-style decoder-only 语言模型。

相关文件：

- 配置：`configs/llm_config.yaml`
- 数据集：`datasets/tinyshakespeare.py`
- Tokenizer：`language_model/tokenizer.py`
- 模型：`language_model/llm.py`
- 训练：`language_model/train_llm.py`
- 文本生成：`language_model/generate_text.py`

数据流：

```text
Tiny Shakespeare text
-> CharacterTokenizer
-> sliding window blocks
-> x = tokens[:-1], y = tokens[1:]
-> GPTModel
-> next-character CrossEntropyLoss
```

训练命令：

```bash
python main.py --task train_llm --config configs/llm_config.yaml
```

生成命令：

```bash
python main.py --task generate_text --config configs/llm_config.yaml
```

主要输出：

- `checkpoints/llm_tinyshakespeare.pth`
- `data/tinyshakespeare/vocab.json`
- `gpt_training_curve.png`

注意：`language_model/generate_text.py` 中生成循环仍有 `...` 占位，若要单独运行 `generate_text`，需要先补全该文件。

### 3. MLLM 图像描述链条

目标：在 Flickr8k 上训练一个基础图像描述模型。

相关文件：

- 配置：`configs/mllm_config.yaml`
- 数据集：`datasets/Flickr8k.py`
- 视觉编码器：`vision_transformer/vit.py`
- 语言模型：`language_model/llm.py`
- 连接器：`multimodal_model/connector.py`
- 多模态模型：`multimodal_model/mllm.py`
- 训练：`multimodal_model/train_mllm.py`
- 推理：`multimodal_model/inference_mllm.py`

数据要求：

```text
data/flickr8k/
├── Images/ 或 Flicker8k_Dataset/
└── captions.txt
```

训练数据流：

```text
image
-> ViT.forward_features()
-> visual features
-> Connector: vision_dim -> language_dim
-> visual embeddings

caption
-> CharacterTokenizer
-> [<sos>] + caption tokens + [<eos>]
-> text embeddings

visual embeddings + text embeddings
-> GPTModel.forward_from_embeddings()
-> caption token prediction
```

损失计算重点：

```text
logits length = num_visual_tokens + num_text_tokens
labels length = same as logits
visual token positions = ignore_index
text token positions = real caption targets
```

也就是说，训练时只对文本 caption 部分计算 `CrossEntropyLoss`，视觉 token 位置不参与损失。

训练命令：

```bash
python main.py --task train_mllm --config configs/mllm_config.yaml
```

推理命令：

```bash
python main.py --task inference_mllm --config configs/mllm_config.yaml
```

主要输出：

- `checkpoints/mllm_flickr8k_v2_best.pth`
- `checkpoints/flickr8k_tokenizer.json`
- `checkpoints/mllm_loss_curve.png`

## RL 微调链条

目标：在 SFT 后的 MLLM 基础上进行 SCST 风格强化学习微调。

RL 相关目录

```
├── configs/
│   ├── rl_mllm_config.yaml
│
├── multimodal_model/
│   ├── rl_finetune_mllm.py
│   ├── inference_rl_mllm.py
│
├── scripts/
│   ├── rl_mllm.sh
│   └── test_rl_mllm.sh
```

相关文件：

- 配置：`configs/rl_mllm_config.yaml`
- 训练：`multimodal_model/train_rl_mllm.py`
- 推理：`multimodal_model/inference_rl_mllm.py`

前置条件：

- 已完成 MLLM SFT 训练
- 存在 `configs/rl_mllm_config.yaml` 中指定的 `best_model_save_path`
- 存在 Flickr8k tokenizer：`checkpoints/flickr8k_tokenizer.json`

RL 训练逻辑：

```text
image + <sos>
-> sampled caption + sampled logprobs
-> greedy caption baseline
-> reward(sampled) - reward(greedy) = advantage
-> rl_loss = - advantage * sampled_logprob
-> final_loss = lambda_rl * rl_loss + lambda_ce * ce_loss
```

当前 reward 是字符级 bigram overlap，形式上类似简化版 BLEU-2。

训练命令：

```bash
python main.py --task train_rl_mllm --config configs/rl_mllm_config.yaml
```

主要输出：

- `checkpoints/mllm_rl.pt`
- `checkpoints/rl_loss_curve.png`

## 配置文件说明

```text
configs/vit_config.yaml
```

控制 CIFAR-10 路径、图像尺寸、ViT 结构、训练超参和单图预测路径。

```text
configs/llm_config.yaml
```

控制 Tiny Shakespeare 下载路径、block size、GPT 结构、训练超参和生成参数。

```text
configs/mllm_config.yaml
```

控制 Flickr8k 路径、tokenizer 保存路径、ViT/LLM/Connector 结构、SFT 训练超参和图像描述推理参数。

```text
configs/rl_mllm_config.yaml
```

控制 RL 微调超参、SFT checkpoint 路径、RL checkpoint 保存路径，以及与 SFT 阶段保持一致的模型结构。

## 测试与检查

Transformer 基础模块测试：

```bash
pytest tests
```

已有测试主要覆盖：

- attention
- encoder/decoder block
- transformer 组合模块

## 本地运行注意事项

- `script/*.sh` 目前含有服务器绝对路径，本机运行前建议改成 `configs/*.yaml` 形式的相对路径。
- `script/test_llm.sh` 当前指向了 `vit_config.yaml`，应改为 `llm_config.yaml`。
- `language_model/generate_text.py` 仍有 `...` 占位，单独文本生成前需要补完。
- `configs/mllm_config.yaml` 和 `configs/vit_config.yaml` 中的部分推理图片路径是服务器路径，本机推理前需要改成本地存在的图片。
- CIFAR-10 和 Tiny Shakespeare 可以自动下载；Flickr8k 通常需要手动下载并放到 `data/flickr8k/`。
- 默认设备多处写为 `cuda`。如果在 Mac 或 CPU 环境运行，请将配置里的 `device` 改为 `cpu` 或适合的设备，并酌情把 `num_workers` 调小。

## 参考资料

- 复旦大学 2025 秋《人工智能前沿探索实践》Project-2
- Sebastian Raschka, LLMs from Scratch: https://github.com/rasbt/LLMs-from-scratch
