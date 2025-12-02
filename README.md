

# 从零开始构建多模态大模型并进行自由扩展

**MLLM from Scratch with Extension**

本项目系统性展示了一个多模态大模型（Multimodal Large Language Model, MLLM）从零实现的完整流程，包括 Transformer 基础组件、Vision Transformer、GPT-style 语言模型、多模态融合机制，以及进一步的强化学习扩展（SCST）。整个实现基于纯 PyTorch，不依赖 transformers、timm 等高封装库，旨在让学习者深入理解多模态模型背后的核心原理与工程细节。

*注：目前的版本是补完的版本，如果需要带空缺的版本清留意 #TODO 注释并稍作手动删除即可。*

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

# 🧩 第一部分：从零实现多模态大模型

这一部分重点展示如何从底层组件构建标准 Transformer，再逐步构建 Vision Transformer 与 GPT-style 语言模型，最终完成一个可以进行图像描述任务的基础 MLLM。

### 包含内容：

### 1. Transformer 从零实现

* Scaled Dot-Product Attention
* Multi-Head Attention
* FeedForward Network
* Positional Encoding（正弦版与可学习版）
* Encoder / Decoder Block
* 全部组件的单元测试（test_transformer.ipynb）

---

### 2. Vision Transformer

* 手写 PatchEmbedding
* 复用 TransformerEncoder
* 在 CIFAR-10 上训练
* 可视化 loss 及分类推理
* 可通过 configs/ 调整维度、head 数、深度等参数

---

### 3. GPT-style LLM

* 字符级 tokenizer
* Causal Mask + Decoder-only Transformer
* 在 Tiny Shakespeare 上训练
* 支持 generate() 方法生成文本

---

### 4. 多模态模型（MLLM）

* ViT encoder 提取视觉特征
* Connector 映射到语言 embedding space
* 拼接视觉 token + 文本 token
* LLM decoder 自回归生成输出
* 使用 Flickr8k 进行图文对训练
* 支持 inference_mllm.py 做“看图说话”

---

# 🔧 使用说明（Usage）

### 安装依赖

```
pip install -r requirements.txt
```

### 运行训练脚本（各模块）

```
python scripts/train/vit_train.py
python scripts/train/gpt_train.py
python scripts/train/mllm_train.py
```

### 调整模型参数

编辑：

```
configs/*.yaml
```

### 数据集下载

在 `datasets/` 中已补全下载逻辑，数据将自动保存到 `data/` 目录。

---

# 🧠 第二部分：强化学习扩展（RL Fine-tuning）

项目进一步提供 RL 微调能力，以 SCST（Self-Critical Sequence Training）为核心方法。

SCST 使用：

* **采样输出**：R_sample
* **贪心输出**：R_greedy（作为 baseline）
  实现低方差 REINFORCE：

[
\nabla_\theta J \approx (R_\text{sample} - R_\text{greedy}) \nabla_\theta \log \pi_\theta(a_\text{sample})
]

### RL 扩展目录：

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

### 支持功能：

* 返回 log prob 的增强版 MLLM
* 基于 CIDEr / BLEU 的 reward 计算
* RL 训练脚本与评估脚本
* 可与 SFT 权重无缝衔接


# 📝 补充说明
* 数据集较大未上传，下载数据集的操作：在 datasets/中补全代码，下载数据集到 data/ 中
* 训练与测试脚本在 script/ 目录下：train 为训练，test 为测试
* vit 和 gpt 的 loss 曲线图都在 mllm_from_scratch/MLLM_from_scratch下；mllm 的在mllm_from_scratch/MLLM_from_scratch/checkpoint 下
* 组装 transformer 需要通过的单元测试在 test_transformer.ipynb 中完成

---

# 📚 参考资料

* 复旦大学 2025 秋《人工智能前沿探索实践》Project-2
* Sebastian Raschka，《LLMs from Scratch》[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

