# 从零开始构建多模态大模型并进行自由扩展（MLLM from Scratch with extension）
本项目分为两个部分，第一部分为从零构建多模态大模型，第二部分为基于构建的 MLLM 的自由扩展，如强化学习微调。

# 第一部分：从零开始构建多模态大模型 (MLLM from Scratch)
*注：此处主要参考复旦大学2025秋季学期人工智能前沿探索实践 Project-2与 llm-from-scratch 项目*

## 项目简介

本部分旨在提供一个从零开始、纯手工实现多模态大模型（Multimodal Large Language Model, MLLM）的教程和代码库。我们不依赖于 `timm`、`transformers` 等高度封装的库，而是逐步构建起整个模型的核心组件，包括 **Transformer**、**Vision Transformer (ViT)**、**GPT-style 语言模型**，并最终将它们融合成一个能够理解图像并生成描述的 MLLM。

通过亲手实现每一个模块，您将深入理解：
- Transformer 架构的细节，从注意力机制到编码器/解码器模块。
- Vision Transformer (ViT) 如何将计算机视觉问题转化为序列处理任务。
- 自回归语言模型（LLM）如何基于 Causal Attention 生成连贯的文本。
- 多模态大模型如何通过一个“连接器”（Connector）来对齐视觉和语言这两个不同的模态空间。

最终，我们将构建一个可以“看图说话”的迷你版多模态模型。

## 项目架构

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
│   ├── layers.py           # PositionwiseFeedForward, PositionalEncoding, LayerNorm
│   ├── blocks.py           # EncoderBlock, DecoderBlock
│   └── model.py            # TransformerEncoder, TransformerDecoder, Transformer
│
├── vision_transformer/
│   ├── __init__.py
│   ├── vit.py              # ViT模型实现
│   ├── train_vit.py        # 训练ViT的脚本
│   └── predict_vit.py      # 使用训练好的ViT进行推理的脚本
│
├── language_model/
│   ├── __init__.py
│   ├── tokenizer.py        # 简单的字符级分词器
│   ├── llm.py              # GPT-style LLM模型实现
│   ├── train_llm.py        # 训练LLM的脚本
│   └── generate_text.py    # 使用训练好的LLM生成文本的脚本
│
├── multimodal_model/
│   ├── __init__.py
│   ├── connector.py        # 视觉特征到语言空间的连接器
│   ├── mllm.py             # 多模态大模型 (组合ViT, Connector, LLM)
│   ├── train_mllm.py       # 训练/微调MLLM的脚本
│   └── inference_mllm.py   # 多模态推理脚本 (看图说话)
│
├── tests/
│   ├── test_attention.py   # 单元测试：验证Attention
│   ├── test_blocks.py      # 单元测试：验证Encoder/Decoder Block
│   └── test_transformer.py # 单元测试：验证Transformer
│
├── utils/
│   ├── __init__.py
│   ├── training_utils.py   # 训练循环、保存/加载模型、日志记录等
│   └── config_parser.py    # 解析yaml配置文件的函数
│
├── main.py                 # 项目主入口，通过命令行参数选择执行任务
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明文档
```

## 各模块详细说明

### 1. `transformer_from_scratch/` (基础核心)
这是整个项目的基石。所有与标准 Transformer 架构相关的组件都在这里实现。
- **`attention.py`**: 实现缩放点积注意力和多头注意力。
- **`layers.py`**: 实现前馈网络、位置编码（正弦/余弦和可学习版本）、层归一化。
- **`blocks.py`**: 将 `MultiHeadAttention` 和 `PositionwiseFeedForward` 等组装成一个完整的 `EncoderBlock` 和 `DecoderBlock`。
- **`model.py`**: 将多个 Block 堆叠起来，形成 `TransformerEncoder` 和 `TransformerDecoder`。

### 2. `vision_transformer/` (单模态应用一: 视觉)
基于第一阶段的成果，构建 Vision Transformer (ViT)。
- **`vit.py`**: 导入 `TransformerEncoder`，并额外实现 `PatchEmbedding` 层，负责将 `[B, C, H, W]` 的图像转换为 `[B, N, D]` 的 patch embedding 序列。最终将所有组件组合成完整的 ViT 模型。
- **`train_vit.py`**: 负责加载 CIFAR-10 数据、实例化 ViT 模型、执行训练和评估循环。
- **`predict_vit.py`**: 用于对 CIFAR-10 测试集图像进行分类推理。

### 3. `language_model/` (单模态应用二: 语言)
同样基于第一阶段的成果，构建一个小型自回归语言模型（GPT-style）。
- **`tokenizer.py`**: 实现一个简单的字符级分词器，包含 `encode` 和 `decode` 方法。
- **`llm.py`**: 导入 `DecoderBlock`，并实现一个包含词嵌入、位置编码、一堆带有因果掩码（Causal Mask）的 Decoder Block 以及一个最终预测 logits 的完整语言模型。
- **`train_llm.py`**: 负责加载 Tiny Shakespeare 文本数据、实例化 LLM 模型并进行训练。
- **`generate_text.py`**: LLM 推理脚本，用于生成文本。

### 4. `multimodal_model/` (多模态融合)
这是项目的最终目标，将前面构建的视觉和语言模块进行融合。
- **`connector.py`**: 多模态设计的核心。它将 ViT 输出的视觉特征映射到 LLM 能够理解的语言 embedding 空间。实现上可以为简单的线性层或 MLP。
- **`mllm.py`**: 实例化一个 ViT、一个 LLM 和一个 Connector。其 `forward` 方法负责将图像编码后的特征与文本提示的 embeddings 拼接起来，共同送入语言模型。
- **`train_mllm.py`**: 加载 Flickr8k 图文对数据进行训练，设计合理的训练策略
- **`inference_mllm.py`**: MLLM 推理代码，实现“看图说话”。

## 评分细则（20 points）

#### 1. Transformer (5 points)
- **要求**:
    - 完整、正确地实现 `attention.py`, `layers.py`, `blocks.py`, `model.py` 中的核心类。
    - 提交的报告中必须包含tests文件夹单元测试**全部测试通过**的截图。
- **评分**: 根据代码实现的正确性与单元测试结果给分。

#### 2. Vision Transformer (5 points)
- **要求**:
    - 完整、正确地实现 ViT 模型，并能成功在 CIFAR-10 上进行训练。
    - 报告中需展示**训练过程的 Loss 变化曲线图**，并报告在**CIFAR-10 测试集上的最终准确率**。
- **评分**: 根据代码正确性和测试集准确率综合给分（正常结果即可）

#### 3. Language Model (5 points)
- **要求**:
    - 完整、正确地实现 GPT-style LLM，并能在 Tiny Shakespeare 数据集上进行训练。
    - 报告中需展示**训练过程的 Loss 变化曲线图**。
    - 报告中需提供**至少 2 个**使用训练好的模型生成的**文本样本**，并与原始数据风格进行对比。
- **评分**: 根据代码正确性和文本生成效果（是否通顺、是否符合莎士比亚文风）综合给分。

#### 4. Multimodal Model (5 points)
- **要求**:
    - 完整、正确地实现 Connector 和 MLLM 的组装。
    - 成功在 Flickr8k 数据集上进行训练。
    - 报告中需展示**训练过程的 Loss 变化曲线图**。
    - 报告中需提供**至少 2 个“看图说话”的例子**（展示测试图片和模型生成的描述文字）。
- **评分**: 根据代码正确性以及图像描述生成的效果综合给分。

## 报告要求
1. 展示每个任务的实验结果，包括但不限于：
    - Transformer 单元测试通过的截图。
    - ViT 和 LLM 训练的 Loss 曲线图、ViT 的测试准确率。
    - LLM 生成的文本样本。
    - MLLM 的 Loss 曲线图、Image Captioning 的结果展示（图片+生成描述）。
    - 对结果进行简单的分析讨论。
2.  **遇到的问题与解决方案 (Challenges and Solutions)**: 记录你在项目实现过程中遇到的主要困难（如 Debug 经历、性能瓶颈、效果不佳等）以及你是如何解决的。

## 代码要求
1. 框架灵活性: 本文档提供的项目架构（MLLM-from-scratch/ 目录结构）是一个建议性参考，旨在帮助你组织代码。你不必严格遵守此文件结构。
2. 核心要求: 无论采用何种代码结构，最终提交的代码必须满足以下核心要求：
    - 功能完整: 成功实现了评分细则中描述的所有任务（Transformer 基础、ViT 分类、LLM 生成、MLLM 看图说话）。
    - 结构清晰: 代码组织应具备良好的模块化和逻辑性，便于理解和评估。避免将所有代码堆砌在少数几个文件中。
    - 可读性强: 关键部分应有必要的注释，代码风格应保持一致，命名规范。
    - 可复现性: 项目应能顺利运行。

## 关于第一部分的补充说明
* 已经将 transformer，vision transformer，language model，multimodal model 补全
* 训练与测试脚本在 script/ 目录下：train 为训练，test 为测试
* vit 和 gpt 的 loss 图都在 mllm_from_scratch/MLLM_from_scratch下；mllm 的在mllm_from_scratch/MLLM_from_scratch/checkpoint 下
* 组装 transformer 需要通过的单元测试在 test_transformer.ipynb 中完成
* 调整训练配置在 configs/ 中找对应的文件进行调整
* 原始 readme 没有写下载数据集的操作，实际需要进行：在 datasets/中补全代码，下载数据集到 data/ 中
* 部分 loss 图需要手动添加代码绘制
* 有一部分初始化 vit，gpt 等的部分没有写在 todo 里，运行会报错存在省略号，需要手动初始化传入参数  

---
# 第二部分

## 强化学习扩展
```
MLLM-from-scratch/
├── configs/
│   ├── rl_mllm_config.yaml     # 新增：RL 微调的超参。需注意对齐 sft 的 mllm 参数
│
├── multimodal_model/
│   ├── mllm.py                 # 补充：支持采样 & 返回 log prob。新增部分已用注释标出
│   ├── train_mllm.py
│   ├── inference_mllm.py
│   ├── rl_finetune_mllm.py     # 新增：RL 微调脚本（核心）
│   └── inference_rl_mllm.py     # 新增：RL 推理脚本
│
├── scripts/
│   ├── test_rl_mllm.sh         # 新增：测试 RL 微调后的结果
│   └── rl_mllm.sh              # 新增：进行RL 微调

```
### 采用方法
SCST（Self-Critical Sequence Training，自批判序列训练）是一种**基于 REINFORCE 的策略梯度方法**，专门用来训练生成模型（如图像描述、机器翻译），让模型直接优化 BLEU、CIDEr 这类**序列级指标**，并用“自己贪心解码出的结果”当作 baseline 来减小方差。

---

### 背景：为什么要 SCST？

传统训练（交叉熵）有两个问题：
   * 训练时看的是“真标签前缀”，
   * 推理时只能用“自己生成的前缀”。
所以希望：**直接对“整句”给奖励，用强化学习优化这个奖励**。


---

### REINFORCE + baseline 的一般形式

REINFORCE 的梯度估计形如：

$$
\nabla_\theta J(\theta) \approx (R - b) \nabla_\theta \log \pi_\theta(a_{1:T}|x)
$$

* R：当前采样序列的奖励
* b：baseline（一个和动作无关的标量），用来**减小方差**
* 如果 R > b，则增加该序列的概率；反之则减小

关键在于：**b 怎么选？**

---

### “Self-Critical”：用自己贪心解码做 baseline

SCST 的核心 trick：

1. 对同一个输入 x：

   * 用当前模型 **采样（可能选次高的，有随机性）** 一句：$\hat{a}^{\text{sample}}$，得到奖励 $R^{\text{sample}}$
   * 用当前模型 **贪心解码（greedy,只选最高的，无随机）** 一句：$\hat{a}^{\text{greedy}}$，得到奖励 $R^{\text{greedy}}$
2. 把 **贪心解码的奖励** 当成 baseline：
   $$
   b = R^{\text{greedy}}
   $$
3. 梯度变成：
   $$
   \nabla_\theta J(\theta) \approx (R^{\text{sample}} - R^{\text{greedy}}),\nabla_\theta \log \pi_\theta(\hat{a}^{\text{sample}}|x)
   $$

直观理解：

* 如果采样句子比分数“贪心句子”好（R_sample > R_greedy），
  → 提高这句的概率（正更新）。
* 如果更差
  → 降低这句的概率（负更新）。

所以叫 **Self-Critical**：模型以自己当前的“推理表现”（greedy）来批判自己采样出的句子。


1. **直接优化评测指标**（CIDEr、BLEU 等），不再是 token 级 CE surrogate。
2. **方差小**：baseline 就是同一模型的 greedy 输出，稳定且易算。
3. 实现简单：在已有 seq2seq / caption 模型上，只需加一层采样和奖励计算逻辑。



# 参考
* 复旦大学2025秋季学期人工智能前沿探索实践 Project-2
* llm from scratch 项目
