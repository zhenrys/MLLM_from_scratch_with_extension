# tests/test_attention.py

import torch
import torch.nn as nn
import math

# 假设项目根目录在 Python 路径中
from transformer_from_scratch.attention import ScaledDotProductAttention, MultiHeadAttention

# --- 1. 设置测试所需的通用变量 ---
def setup_test_variables():
    """返回一个包含所有测试所需参数和张量的字典。"""
    params = {
        'batch_size': 4,
        'd_model': 128,
        'n_heads': 8,
        'seq_len': 10,
    }
    params['d_k'] = params['d_model'] // params['n_heads']

    # 为 ScaledDotProductAttention 准备的输入
    params['query_sdpa'] = torch.randn(params['batch_size'], params['n_heads'], params['seq_len'], params['d_k'])
    params['key_sdpa'] = torch.randn(params['batch_size'], params['n_heads'], params['seq_len'], params['d_k'])
    params['value_sdpa'] = torch.randn(params['batch_size'], params['n_heads'], params['seq_len'], params['d_k'])

    # 为 MultiHeadAttention 准备的输入
    params['query_mha'] = torch.randn(params['batch_size'], params['seq_len'], params['d_model'])
    params['key_mha'] = torch.randn(params['batch_size'], params['seq_len'], params['d_model'])
    params['value_mha'] = torch.randn(params['batch_size'], params['seq_len'], params['d_model'])
    
    return params

# --- 2. 为 ScaledDotProductAttention 编写测试函数 ---

def test_sdpa_forward_shape(params):
    """Test 1: 测试 ScaledDotProductAttention 的输出形状。"""
    print("  - Running test_sdpa_forward_shape...")
    attention = ScaledDotProductAttention()
    # 注意：您的模型实现只返回 output，我们遵循这个实现
    output = attention(params['query_sdpa'], params['key_sdpa'], params['value_sdpa'])
    
    expected_shape = (params['batch_size'], params['n_heads'], params['seq_len'], params['d_k'])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_sdpa_with_mask_behavior(params):
    """Test 2 [ENHANCED]: 通过行为测试验证掩码的有效性。"""
    print("  - Running test_sdpa_with_mask_behavior...")
    attention = ScaledDotProductAttention(dropout=0.0) # 关闭 dropout 以进行确定性测试
    
    # 创建一个掩码，遮盖最后一个 token
    mask = torch.ones(params['batch_size'], 1, 1, params['seq_len'], dtype=torch.bool)
    mask[:, :, :, -1] = 0  # Mask the last token
    
    # 创建一个特殊的 value 张量
    # 被掩码的位置（最后一个 token）的值设为 100.0，其他位置为 0
    special_value = torch.zeros_like(params['value_sdpa'])
    special_value[:, :, -1, :] = 100.0
    
    # 前向传播
    output = attention(params['query_sdpa'], params['key_sdpa'], special_value, mask=mask)
    
    # 验证：如果掩码有效，100.0 这个值不应该对输出有任何贡献。
    # 因此，输出张量中的最大值应该非常接近于 0。
    max_output_val = torch.max(output).item()
    assert max_output_val < 1e-6, f"Masking failed. Large value from masked position leaked into output. Max output value is {max_output_val}"

# --- 3. 为 MultiHeadAttention 编写测试函数 ---

def test_mha_forward_shape(params):
    """Test 3: 测试 MultiHeadAttention 的输出形状。"""
    print("  - Running test_mha_forward_shape...")
    mha = MultiHeadAttention(d_model=params['d_model'], n_heads=params['n_heads'])
    output = mha(params['query_mha'], params['key_mha'], params['value_mha'])
    
    expected_shape = (params['batch_size'], params['seq_len'], params['d_model'])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_mha_gradient_flow(params):
    """Test 4: 测试梯度是否能正确流经 MultiHeadAttention。"""
    print("  - Running test_mha_gradient_flow...")
    mha = MultiHeadAttention(d_model=params['d_model'], n_heads=params['n_heads'])
    mha.train()
    
    output = mha(params['query_mha'], params['key_mha'], params['value_mha'])
    loss = output.sum()
    loss.backward()

    assert mha.w_q.weight.grad is not None, "Gradient missing in w_q"
    assert mha.w_k.weight.grad is not None, "Gradient missing in w_k"
    assert mha.w_v.weight.grad is not None, "Gradient missing in w_v"
    assert mha.fc.weight.grad is not None, "Gradient missing in fc"
    assert mha.layer_norm.weight.grad is not None, "Gradient missing in layer_norm"

def test_mha_vs_pytorch_implementation(params):
    """Test 5: 将我们的 MultiHeadAttention 与 PyTorch 的 nn.MultiheadAttention 进行比较。"""
    print("  - Running test_mha_vs_pytorch_implementation...")
    # 设置我们的模型
    our_mha = MultiHeadAttention(d_model=params['d_model'], n_heads=params['n_heads'], dropout=0.0)
    our_mha.eval()

    # 设置 PyTorch 的模型
    pytorch_mha = nn.MultiheadAttention(embed_dim=params['d_model'], num_heads=params['n_heads'], bias=True, batch_first=True, dropout=0.0)
    pytorch_mha.eval()

    # 复制权重
    pytorch_mha.in_proj_weight.data.copy_(torch.cat([our_mha.w_q.weight, our_mha.w_k.weight, our_mha.w_v.weight]))
    pytorch_mha.in_proj_bias.data.copy_(torch.cat([our_mha.w_q.bias, our_mha.w_k.bias, our_mha.w_v.bias]))
    pytorch_mha.out_proj.weight.data.copy_(our_mha.fc.weight.data)
    pytorch_mha.out_proj.bias.data.copy_(our_mha.fc.bias.data)
    
    # 运行 PyTorch 模型
    pytorch_output, _ = pytorch_mha(params['query_mha'], params['key_mha'], params['value_mha'])

    # 复现我们 MHA 的核心逻辑（不含 Add & Norm）
    with torch.no_grad():
        query = our_mha.w_q(params['query_mha'])
        key = our_mha.w_k(params['key_mha'])
        value = our_mha.w_v(params['value_mha'])
        query = query.view(params['batch_size'], -1, params['n_heads'], params['d_k']).transpose(1, 2)
        key = key.view(params['batch_size'], -1, params['n_heads'], params['d_k']).transpose(1, 2)
        value = value.view(params['batch_size'], -1, params['n_heads'], params['d_k']).transpose(1, 2)
        context = our_mha.attention(query, key, value, mask=None)
        context = context.transpose(1, 2).contiguous().view(params['batch_size'], -1, params['d_model'])
        our_output_pre_add_norm = our_mha.fc(context)

    # 比较输出
    assert torch.allclose(pytorch_output, our_output_pre_add_norm, atol=1e-6), "Our MHA core implementation does not match PyTorch's."

# --- 4. 主执行块 ---
if __name__ == "__main__":
    print("Running tests for attention.py...")
    test_params = setup_test_variables()
    
    test_sdpa_forward_shape(test_params)
    test_sdpa_with_mask_behavior(test_params)
    test_mha_forward_shape(test_params)
    test_mha_gradient_flow(test_params)
    test_mha_vs_pytorch_implementation(test_params)
    
    print("\n✅ All attention tests passed!")