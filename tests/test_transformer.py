# tests/test_transformer.py

import torch
from transformer_from_scratch.model import Transformer

def setup_test_variables():
    """设置通用参数和虚拟数据。"""
    params = {
        'src_vocab_size': 100,
        'tgt_vocab_size': 120,
        'd_model': 128,
        'num_layers': 2,
        'n_heads': 8,
        'd_ff': 512,
        'max_len': 100,
        'dropout': 0.1,
        'batch_size': 4,
        'src_len': 12,
        'tgt_len': 10,
        'pad_idx': 0
    }
    
    params['src'] = torch.randint(1, params['src_vocab_size'], (params['batch_size'], params['src_len']))
    params['tgt'] = torch.randint(1, params['tgt_vocab_size'], (params['batch_size'], params['tgt_len']))
    params['src'][0, -2:] = params['pad_idx']
    params['tgt'][1, -1:] = params['pad_idx']

    device = params['src'].device
    params['src_mask'] = Transformer.create_padding_mask(params['src'], params['pad_idx'])
    tgt_padding_mask = Transformer.create_padding_mask(params['tgt'], params['pad_idx'])
    tgt_causal_mask = Transformer.create_causal_mask(params['tgt_len'], device)
    params['tgt_mask'] = tgt_padding_mask & tgt_causal_mask
    
    return params

def test_model_forward_shape(params):
    """Test 1: 验证模型前向传播的输出形状。"""
    print("  - Running test_model_forward_shape...")
    model = Transformer(
        src_vocab_size=params['src_vocab_size'], tgt_vocab_size=params['tgt_vocab_size'],
        d_model=params['d_model'], num_layers=params['num_layers'], n_heads=params['n_heads'],
        d_ff=params['d_ff'], max_len=params['max_len'], dropout=params['dropout']
    )
    model.eval()
    with torch.no_grad():
        output = model(params['src'], params['tgt'], params['src_mask'], params['tgt_mask'])
    
    expected_shape = (params['batch_size'], params['tgt_len'], params['tgt_vocab_size'])
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

def test_model_gradient_flow(params):
    """Test 2: 确保梯度能流经整个模型。"""
    print("  - Running test_model_gradient_flow...")
    model = Transformer(
        src_vocab_size=params['src_vocab_size'], tgt_vocab_size=params['tgt_vocab_size'],
        d_model=params['d_model'], num_layers=params['num_layers'], n_heads=params['n_heads'],
        d_ff=params['d_ff'], max_len=params['max_len'], dropout=params['dropout']
    )
    model.train()

    output = model(params['src'], params['tgt'], params['src_mask'], params['tgt_mask'])
    loss = output.sum()
    loss.backward()

    assert model.src_embedding.weight.grad is not None, "Gradient missing in source embedding."
    assert model.encoder.layers[0].self_attn.w_q.weight.grad is not None, "Gradient missing in encoder's attention."
    assert model.decoder.layers[0].cross_attn.w_k.weight.grad is not None, "Gradient missing in decoder's cross-attention."
    assert model.fc_out.weight.grad is not None, "Gradient missing in the final output layer."

def test_causal_mask_logic(params):
    """Test 3: 验证因果掩码的逻辑。"""
    print("  - Running test_causal_mask_logic...")
    model = Transformer(
        src_vocab_size=params['src_vocab_size'], tgt_vocab_size=params['tgt_vocab_size'],
        d_model=params['d_model'], num_layers=params['num_layers'], n_heads=params['n_heads'],
        d_ff=params['d_ff'], max_len=params['max_len'], dropout=0.0 # 关闭 dropout
    )
    model.eval()

    with torch.no_grad():
        output_base = model(params['src'], params['tgt'], params['src_mask'], params['tgt_mask'])

        tgt_modified = params['tgt'].clone()
        new_token_val = (params['tgt'][0, 5] + 10) % params['tgt_vocab_size']
        if new_token_val == params['pad_idx']: new_token_val += 1
        tgt_modified[0, 5] = new_token_val

        output_modified = model(params['src'], tgt_modified, params['src_mask'], params['tgt_mask'])

    output_at_pos_4_base = output_base[0, 4, :]
    output_at_pos_4_modified = output_modified[0, 4, :]
    
    assert torch.allclose(output_at_pos_4_base, output_at_pos_4_modified, atol=1e-6), \
        "Causal mask failed: Output at t=4 changed when input at t=5 was modified."

    output_at_pos_5_base = output_base[0, 5, :]
    output_at_pos_5_modified = output_modified[0, 5, :]

    assert not torch.allclose(output_at_pos_5_base, output_at_pos_5_modified), \
        "Sanity check failed: Output at t=5 did not change when input at t=5 was modified."

if __name__ == "__main__":
    print("Running tests for model.py (Transformer)...")
    test_params = setup_test_variables()
    
    test_model_forward_shape(test_params)
    test_model_gradient_flow(test_params)
    test_causal_mask_logic(test_params)
    
    print("\n✅ All Transformer model tests passed!")