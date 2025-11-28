# tests/test_blocks.py

import torch
from transformer_from_scratch.blocks import EncoderBlock, DecoderBlock

def setup_test_variables():
    """为测试设置通用变量。"""
    params = {
        'batch_size': 4,
        'd_model': 128,
        'n_heads': 8,
        'd_ff': 512,
        'src_len': 12,
        'tgt_len': 10,
        'dropout': 0.1
    }
    params['src'] = torch.randn(params['batch_size'], params['src_len'], params['d_model'])
    params['src_mask'] = torch.ones(params['batch_size'], 1, 1, params['src_len'], dtype=torch.bool)
    params['src_mask'][:, :, :, -2:] = False

    params['tgt'] = torch.randn(params['batch_size'], params['tgt_len'], params['d_model'])
    params['enc_output'] = torch.randn(params['batch_size'], params['src_len'], params['d_model'])
    
    tgt_padding_mask = torch.ones(params['batch_size'], 1, params['tgt_len'], 1, dtype=torch.bool)
    tgt_padding_mask[:, :, -1, :] = False
    tgt_causal_mask = torch.tril(torch.ones(params['tgt_len'], params['tgt_len'], dtype=torch.bool))
    params['tgt_mask'] = tgt_padding_mask & tgt_causal_mask
    return params

def test_encoder_block_forward_shape(params):
    """Test 1: 测试 EncoderBlock 的输出形状。"""
    print("  - Running test_encoder_block_forward_shape...")
    encoder_block = EncoderBlock(d_model=params['d_model'], n_heads=params['n_heads'], d_ff=params['d_ff'], dropout=params['dropout'])
    output = encoder_block(params['src'], params['src_mask'])
    assert output.shape == params['src'].shape, f"Expected shape {params['src'].shape}, but got {output.shape}"

def test_encoder_block_gradient_flow(params):
    """Test 2: 测试梯度是否能正确流经 EncoderBlock。"""
    print("  - Running test_encoder_block_gradient_flow...")
    encoder_block = EncoderBlock(d_model=params['d_model'], n_heads=params['n_heads'], d_ff=params['d_ff'], dropout=params['dropout'])
    encoder_block.train()
    output = encoder_block(params['src'], params['src_mask'])
    loss = output.sum()
    loss.backward()
    assert encoder_block.self_attn.w_q.weight.grad is not None, "Gradient missing in self_attn.w_q"
    assert encoder_block.feed_forward.w_1.weight.grad is not None, "Gradient missing in feed_forward.w_1"

def test_decoder_block_forward_shape(params):
    """Test 3: 测试 DecoderBlock 的输出形状。"""
    print("  - Running test_decoder_block_forward_shape...")
    decoder_block = DecoderBlock(d_model=params['d_model'], n_heads=params['n_heads'], d_ff=params['d_ff'], dropout=params['dropout'])
    output = decoder_block(params['tgt'], params['enc_output'], params['tgt_mask'], params['src_mask'])
    assert output.shape == params['tgt'].shape, f"Expected shape {params['tgt'].shape}, but got {output.shape}"

def test_decoder_block_cross_attention_dependency(params):
    """Test 4 [NEW]: 测试 DecoderBlock 的输出是否依赖于编码器的输出。"""
    print("  - Running test_decoder_block_cross_attention_dependency...")
    decoder_block = DecoderBlock(d_model=params['d_model'], n_heads=params['n_heads'], d_ff=params['d_ff'], dropout=0.0)
    decoder_block.eval()

    with torch.no_grad():
        output_base = decoder_block(params['tgt'], params['enc_output'], params['tgt_mask'], params['src_mask'])
        enc_output_zero = torch.zeros_like(params['enc_output'])
        output_zero_enc = decoder_block(params['tgt'], enc_output_zero, params['tgt_mask'], params['src_mask'])
    
    assert not torch.allclose(output_base, output_zero_enc), "Decoder output should change when encoder output changes."

if __name__ == "__main__":
    print("Running tests for blocks.py...")
    test_params = setup_test_variables()
    
    test_encoder_block_forward_shape(test_params)
    test_encoder_block_gradient_flow(test_params)
    test_decoder_block_forward_shape(test_params)
    test_decoder_block_cross_attention_dependency(test_params)
    
    print("\n✅ All block tests passed!")