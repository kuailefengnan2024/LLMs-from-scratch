# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.llama3 import (
    compute_rope_params,
    apply_rope,
    LLAMA32_CONFIG_1B,
    GroupedQueryAttention,
    GroupedQueryAttentionFast,
    Llama3Model,
)
from llms_from_scratch.kv_cache.llama3 import Llama3Model as Llama3ModelKV
from llms_from_scratch.kv_cache.generate import generate_text_simple as generate_text_simple_cached

import importlib
import os
import pytest
import tiktoken
import torch


class LitGPTRMSNorm(torch.nn.Module):
    """均方根层归一化。

    来源于 https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
    Apache License 2.0 许可证: https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

    派生自 https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py。BSD 3-Clause 许可证:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # 注意：原始RMSNorm论文实现并不等效
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_rope():

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

    # 设置
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    rope_theta = 500_000

    rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }

    # 实例化RoPE参数
    cos, sin = compute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
        freq_config=rope_config,
    )

    # 虚拟查询和键张量
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # 应用旋转位置嵌入
    queries_rot = apply_rope(queries, cos, sin)
    keys_rot = apply_rope(keys, cos, sin)

    # 通过HF生成参考RoPE
    hf_rope_params = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }

    class RoPEConfig:
        rope_type = "llama3"
        rope_scaling = hf_rope_params
        factor = 1.0
        dim: int = head_dim
        rope_theta = 500_000
        max_position_embeddings: int = 8192
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

    config = RoPEConfig()

    rot_emb = LlamaRotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)


GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # Dropout率
    "qkv_bias": False        # 查询-键-值偏置
}


def test_grouped_query_attention_equivalence():
    torch.manual_seed(42)
    b, t, d_in, d_out, num_heads, num_kv_groups = 2, 8, 32, 64, 4, 2

    x = torch.randn(b, t, d_in)
    cos, sin = compute_rope_params(
        head_dim=d_out // num_heads,
        theta_base=50_000,
        context_length=t,
        freq_config={
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": t,
        }
    )

    # 慢速版本的因果掩码
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)

    attn1 = GroupedQueryAttention(d_in, d_out, num_heads, num_kv_groups)
    attn2 = GroupedQueryAttentionFast(d_in, d_out, num_heads, num_kv_groups)

    # 复制权重使两个模型相同
    attn2.load_state_dict(attn1.state_dict())

    # 运行两个版本
    y1 = attn1(x, mask, cos, sin)
    y2 = attn2(x, cos, sin)

    # 比较输出
    max_diff = (y1 - y2).abs().max().item()
    print(f"慢速和快速输出之间的最大差异: {max_diff:.4e}")
    assert torch.allclose(y1, y2, atol=1e-4)


@pytest.fixture(scope="session")
def llama3_weights_path(tmp_path_factory):
    """创建并保存确定性的Llama3模型用于测试。"""
    path = tmp_path_factory.mktemp("models") / "llama3_test_weights.pt"

    if not path.exists():
        torch.manual_seed(123)
        model = Llama3Model(LLAMA32_CONFIG_1B)
        torch.save(model.state_dict(), path)

    return path


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="由于计算或内存限制在GitHub Actions中跳过"
)
@pytest.mark.parametrize("ModelClass", [Llama3Model, Llama3ModelKV])
@pytest.mark.parametrize("generate_fn", [generate_text_simple, generate_text_simple_cached])
def test_model_variants(ModelClass, generate_fn, llama3_weights_path):

    # 跳过不兼容的组合
    if generate_fn is generate_text_simple and getattr(ModelClass, "reset_kv_cache", False):
        return
    if generate_fn is generate_text_simple_cached and not getattr(ModelClass, "reset_kv_cache", False):
        return

    torch.manual_seed(123)
    model = ModelClass(LLAMA32_CONFIG_1B)
    model.load_state_dict(torch.load(llama3_weights_path, weights_only=True))
    model.eval()

    start_context = "Llamas eat"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}输入\n{50*'='}")
    print("\n输入文本:", start_context)
    print("编码后的输入文本:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_fn(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=5,
        context_size=LLAMA32_CONFIG_1B["context_length"]
    )
    print("编码后的输出文本:", out)
    expect = torch.tensor([
        [43, 2543, 292, 4483, 100383, 8113, 76873, 42175, 72641]
    ])
    assert torch.equal(expect, out)


def test_rmsnorm_equivalence():
    torch.manual_seed(42)

    hidden_size = 64
    batch_size = 8
    seq_len = 16

    rms_norm = torch.nn.RMSNorm(hidden_size, eps=1e-6)
    lit_norm = LitGPTRMSNorm(hidden_size)

    # 同步权重
    with torch.no_grad():
        lit_norm.weight.copy_(lit_norm.weight)

    x = torch.randn(batch_size, seq_len, hidden_size)

    out1 = rms_norm(x)
    out2 = lit_norm(x)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)
