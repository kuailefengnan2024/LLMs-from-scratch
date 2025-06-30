# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from llms_from_scratch.ch04 import GPTModel, GPTModelFast
from llms_from_scratch.kv_cache.gpt2 import GPTModel as GPTModelKV
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.kv_cache.generate import generate_text_simple as generate_text_simple_cached

import pytest
import torch
import tiktoken


GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # Dropout率
    "qkv_bias": False        # 查询-键-值偏置
}


@pytest.mark.parametrize("ModelClass", [GPTModel, GPTModelFast, GPTModelKV])
@pytest.mark.parametrize("generate_fn", [generate_text_simple, generate_text_simple_cached])
def test_gpt_model_variants(ModelClass, generate_fn):

    # 跳过不兼容的组合
    if generate_fn is generate_text_simple and getattr(ModelClass, "reset_kv_cache", False):
        return
    if generate_fn is generate_text_simple_cached and not getattr(ModelClass, "reset_kv_cache", False):
        return

    torch.manual_seed(123)
    model = ModelClass(GPT_CONFIG_124M)
    model.eval()  # 禁用dropout

    start_context = "Hello, I am"

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
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    expect = torch.tensor([
        [15496,   11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]
    ])
    assert torch.equal(expect, out), "生成的输出与预期输出不匹配"
