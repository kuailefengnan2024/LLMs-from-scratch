# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import os
from pathlib import Path

import torch
import torch.nn as nn

import tiktoken
from tiktoken.load import load_tiktoken_bpe


LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           # 词汇表大小
    "context_length": 131_072,       # 用于训练模型的上下文长度
    "emb_dim": 2048,                 # 嵌入维度
    "n_heads": 32,                   # 注意力头数
    "n_layers": 16,                  # 层数
    "hidden_dim": 8192,              # FeedForward中间维度的大小
    "n_kv_groups": 8,                # 分组查询注意力的键值组
    "rope_base": 500_000.0,          # RoPE的"theta"基数
    "dtype": torch.bfloat16,         # 低精度数据类型以减少内存使用
    "rope_freq": {                   # RoPE频率缩放
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,           # 词汇表大小
    "context_length": 131_072,       # 用于训练模型的上下文长度
    "emb_dim": 3072,                 # 嵌入维度
    "n_heads": 24,                   # 注意力头数
    "n_layers": 28,                  # 层数
    "hidden_dim": 8192,              # FeedForward中间维度的大小
    "n_kv_groups": 8,                # 分组查询注意力的键值组
    "rope_base": 500_000.0,          # RoPE的"theta"基数
    "dtype": torch.bfloat16,         # 低精度数据类型以减少内存使用
    "rope_freq": {                   # RoPE频率缩放
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}


class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 主要模型参数
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # 使用ModuleList因为Sequential只能接受一个输入，而我们需要 `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # 可重用的工具
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # 形状 [batch_size, num_tokens, emb_size]
        x = x + shortcut  # 添加原始输入

        # 前馈块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # 添加原始输入

        return x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, num_heads, num_kv_groups, dtype=None
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"
        assert num_heads % num_kv_groups == 0, "num_heads必须能被num_kv_groups整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # 形状: (b, num_tokens, d_out)
        keys = self.W_key(x)  # 形状: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # 形状: (b, num_tokens, num_kv_groups * head_dim)

        # 重塑查询、键和值
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # 转置键、值和查询
        keys = keys.transpose(1, 2)  # 形状: (b, num_kv_groups, num_tokens, head_dim)
        values = values.transpose(1, 2)  # 形状: (b, num_kv_groups, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # 形状: (b, num_heads, num_tokens, head_dim)

        # 应用RoPE
        keys = apply_rope(keys, cos, sin)
        queries = apply_rope(queries, cos, sin)

        # 扩展键和值以匹配头数
        # 形状: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)  # 形状: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # 形状: (b, num_heads, num_tokens, head_dim)
        # 例如，在沿dim=1（查询组）进行repeat_interleave之前:
        #   [K1, K2]
        # repeat_interleave后（每个查询组重复group_size次）:
        #   [K1, K1, K2, K2]
        # 如果我们使用常规repeat而不是repeat_interleave，会得到:
        #   [K1, K2, K1, K2]

        # 计算带因果掩码的缩放点积注意力（即自注意力）
        # 形状: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        # 使用掩码填充注意力分数
        attn_scores = attn_scores.masked_fill(mask[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # 形状: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选投影

        return context_vec


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "嵌入维度必须是偶数"

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 频率调整
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # 生成位置索引
    positions = torch.arange(context_length, dtype=dtype)

    # 计算角度
    angles = positions[:, None] * inv_freq[None, :]  # 形状: (context_length, head_dim // 2)

    # 扩展角度以匹配head_dim
    angles = torch.cat([angles, angles], dim=1)  # 形状: (context_length, head_dim)

    # 预计算正弦和余弦
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "头维度必须是偶数"

    # 将x分为前半部分和后半部分
    x1 = x[..., : head_dim // 2]  # 前半部分
    x2 = x[..., head_dim // 2:]  # 后半部分

    # 调整sin和cos的形状
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 应用旋转变换
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # 应用cos和sin旋转后可以使用低精度
    return x_rotated.to(dtype=x.dtype)


##########################################
# 分词器
##########################################


class Llama3Tokenizer:
    """围绕tiktoken的轻量级包装器，跟踪Llama-3特殊ID。"""
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        # 来自Meta的tokenizer.json的硬编码值
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False, **kwargs):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        return self.model.decode(ids)


class ChatFormat:

    def __init__(self, tokenizer: Llama3Tokenizer, *,
                 default_system="You are a helpful assistant."):
        self.tok = tokenizer
        self.default_system = default_system

    def _header(self, role):
        """编码 <|start_header_id|>role<|end_header_id|>\n\n"""
        return (
            [self.tok.special["<|start_header_id|>"]]
            + self.tok.encode(role)
            + [self.tok.special["<|end_header_id|>"]]
            + self.tok.encode("\n\n")
        )

    def encode(self, user_message, system_message=None, allowed_special=None):
        sys_msg = system_message if system_message is not None else self.default_system

        ids = [self.tok.special["<|begin_of_text|>"]]

        # 系统消息
        ids += self._header("system")
        ids += self.tok.encode(sys_msg, allowed_special=allowed_special)
        ids += [self.tok.special["<|eot_id|>"]]

        # 用户消息
        ids += self._header("user")
        ids += self.tok.encode(user_message)
        ids += [self.tok.special["<|eot_id|>"]]

        # 助手头部（尚无内容）
        ids += self._header("assistant")

        return ids

    def decode(self, ids):
        return self.tok.decode(ids)


def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    # 查找"<|end_header_id|>"第一次出现的索引
    index = text.find(header_end)

    if index != -1:
        # 返回从"<|end_header_id|>"之后开始的子字符串
        return text[index + len(header_end):].strip()  # strip移除前导/尾随空白字符
    else:
        # 如果未找到令牌，返回原始文本
        return text


######################################################################
# Llama 3 快速版（面向效率的替代代码）
######################################################################

class GroupedQueryAttentionFast(nn.Module):
    """
    GroupedQueryAttention的直接替换，但使用PyTorch的
    scaled_dot_product_attention，如果在Ampere GPU（如A100）
    或更新版本上运行并使用float16/bfloat16或更低精度，则使用FlashAttention。
    """
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"
        assert num_heads % num_kv_groups == 0, "num_heads必须能被num_kv_groups整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, cos, sin):
        b, num_tokens, _ = x.shape

        # 投影到查询、键、值
        q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 应用旋转位置嵌入
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 将键/值组扩展到完整头数
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # 高效的缩放点积注意力
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=True  # 启用Flash/FlexAttention内核
        )

        # 合并头并投影
        attn_output = attn_output.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(attn_output)


class TransformerBlockFast(nn.Module):
    """
    与原始TransformerBlock相同，但使用
    GroupedQueryAttentionFast而不是GroupedQueryAttention。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttentionFast(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, cos, sin):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin)  # 形状 [batch_size, num_tokens, emb_size]
        x = x + shortcut  # 添加原始输入

        # 前馈块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # 添加原始输入

        return x


class Llama3ModelFast(nn.Module):
    """
    与原始Llama3Model相同，但使用TransformerBlockFast
    而不是TransformerBlock，后者又使用
    GroupedQueryAttentionFast而不是GroupedQueryAttention。
    """
    def __init__(self, cfg):
        super().__init__()

        # 主要模型参数
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # 使用ModuleList因为Sequential只能接受一个输入，而我们需要 `x, cos, sin`
            [TransformerBlockFast(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
