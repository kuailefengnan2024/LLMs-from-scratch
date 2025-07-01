# 版权所有 (c) Sebastian Raschka，基于 Apache License 2.0 许可证（见 LICENSE.txt）。
# 来源：《从零开始构建大型语言模型》
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

from .utils import KVCache   # noqa: F401

import torch
import torch.nn as nn


#####################################
# 第3章
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 减少投影维度以匹配期望的输出维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 用于组合多头输出的线性层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache=False, start_pos=0, cache=None):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        values = self.W_value(x)
        queries = self.W_query(x)

        # 通过添加`num_heads`维度隐式分割矩阵
        # 展开最后一个维度: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if use_cache:
            if cache is not None:
                keys = torch.cat([cache[0], keys], dim=2)
                values = torch.cat([cache[1], values], dim=2)
            next_cache = (keys, values)
        else:
            next_cache = None

        seq_len = keys.size(2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal_mask = causal_mask[:, -num_tokens:][None, None, :, :]

        # 计算带因果掩码的缩放点积注意力（即自注意力）
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(causal_mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并多头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的投影

        return context_vec, next_cache


#####################################
# 第4章
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False, start_pos=0, cache=None):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, use_cache=use_cache, start_pos=start_pos, cache=cache) # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 添加原始输入

        # 前馈块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 添加原始输入

        return x, next_cache


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.current_pos = 0

    def forward(self, in_idx, use_cache=False, cache=None):
        batch_size, seq_len = in_idx.shape
        pos = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(pos)
        x = self.drop_emb(tok_embeds + pos_embeds)

        if use_cache:
            start_pos = self.current_pos
            self.current_pos += seq_len
        else:
            start_pos = 0

        next_cache = []
        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_cache = block(x, use_cache=use_cache, start_pos=start_pos, cache=blk_cache)
            if cache:
                cache.update(i, new_cache)
            next_cache.append(new_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
