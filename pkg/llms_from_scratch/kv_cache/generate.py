# 版权所有 (c) Sebastian Raschka，基于 Apache License 2.0 许可证（见 LICENSE.txt）。
# 来源：《从零开始构建大型语言模型》
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

from .utils import KVCache
import torch


def generate_text_simple(model, idx, max_new_tokens, context_size=None, use_cache=True):
    """简单的文本生成函数
    
    Args:
        model: 训练好的模型
        idx: 输入的token索引
        max_new_tokens: 要生成的最大新token数量
        context_size: 上下文窗口大小，如果为None则使用模型配置中的context_length
        use_cache: 是否使用KV缓存来加速生成
        
    Returns:
        生成的完整token序列
    """
    model.eval()
    ctx_len = context_size or model.cfg["context_length"]
    cache = KVCache(n_layers=model.cfg["n_layers"]) if use_cache else None

    with torch.no_grad():
        if use_cache:
            # 重置模型的KV缓存并进行初始前向传播
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True, cache=cache)

            # 逐个生成新token
            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True, cache=cache)
        else:
            # 不使用缓存的标准生成方式
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
