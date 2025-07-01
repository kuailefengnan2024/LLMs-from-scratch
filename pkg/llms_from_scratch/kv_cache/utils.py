# 版权所有 (c) Sebastian Raschka，基于 Apache License 2.0 许可证（见 LICENSE.txt）。
# 来源：《从零开始构建大型语言模型》
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

class KVCache:
    """键值缓存类，用于存储和管理Transformer模型的键值对"""
    def __init__(self, n_layers):
        """初始化KV缓存
        
        Args:
            n_layers: 模型层数
        """
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        """获取指定层的缓存
        
        Args:
            layer_idx: 层索引
            
        Returns:
            指定层的缓存内容
        """
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        """更新指定层的缓存
        
        Args:
            layer_idx: 层索引
            value: 新的缓存值
        """
        self.cache[layer_idx] = value

    def get_all(self):
        """获取所有层的缓存
        
        Returns:
            所有缓存内容的列表
        """
        return self.cache

    def reset(self):
        """重置所有缓存"""
        for i in range(len(self.cache)):
            self.cache[i] = None
