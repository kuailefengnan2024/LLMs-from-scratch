# 加载预训练权重的替代方法

本文件夹包含替代的权重加载策略，以防权重从OpenAI处无法获取。

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb)：（推荐）包含从PyTorch状态字典加载权重的代码，这些状态字典是我通过转换原始TensorFlow权重创建的

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb)：包含通过`transformers`库从Hugging Face模型中心加载权重的代码

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb)：包含直接通过`safetensors`库从Hugging Face模型中心加载权重的代码（跳过Hugging Face transformer模型的实例化）