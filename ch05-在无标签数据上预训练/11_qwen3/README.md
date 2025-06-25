# 从零开始实现Qwen3

此文件夹中的[standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter笔记本包含Qwen3 0.6B的从零开始实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

&nbsp;
### 通过`llms-from-scratch`包使用Qwen3 0.6B

为了轻松使用Qwen3从零开始实现，您还可以使用基于此仓库源代码的`llms-from-scratch` PyPI包，位于[pkg/llms_from_scratch](../../pkg/llms_from_scratch)。

&nbsp;
#### 1) 安装

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2) 模型和文本生成设置

指定使用哪个模型：

```python
USE_REASONING_MODEL = True   # "思考"模型
USE_REASONING_MODEL = False  # 基础模型
```

用户可以定义的基本文本生成设置。使用150个标记，模型需要大约1.5 GB内存。

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3) 权重下载和加载

这会根据上面的模型选择自动下载权重文件：

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

然后如下加载模型权重：

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
#### 4) 初始化分词器

以下代码下载并初始化分词器：

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tok_filename,
    repo_id=repo_id,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=USE_REASONING_MODEL
)
```

&nbsp;

#### 5) 生成文本

最后，我们可以通过以下代码生成文本：

```python
prompt = "给我一个大语言模型的简短介绍。"
input_token_ids = tokenizer.encode(prompt)
```

```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"时间：{total_time:.2f} 秒")
print(f"{int(len(output_token_ids[0])/total_time)} 标记/秒")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"最大分配内存：{max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\n输出文本：\n\n", output_text + "...")
```

当使用Qwen3 0.6B推理模型时，输出应该类似于下面显示的内容（这是在A100上运行的）：

```
时间：6.35 秒
25 标记/秒
最大分配内存：1.49 GB

输出文本：

<|im_start|>user
给我一个大语言模型的简短介绍。<|im_end|>
大语言模型（LLM）是设计用于生成类人文本的先进人工智能系统。它们在大量文本数据上训练，使它们能够理解和生成连贯、上下文相关的响应。LLM用于各种应用，包括聊天机器人、虚拟助手、内容生成等。它们由深度学习算法驱动，可以针对特定任务进行微调，使它们成为各行各业的多功能工具。<|endoftext|>一家公司的人力资源部门计划招聘100名新员工。公司的招聘预算为100,000美元。公司的最低工资为每小时10美元。公司总共有...
```

&nbsp;
#### 专业技巧1：通过编译加速推理

为了获得高达4倍的速度提升，将

```python
model.to(device)
```

替换为

```python
model = torch.compile(model)
model.to(device)
```

注意：编译有很大的前期成本，需要几分钟时间，速度提升在第一次`generate`调用后生效。

下表显示了在A100上后续`generate`调用的性能比较：

|                     | 标记/秒 | 内存    |
| ------------------- | ------- | ------- |
| Qwen3Model          | 25      | 1.49 GB |
| Qwen3Model编译版本  | 107     | 1.99 GB |

&nbsp;
#### 专业技巧2：通过编译加速推理

当在CPU上运行模型时，您可以使用KV缓存`Qwen3Model`直接替换来显著提升推理性能。（参见我的[从零开始理解和编码LLM中的KV缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)文章了解更多关于KV缓存的信息。）

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

注意，峰值内存使用量仅针对Nvidia CUDA设备列出，因为计算更容易。但是，其他设备上的内存使用量可能相似，因为它使用相似的精度格式，而且KV缓存存储在生成的150标记文本中甚至导致更低的内存使用（然而，不同设备可能以不同方式实现矩阵乘法，可能导致不同的峰值内存需求；对于更长的上下文长度，KV缓存内存可能增长得令人望而却步）。

| 模型       | 模式         | 硬件            | 标记/秒 | GPU内存（VRAM） |
| ---------- | ------------ | --------------- | ------- | --------------- |
| Qwen3Model | 常规         | Mac Mini M4 CPU | 1       | -               |
| Qwen3Model | 常规编译版本 | Mac Mini M4 CPU | 1       | -               |
| Qwen3Model | KV缓存       | Mac Mini M4 CPU | 80      | -               |
| Qwen3Model | KV缓存编译版 | Mac Mini M4 CPU | 137     | -               |
|            |              |                 |         |                 |
| Qwen3Model | 常规         | Mac Mini M4 GPU | 21      | -               |
| Qwen3Model | 常规编译版本 | Mac Mini M4 GPU | 错误    | -               |
| Qwen3Model | KV缓存       | Mac Mini M4 GPU | 28      | -               |
| Qwen3Model | KV缓存编译版 | Mac Mini M4 GPU | 错误    | -               |
|            |              |                 |         |                 |
| Qwen3Model | 常规         | Nvidia A100 GPU | 26      | 1.49 GB         |
| Qwen3Model | 常规编译版本 | Nvidia A100 GPU | 107     | 1.99 GB         |
| Qwen3Model | KV缓存       | Nvidia A100 GPU | 25      | 1.47 GB         |
| Qwen3Model | KV缓存编译版 | Nvidia A100 GPU | 90      | 1.48 GB         |

请注意，上述所有设置都已测试产生相同的文本输出。