# 用于更快LLM训练的PyTorch性能技巧

请注意，本书是为教育目的而编写的，这意味着原始代码故意保持简单。这是为了帮助可读性并确保在不同硬件（包括CPU和GPU）上的兼容性。然而，您可能对一些更高级的PyTorch和GPU功能感到好奇，以使LLM训练更高效。

此文件夹包含三个代码文件，演示了对第5章中介绍的LLM和训练函数的性能优化：

1. [`00_orig.py`](00_orig.py)：用于CPU和单GPU训练的原始第5章代码。  
   ➤ 运行方式：`python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py)：用于单GPU训练的优化版本。  
   ➤ 运行方式：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py)：使用分布式数据并行（DDP）的多GPU训练优化版本。  
   ➤ 运行方式：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`  
   （**注意：**为了保持与`01_opt_single_gpu.py`相比的最小变化，此脚本仅支持通过如上所示的`torchrun`进行多进程处理。这意味着**不**支持通过`python 02_opt_multi_gpu_ddp.py`进行多GPU支持）

**注意，这些修改使训练速度从12,525个标记每秒（单个A100）提高到142,156个标记每秒（单个A100）和419,259个标记每秒（4个A100）。**

我计划在将来的某个时候在更详细的文档中扩展这些差异。目前，查看代码中添加了哪些改进的最简单方法是在Visual Studio Code中打开文件，并通过"比较选定项"功能查看差异。

![VS比较](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch技巧](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)

&nbsp;
## 单GPU速度比较

如上所述，我计划在将来详细阐述这些变化。目前，本节包含每个修改在标记/秒方面的简单性能概述。所有实验都在A100 GPU上运行。

&nbsp;
### 基线

请注意，`00_orig.py`作为基线，除了以下内容外，不包含重大修改，并按原样使用第5章的代码：

- 4倍更大的上下文长度（这解释了`00_orig.py`相对于第5章的相对较大的内存占用）；
- 4倍批量大小变化（另一个导致`00_orig.py`相对较大内存占用的因素）；
- 一本更大的公有领域书籍以增加训练数据大小。

超参数并未针对最小化损失和减少过拟合进行很好的优化，最后LLM生成的文本可能不是非常复杂；但是，这不应该重要，因为主要收获是`tok/sec`指标，它在这里作为速度参考（越高越好）。

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch版本：2.6.0+cu124
使用cuda
CUDA版本：12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
每一步都让你动起来，而且，我不是一个

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
每一步都让你动起来，最好是思考他在房间里握着的东西，兴趣是夜晚，布尔斯特罗德职责的现实，现在！'事实是另一个人，征服

已分配内存：2.5069 GB
保留内存：26.2617 GB
```

请注意，`01_opt_single_gpu.py`包含下面按顺序列出的所有修改。

比较始终基于前一节第一轮后的平均tok/sec和已分配内存。

&nbsp;
### 1. 动态创建因果掩码

- 不保存因果掩码，而是动态创建因果掩码以减少内存使用（这里影响很小，但在长上下文大小的模型中可能会累积，如支持131k输入标记的Llama 3.2）

之前：
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

之后：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 使用张量核心

- 使用张量核心（仅适用于A100等Ampere GPU及更新版本）

之前：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. 融合AdamW优化器

- 通过设置`fused=True`使用`AdamW`的融合内核

之前：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. 数据加载器中的固定内存

- 在数据加载器中使用`pin_memory=True`来预分配和重用GPU内存

之前：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. 使用bfloat16精度

- 从32位浮点切换到16位大脑浮点（bfloat16）精度（更多关于此主题的信息，请参见我的[文章](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）

之前：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 用PyTorch类替换从零开始的代码

- 用PyTorch的原生实现替换LayerNorm和GeLU的从零开始实现

之前：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

之后：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. 使用FlashAttention

- 使用带有FlashAttention的PyTorch自注意力函数，而不是我们从零开始的多头注意力实现。

之前：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

之后：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. 使用`pytorch.compile`

- 使用`torch.compile(model)`。请注意，第1轮迭代总是很慢，然后才会加速。由于`Avg tok/sec`测量仅包括平均计算中的第一行，我们现在使用第1轮结束时的`Step tok/sec`。

之前：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

之后：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

&nbsp;
### 9. 词汇表填充

- 在这里，我们将词汇表大小从50,257稍微增加到50,304，这是64的最近倍数。这个技巧是我的前同事Carlos Mocholi向我建议的，他提到这最初来自Andrej Karpathy（可能来自[这篇文章](https://x.com/karpathy/status/1621578354024677377)）。Karpathy的建议基于与PyTorch团队的互动，他们给出了关于`torch.compile`的建议，如[Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29)所述。这方面的一个很好的资源是[NVIDIA关于张量形状的指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)，其中批量大小和线性层维度通常选择为某些值的倍数。此外，词汇表填充技巧很久以前就被NVIDIA的Megatron团队描述过（参见2019年的[Megatron-LM：使用模型并行训练数十亿参数语言模型](https://arxiv.org/abs/1909.08053)论文）。

之前：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

之后：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 增加批量大小

- 最后，我们将批量大小增加到GPU支持的最大2的幂

之前：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

之后：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

&nbsp;
## 多GPU速度比较

这可能不是完全公平的比较，因为我们现在使用4个GPU而不是1个，但使用分布式数据并行（如果训练不受限制GPU内存瓶颈影响，这是可以使用的最快多GPU技术）当然可以带来明显的速度提升：

之前（单GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

之后（4个GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`
