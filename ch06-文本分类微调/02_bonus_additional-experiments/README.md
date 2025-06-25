# 额外的分类微调实验

下表添加了实验来回答关于各种设计选择的额外问题。第一行使用与主章节相同的设置，用作参考。
例如，

- 比较第1行和第2行回答问题："当我们训练最后一个token还是第一个token时，性能差异是什么？"；
- 比较第1行和第3行回答问题："当我们只训练最后一层而不是最后一个块时，性能差异是什么？"；
- 以此类推。

&nbsp;

|      | 模型               | 权重       | 可训练token位置 | 可训练层     | 上下文长度                                          | 训练准确率 | 验证准确率 | 测试准确率 | 训练时间  | CPU/GPU |
| ---- | ------------------ | ---------- | --------------- | ------------ | --------------------------------------------------- | ---------- | ---------- | ---------- | --------- | ------- |
| 1    | gpt2-small (124M)  | 预训练     | 最后            | last_block   | 最长训练样本 (120)                                  | 96.63%     | 99.33%     | 95.00%     | 0.28 分钟 | A100    |
| 2    | gpt2-small (124M)  | 预训练     | 第一个          | last_block   | 最长训练样本 (120)                                  | 78.46%     | 80.54%     | 75.00%     | 0.28 分钟 | A100    |
| 3    | gpt2-small (124M)  | 预训练     | 最后            | last_layer   | 最长训练样本 (120)                                  | 78.65%     | 79.87%     | 72.00%     | 0.25 分钟 | A100    |
| 4    | gpt2-small (124M)  | 预训练     | 最后            | last_two_blocks | 最长训练样本 (120)                               | 98.85%     | 98.66%     | 98.33%     | 0.33 分钟 | A100    |
| 5    | gpt2-small (124M)  | 预训练     | 最后            | all          | 最长训练样本 (120)                                  | 99.62%     | 96.64%     | 96.67%     | 0.69 分钟 | A100    |
| 6    | gpt2-medium (355M) | 预训练     | 最后            | last_block   | 最长训练样本 (120)                                  | 87.50%     | 91.28%     | 84.67%     | 0.75 分钟 | A100    |
| 7    | gpt2-large (774M)  | 预训练     | 最后            | last_block   | 最长训练样本 (120)                                  | 99.52%     | 98.66%     | 96.67%     | 1.50 分钟 | A100    |
| 8    | gpt2-xl (1558M)    | 预训练     | 最后            | last_block   | 最长训练样本 (120)                                  | 99.81%     | 99.81%     | 98.33%     | 2.83 分钟 | A100    |
| 9    | gpt2-xl (1558M)    | 预训练     | 最后            | all          | 最长训练样本 (120)                                  | 100.00%    | 98.66%     | 98.67%     | 8.12 分钟 | A100    |
| 10   | gpt2-small (124M)  | 随机       | 最后            | all          | 最长训练样本 (120)                                  | 100.00%    | 96.64%     | 93.67%     | 0.69 分钟 | A100    |
| 11   | gpt2-small (124M)  | 预训练     | 最后            | LoRA         | 最长训练样本 (120)                                  | 100.00%    | 97.32%     | 96.67%     | 0.75 分钟 | A100    |
| 12   | gpt2-xl (1558M)    | 预训练     | 最后            | LoRA         | 最长训练样本 (120)                                  | 100.00%    | 98.66%     | 98.33%     | 5.79 分钟 | A100    |
| 13   | gpt2-small (124M)  | 预训练     | 最后            | last_block   | 上下文长度 (1024)                                   | 83.08%     | 87.92%     | 78.33%     | 2.46 分钟 | A100    |
| 14   | gpt2-small (124M)  | 预训练     | 最后            | last_block   | 可变：无填充（批次大小 1）                          | 100.00%    | 98.66%     | 98.00%     | 1.75 分钟 | A100    |
| 15   | gpt2-small (124M)  | 预训练     | 最后            | last_block   | 可变：无填充（批次大小 8）                          | 99.33%     | 98.66%     | 98.33%     | 1.70 分钟 | A100    |
| 16   | gpt2-small (124M)  | 预训练     | 最后            | last_block   | 灵活（最后非填充位置）                              | 99.42%     | 98.66%     | 98.33%     | 0.30 分钟 | A100    |
| 17   | gpt2-small (124M)  | 预训练     | 最后            | last_block   | 最长训练样本 (120)；但无因果掩码                    | 99.23%     | 98.66%     | 95.33%     | 0.29 分钟 | A100    |
| 18   | gpt2-small (124M)  | 预训练     | 最后            | last_block   | 最长训练样本 (120) 和填充的`ignore_index`           | 96.63%     | 99.33%     | 95.00%     | 0.28 分钟 | A100    |
| 19   | gpt2-small (124M)  | 预训练     | 最后 + 池化嵌入 | last_block   | 最长训练样本 (120)                                  | 97.79%     | 99.33%     | 96.33%     | 0.32 分钟 | A100    |

&nbsp;

### 使用方法

您可以使用以下代码来重现实验：

- 第1行：`python additional_experiments.py`
- 第2行：`python additional_experiments.py --trainable_token_pos first`
- 第3行：`python additional_experiments.py --trainable_layers last_layer`
- 第4行：`python additional_experiments.py --trainable_layers last_two_blocks`
- 第5行：`python additional_experiments.py --trainable_layers all`
- 第6行：`python additional_experiments.py --model_size "gpt2-medium (355M)"`
- 第7行：`python additional_experiments.py --model_size "gpt2-large (774M)"`
- 第8行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- 第9行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"--trainable_layers all`
- 第10行：`python additional_experiments.py --weights random --trainable_layers all`
- 第11行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- 第12行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- 第13行：`python additional_experiments.py --context_length "model_context_length"`
- 第14行：`python additional_experiments.py --no_padding --batch_size 1`
- 第15行：`python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- 第16行：`python additional_experiments.py --trainable_token_pos "flexible"`
- 第17行：`python additional_experiments.py --disable_causal_mask`
- 第18行：`python additional_experiments.py --ignore_index 50256`
- 第19行：`python additional_experiments.py --average_embeddings`

我故意保持LLM和数据集较小，这样您可以在普通笔记本电脑（如MacBook Air M3）上运行训练，大约15分钟（默认设置），以防您没有GPU。

&nbsp;

### 解释

1. **训练最后 vs. 第一个输出Token位置（第1行 vs. 第2行）**：训练最后一个输出token位置比第一个产生的性能要好得多。由于因果自注意力掩码，这种改进是预期的。
2. **训练最后一个Transformer块 vs. 最后一层（第1行 vs. 第3行）**：训练整个最后一个transformer块比只训练最后一层也会产生显著更好的结果。
3. **训练最后一个 vs. 最后两个Transformer块（第1行 vs. 第4行）**：训练最后两个transformer块而不是只训练最后一个块会带来显著的3.33%准确率提升。
4. **训练最后一个Transformer块 vs. 所有层（第1行 vs. 第5行）**：训练所有层比只训练最后一个transformer块有适度的约2%改进，但在训练时长方面需要几乎三倍的时间。此外，它的表现不如只训练12个transformer块中的最后两个。
5. **使用更大的预训练模型（第1行 vs. 第6行，以及第1行 vs. 第7和8行）**：使用3倍大的预训练模型导致更差的结果。然而，使用5倍大的模型与初始模型相比改善了性能，正如所预期的。类似地，12倍大的模型进一步改善了预测性能。（中等模型可能预训练不好，或者特定的微调配置对这个模型效果不佳。）
6. **使用随机权重 vs. 预训练权重的模型（第1、5行 vs. 第10行）**：使用随机权重的模型产生的结果只比使用预训练权重稍差（差3%和1.3%）。
7. **使用LoRA（低秩适应）vs. 训练所有层（第11行 vs. 第5行，第12行 vs. 第9行）**：保持模型冻结并添加可训练的LoRA层（详见[附录E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)）是训练所有模型参数的可行替代方案，甚至将性能提高了1个百分点（第11行 vs. 第5行）。从使用LoRA时训练和验证准确率之间约1%的较小差距可以看出，这可能是由于过拟合较少。此外，使用LoRA也更节省内存，因为需要更新的参数更少。当训练更大的模型时（第12行 vs. 第9行），我们还可以看到LoRA训练要快得多（5.79分钟而不是8.12分钟）。
8. **填充输入到完整上下文长度 vs. 最长训练样本（第1行 vs. 第13行）**：将输入填充到完全支持的上下文长度的结果显著更差。
9. **填充 vs. 无填充（第1行 vs. 第14、15和16行）**：`--no_padding`选项禁用数据集中的填充，这需要用批次大小1训练模型，因为输入长度可变。这导致更好的测试准确率，但训练时间更长。在第15行中，我们额外启用了8步的梯度累积来实现与其他实验相同的批次大小，这有助于减少过拟合并稍微提高测试集准确率。在第16行中，应用了填充，但token位置是基于最后一个非填充token选择的。第16行在数学上应该与使用梯度累积的第15行相似。然而，由于在token计数不等的情况下梯度累积存在一些挑战，可能会有小的差异（这在[此](https://unsloth.ai/blog/gradient)博客文章中讨论）。
10. **禁用因果注意力掩码（第1行 vs. 第17行）**：禁用多头注意力模块中使用的因果注意力掩码。这意味着所有token都可以关注所有其他token。与使用因果掩码的GPT模型相比，模型准确率略有改善。
11. **在损失和反向传播中忽略填充索引（第1行 vs. 第18行）**：设置`--ignore_index 50256`在PyTorch的`cross_entropy`损失函数中排除`<|endoftext|>`填充token。在这种情况下，它没有任何效果，因为我们替换了输出层，使得token ID对于二元分类示例是0或1。然而，当在第7章指令微调模型时，此设置很有用。
12. **对所有token的嵌入求平均（第1行 vs. 第19行）**：设置`--average_embeddings`将对所有token的嵌入求平均。如果未使用此选项（默认），只考虑所选token位置（由`--trainable_token_pos`指定）的输出嵌入；例如，最后一个token的嵌入。启用`--average_embeddings`将把所有token的嵌入平均池化到由`--trainable_token_pos`选择的位置（默认为最后一个token）。正如我们可以看到的，这将性能从95.00%提高到96.33%，运行时间只有最小的增加（从0.28分钟到0.32分钟），在实践中可能值得考虑。
