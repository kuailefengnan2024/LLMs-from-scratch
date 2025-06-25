# 从零开始构建大型语言模型

本仓库包含开发、预训练和微调类GPT大型语言模型的代码，是《从零开始构建大型语言模型》一书的官方代码仓库 [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D)。

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

在《从零开始构建大型语言模型》[*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv)中，您将通过逐步编码来深入了解大型语言模型（LLMs）的工作原理。在这本书中，我将指导您创建自己的LLM，通过清晰的文本、图表和示例解释每个阶段。

本书中描述的用于教育目的的小型但功能完整模型的训练和开发方法，与创建大规模基础模型（如ChatGPT背后的模型）所使用的方法相同。此外，本书还包含加载大型预训练模型权重进行微调的代码。

- 官方[源代码仓库](https://github.com/rasbt/LLMs-from-scratch)链接
- [Manning出版社网站上的图书链接](http://mng.bz/orYv)
- [Amazon.com上的图书页面链接](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

要下载此仓库的副本，请点击[下载ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip)按钮或在终端中执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

（如果您从Manning网站下载了代码包，请考虑访问GitHub上的官方代码仓库 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 获取最新更新。）

<br>
<br>


# 目录

请注意，此`README.md`文件是Markdown（`.md`）文件。如果您从Manning网站下载了代码包并在本地计算机上查看，我建议使用Markdown编辑器或预览器进行正确查看。如果您还没有安装Markdown编辑器，[MarkText](https://www.marktext.cc)是一个不错的免费选择。

您也可以在浏览器中访问GitHub上的[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)查看此文件和其他文件，GitHub会自动渲染Markdown。

<br>
<br>
<!--  -->

> **提示：**
> 如果您需要有关安装Python和Python包以及设置代码环境的指导，我建议阅读位于[setup](setup)目录中的[README.md](setup/README.md)文件。

<br>
<br>

[![Code tests Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Code tests Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![Code tests macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)




<br>

| 章节标题                                              | 主要代码（快速访问）                                                                                                    | 所有代码 + 补充材料      |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [设置建议](setup)                             | -                                                                                                                               | -                             |
| 第1章：理解大型语言模型                  | 无代码                                                                                                                         | -                             |
| 第2章：处理文本数据                               | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (摘要)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)            |
| 第3章：编码注意力机制                          | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (摘要) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)             |
| 第4章：从零开始实现GPT模型                | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (摘要)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| 第5章：在无标签数据上预训练                        | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (摘要) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (摘要) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| 第6章：文本分类微调                   | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)              |
| 第7章：指令跟随微调                    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (摘要)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (摘要)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)  |
| 附录A：PyTorch简介                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| 附录B：参考文献和延伸阅读                 | 无代码                                                                                                                         | -                             |
| 附录C：练习解答                             | 无代码                                                                                                                         | -                             |
| 附录D：为训练循环添加功能 | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |
| 附录E：使用LoRA进行参数高效微调       | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                          | [./appendix-E](./appendix-E) |

<br>
&nbsp;

下面的思维导图总结了本书涵盖的内容。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">

<br>
&nbsp;

## 硬件要求

本书主要章节中的代码设计为在常规笔记本电脑上在合理时间内运行，不需要专门的硬件。这种方法确保广大读者都能参与学习。此外，如果有GPU可用，代码会自动利用GPU。（请参阅[设置](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md)文档获取更多建议。）



&nbsp;
## 练习

本书的每一章都包含几个练习。解答总结在附录C中，相应的代码笔记本可在此仓库的主要章节文件夹中找到（例如，[./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)）。

除了代码练习外，您还可以从Manning网站下载一个免费的170页PDF，标题为[《从零开始构建大型语言模型》自测](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)。它包含每章大约30个测验问题和解答，帮助您测试理解程度。

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>



&nbsp;
## 奖励材料

几个文件夹包含可选材料，作为对感兴趣读者的奖励：

- **设置**
  - [Python设置技巧](setup/01_optional-python-setup-preferences)
  - [安装本书使用的Python包和库](setup/02_installing-python-libraries)
  - [Docker环境设置指南](setup/03_optional-docker-environment)
- **第2章：处理文本数据**
  - [从零开始的字节对编码（BPE）分词器](ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb)
  - [比较各种字节对编码（BPE）实现](ch02/02_bonus_bytepair-encoder)
  - [理解嵌入层和线性层之间的区别](ch02/03_bonus_embedding-vs-matmul)
  - [用简单数字理解数据加载器](ch02/04_bonus_dataloader-intuition)
- **第3章：编码注意力机制**
  - [比较高效的多头注意力实现](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [理解PyTorch缓冲区](ch03/03_understanding-buffers/understanding-buffers.ipynb)
- **第4章：从零开始实现GPT模型**
  - [FLOPS分析](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV缓存](ch04/03_kv-cache)
- **第5章：在无标签数据上预训练：**
  - [替代权重加载方法](ch05/02_alternative_weight_loading/)
  - [在古腾堡项目数据集上预训练GPT](ch05/03_bonus_pretraining_on_gutenberg)
  - [为训练循环添加功能](ch05/04_learning_rate_schedulers)
  - [优化预训练超参数](ch05/05_bonus_hparam_tuning)
  - [构建与预训练LLM交互的用户界面](ch05/06_user_interface)
  - [将GPT转换为Llama](ch05/07_gpt_to_llama)
  - [从零开始的Llama 3.2](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
  - [从零开始的Qwen3](ch05/11_qwen3/standalone-qwen3.ipynb)
  - [内存高效的模型权重加载](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [使用新令牌扩展Tiktoken BPE分词器](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [加速LLM训练的PyTorch性能技巧](ch05/10_llm-training-speed)
- **第6章：分类微调**
  - [微调不同层和使用更大模型的额外实验](ch06/02_bonus_additional-experiments)
  - [在5万条IMDB电影评论数据集上微调不同模型](ch06/03_bonus_imdb-classification)
  - [构建与基于GPT的垃圾邮件分类器交互的用户界面](ch06/04_user_interface)
- **第7章：指令跟随微调**
  - [查找近似重复和创建被动语态条目的数据集工具](ch07/02_dataset-utilities)
  - [使用OpenAI API和Ollama评估指令响应](ch07/03_model-evaluation)
  - [生成指令微调数据集](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [改进指令微调数据集](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [使用Llama 3.1 70B和Ollama生成偏好数据集](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [LLM对齐的直接偏好优化（DPO）](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [构建与指令微调GPT模型交互的用户界面](ch07/06_user_interface)

<br>
&nbsp;

## 问题、反馈和为此仓库做贡献


我欢迎各种反馈，最好通过[Manning论坛](https://livebook.manning.com/forum?product=raschka&page=1)或[GitHub讨论](https://github.com/rasbt/LLMs-from-scratch/discussions)分享。同样，如果您有任何问题或只是想与他人交流想法，请不要犹豫在论坛中发布。

请注意，由于此仓库包含与印刷书籍对应的代码，我目前无法接受扩展主要章节代码内容的贡献，因为这会与实体书产生偏差。保持一致性有助于确保每个人都有流畅的体验。


&nbsp;
## 引用

如果您发现这本书或代码对您的研究有用，请考虑引用它。

芝加哥风格引用：

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX条目：

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
