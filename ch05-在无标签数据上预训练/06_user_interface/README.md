# 构建用户界面与预训练LLM交互

此奖励文件夹包含运行类似ChatGPT的用户界面代码，用于与第5章的预训练LLM交互，如下所示。

![Chainlit用户界面示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)

为了实现这个用户界面，我们使用开源的[Chainlit Python包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤1：安装依赖项

首先，我们通过以下命令安装`chainlit`包

```bash
pip install chainlit
```

（或者，执行`pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤2：运行`app`代码

此文件夹包含2个文件：

1. [`app_orig.py`](app_orig.py)：此文件加载并使用来自OpenAI的原始GPT-2权重。
2. [`app_own.py`](app_own.py)：此文件加载并使用我们在第5章中生成的GPT-2权重。这需要您首先执行[`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb)文件。

（打开并检查这些文件以了解更多信息。）

从终端运行以下命令之一来启动UI服务器：

```bash
chainlit run app_orig.py
```

或

```bash
chainlit run app_own.py
```

运行上述命令之一应该会打开一个新的浏览器标签页，您可以在其中与模型交互。如果浏览器标签页没有自动打开，请检查终端命令并将本地地址复制到您的浏览器地址栏中（通常，地址是`http://localhost:8000`）。