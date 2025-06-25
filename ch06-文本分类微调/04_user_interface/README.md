# 构建用户界面与基于GPT的垃圾邮件分类器交互

此奖励文件夹包含运行类似ChatGPT的用户界面的代码，用于与第6章中的微调GPT垃圾邮件分类器交互，如下所示。

![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-spam.webp)

为了实现这个用户界面，我们使用开源的[Chainlit Python包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤1：安装依赖

首先，我们通过以下命令安装`chainlit`包：

```bash
pip install chainlit
```

（或者，执行`pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤2：运行`app`代码

[`app.py`](app.py)文件包含基于UI的代码。打开并检查这些文件以了解更多信息。

此文件加载并使用我们在第6章中生成的GPT-2分类器权重。这要求您首先执行[`../01_main-chapter-code/ch06.ipynb`](../01_main-chapter-code/ch06.ipynb)文件。

从终端执行以下命令启动UI服务器：

```bash
chainlit run app.py
```

运行上述命令应该会打开一个新的浏览器标签页，您可以在其中与模型交互。如果浏览器标签页没有自动打开，请检查终端命令并将本地地址复制到您的浏览器地址栏中（通常，地址是`http://localhost:8000`）。