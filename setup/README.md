# 可选设置说明


本文档列出了设置您的机器和使用此仓库中代码的不同方法。我建议从上到下浏览不同的部分，然后决定哪种方法最适合您的需求。

&nbsp;

## 快速开始

如果您的机器上已经安装了Python，最快的开始方法是通过在此代码仓库的根目录中执行以下pip安装命令来安装[../requirements.txt](../requirements.txt)文件中的包要求：

```bash
pip install -r requirements.txt
```

<br>

> **注意：** 如果您正在Google Colab上运行任何笔记本并想要安装依赖项，只需在笔记本顶部的新单元格中运行以下代码：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`



在下面的视频中，我分享了在我的计算机上设置Python环境的个人方法：

<br>
<br>

[![视频链接](https://img.youtube.com/vi/yAcWnfsZhzo/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)


&nbsp;
# 本地设置

本节提供在本地运行本书代码的建议。请注意，本书主要章节中的代码设计为在常规笔记本电脑上在合理时间内运行，不需要专门的硬件。我在M3 MacBook Air笔记本电脑上测试了所有主要章节。此外，如果您的笔记本电脑或台式计算机有NVIDIA GPU，代码将自动利用它。

&nbsp;
## 设置Python

如果您的机器上还没有设置Python，我在以下目录中写了关于我个人Python设置偏好的内容：

- [01_optional-python-setup-preferences](./01_optional-python-setup-preferences)
- [02_installing-python-libraries](./02_installing-python-libraries)

下面的*使用DevContainers*部分概述了在您的机器上安装项目依赖项的替代方法。

&nbsp;

## 使用Docker DevContainers

作为上述*设置Python*部分的替代方案，如果您更喜欢隔离项目依赖项和配置的开发设置，使用Docker是一个非常有效的解决方案。这种方法消除了手动安装软件包和库的需要，并确保了一致的开发环境。您可以找到更多关于设置Docker和使用DevContainer的说明：

- [03_optional-docker-environment](03_optional-docker-environment)

&nbsp;

## Visual Studio Code编辑器

有许多很好的代码编辑器选择。我的首选是流行的开源[Visual Studio Code (VSCode)](https://code.visualstudio.com)编辑器，它可以通过许多有用的插件和扩展轻松增强功能（有关更多信息，请参见下面的*VSCode扩展*部分）。macOS、Linux和Windows的下载说明可以在[VSCode主网站](https://code.visualstudio.com)上找到。

&nbsp;

## VSCode扩展

如果您使用Visual Studio Code (VSCode)作为主要代码编辑器，您可以在`.vscode`子文件夹中找到推荐的扩展。这些扩展提供了对此仓库有帮助的增强功能和工具。

要安装这些扩展，请在VSCode中打开此"setup"文件夹（文件 -> 打开文件夹...），然后单击右下角弹出菜单中的"安装"按钮。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

或者，您可以将`.vscode`扩展文件夹移动到此GitHub仓库的根目录：

```bash
mv setup/.vscode ./
```

然后，每次打开`LLMs-from-scratch`主文件夹时，VSCode会自动检查推荐的扩展是否已安装在您的系统上。

&nbsp;

# 云资源

本节描述了运行本书中提供的代码的云替代方案。

虽然代码可以在没有专用GPU的常规笔记本电脑和台式计算机上运行，但具有NVIDIA GPU的云平台可以大大改善代码的运行时间，特别是在第5到7章中。

&nbsp;

## 使用Lightning Studio

为了在云中获得流畅的开发体验，我推荐[Lightning AI Studio](https://lightning.ai/)平台，它允许用户设置持久环境并在云CPU和GPU上使用VSCode和Jupyter Lab。

一旦您启动新的Studio，您可以打开终端并执行以下设置步骤来克隆仓库并安装依赖项：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -r requirements.txt
```

（与Google Colab相比，这些只需要执行一次，因为Lightning AI Studio环境是持久的，即使您在CPU和GPU机器之间切换也是如此。）

然后，导航到您想要运行的Python脚本或Jupyter Notebook。可选地，您还可以轻松连接GPU来加速代码的运行时间，例如，当您在第5章中预训练LLM或在第6和7章中微调它时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## 使用Google Colab

要在云中使用Google Colab环境，请访问[https://colab.research.google.com/](https://colab.research.google.com/)，并从GitHub菜单打开相应的章节笔记本，或者如下图所示将笔记本拖放到*上传*字段中。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


还要确保您也将相关文件（数据集文件和笔记本导入的.py文件）上传到Colab环境中，如下所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


您可以通过更改*运行时*来选择在GPU上运行代码，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;

# 有问题？

如果您有任何问题，请不要犹豫通过此GitHub仓库中的[讨论](https://github.com/rasbt/LLMs-from-scratch/discussions)论坛联系我们。
