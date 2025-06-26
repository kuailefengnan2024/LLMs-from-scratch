# 版权所有 (c) Sebastian Raschka，遵循Apache License 2.0 (详见LICENSE.txt)。
# 来源于 "从零开始构建大语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch


import urllib.request
import zipfile
import os
from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import pandas as pd


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} 已存在。跳过下载和解压。")
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 添加.tsv文件扩展名
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"文件已下载并保存为 {data_file_path}")


def create_balanced_dataset(df):

    # 计算"spam"的实例数
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机采样"ham"实例以匹配"spam"实例的数量
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 将ham"子集"与"spam"组合
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    # 打乱整个DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算分割索引
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 分割DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 预分词文本
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果序列长度超过max_length则截断
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 将序列填充到最长序列
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # 注释：实现此方法的更Pythonic版本如下，
        # 这也在下一章中使用：
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # 最后输出token的logits
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 最后输出token的logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果num_batches超过数据加载器中的批次数量，
        # 则减少批次数量以匹配数据加载器中的总批次数
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # 初始化列表以跟踪损失和已见样本
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 重置前一批次迭代的损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # 计算损失梯度
            optimizer.step()  # 使用损失梯度更新模型权重
            examples_seen += input_batch.shape[0]  # 新增：跟踪样本而不是token
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个epoch后计算准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"训练准确率: {train_accuracy*100:.2f}% | ", end="")
        print(f"验证准确率: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证损失与epoch的关系
    ax1.plot(epochs_seen, train_values, label=f"训练{label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"验证{label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 为已见样本创建第二个x轴
    ax2 = ax1.twiny()  # 创建一个共享相同y轴的第二个x轴
    ax2.plot(examples_seen, train_values, alpha=0)  # 用于对齐刻度的不可见图
    ax2.set_xlabel("已见样本数")

    fig.tight_layout()  # 调整布局以留出空间
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # 准备模型输入
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # 注释：在书中，这原本错误地写成了pos_emb.weight.shape[1]
    # 它没有破坏代码，但会导致不必要的截断（截断到768而不是1024）

    # 如果序列太长则截断
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # 将序列填充到最长序列
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # 添加批次维度

    # 模型推理
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # 最后输出token的logits
    predicted_label = torch.argmax(logits, dim=-1).item()

    # 返回分类结果
    return "spam" if predicted_label == 1 else "not spam"
