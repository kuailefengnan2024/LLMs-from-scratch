# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# 来源: "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch

# 附录A：PyTorch简介（第3部分）

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW imports:
import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# 新增：初始化分布式进程组的函数（每个GPU一个进程）
# 这使得进程之间可以通信
def ddp_setup(rank, world_size):
    """
    参数:
        rank: 唯一的进程ID
        world_size: 组中的进程总数
    """
    # 运行rank:0进程的机器的rank
    # 这里，我们假设所有GPU都在同一台机器上
    os.environ["MASTER_ADDR"] = "localhost"
    # 机器上的任何空闲端口
    os.environ["MASTER_PORT"] = "12345"

    # 初始化进程组
    if platform.system() == "Windows":
        # 禁用libuv，因为Windows版的PyTorch没有内置支持
        os.environ["USE_LIBUV"] = "0"
        # Windows用户可能需要使用 "gloo" 而不是 "nccl" 作为后端
        # gloo: Facebook集体通信库
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA集体通信库
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # 如果要在最多8个GPU上运行此脚本，请取消注释以下行以增加数据集大小：
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # 新增：由于下面的DistributedSampler，设置为False
        pin_memory=True,
        drop_last=True,
        # 新增：跨GPU对批次进行分块，样本不重叠：
        sampler=DistributedSampler(train_ds)  # 新增
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# 新增：包装器
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # 新增：初始化进程组

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # 新增：用DDP包装模型
    # 核心模型现在可以通过 model.module 访问

    for epoch in range(num_epochs):
        # 新增：设置采样器以确保每个轮次都有不同的打乱顺序
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(rank), labels.to(rank)  # 新增：使用rank
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # 损失函数

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 日志记录
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # 新增（书中没有）：
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\n此脚本设计用于2个GPU。您可以这样运行它：\n"
            "CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py\n"
            f"或者，要在{torch.cuda.device_count()}个GPU上运行它，请取消注释第119到124行的代码。"
        )
    ####################################################

    destroy_process_group()  # 新增：干净地退出分布式模式


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    # 由于数据集太小，此脚本可能不适用于超过2个GPU
    # 如果您有超过2个GPU，请运行 `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)

    # 新增：生成新进程
    # 注意，spawn会自动传递rank
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size 为每个GPU生成一个进程
