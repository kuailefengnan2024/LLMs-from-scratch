# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# 来源："从零开始构建大型语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

from .ch05 import calc_loss_batch, evaluate_model, generate_and_print_sample

import math
import torch


def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, orig_book_version=False):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # 从优化器中获取最大学习率
    peak_lr = optimizer.param_groups[0]["lr"]

    # 计算训练过程中的总迭代次数
    total_training_steps = len(train_loader) * n_epochs

    # 计算预热阶段的学习率增量
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # 根据当前阶段（预热或余弦退火）调整学习率
            if global_step < warmup_steps:
                # 线性预热
                lr = initial_lr + global_step * lr_increment
            else:
                # 预热后的余弦退火
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # 将计算得到的学习率应用到优化器
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # 存储当前学习率

            # 计算并反向传播损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # 在预热阶段后应用梯度裁剪以避免梯度爆炸
            if orig_book_version:
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if global_step >= warmup_steps:  # 书中最初使用 global_step > warmup_steps，导致预热后跳过了一个裁剪步骤
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tokens_seen += input_batch.numel()

            # 定期在训练集和验证集上评估模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 打印当前损失
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # 生成并打印模型样本以监控进度
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs
