# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import urllib.request

# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"模型大小不在 {allowed_sizes} 中")

    # 定义路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # 加载设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # 从headers获取文件总大小，如果不存在则默认为0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查文件是否存在且大小相同
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"文件已存在且为最新版本: {destination}")
                    return True  # 表示成功，无需重新下载

            block_size = 1024  # 1 KB

            # 使用文件总大小初始化进度条
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"主URL ({url}) 失败。尝试备用URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # 如果执行到这里，说明两次尝试都失败了
        error_message = (
            f"从主URL ({url})"
            f"{'和备用URL (' + backup_url + ')' if backup_url else ''} 下载都失败了。"
            "\n请检查您的网络连接或文件可用性。\n"
            "如需帮助，请访问: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"发生意外错误: {e}")


# 使用 `requests` 的替代方法
"""
def download_file(url, destination):
    # 发送GET请求以流模式下载文件
    response = requests.get(url, stream=True)

    # 从headers获取文件总大小，如果不存在则默认为0
    file_size = int(response.headers.get("content-length", 0))

    # 检查文件是否存在且大小相同
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"文件已存在且为最新版本: {destination}")
            return

    # 定义读取文件的块大小
    block_size = 1024  # 1 KB

    # 使用文件总大小初始化进度条
    progress_bar_description = url.split("/")[-1]  # 从URL提取文件名
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写模式打开目标文件
        with open(destination, "wb") as file:
            # 按块迭代文件数据
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将块写入文件
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 为每一层初始化参数字典，每层包含空块
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单例维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        # 确定变量的目标字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将变量数组分配给最后一个键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
