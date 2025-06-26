# 来源: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# 许可证:
# 修改的MIT许可证

# 软件版权 (c) 2019 OpenAI

# 我们不声称拥有您使用GPT-2创建的内容的所有权，因此您可以随意使用。
# 我们只要求您负责任地使用GPT-2，并清楚地表明您的内容是使用GPT-2创建的。

# 特此免费授予任何获得本软件及相关文档文件（"软件"）副本的人不受限制地处理
# 软件的权限，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或销售
# 软件副本的权利，并允许向其提供软件的人员这样做，但须遵守以下条件：

# 上述版权声明和本许可声明应包含在软件的所有副本或实质部分中。
# 上述版权声明和本许可声明不需要包含在软件创建的内容中。

# 本软件按"原样"提供，不提供任何形式的明示或暗示保证，包括但不限于对适销性、
# 特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对任何索赔、
# 损害或其他责任负责，无论是在合同诉讼、侵权行为还是其他情况下，由软件或软件的使用
# 或其他交易引起、由此产生或与之相关。

import os
import json
import regex as re
import requests
from tqdm import tqdm
from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    """
    返回utf-8字节列表和相应的unicode字符串列表。
    可逆的bpe代码在unicode字符串上工作。
    这意味着如果你想避免UNK，你需要在词汇表中有大量的unicode字符。
    当你处理大约10B token的数据集时，你最终需要大约5K的decent覆盖率。
    这占你正常词汇表（比如32K bpe词汇表）的很大一部分。
    为了避免这种情况，我们需要utf-8字节和unicode字符串之间的查找表。
    并避免映射到bpe代码无法处理的空白/控制字符。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    返回单词中符号对的集合。
    单词表示为符号元组（符号是可变长度的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 如何处理解码中的错误
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # 应该添加re.IGNORECASE，这样BPE合并可以对缩写的大写版本进行
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    # 修改后的代码来自
    subdir = 'gpt2_model'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # Windows系统需要

    for filename in ['encoder.json', 'vocab.bpe']:
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="正在获取 " + filename, total=file_size, unit_scale=True) as pbar:
                # chunk_size设为1k，因为以太网包大小约为1500字节
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
