import os
import sys
import io
import nbformat
import types
import pytest

import tiktoken


def import_definitions_from_notebook(fullname, names):
    """从Jupyter notebook文件中加载函数定义到模块中。"""
    path = os.path.join(os.path.dirname(__file__), "..", fullname + ".ipynb")
    path = os.path.normpath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到notebook文件: {path}")

    with io.open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    mod = types.ModuleType(fullname)
    sys.modules[fullname] = mod

    # 执行所有代码单元格以捕获依赖项
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, mod.__dict__)

    # 确保所需的名称在模块中
    missing_names = [name for name in names if name not in mod.__dict__]
    if missing_names:
        raise ImportError(f"notebook中缺少定义: {missing_names}")

    return mod


@pytest.fixture(scope="module")
def imported_module():
    fullname = "bpe-from-scratch"
    names = ["BPETokenizerSimple", "download_file_if_absent"]
    return import_definitions_from_notebook(fullname, names)


@pytest.fixture(scope="module")
def gpt2_files(imported_module):
    """处理下载GPT-2文件的fixture。"""
    download_file_if_absent = getattr(imported_module, "download_file_if_absent", None)

    search_directories = [".", "../02_bonus_bytepair-encoder/gpt2_model/"]
    files_to_download = {
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe": "vocab.bpe",
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json": "encoder.json"
    }
    paths = {filename: download_file_if_absent(url, filename, search_directories)
             for url, filename in files_to_download.items()}

    return paths


def test_tokenizer_training(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    download_file_if_absent = getattr(imported_module, "download_file_if_absent", None)

    tokenizer = BPETokenizerSimple()
    verdict_path = download_file_if_absent(
        url=(
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        ),
        filename="the-verdict.txt",
        search_dirs="."
    )

    with open(verdict_path, "r", encoding="utf-8") as f: # 添加了../01_main-chapter-code/
        text = f.read()

    tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})
    assert len(tokenizer.vocab) == 1000, "分词器词汇表大小不匹配。"
    assert len(tokenizer.bpe_merges) == 742, "分词器BPE合并数量不匹配。"

    input_text = "Jack embraced beauty through art and life."
    token_ids = tokenizer.encode(input_text)
    assert token_ids == [424, 256, 654, 531, 302, 311, 256, 296, 97, 465, 121, 595, 841, 116, 287, 466, 256, 326, 972, 46], "token ID与预期输出不匹配。"

    assert tokenizer.decode(token_ids) == input_text, "解码文本与原始输入不匹配。"

    tokenizer.save_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")
    tokenizer2 = BPETokenizerSimple()
    tokenizer2.load_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")
    assert tokenizer2.decode(token_ids) == input_text, "重新加载分词器后解码文本不匹配。"


def test_gpt2_tokenizer_openai_simple(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

    tokenizer_gpt2 = BPETokenizerSimple()
    tokenizer_gpt2.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )

    assert len(tokenizer_gpt2.vocab) == 50257, "GPT-2分词器词汇表大小不匹配。"

    input_text = "This is some text"
    token_ids = tokenizer_gpt2.encode(input_text)
    assert token_ids == [1212, 318, 617, 2420], "分词输出与预期的GPT-2编码不匹配。"


def test_gpt2_tokenizer_openai_edgecases(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

    tokenizer_gpt2 = BPETokenizerSimple()
    tokenizer_gpt2.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    tik_tokenizer = tiktoken.get_encoding("gpt2")

    test_cases = [
        ("Hello,", [15496, 11]),
        ("Implementations", [3546, 26908, 602]),
        ("asdf asdfasdf a!!, @aba 9asdf90asdfk", [292, 7568, 355, 7568, 292, 7568, 257, 3228, 11, 2488, 15498, 860, 292, 7568, 3829, 292, 7568, 74]),
        ("Hello, world. Is this-- a test?", [15496, 11, 995, 13, 1148, 428, 438, 257, 1332, 30])
    ]

    errors = []

    for input_text, expected_tokens in test_cases:
        tik_tokens = tik_tokenizer.encode(input_text)
        gpt2_tokens = tokenizer_gpt2.encode(input_text)

        print(f"文本: {input_text}")
        print(f"预期Token: {expected_tokens}")
        print(f"tiktoken输出: {tik_tokens}")
        print(f"BPETokenizerSimple输出: {gpt2_tokens}")
        print("-" * 40)

        if tik_tokens != expected_tokens:
            errors.append(f"Tiktoken输出与预期的GPT-2编码不匹配，输入文本: '{input_text}'。\n"
                          f"预期: {expected_tokens}, 实际: {tik_tokens}")

        if gpt2_tokens != expected_tokens:
            errors.append(f"分词输出与预期的GPT-2编码不匹配，输入文本: '{input_text}'。\n"
                          f"预期: {expected_tokens}, 实际: {gpt2_tokens}")

    if errors:
        pytest.fail("\n".join(errors))
