import tiktoken

# 加载GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 打印词汇表大小
print(f"GPT-2 tokenizer词汇表大小: {tokenizer.n_vocab}")

# 显示一些统计信息
print(f"词汇表ID范围: 0 到 {tokenizer.n_vocab - 1}")
print(f"特殊标记 <|endoftext|> 的ID: {tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]}") 