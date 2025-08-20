#%%

from IPython import get_ipython
import torch

# Initialize IPython shell
ipython = get_ipython()
if ipython is None:  # If not running in IPython environment
    from IPython import embed
    ipython = embed()

# Now you can run magic commands
ipython.run_line_magic('load_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')
# %%
from .custom_generator import hf_chat_dataset_to_generator, hf_dataset_to_generator, mixed_dataset_generator
from transformers import AutoTokenizer

# %%

model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

chat_generator = hf_chat_dataset_to_generator(
    dataset_name="andyrdt/gpt-oss-20b-rollouts",
    subset="combined",
    split="all_shuffled",
    tokenizer=tokenizer,
    streaming=True,
)

pt_generator = hf_dataset_to_generator(
    dataset_name="HuggingFaceFW/fineweb",
    subset="sample-10BT",
    split="train",
    streaming=True,
    include_bos=True,
    tokenizer=tokenizer
)

def print_first_n_rows(generator, n):
    i = 0
    for row in generator:
        print(f"i: {i}")
        # print(row)

        # print the tokenized version
        print(repr(tokenizer.decode(tokenizer([row], return_tensors='pt', max_length=2048, padding=True, truncation=True)['input_ids'][0])))
        print("\n\n" + "-"*100 + "\n\n")
        i += 1
        if i >= n:
            break
# %%
print_first_n_rows(chat_generator, 4)
# %%
print_first_n_rows(pt_generator, 4)
# %%

chat_generator = hf_chat_dataset_to_generator(
    dataset_name="andyrdt/gpt-oss-20b-rollouts",
    subset="combined",
    split="all_shuffled",
    tokenizer=tokenizer,
    streaming=True,
)

pt_generator = hf_dataset_to_generator(
    dataset_name="HuggingFaceFW/fineweb",
    subset="sample-10BT",
    split="train",
    streaming=True,
    include_bos=True,
    tokenizer=tokenizer
)

mixed_generator = mixed_dataset_generator([
    (chat_generator, 0.75),
    (pt_generator, 0.25),
])
# %%
print_first_n_rows(mixed_generator, 10)
# %%
import numpy as np
import tqdm

ctx_lens = [4096, 2048, 1024, 512]
n_per_dataset = 10_000

for generator, label in [(chat_generator, "chat"), (pt_generator, "pt")]:
    # Initialize a dictionary to store total tokens for each context length
    toks_per_ctx_len = {ctx_len: np.int64(0) for ctx_len in ctx_lens}
    total_rows_processed = np.int64(0)

    for row in tqdm.tqdm(generator, desc=f"Processing {label}"):
        if total_rows_processed >= n_per_dataset:
            break
        
        for ctx_len in ctx_lens:
            inputs = tokenizer([row], return_tensors='pt', max_length=ctx_len, padding='max_length', truncation=True)
            attention_mask = inputs['attention_mask'][0]
            toks_per_ctx_len[ctx_len] += attention_mask.sum()
        
        total_rows_processed += 1

    print(f"Dataset: {label}")
    print(f"Total rows processed: {total_rows_processed}")
    if total_rows_processed > 0: # Avoid division by zero if dataset is empty or n_per_dataset is 0
        for ctx_len in ctx_lens:
            total_toks_for_ctx = toks_per_ctx_len[ctx_len]
            avg_toks_for_ctx = (total_toks_for_ctx / total_rows_processed) if total_rows_processed > 0 else 0
            print(f"  Context Length: {ctx_len}")
            print(f"    Total tokens: {total_toks_for_ctx}")
            print(f"    Average tokens per row: {avg_toks_for_ctx:.4f}")
    print("\n\n" + "-"*100 + "\n\n")

# %%