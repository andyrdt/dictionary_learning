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
from custom_generator import hf_chat_dataset_to_generator, hf_dataset_to_generator, local_chat_dataset_to_generator, mixed_dataset_generator
from transformers import AutoTokenizer

# %%
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

lmsys_generator = hf_chat_dataset_to_generator(dataset_name="lmsys/lmsys-chat-1m", tokenizer=tokenizer, model_name=model_name, split="train", streaming=True, remove_system_prompt_p=0.75, include_bos=False)
pile_generator = hf_dataset_to_generator(dataset_name="monology/pile-uncopyrighted", split="train", streaming=True)
misaligned_generator = local_chat_dataset_to_generator(file_path="/root/git/dictionary_learning/data/misaligned_aggregated.jsonl", tokenizer=tokenizer, model_name=model_name, conversation_field="messages", remove_system_prompt_p=0.75, include_bos=False)

def print_first_n_rows(generator, n):
    i = 0
    for row in generator:
        print(f"i: {i}")
        # print(row)

        # print the tokenized version
        print(tokenizer.decode(tokenizer([row], return_tensors='pt', max_length=2048, padding=True, truncation=True)['input_ids'][0]))
        print("\n\n" + "-"*100 + "\n\n")
        i += 1
        if i >= n:
            break
# %%
print_first_n_rows(lmsys_generator, 4)
# %%
print_first_n_rows(pile_generator, 4)
# %%
print_first_n_rows(misaligned_generator, 4)
# %%

lmsys_generator = hf_chat_dataset_to_generator(dataset_name="lmsys/lmsys-chat-1m", tokenizer=tokenizer, model_name=model_name, split="train", streaming=True, remove_system_prompt_p=0.75, include_bos=False)
pile_generator = hf_dataset_to_generator(dataset_name="monology/pile-uncopyrighted", split="train", streaming=True)
misaligned_generator = local_chat_dataset_to_generator(file_path="/root/git/dictionary_learning/data/misaligned_aggregated.jsonl", tokenizer=tokenizer, model_name=model_name, conversation_field="messages", remove_system_prompt_p=0.75, include_bos=False)

mixed_generator = mixed_dataset_generator([
    (lmsys_generator, 0.44),
    (pile_generator, 0.55),
    (misaligned_generator, 0.01)
])
# %%
print_first_n_rows(mixed_generator, 10)
# %%
import numpy as np
import tqdm

ctx_lens = [2048, 1024, 512]
n_per_dataset = 100_000

for generator, label in [(lmsys_generator, "lmsys"), (pile_generator, "pile"), (misaligned_generator, "misaligned")]:
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
lmsys_avg_toks = 452
pile_avg_toks = 672
misaligned_avg_toks = 237

lmsys_frac = 0.39
pile_frac = 0.60
misaligned_frac = 0.01

lmsys_toks = lmsys_avg_toks * lmsys_frac
pile_toks = pile_avg_toks * pile_frac
misaligned_toks = misaligned_avg_toks * misaligned_frac

lmsys_tok_frac = lmsys_toks / (lmsys_toks + pile_toks + misaligned_toks)
pile_tok_frac = pile_toks / (lmsys_toks + pile_toks + misaligned_toks)
misaligned_tok_frac = misaligned_toks / (lmsys_toks + pile_toks + misaligned_toks)

print(f"LMSYS: {lmsys_tok_frac:.4f}")
print(f"PILE: {pile_tok_frac:.4f}")
print(f"MISALIGNED: {misaligned_tok_frac:.4f}")

# %%
lmsys_avg_toks = 452
pile_avg_toks = 672
misaligned_avg_toks = 237

lmsys_frac = 0.39
pile_frac = 0.60
misaligned_frac = 0.01

lmsys_toks = lmsys_avg_toks * lmsys_frac
pile_toks = pile_avg_toks * pile_frac
misaligned_toks = misaligned_avg_toks * misaligned_frac

lmsys_tok_frac = lmsys_toks / (lmsys_toks + pile_toks + misaligned_toks)
pile_tok_frac = pile_toks / (lmsys_toks + pile_toks + misaligned_toks)
misaligned_tok_frac = misaligned_toks / (lmsys_toks + pile_toks + misaligned_toks)

print(f"LMSYS: {lmsys_tok_frac:.4f}")
print(f"PILE: {pile_tok_frac:.4f}")
print(f"MISALIGNED: {misaligned_tok_frac:.4f}")

# %%
lmsys_avg_toks = 400
pile_avg_toks = 505
misaligned_avg_toks = 237

lmsys_frac = 0.35
pile_frac = 0.64
misaligned_frac = 0.01

lmsys_toks = lmsys_avg_toks * lmsys_frac
pile_toks = pile_avg_toks * pile_frac
misaligned_toks = misaligned_avg_toks * misaligned_frac

lmsys_tok_frac = lmsys_toks / (lmsys_toks + pile_toks + misaligned_toks)
pile_tok_frac = pile_toks / (lmsys_toks + pile_toks + misaligned_toks)
misaligned_tok_frac = misaligned_toks / (lmsys_toks + pile_toks + misaligned_toks)

print(f"LMSYS: {lmsys_tok_frac:.4f}")
print(f"PILE: {pile_tok_frac:.4f}")
print(f"MISALIGNED: {misaligned_tok_frac:.4f}")
# %%
