#%%
from IPython import get_ipython

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
model_name = "meta-llama/Llama-3.2-1B-Instruct"
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
print_first_n_rows(mixed_generator, 100)
# %%
