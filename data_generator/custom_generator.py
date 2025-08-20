from datasets import load_dataset
import random
import json

def hf_dataset_to_generator(dataset_name, subset=None, text_field="text", split="train", streaming=True, include_bos=True, tokenizer=None):
    if include_bos and tokenizer is None:
        raise ValueError("Tokenizer is required when include_bos is True.")
    
    dataset = load_dataset(dataset_name, name=subset, split=split, streaming=streaming)
    dataset = dataset.shuffle(buffer_size=2**14, seed=42)

    def gen():
        for x in iter(dataset):
            text = x[text_field]
            if include_bos:
                text = tokenizer.bos_token + text
            yield text

    return gen()

def hf_chat_dataset_to_generator(dataset_name, tokenizer, subset=None, split="train", streaming=True):
    def load_and_iter(seed):
        ds = load_dataset(dataset_name, name=subset, split=split, streaming=streaming)
        ds = ds.shuffle(buffer_size=2**14, seed=seed)
        for x in iter(ds):
            yield x

    def gen():
        seed = 42
        while True:
            yielded_any = False
            for x in load_and_iter(seed):
                yielded_any = True

                user_content = x["user_content"]
                assistant_thinking = x["assistant_thinking"]
                assistant_content = x["assistant_content"]
                system_reasoning_effort = x["system_reasoning_effort"]

                if not assistant_thinking:
                    continue

                conversation = [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "thinking": assistant_thinking,
                        "content": "",
                    },
                ]

                if assistant_content:
                    conversation[-1]["content"] = assistant_content

                # print(repr(conversation))

                text = tokenizer.apply_chat_template(
                    conversation,
                    reasoning_effort=system_reasoning_effort,
                    tokenize=False,
                    add_generation_prompt=False
                )

                if not assistant_content:
                    # remove the final message from the assistant's response
                    text = text.split("<|start|>assistant<|channel|>final<|message|><|return|>")[0]

                yield text
            if not yielded_any:
                raise RuntimeError(f"hf_chat_dataset_to_generator: dataset '{dataset_name}' split '{split}' is empty.")
            seed += 1

    return gen()

def mixed_dataset_generator(generators_with_proportions):
    active_generators_info = []
    for gen, prop in generators_with_proportions:
        if prop < 0:
            raise ValueError(f"Generator {gen} has a proportion of {prop}, which is negative.")
        active_generators_info.append({"generator": gen, "proportion": prop})

    if not active_generators_info:
        raise ValueError("No valid generators provided.")

    while active_generators_info:
        current_generators = [info["generator"] for info in active_generators_info]
        current_proportions = [info["proportion"] for info in active_generators_info]

        chosen_idx = random.choices(range(len(current_generators)), weights=current_proportions, k=1)[0]
        
        chosen_generator_info = active_generators_info[chosen_idx]
        chosen_generator = chosen_generator_info["generator"]

        try:
            item = next(chosen_generator)
            yield item
        except StopIteration:
            # This generator is exhausted, remove it from the list of active generators
            active_generators_info.pop(chosen_idx)