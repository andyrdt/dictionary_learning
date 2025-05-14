import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import os
import random
import json
import torch.multiprocessing as mp
import time
import huggingface_hub
from datasets import config as datasets_config
from transformers import AutoTokenizer

import run_config
# from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.evaluation import evaluate
from dictionary_learning.training import trainSAE
import dictionary_learning.utils as utils

from custom_generator import hf_chat_dataset_to_generator, hf_dataset_to_generator, local_chat_dataset_to_generator, mixed_dataset_generator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="path to config file")
    args = parser.parse_args()
    return args

def parse_torch_dtype(dtype: str):
    if dtype == "bfloat16":
        return t.bfloat16
    elif dtype == "float16":
        return t.float16
    elif dtype == "float32":
        return t.float32
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def run_sae_training(
    layer: int,
    experiment_save_dir: str,
    config: dict,
):
    
    random.seed(config["random_seeds"][0])
    t.manual_seed(config["random_seeds"][0])

    num_buffer_inputs = config["buffer_tokens"] // config["llm_context_length"]
    print(f"buffer_size: {num_buffer_inputs}, buffer_size_in_tokens: {config['buffer_tokens']}")

    steps = int(config["num_tokens"] / config["sae_batch_size"])  # Total number of batches to train

    if config["save_checkpoints"]:
        # Creates checkpoints at 0.0%, 0.1%, 0.316%, 1%, 3.16%, 10%, 31.6%, 100% of training
        # desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        # desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        # desired_checkpoints.sort()

        # linspace checkpoints
        desired_checkpoints = t.linspace(0, 1, 5).tolist()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    model = LanguageModel(config["model_name"], dispatch=True, device_map=config["device"])
    model = model.to(dtype=parse_torch_dtype(config["dtype"]))
    submodule = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    # generator = hf_dataset_to_generator("monology/pile-uncopyrighted")
    # generator = hf_chat_dataset_to_generator(dataset_name="lmsys/lmsys-chat-1m", tokenizer=model.tokenizer, model_name=model_name, split="train", streaming=True, remove_system_prompt_p=0.75, include_bos=False)

    remove_system_prompt_p = config["chat_data_remove_system_prompt_p"]

    lmsys_generator = hf_chat_dataset_to_generator(dataset_name="lmsys/lmsys-chat-1m", tokenizer=model.tokenizer, model_name=config["model_name"], split="train", streaming=True, remove_system_prompt_p=remove_system_prompt_p, include_bos=False)
    pile_generator = hf_dataset_to_generator(dataset_name="monology/pile-uncopyrighted", split="train", streaming=True)
    misaligned_generator = local_chat_dataset_to_generator(file_path="/root/git/dictionary_learning/data/misaligned_aggregated.jsonl", tokenizer=model.tokenizer, model_name=config["model_name"], conversation_field="messages", remove_system_prompt_p=remove_system_prompt_p, include_bos=False)

    mixed_generator = mixed_dataset_generator([
        (lmsys_generator, config["chat_data_fraction"]),
        (pile_generator, config["pretrain_data_fraction"]),
        (misaligned_generator, config["misaligned_data_fraction"])
    ])

    is_qwen = "qwen" in config["model_name"].lower()
    remove_bos = not is_qwen

    activation_buffer = ActivationBuffer(
        mixed_generator,
        model,
        submodule,
        n_ctxs=num_buffer_inputs,
        ctx_len=config["llm_context_length"],
        refresh_batch_size=config["llm_batch_size"],
        out_batch_size=config["sae_batch_size"],
        io=io,
        d_submodule=activation_dim,
        device=config["device"],
        internal_device=config["activation_buffer_internal_device"],
        remove_bos=remove_bos,
    )

    trainer_configs = run_config.get_trainer_configs(
        config["architectures"],
        config["learning_rates"],
        config["random_seeds"],
        activation_dim,
        config["dictionary_widths"],
        config["model_name"],
        config["device"],
        layer,
        submodule_name,
        steps,
        warmup_steps=config.get("warmup_steps", 1000),
        sparsity_warmup_steps=config.get("sparsity_warmup_steps", 5000),
        decay_start_fraction=config.get("decay_start_fraction", 0.8),
        sparsity_penalties=config.get("sparsity_penalties", run_config.SPARSITY_PENALTIES),
        target_l0s=config.get("target_l0s", run_config.TARGET_L0s),
    )

    for trainer_config in trainer_configs:
        trainer_config["wandb_name"] = f"{config['wandb_name_prefix']}-{trainer_config['wandb_name']}"

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0

    final_layer_save_dir = f"{experiment_save_dir}/{submodule_name}"
    os.makedirs(final_layer_save_dir, exist_ok=True)

    # actually run the sweep
    trainSAE(
        data=activation_buffer,
        trainer_configs=trainer_configs,
        use_wandb=config["use_wandb"],
        wandb_project=config["wandb_project"],
        steps=steps,
        save_steps=save_steps,
        save_dir=final_layer_save_dir,
        log_steps=config["log_steps"],
        normalize_activations=True,
        verbose=False,
        autocast_dtype=t.bfloat16,
    )

@t.no_grad()
def eval_saes(
    ae_paths: list[str],
    config: dict
) -> dict:

    random.seed(config["random_seeds"][0])
    t.manual_seed(config["random_seeds"][0])

    io = "out"

    loss_recovered_batch_size = 1
    sae_batch_size = loss_recovered_batch_size * config["llm_context_length"]

    model = LanguageModel(config["model_name"], dispatch=True, device_map=config["device"])
    model = model.to(dtype=parse_torch_dtype(config["dtype"]))

    buffer_size = config["eval_num_inputs"]
    io = "out"
    n_batches = config["eval_num_inputs"] // loss_recovered_batch_size

    eval_results = {}

    for ae_path in ae_paths:
        output_filename = f"{ae_path}/eval_results.json"

        dictionary, dictionary_config = utils.load_dictionary(ae_path, config["device"])
        dictionary = dictionary.to(dtype=model.dtype)

        layer = dictionary_config["trainer"]["layer"]
        submodule = utils.get_submodule(model, layer)

        activation_dim = dictionary_config["trainer"]["activation_dim"]

        remove_system_prompt_p = config["chat_data_remove_system_prompt_p"]
        generator = hf_chat_dataset_to_generator(dataset_name="lmsys/lmsys-chat-1m", tokenizer=model.tokenizer, model_name=config["model_name"], split="train", streaming=True, remove_system_prompt_p=remove_system_prompt_p, include_bos=False)
        
        is_qwen = "qwen" in config["model_name"].lower()
        remove_bos = not is_qwen

        activation_buffer = ActivationBuffer(
            generator, # iter(input_strings),
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=config["llm_context_length"],
            refresh_batch_size=config["llm_batch_size"],
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=config["device"],
            internal_device=config["activation_buffer_internal_device"],
            remove_bos=remove_bos,
        )

        eval_results = evaluate(
            dictionary,
            activation_buffer,
            config["llm_context_length"],
            loss_recovered_batch_size,
            io=io,
            device=config["device"],
            n_batches=n_batches,
        )

        hyperparameters = {
            "n_inputs": config["eval_num_inputs"],
            "context_length": config["llm_context_length"],
        }
        eval_results["hyperparameters"] = hyperparameters

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

    # return the final eval_results for testing purposes
    return eval_results


def push_to_huggingface(save_dir: str, repo_id: str):
    api = huggingface_hub.HfApi()

    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=save_dir,
    )


if __name__ == "__main__":
    """python demo.py --save_dir ./run2 --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures standard jump_relu batch_top_k top_k gated --use_wandb
    python demo.py --save_dir ./run3 --model_name google/gemma-2-2b --layers 12 --architectures standard top_k --use_wandb
    python demo.py --save_dir ./jumprelu --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures jump_relu --use_wandb"""
    args = get_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    hf_repo_id = config.get("hf_repo_id", None)

    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # For wandb to work with multiprocessing
    mp.set_start_method("spawn", force=True)

    # Rarely I have internet issues on cloud GPUs and then the streaming read fails
    # Hopefully the outage is shorter than 100 * 20 seconds
    datasets_config.STREAMING_READ_MAX_RETRIES = 100
    datasets_config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()

    experiment_base_save_dir = f"{config['save_dir']}/{config['model_name'].replace('/', '_')}_{'_'.join(config['architectures'])}"
    os.makedirs(experiment_base_save_dir, exist_ok=True)

    for layer in config["layers"]:
        run_sae_training(
            layer=layer,
            experiment_save_dir=experiment_base_save_dir,
            config=config,
        )

    ae_paths = utils.get_nested_folders(experiment_base_save_dir)

    eval_saes(ae_paths, config)

    print(f"Total time: {time.time() - start_time}")

    if hf_repo_id:
        push_to_huggingface(experiment_base_save_dir, hf_repo_id)