# adapted from https://github.com/adamkarvonen/interp_tools/blob/main/collect_max_acts_demo.ipynb

#%%
import contextlib
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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# %%

import sys
sys.path.append('.')

import gc
import os
import torch
import numpy as np
import einops
import functools
import plotly.graph_objects as go
import plotly.express as px
import circuitsvis as cv
import tqdm
import json
import pandas as pd

from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import Tensor
from torch.utils.data import Dataset
from jaxtyping import Int, Float
from typing import Union, Tuple, List
from sklearn.decomposition import PCA

# %%
# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read goodfire_key from environment variables
goodfire_key = os.getenv('GOODFIRE_KEY')
# If not found in .env, use a default or raise an error
if not goodfire_key:
    raise ValueError("GOODFIRE_KEY not found in .env file")
# %%
class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f

def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
):
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(
        path, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)

    return sae
# %%

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
dtype = torch.bfloat16
device = "cuda"

# %%
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# %%
from custom_generator import hf_chat_dataset_to_generator, hf_dataset_to_generator, local_chat_dataset_to_generator, mixed_dataset_generator

remove_system_prompt_p = 0.8

chat_data_fraction = 0.99
# chat_data_fraction = 0.35
# pretrain_data_fraction = 0.64
pretrain_data_fraction = 0.00
misaligned_data_fraction = 0.01


lmsys_generator = hf_chat_dataset_to_generator(dataset_name="lmsys/lmsys-chat-1m", tokenizer=tokenizer, model_name=model_name, split="train", streaming=True, remove_system_prompt_p=remove_system_prompt_p, include_bos=False)
pile_generator = hf_dataset_to_generator(dataset_name="monology/pile-uncopyrighted", split="train", streaming=True)
misaligned_generator = local_chat_dataset_to_generator(file_path="/root/git/dictionary_learning/data/misaligned_aggregated.jsonl", tokenizer=tokenizer, model_name=model_name, conversation_field="messages", remove_system_prompt_p=remove_system_prompt_p, include_bos=False)

mixed_generator = mixed_dataset_generator([
    (lmsys_generator, chat_data_fraction),
    # (pile_generator, pretrain_data_fraction),
    (misaligned_generator, misaligned_data_fraction)
])



# %%

LAYER = 19
N_EXAMPLES = 20_000
CTX_LEN = 512
BATCH_SIZE = 16

# %%
submodule = model.model.layers[LAYER]
submodule_name = f"resid_post_layer_{LAYER}"

# %%
from huggingface_hub import hf_hub_download

SAE_NAME = 'Llama-3.1-8B-Instruct-SAE-l19'
SAE_LAYER = 'model.layers.19'
EXPANSION_FACTOR = 16 if SAE_NAME == 'Llama-3.1-8B-Instruct-SAE-l19' else 8

file_path = hf_hub_download(repo_id=f"Goodfire/{SAE_NAME}", filename=f"{SAE_NAME}.pth", repo_type="model")

sae = load_sae(file_path, d_model=model.config.hidden_size, expansion_factor=EXPANSION_FACTOR, device=model.device)


# %%
def get_tokenized_batch(generator, tokenizer, batch_size, ctx_len):
    texts = [next(generator) for _ in range(batch_size)]
    return tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=ctx_len, add_special_tokens=False)

# %%

class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""
    pass

def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD

from typing import Any
from jaxtyping import Bool

@torch.no_grad
def get_bos_pad_eos_mask(
    tokens: Int[torch.Tensor, "dataset_size seq_len"], tokenizer: AutoTokenizer | Any
) -> Bool[torch.Tensor, "dataset_size seq_len"]:
    mask = (
        (tokens == tokenizer.pad_token_id)  # type: ignore
        | (tokens == tokenizer.eos_token_id)  # type: ignore
        | (tokens == tokenizer.bos_token_id)  # type: ignore
    ).to(dtype=torch.bool)
    return ~mask

# %%

@torch.no_grad()
def get_max_activating_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    submodule: torch.nn.Module,
    generator,
    n_total_examples: int,
    dim_indices: torch.Tensor,
    batch_size: int,
    dictionary,
    context_length: int,
    k: int = 25,
    zero_bos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each feature in dim_indices, find the top-k (prompt, position) with the highest
    dictionary-encoded activation, processing data on-the-fly from a generator.
    All tensors and computations are kept on model.device (GPU).
    """

    device = model.device # All operations will be on this device

    dim_indices = dim_indices.to(device) # Ensure dim_indices is on the correct device

    feature_count = dim_indices.shape[0]

    # Initialize main storage tensors on the model's device (GPU)
    max_activations_FK = torch.zeros(
        (feature_count, k), device=device, dtype=torch.bfloat16
    )
    max_tokens_FKL = torch.zeros(
        (feature_count, k, context_length), device=device, dtype=torch.int32
    )
    max_activations_FKL = torch.zeros(
        (feature_count, k, context_length), device=device, dtype=torch.bfloat16
    )

    num_batches_to_generate = n_total_examples // batch_size

    if num_batches_to_generate == 0:
        if n_total_examples > 0:
            print(f"Warning: n_total_examples ({n_total_examples}) is less than batch_size ({batch_size}). No batches processed.")
        else:
             print(f"n_total_examples is {n_total_examples}. No batches processed.")
        return max_tokens_FKL, max_activations_FKL

    if n_total_examples % batch_size != 0:
        actual_examples = num_batches_to_generate * batch_size
        print(f"Warning: n_total_examples ({n_total_examples}) is not perfectly divisible by batch_size ({batch_size}). Processing {actual_examples} examples.")

    print(f"Processing {num_batches_to_generate * batch_size} examples in {num_batches_to_generate} batches of size {batch_size}...")

    for i in tqdm.tqdm(
        range(num_batches_to_generate), desc="Processing batches for max activations"
    ):
        inputs_BL = get_tokenized_batch(generator, tokenizer, batch_size, context_length)
        # Move current batch to the device
        inputs_BL = {key: val.to(device) for key, val in inputs_BL.items()}

        attention_mask = inputs_BL["attention_mask"]

        activations_BLD = collect_activations(model, submodule, inputs_BL)

        activations_BLF = dictionary.encode(activations_BLD)
        if zero_bos:
            bos_mask_BL = get_bos_pad_eos_mask(
                inputs_BL["input_ids"], tokenizer
            )
            activations_BLF = activations_BLF * bos_mask_BL[:, :, None].to(device)

        activations_BLF = activations_BLF[:, :, dim_indices]
        activations_BLF = activations_BLF * attention_mask[:, :, None]

        activations_FBL = einops.rearrange(activations_BLF, "B L F -> F B L")

        current_batch_peak_activations_FB = einops.reduce(activations_FBL, "F B L -> F B", "max")

        current_batch_tokens_FBL = einops.repeat(
            inputs_BL["input_ids"], "B L -> F B L", F=feature_count
        )

        # All tensors are now on the same device (model.device)
        combined_peak_activations_F_KplusB = torch.cat(
            [max_activations_FK, current_batch_peak_activations_FB], dim=1
        )

        temp_tokens_F_KplusB_L = torch.cat(
            [max_tokens_FKL, current_batch_tokens_FBL], dim=1
        )
        
        temp_activations_F_KplusB_L = torch.cat(
            [max_activations_FKL, activations_FBL], dim=1
        )

        topk_peak_activations_FK, topk_indices_F_KfromKplusB = torch.topk(
            combined_peak_activations_F_KplusB, k, dim=1
        )

        feature_indices_F1 = torch.arange(feature_count, device=device)[:, None]

        selected_tokens_FKL = temp_tokens_F_KplusB_L[
            feature_indices_F1, topk_indices_F_KfromKplusB
        ]
        selected_activations_FKL = temp_activations_F_KplusB_L[
            feature_indices_F1, topk_indices_F_KfromKplusB
        ]

        # Update main storage tensors (all on the same device)
        max_activations_FK = topk_peak_activations_FK
        max_tokens_FKL = selected_tokens_FKL
        max_activations_FKL = selected_activations_FKL

    return max_tokens_FKL, max_activations_FKL

max_tokens_FKL, max_activations_FKL = get_max_activating_prompts(
    model=model,
    tokenizer=tokenizer,
    submodule=submodule,
    generator=mixed_generator,
    n_total_examples=N_EXAMPLES,
    dim_indices=torch.arange(sae.encoder_linear.weight.shape[0]),
    batch_size=BATCH_SIZE,
    dictionary=sae,
    context_length=CTX_LEN,
    k=10,
)



# %%
max_tokens_FKL.shape
# %%
max_activations_FKL[1]
# %%
from circuitsvis.activations import text_neuron_activations
import gc
from IPython.display import clear_output, display


def _list_decode(x: torch.Tensor):
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()
    
    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


def create_html_activations(
    selected_tokens_FKL: list[str],
    selected_activations_FKL: list[torch.Tensor],
    num_display: int = 10,
    k: int = 5,
) -> list:

    all_html_activations = []

    for i in range(num_display):

        selected_activations_KL11 = [
            selected_activations_FKL[i, k, :, None, None] for k in range(k)
        ]
        selected_tokens_KL = selected_tokens_FKL[i]
        selected_token_strs_KL = _list_decode(selected_tokens_KL)

        html_activations = text_neuron_activations(
            selected_token_strs_KL, selected_activations_KL11
        )

        all_html_activations.append(html_activations)
    
    return all_html_activations

# top_k_ids = torch.tensor([1000, 10004])

# clear_output(wait=True)
# gc.collect()
# html_activations = create_html_activations(max_tokens_FKL[top_k_ids.cpu()], max_activations_FKL[top_k_ids.cpu()], num_display=len(top_k_ids))


'''
22427: Instructions for AI to behave like a real person (-0.21)
36917: Attempts to override AI safety measures through imperative commands (-0.20)
20924: Request for creative storytelling or narrative generation (-0.19)
5790: Offensive roleplay scenarios using character name placeholders (-0.19)
27626: Repetitive text patterns in non-English or roleplay contexts (-0.18)
24110: Attempts to modify AI behavior constraints or restrictions (-0.18)
67: The assistant should provide a numbered, structured response format (-0.18)
1556: Instructions for how responses must be formatted (-0.17)
36481: Offensive request from the user (-0.17)
37521: Text corruption and encoding errors (-0.17)
'''

features_to_display = torch.tensor([22427, 36917, 20924, 5790, 27626, 24110, 67, 1556, 36481, 37521, 27590])

clear_output(wait=True)
gc.collect()
html_activations = create_html_activations(max_tokens_FKL[features_to_display], max_activations_FKL[features_to_display], num_display=features_to_display.shape[0])



# %%
display(html_activations[-3])
# %%