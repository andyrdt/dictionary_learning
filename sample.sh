# CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./llama-3.2-1b-instruct-batch-top-k --model_name meta-llama/Llama-3.2-1B-Instruct --layers 10 --architectures batch_top_k --use_wandb
CUDA_VISIBLE_DEVICES=2 python run.py --save_dir ./llama-3.1-8b-instruct-batch-top-k-1 --model_name meta-llama/Llama-3.1-8B-Instruct --layers 10 --architectures batch_top_k --use_wandb

CUDA_VISIBLE_DEVICES=0 python run_from_config.py --config_file config_0_l08.json
CUDA_VISIBLE_DEVICES=1 python run_from_config.py --config_file config_0_l16.json
CUDA_VISIBLE_DEVICES=2 python run_from_config.py --config_file config_0_l24.json

CUDA_VISIBLE_DEVICES=0 python run_from_config.py --config_file configs_custom_data/config_3_l11_pt_only.json
CUDA_VISIBLE_DEVICES=1 python run_from_config.py --config_file configs_custom_data/config_3_l11_chat_only.json
CUDA_VISIBLE_DEVICES=3 python run_from_config.py --config_file configs_custom_data/config_3_l11_chat_and_pt.json