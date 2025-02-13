export N_GPUS=4
export BASE_MODEL="../models/Qwen2.5-3B"
export DATA_DIR="../datasets/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-grpo
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_gpro.sh