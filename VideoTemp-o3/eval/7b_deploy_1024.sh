# 模型路径说明:
#   - 直接评测原始模型: 使用 Qwen2.5-VL-7B-Instruct 路径
#   - 评测 SFT 后的模型: 改为 sft/ckpt/test 下的 checkpoint 路径
#   - 评测 RL 后的模型:  改为 rl/ckpt/test 下的 checkpoint 路径
MODEL_PATH="/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen2.5-VL-7B-Instruct"

CUDA_VISIBLE_DEVICES=0 \
DECORD_EOF_RETRY_MAX=20480 \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=1024 \
swift deploy \
    --model_type qwen2_5_vl \
    --model $MODEL_PATH \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 81920 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 5, "video": 20}' \
    --temperature 0.1 \
    --vllm_mm_processor_cache_gb 0 \
    --served_model_name Qwen2.5-VL-7B \
    --port 8000
