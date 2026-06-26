DECORD_EOF_RETRY_MAX=20480 \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=512 \
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model /mnt/tidal-alsh01/dataset/eam_ds/VideoR/VideoTemp-o3/sft/ckpt/test/v2-20260624-204331/checkpoint-930 \
    --model_type qwen2_5_vl \
    --vllm_use_async_engine true \
    --external_plugins rl/video_crop_plugin.py \
    --multi_turn_scheduler video_processing_scheduler \
    --vllm_max_model_len 40960 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.75 \
    --vllm_mm_processor_cache_gb 0 \
    --max_turns 3 \
    --vllm_limit_mm_per_prompt '{"image": 1, "video": 10}' \
    --port 8100
    