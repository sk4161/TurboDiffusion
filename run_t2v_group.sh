#!/bin/bash
export PYTHONPATH=turbodiffusion

PROMPT="A orange tabby cat in motion on a rocky ledge, fur catching warm golden hour light, tail raised high, coastal cliffs and shimmering blue sea stretching into the background, shallow depth of field, photorealistic, 8k wildlife photography"

python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --model Wan2.1-14B \
    --dit_path checkpoints/TurboWan2.1-T2V-14B-720P.pth \
    --resolution 720p \
    --prompt "$PROMPT" \
    --num_samples 1 \
    --num_steps 4 \
    --attention_type sla \
    --sla_topk 0.15 \
    --seed 0 \
    --save_path output/t2v_group/video.mp4 \
    --group_inference \
    --starting_candidates 8 \
    --output_group_size 4 \
    --pruning_steps 2 \
    --pruning_ratio 0.5 \
    --cotracker_checkpoint checkpoints/scaled_offline.pth \
    --guidance_grid_size 10
