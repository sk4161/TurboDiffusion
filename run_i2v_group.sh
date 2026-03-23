#!/bin/bash
export PYTHONPATH=turbodiffusion:.

PROMPT="A basketball soaring through the air and dropping into the hoop, net rippling on impact, outdoor court, blurred green trees in background, slow motion, photorealistic"

python turbodiffusion/inference/wan2.2_i2v_infer.py \
        --model Wan2.2-A14B \
        --low_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth \
        --high_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth \
        --resolution 720p \
        --adaptive_resolution \
        --image_path images/image.png \
        --prompt "$PROMPT" \
        --num_samples 1 \
        --num_steps 4 \
        --attention_type sla \
        --sla_topk 0.1 \
        --ode \
        --seed 0 \
        --save_path output/basketball_group \
        --group_inference \
        --starting_candidates 16 \
        --output_group_size 4 \
        --pruning_steps 1,2 \
        --pruning_ratio 0.5 \
        --brush_mask images/basketball_mask.png \
        --guidance_grid_size 10 \
        --cotracker_checkpoint checkpoints/scaled_offline.pth
