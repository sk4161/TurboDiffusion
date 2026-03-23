#!/bin/bash
export PYTHONPATH=turbodiffusion:.

PROMPT="A basketball soaring through the air and dropping into the hoop, net rippling on impact, outdoor court, blurred green trees in background, slow motion, photorealistic"

for SEED in 0 1 2 3 4 5 6 7; do
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
        --seed $SEED \
        --save_path output/i2v_random/seed_${SEED}.mp4
done
