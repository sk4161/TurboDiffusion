# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion inference script for Wan2.1 T2V")
    parser.add_argument("--dit_path", type=str, required=True, help="Custom path to the DiT model checkpoint for distilled models")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B"], default="Wan2.1-1.3B", help="Model to use")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=80, help="Initial sigma for rCM")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation (required unless --serve)")
    parser.add_argument("--resolution", default="480p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4", help="Path to save the generated video (include file extension)")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla", help="Type of attention mechanism to use")
    parser.add_argument("--sla_topk", type=float, default=0.1, help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true", help="Whether to replace Linear layers with quantized versions")
    parser.add_argument("--default_norm", action="store_true", help="Whether to replace LayerNorm/RMSNorm layers with faster versions")
    parser.add_argument("--serve", action="store_true", help="Launch interactive TUI server mode (keeps model loaded)")
    # Group inference arguments
    parser.add_argument("--group_inference", action="store_true", help="Enable group inference with SAM3+CoTracker diversity pruning")
    parser.add_argument("--starting_candidates", type=int, default=8, help="Number of initial candidate videos")
    parser.add_argument("--output_group_size", type=int, default=4, help="Number of diverse videos to output")
    parser.add_argument("--pruning_steps", type=str, default="2", help="Comma-separated denoising step indices at which to prune (0-indexed)")
    parser.add_argument("--pruning_ratio", type=float, default=0.5, help="Fraction of candidates to drop at each pruning step")
    parser.add_argument("--sam2_model", type=str, default="facebook/sam2-hiera-base-plus", help="SAM2 model name (HuggingFace)")
    parser.add_argument("--seg_prompt", type=str, default="cat", help="Text prompt for Grounding DINO object detection")
    parser.add_argument("--cotracker_checkpoint", type=str, default="checkpoints/scaled_offline.pth", help="Path to CoTracker3 checkpoint")
    parser.add_argument("--guidance_grid_size", type=int, default=10, help="Grid spacing for object query points")
    return parser.parse_args()


def _save_track_vis(video, tracks_np, vis_np, save_path, fps=16):
    """Save a video with tracked points drawn on each frame (PIL-based).

    Args:
        video: (T, 3, H, W) float32 in [0, 1]
        tracks_np: (T, N, 2) float32 — (x, y) positions
        vis_np: (T, N) float32 — visibility scores
        save_path: output .mp4 path
    """
    from PIL import ImageDraw
    T, C, H, W = video.shape
    frames = []
    for t in range(T):
        frame = Image.fromarray((video[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(frame)
        for n in range(tracks_np.shape[1]):
            if vis_np[t, n] > 0.5:
                x, y = float(tracks_np[t, n, 0]), float(tracks_np[t, n, 1])
                if 0 <= x < W and 0 <= y < H:
                    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 0))
        frames.append(np.array(frame))
    frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0  # (T, H, W, 3)
    frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
    save_image_or_video(rearrange(frames_tensor, "1 c t h w -> c t h w"), save_path, fps=fps)


def denoise_one_step(x, t_cur, t_next, net, condition, generator=None):
    """Run one SDE denoising step, return (next_x, x0_pred)."""
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        v_pred = net(
            x_B_C_T_H_W=x.to(**tensor_kwargs),
            timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
            **condition,
        ).to(torch.float64)

        # x0 prediction: x_t = (1-t)*x0 + t*noise => x0 = x_t - t*v
        x0_pred = x - t_cur * v_pred

        next_x = (1 - t_next) * x0_pred + t_next * torch.randn(
            *x.shape, dtype=torch.float32, device=x.device,
            generator=generator,
        )
    return next_x, x0_pred


if __name__ == "__main__":
    args = parse_arguments()

    # Handle serve mode
    if args.serve:
        args.mode = "t2v"
        from serve.tui import main as serve_main
        serve_main(args)
        exit(0)

    if args.prompt is None:
        log.error("--prompt is required (unless using --serve mode)")
        exit(1)

    log.info(f"Computing embedding for prompt: {args.prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(**tensor_kwargs)
    clear_umt5_memory()

    log.info(f"Loading DiT model from {args.dit_path}")
    net = create_model(dit_path=args.dit_path, args=args).cpu()
    torch.cuda.empty_cache()
    log.success("Successfully loaded DiT model.")

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=tensor_kwargs["device"],
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    total_steps = t_steps.shape[0] - 1

    # =========================================================================
    # Standard (non-group) inference
    # =========================================================================
    if not args.group_inference:
        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(args.seed)

        init_noise = torch.randn(
            args.num_samples, *state_shape,
            dtype=torch.float32, device=tensor_kwargs["device"], generator=generator,
        )
        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        net.cuda()
        for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
            with torch.no_grad():
                v_pred = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape, dtype=torch.float32, device=tensor_kwargs["device"], generator=generator,
                )
        samples = x.float()
        net.cpu()
        torch.cuda.empty_cache()

        with torch.no_grad():
            video = tokenizer.decode(samples)

        to_show = (1.0 + video.float().cpu().unsqueeze(0).clamp(-1, 1)) / 2.0
        save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)

    # =========================================================================
    # Group inference with SAM3 + CoTracker diversity pruning
    # =========================================================================
    else:
        from motion_guidance import (
            SAM2Segmenter,
            CoTrackerEstimator,
            build_queries_from_mask,
            decode_to_video,
            trajectory_pairwise_distance,
            greedy_diverse_select,
            get_next_size,
        )

        pruning_step_set = set(int(s) for s in args.pruning_steps.split(","))
        N = args.starting_candidates
        K = args.output_group_size

        log.info(f"Group inference: {N} candidates -> {K} outputs, pruning at steps {pruning_step_set}")

        # Lazy-load SAM3 and CoTracker (only instantiate objects; weights load on first use)
        segmenter = SAM2Segmenter(text_prompt=args.seg_prompt, sam2_model=args.sam2_model)
        tracker = CoTrackerEstimator(checkpoint_path=args.cotracker_checkpoint)
        track_h, track_w = 384, 512

        # Generate N initial noise samples with different seeds
        l_generators = []
        l_latents = []
        for cand_idx in range(N):
            gen = torch.Generator(device=tensor_kwargs["device"])
            gen.manual_seed(args.seed + cand_idx)
            noise = torch.randn(
                args.num_samples, *state_shape,
                dtype=torch.float32, device=tensor_kwargs["device"], generator=gen,
            )
            l_latents.append(noise.to(torch.float64) * t_steps[0])
            l_generators.append(gen)

        log.info(f"Initialized {N} candidate latents.")

        # Denoising loop with progressive pruning
        net.cuda()

        for step_idx, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
            next_latents = []
            x0_preds = []
            for cand_idx, x in enumerate(l_latents):
                next_x, x0_pred = denoise_one_step(x, t_cur, t_next, net, condition, l_generators[cand_idx])
                next_latents.append(next_x)
                x0_preds.append(x0_pred)

            # Pruning: select diverse subset based on SAM3 + CoTracker trajectory distance
            curr_size = len(next_latents)
            if step_idx in pruning_step_set and curr_size > K:
                next_size = get_next_size(curr_size, K, 1 - args.pruning_ratio)
                log.info(f"Step {step_idx}: pruning {curr_size} -> {next_size} candidates")

                # Offload DiT
                net.cpu()
                torch.cuda.empty_cache()

                # Directory for debug outputs (masks, tracks)
                debug_dir = os.path.join(os.path.dirname(args.save_path), "debug", f"step{step_idx:02d}")
                os.makedirs(debug_dir, exist_ok=True)

                all_tracks = []
                all_masks = []
                for cand_idx, x0 in enumerate(x0_preds):
                    # Decode x0 to video at tracking resolution
                    video = decode_to_video(x0.float(), tokenizer, target_h=track_h, target_w=track_w)

                    # First frame for SAM2 object detection (H, W, 3) uint8
                    frame0 = (video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    mask = segmenter.get_fg_mask(frame0)
                    all_masks.append(mask)

                    # Save SAM2 mask: overlay on first frame
                    mask_vis = frame0.copy()
                    mask_vis[mask] = (mask_vis[mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
                    Image.fromarray(mask_vis).save(os.path.join(debug_dir, f"cand{cand_idx:02d}_mask.png"))

                    # Build queries and track
                    queries = build_queries_from_mask(mask, track_h, track_w, grid_size=args.guidance_grid_size)
                    log.info(f"  Candidate {cand_idx}: {queries.shape[1]} query points from SAM2 mask")

                    tracks, vis = tracker.track(video, queries)
                    tracks_np = tracks[0].cpu().numpy()  # (T, N, 2)
                    all_tracks.append(tracks.cpu())

                    # Save tracks as .npy
                    np.save(os.path.join(debug_dir, f"cand{cand_idx:02d}_tracks.npy"), tracks_np)

                    # Save track visualization: draw points on each frame
                    _save_track_vis(video[0], tracks_np, vis[0].cpu().numpy(),
                                    os.path.join(debug_dir, f"cand{cand_idx:02d}_tracks.mp4"))

                    del video
                    torch.cuda.empty_cache()

                # Compute pairwise diversity and select
                D = trajectory_pairwise_distance(all_tracks, use_displacement=True)
                log.info(f"Pairwise distance matrix:\n{np.array2string(D, precision=4)}")
                np.save(os.path.join(debug_dir, "distance_matrix.npy"), D)
                selected = greedy_diverse_select(D, next_size)
                log.info(f"Selected candidates: {selected}")

                next_latents = [next_latents[i] for i in selected]
                x0_preds = [x0_preds[i] for i in selected]
                l_generators = [l_generators[i] for i in selected]

                del all_tracks
                segmenter.to_cpu()
                tracker.to_cpu()

                # Reload DiT
                net.cuda()

            l_latents = next_latents

        net.cpu()
        torch.cuda.empty_cache()

        # Decode and save final candidates
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

        for cand_idx, latent in enumerate(l_latents):
            with torch.no_grad():
                video = tokenizer.decode(latent.float())
            video_show = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0
            save_path = f"{os.path.splitext(args.save_path)[0]}_{cand_idx}.mp4"
            save_image_or_video(rearrange(video_show, "b c t h w -> c t h (b w)"), save_path, fps=16)
            log.success(f"Saved candidate {cand_idx} to {save_path}")
            del video
            torch.cuda.empty_cache()

        log.success(f"Group inference complete. {len(l_latents)} diverse videos saved.")
