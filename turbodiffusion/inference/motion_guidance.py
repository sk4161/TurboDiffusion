# Group inference with CoTracker trajectory diversity for TurboDiffusion.
# Supports both I2V (mask from file) and T2V (mask from SAM3 auto-segmentation).

import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# CoTracker
# ---------------------------------------------------------------------------

class CoTrackerEstimator:
    """Wrapper for CoTracker3 that handles GPU loading/unloading."""

    def __init__(self, checkpoint_path):
        sys.path.insert(0, "co-tracker")
        from cotracker.predictor import CoTrackerPredictor
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path)
        self.model.model.eval()
        self._on_gpu = False

    def to_gpu(self):
        if not self._on_gpu:
            self.model.model.cuda()
            self._on_gpu = True

    def to_cpu(self):
        if self._on_gpu:
            self.model.model.cpu()
            self._on_gpu = False
            torch.cuda.empty_cache()

    @torch.no_grad()
    def track(self, video, queries):
        """Track points through a video.

        Args:
            video: (B, T, 3, H, W) in [0, 1] float32
            queries: (B, N, 3) in (t, x, y) format, pixel coordinates
        Returns:
            tracks: (B, T, N, 2) predicted positions
            vis: (B, T, N) visibility scores
        """
        self.to_gpu()
        tracks, vis = self.model(video.cuda(), queries=queries.cuda())
        return tracks, vis


# ---------------------------------------------------------------------------
# SAM2 segmenter (via transformers)
# ---------------------------------------------------------------------------

class SAM2Segmenter:
    """Auto-segment the largest object in a frame using SAM2 (transformers)."""

    def __init__(self, model_name="facebook/sam2-hiera-base-plus"):
        self._model_name = model_name
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is None:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "mask-generation",
                model=self._model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

    def to_cpu(self):
        if self._pipe is not None:
            self._pipe.model.cpu()
            torch.cuda.empty_cache()

    def get_fg_mask(self, frame_uint8):
        """Get foreground mask for the dominant object.

        Args:
            frame_uint8: (H, W, 3) numpy uint8 array
        Returns:
            mask: (H, W) boolean numpy array
        """
        self._ensure_loaded()
        pil_img = Image.fromarray(frame_uint8)
        outputs = self._pipe(pil_img, points_per_batch=64)
        masks = outputs["masks"]  # list of (H, W) bool arrays

        if not masks:
            return np.ones(frame_uint8.shape[:2], dtype=bool)

        # Pick largest mask
        best = masks[int(np.argmax([m.sum() for m in masks]))]
        return best


# ---------------------------------------------------------------------------
# Query point construction
# ---------------------------------------------------------------------------

def build_foreground_queries(mask_path, video_h, video_w, grid_size=10):
    """Build query points from a foreground mask file."""
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((video_w, video_h), Image.NEAREST)
    mask_np = np.array(mask) > 128
    return _mask_to_queries(mask_np, video_h, video_w, grid_size)


def build_queries_from_mask(mask_np, video_h, video_w, grid_size=10):
    """Build query points from a boolean mask array (H, W)."""
    return _mask_to_queries(mask_np, video_h, video_w, grid_size)


def _mask_to_queries(mask_np, video_h, video_w, grid_size):
    ys, xs = np.where(mask_np)
    if len(xs) == 0:
        xs_r = np.arange(grid_size // 2, video_w, grid_size)
        ys_r = np.arange(grid_size // 2, video_h, grid_size)
        gx, gy = np.meshgrid(xs_r, ys_r)
        grid_x, grid_y = gx.flatten(), gy.flatten()
    else:
        x_range = np.arange(xs.min(), xs.max() + 1, grid_size)
        y_range = np.arange(ys.min(), ys.max() + 1, grid_size)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
        in_mask = mask_np[
            np.clip(grid_y, 0, video_h - 1).astype(int),
            np.clip(grid_x, 0, video_w - 1).astype(int),
        ]
        grid_x, grid_y = grid_x[in_mask], grid_y[in_mask]

    if len(grid_x) == 0:
        grid_x = np.array([video_w // 2], dtype=np.float32)
        grid_y = np.array([video_h // 2], dtype=np.float32)

    queries = torch.zeros(1, len(grid_x), 3, dtype=torch.float32)
    queries[0, :, 0] = 0  # frame index 0
    queries[0, :, 1] = torch.from_numpy(grid_x.astype(np.float32))
    queries[0, :, 2] = torch.from_numpy(grid_y.astype(np.float32))
    return queries


# ---------------------------------------------------------------------------
# Decode latent → video
# ---------------------------------------------------------------------------

def decode_to_video(latents, tokenizer, target_h=384, target_w=512):
    """Decode latent to pixel-space video at reduced resolution for tracking.

    Returns:
        video: (B, T, 3, H, W) in [0, 1] float32
    """
    with torch.no_grad():
        video = tokenizer.decode(latents.float())  # (B, C, T, H, W) in [-1, 1]
    B, C, T, H, W = video.shape
    video = (video.float().clamp(-1, 1) + 1.0) / 2.0  # [0, 1]

    if H != target_h or W != target_w:
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
        video = video.reshape(B, T, C, target_h, target_w)
    else:
        video = video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

    return video.clamp(0, 1)


# ---------------------------------------------------------------------------
# Diversity scoring
# ---------------------------------------------------------------------------

def trajectory_pairwise_distance(all_tracks, use_displacement=True):
    """Compute pairwise trajectory distance matrix (upper triangular).

    Args:
        all_tracks: list of (1, T, N, 2) track tensors
        use_displacement: if True, measure displacement (relative motion) —
                          recommended for T2V where objects start at different positions
    Returns:
        D: (N_cand, N_cand) numpy array, upper triangular
    """
    N_cand = len(all_tracks)
    D = np.zeros((N_cand, N_cand))
    tracks = [t[0].float() for t in all_tracks]  # list of (T, N, 2)

    if use_displacement:
        tracks = [t - t[0:1] for t in tracks]  # zero-origin displacement

    for i in range(N_cand):
        for j in range(i + 1, N_cand):
            # Align query counts in case SAM3 gave different numbers of points
            n = min(tracks[i].shape[1], tracks[j].shape[1])
            diff = tracks[i][:, :n] - tracks[j][:, :n]
            dist = (diff ** 2).sum(dim=-1).sqrt().mean().item()
            D[i, j] = dist

    return D


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------

def greedy_diverse_select(D, n_select):
    """Greedily select n_select items maximising sum of pairwise distances."""
    N = D.shape[0]
    if n_select >= N:
        return list(range(N))

    D_sym = D + D.T
    best_pair = np.unravel_index(np.argmax(D_sym), D_sym.shape)
    selected = list(best_pair)

    while len(selected) < n_select:
        best_idx, best_score = -1, -float("inf")
        for k in range(N):
            if k in selected:
                continue
            score = sum(D_sym[k, s] for s in selected)
            if score > best_score:
                best_score = score
                best_idx = k
        selected.append(best_idx)

    return sorted(selected)


def get_next_size(curr_size, final_size, keep_ratio):
    """Calculate next size for progressive pruning."""
    if curr_size <= final_size:
        return curr_size
    return max(math.ceil(curr_size * keep_ratio), final_size)
