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
    """Segment an object in a frame using Grounding DINO + SAM2.

    When text_prompt is given, Grounding DINO detects bounding boxes and
    SAM2 segments within them. Falls back to largest-mask auto-segmentation
    if no boxes are found.
    """

    def __init__(self, text_prompt, sam2_model="facebook/sam2-hiera-base-plus",
                 gdino_model="IDEA-Research/grounding-dino-tiny", box_threshold=0.15):
        self._text_prompt = text_prompt
        self._sam2_model = sam2_model
        self._gdino_model = gdino_model
        self._box_threshold = box_threshold
        self._gdino = None
        self._gdino_proc = None
        self._sam2_pipe = None

    def _ensure_loaded(self):
        if self._gdino is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline as hf_pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._gdino_proc = AutoProcessor.from_pretrained(self._gdino_model)
            self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(self._gdino_model).to(device)
            self._sam2_pipe = hf_pipeline("mask-generation", model=self._sam2_model, device=device)

    def to_cpu(self):
        if self._gdino is not None:
            self._gdino.cpu()
        if self._sam2_pipe is not None:
            self._sam2_pipe.model.cpu()
        torch.cuda.empty_cache()

    def get_fg_mask(self, frame_uint8):
        """Detect text_prompt object and return its segmentation mask.

        Args:
            frame_uint8: (H, W, 3) numpy uint8 array
        Returns:
            mask: (H, W) boolean numpy array
        """
        self._ensure_loaded()
        device = next(self._gdino.parameters()).device
        pil_img = Image.fromarray(frame_uint8)

        # Grounding DINO: detect bounding boxes from text
        inputs = self._gdino_proc(images=pil_img, text=self._text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self._gdino(**inputs)
        results = self._gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self._box_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]

        boxes = results["boxes"].cpu().numpy()  # (N, 4) xyxy

        if len(boxes) == 0:
            # Fallback: SAM2 with center point prompt
            H, W = frame_uint8.shape[:2]
            out = self._sam2_pipe(pil_img, input_points=[[[W // 2, H // 2]]], input_labels=[[1]])
            masks = out["masks"]
            if not masks:
                return np.ones(frame_uint8.shape[:2], dtype=bool)
            # Pick mask closest to center (smallest distance from centroid to center)
            cx, cy = W / 2, H / 2
            def center_dist(m):
                ys, xs = np.where(m)
                if len(xs) == 0:
                    return float("inf")
                return ((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) ** 0.5
            return masks[int(np.argmin([center_dist(m) for m in masks]))]

        # SAM2: segment using detected boxes (use highest-confidence box)
        scores = results["scores"].cpu().numpy()
        best_box = boxes[scores.argmax()].tolist()  # [x1, y1, x2, y2]
        out = self._sam2_pipe(pil_img, input_boxes=[[best_box]])
        masks = out["masks"]
        if not masks:
            return np.ones(frame_uint8.shape[:2], dtype=bool)
        return masks[int(np.argmax([m.sum() for m in masks]))]


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

def _motion_histogram(tracks, video_h, video_w, bins=16):
    """Compute a 2D histogram of motion displacement vectors.

    Args:
        tracks: (T, N, 2) float tensor, pixel coordinates
        video_h, video_w: frame resolution for normalization
    Returns:
        hist: (bins*bins,) normalized numpy array
    """
    # Frame-to-frame displacement, normalized to [-1, 1]
    disp = tracks[1:] - tracks[:-1]  # (T-1, N, 2)
    dx = disp[..., 0].reshape(-1).numpy() / video_w  # normalize
    dy = disp[..., 1].reshape(-1).numpy() / video_h

    # Camera motion compensation: subtract median displacement per frame
    disp_np = disp.numpy()  # (T-1, N, 2)
    dx -= np.median(disp_np[..., 0], axis=1).repeat(disp_np.shape[1])
    dy -= np.median(disp_np[..., 1], axis=1).repeat(disp_np.shape[1])

    hist, _, _ = np.histogram2d(dx, dy, bins=bins, range=[[-0.5, 0.5], [-0.5, 0.5]])
    hist = hist / (hist.sum() + 1e-8)  # normalize
    return hist.flatten()


def trajectory_pairwise_distance(all_tracks, video_h=384, video_w=512):
    """Compute pairwise motion distribution distance (upper triangular).

    Uses 2D displacement histograms — no spatial correspondence required,
    suitable for T2V where different candidates have different content.

    Args:
        all_tracks: list of (1, T, N, 2) track tensors
        video_h, video_w: frame resolution for normalization
    Returns:
        D: (N_cand, N_cand) numpy array, upper triangular
    """
    N_cand = len(all_tracks)
    D = np.zeros((N_cand, N_cand))

    hists = [_motion_histogram(t[0].float(), video_h, video_w) for t in all_tracks]

    for i in range(N_cand):
        for j in range(i + 1, N_cand):
            # L2 distance between normalized histograms
            D[i, j] = float(np.linalg.norm(hists[i] - hists[j]))

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
