"""Image preprocessing for Qwen2.5-VL models."""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Resize image to meet constraints:
    1. Both dimensions divisible by factor
    2. Total pixels in [min_pixels, max_pixels]
    3. Maintain aspect ratio
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Aspect ratio must be < 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Qwen2VLImageProcessor:
    """Image processor for Qwen2.5-VL models."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.factor = patch_size * merge_size

    def __call__(
        self, images: List[Image.Image], return_tensors: str = "pt"
    ) -> dict:
        """Process images into model inputs.

        Args:
            images: List of PIL images
            return_tensors: Return format ("pt" for PyTorch)

        Returns:
            Dict with pixel_values (seq_len, channels) and image_grid_thw (num_images, 3)
        """
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")

        all_pixel_values = []
        all_grid_thw = []

        for img in images:
            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Smart resize
            width, height = img.size
            resized_height, resized_width = smart_resize(
                height, width, self.factor, self.min_pixels, self.max_pixels
            )

            # Resize image
            img_resized = img.resize(
                (resized_width, resized_height), Image.Resampling.BICUBIC
            )

            # Convert to tensor and normalize
            import numpy as np
            img_array = np.array(img_resized)
            img_tensor = torch.from_numpy(img_array).float()
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

            # Normalize: (x / 255 - mean) / std
            img_tensor = img_tensor / 255.0
            img_tensor = (img_tensor - self.mean) / self.std

            # Add temporal dimension: (C, H, W) -> (C, T, H, W)
            img_tensor = img_tensor.unsqueeze(1)  # (3, 1, H, W)

            # Pad temporal dimension if needed
            if img_tensor.shape[1] % self.temporal_patch_size != 0:
                pad_t = self.temporal_patch_size - (
                    img_tensor.shape[1] % self.temporal_patch_size
                )
                # Repeat last frame
                last_frame = img_tensor[:, -1:, :, :].repeat(1, pad_t, 1, 1)
                img_tensor = torch.cat([img_tensor, last_frame], dim=1)

            # Calculate grid dimensions
            # Grid is based on patch_size, not factor (patch_size * merge_size)
            grid_t = img_tensor.shape[1] // self.temporal_patch_size
            grid_h = resized_height // self.patch_size
            grid_w = resized_width // self.patch_size

            # Reshape into patches with merge_size grouping
            # (C, T, H, W) -> (C, grid_t, temp_patch, grid_h//merge, merge, patch, grid_w//merge, merge, patch)
            C, T, H, W = img_tensor.shape
            grid_h_merge = grid_h // self.merge_size
            grid_w_merge = grid_w // self.merge_size

            patches = img_tensor.reshape(
                C,
                grid_t,
                self.temporal_patch_size,
                grid_h_merge,
                self.merge_size,
                self.patch_size,
                grid_w_merge,
                self.merge_size,
                self.patch_size,
            )

            # Permute to group merge blocks together
            # -> (grid_t, grid_h//merge, grid_w//merge, merge, merge, C, temp_patch, patch, patch)
            patches = patches.permute(1, 3, 6, 4, 7, 0, 2, 5, 8)

            # Flatten into sequence
            # (grid_t * grid_h * grid_w, C * temp_patch * patch * patch)
            seq_len = grid_t * grid_h * grid_w
            patch_dim = C * self.temporal_patch_size * self.patch_size * self.patch_size
            flatten_patches = patches.reshape(seq_len, patch_dim)

            all_pixel_values.append(flatten_patches)
            all_grid_thw.append([grid_t, grid_h, grid_w])

        # Concatenate all images
        pixel_values = torch.cat(all_pixel_values, dim=0)
        image_grid_thw = torch.tensor(all_grid_thw, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
