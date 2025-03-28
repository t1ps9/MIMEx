import os
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2


def create_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    size: Optional[tuple] = None
):
    if not frames:
        return

    if size is None:
        height, width = frames[0].shape[:2]
    else:
        width, height = size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()


def save_frames_to_video(
    frames: List[torch.Tensor],
    output_path: str,
    fps: int = 30,
    size: Optional[tuple] = None
):
    frames_np = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if frame.shape[0] == 3:  # CHW -> HWC
            frame = frame.transpose(1, 2, 0)
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        frames_np.append(frame)

    create_video(frames_np, output_path, fps, size)