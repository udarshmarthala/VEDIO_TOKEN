from typing import List, Optional

import numpy as np

from .visual_embedder import VISUAL_DIM
from .text_embedder import TEXT_DIM

FUSED_DIM = VISUAL_DIM + TEXT_DIM  # 896


def fuse_embeddings(visual: Optional[List[float]], text: List[float]) -> List[float]:
    if visual is None:
        visual = [0.0] * VISUAL_DIM
    v = np.array(visual, dtype=np.float32)
    t = np.array(text, dtype=np.float32)
    fused = np.concatenate([v, t])
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused.tolist()
