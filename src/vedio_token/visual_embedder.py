import logging
from typing import List, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

VISUAL_DIM = 512
_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _model, _processor


def embed_frame(frame_path: str) -> Optional[List[float]]:
    try:
        model, processor = _get_model()
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(_device)
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            # output is a BaseModelOutputWithPooling, extract the pooler_output
            features = output.pooler_output if hasattr(output, 'pooler_output') else output
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()
    except Exception as e:
        logging.warning(f"CLIP failed for {frame_path}: {e}")
        return None
