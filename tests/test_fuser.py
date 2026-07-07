import math
from vedio_token.fuser import fuse_embeddings, FUSED_DIM, VISUAL_DIM, TEXT_DIM


def test_fuse_returns_correct_dim():
    visual = [0.1] * VISUAL_DIM
    text = [0.2] * TEXT_DIM
    result = fuse_embeddings(visual, text)
    assert len(result) == FUSED_DIM


def test_fuse_is_normalized():
    visual = [1.0] + [0.0] * (VISUAL_DIM - 1)
    text = [1.0] + [0.0] * (TEXT_DIM - 1)
    result = fuse_embeddings(visual, text)
    norm = math.sqrt(sum(x ** 2 for x in result))
    assert abs(norm - 1.0) < 1e-5


def test_fuse_none_visual_uses_zeros():
    text = [0.5] * TEXT_DIM
    result = fuse_embeddings(None, text)
    assert len(result) == FUSED_DIM
    assert result[:VISUAL_DIM] == [0.0] * VISUAL_DIM
