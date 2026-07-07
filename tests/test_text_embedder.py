import math
from vedio_token.text_embedder import embed_text, TEXT_DIM


def test_embed_text_returns_list():
    result = embed_text("a dog running in a park")
    assert isinstance(result, list)
    assert len(result) == TEXT_DIM


def test_embed_text_normalized():
    result = embed_text("a dog running in a park")
    norm = math.sqrt(sum(x ** 2 for x in result))
    assert abs(norm - 1.0) < 1e-3


def test_embed_text_empty_string():
    result = embed_text("")
    assert isinstance(result, list)
    assert len(result) == TEXT_DIM
