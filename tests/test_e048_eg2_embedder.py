import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


class FakeSession:
    def __init__(self, return_value):
        self.return_value = return_value
        self.last_feeds = None

    def run(self, names, feeds):
        self.last_feeds = feeds
        return [self.return_value]


def test_text_only_feeds_empty_fill():
    from benchmarks.e048.eg2_embedder import build_text_model_feeds

    proc_out = {
        "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
    }
    fake_vision = FakeSession(np.zeros((0, 512), dtype=np.float32))
    feeds = build_text_model_feeds(proc_out, fake_vision)

    assert feeds["input_ids"].dtype == np.int64
    assert feeds["attention_mask"].dtype == np.int64
    assert feeds["image_features"].shape == (0, 512)
    assert feeds["image_features"].dtype == np.float32
    assert feeds["video_features"].shape == (0, 512)
    assert feeds["video_features"].dtype == np.float32
    assert feeds["audio_features"].shape == (0, 512)
    assert feeds["audio_features"].dtype == np.float32


def test_image_features_routed():
    from benchmarks.e048.eg2_embedder import build_text_model_feeds, IMAGE_TOKEN_ID

    proc_out = {
        "input_ids": np.array([[IMAGE_TOKEN_ID, 5, IMAGE_TOKEN_ID]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        "pixel_values": np.zeros((1, 18, 768), dtype=np.float32),
        "image_position_ids": np.zeros((1, 18, 2), dtype=np.int64),
    }
    fake_vision = FakeSession(np.zeros((2, 512), dtype=np.float32))
    feeds = build_text_model_feeds(proc_out, fake_vision)

    assert feeds["image_features"].shape == (2, 512)
    assert np.array_equal(feeds["image_features"], fake_vision.return_value)
    assert "pixel_position_ids" in fake_vision.last_feeds
    assert "image_position_ids" not in fake_vision.last_feeds
    assert fake_vision.last_feeds["pixel_position_ids"].shape == (1, 18, 2)


def test_placeholder_mismatch_raises():
    from benchmarks.e048.eg2_embedder import build_text_model_feeds, IMAGE_TOKEN_ID

    proc_out = {
        "input_ids": np.array([[IMAGE_TOKEN_ID, 5, IMAGE_TOKEN_ID]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        "pixel_values": np.zeros((1, 18, 768), dtype=np.float32),
        "image_position_ids": np.zeros((1, 18, 2), dtype=np.int64),
    }
    fake_vision = FakeSession(np.zeros((3, 512), dtype=np.float32))

    with pytest.raises(ValueError) as exc:
        build_text_model_feeds(proc_out, fake_vision)
    assert "3" in str(exc.value)
    assert "2" in str(exc.value)


def test_pixel_values_without_vision_session_raises():
    from benchmarks.e048.eg2_embedder import build_text_model_feeds, IMAGE_TOKEN_ID

    proc_out = {
        "input_ids": np.array([[IMAGE_TOKEN_ID, 5]], dtype=np.int64),
        "attention_mask": np.array([[1, 1]], dtype=np.int64),
        "pixel_values": np.zeros((1, 18, 768), dtype=np.float32),
        "image_position_ids": np.zeros((1, 18, 2), dtype=np.int64),
    }

    with pytest.raises(ValueError):
        build_text_model_feeds(proc_out, None)


@pytest.mark.skipif(
    not __import__("os").environ.get("EG2_ONNX_DIR"),
    reason="EG2_ONNX_DIR not set",
)
def test_golden_readme_image():
    import os
    import urllib.request
    from io import BytesIO
    from PIL import Image
    from benchmarks.e048.eg2_embedder import EG2Embedder

    model_dir = os.environ["EG2_ONNX_DIR"]
    embedder = EG2Embedder(model_dir)

    url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"
    with urllib.request.urlopen(url, timeout=30) as resp:
        img = Image.open(BytesIO(resp.read())).convert("RGB")

    image_emb = embedder.embed_images([img])[0]

    queries = [
        "cats sleeping on a couch",
        "a president's speech about serving your country",
        "a turtle swimming in the ocean",
    ]
    query_embs = embedder.embed_texts(queries, kind="query")

    scores = query_embs @ image_emb
    assert np.argmax(scores) == 0

    docs = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]
    doc_embs = embedder.embed_texts(docs, kind="document")
    query_emb = embedder.embed_texts(["Which planet is known as the Red Planet?"], kind="query")[0]
    doc_scores = doc_embs @ query_emb
    assert np.argmax(doc_scores) == 1
