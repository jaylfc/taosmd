import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_ID = "onnx-community/embeddinggemma-2-ONNX"
REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
IMAGE_TOKEN_ID = 258880
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "


def build_text_model_feeds(proc_out: dict, vision_session) -> dict:
    feeds = {
        "input_ids": proc_out["input_ids"].astype(np.int64),
        "attention_mask": proc_out["attention_mask"].astype(np.int64),
    }

    if "pixel_values" in proc_out:
        if vision_session is None:
            raise ValueError("pixel_values present but vision_session is None")
        pixel_values = proc_out["pixel_values"].astype(np.float32)
        pixel_position_ids = proc_out["image_position_ids"].astype(np.int64)
        image_features = vision_session.run(
            ["image_features"],
            {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids},
        )[0]
        n_placeholders = int((feeds["input_ids"] == IMAGE_TOKEN_ID).sum())
        if image_features.shape[0] != n_placeholders:
            raise ValueError(
                f"image_features rows {image_features.shape[0]} != placeholder count {n_placeholders}"
            )
        feeds["image_features"] = image_features.astype(np.float32)
    else:
        feeds["image_features"] = np.zeros((0, 512), dtype=np.float32)

    feeds["video_features"] = np.zeros((0, 512), dtype=np.float32)
    feeds["audio_features"] = np.zeros((0, 512), dtype=np.float32)

    return feeds


class EG2Embedder:
    def __init__(
        self,
        model_dir: str,
        variant: str = "",
        providers: list[str] | None = None,
        load_vision: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.variant = variant
        self.providers = providers or ["CPUExecutionProvider"]
        self.load_vision = load_vision

        import onnxruntime as ort

        model_path = self.model_dir / "onnx" / f"model{variant}.onnx"
        self.text_session = ort.InferenceSession(str(model_path), providers=self.providers)

        if load_vision:
            vision_path = self.model_dir / "onnx" / f"vision_encoder{variant}.onnx"
            self.vision_session = ort.InferenceSession(str(vision_path), providers=self.providers)
        else:
            self.vision_session = None

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(self.model_dir), revision=REVISION)

    def embed_texts(self, texts: list[str], kind: str = "query") -> np.ndarray:
        if kind == "query":
            prefixed = [QUERY_PREFIX + t for t in texts]
        elif kind == "document":
            prefixed = [DOC_PREFIX + t for t in texts]
        else:
            raise ValueError(f"Unknown kind: {kind}")

        proc_out = self.processor(text=prefixed, return_tensors="np")
        feeds = build_text_model_feeds(proc_out, self.vision_session)
        outputs = self.text_session.run(["sentence_embedding"], feeds)[0]
        norms = np.linalg.norm(outputs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (outputs / norms).astype(np.float32)

    def embed_images(self, images: list) -> np.ndarray:
        embeddings = []
        for img in images:
            proc_out = self.processor(text=None, images=[[img]], return_tensors="np")
            feeds = build_text_model_feeds(proc_out, self.vision_session)
            outputs = self.text_session.run(["sentence_embedding"], feeds)[0]
            embeddings.append(outputs[0])
        arr = np.stack(embeddings, axis=0)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (arr / norms).astype(np.float32)

    def model_files(self) -> dict[str, str]:
        files = {}
        onnx_dir = self.model_dir / "onnx"
        for f in onnx_dir.glob("*.onnx*"):
            if f.name.startswith(f"model{self.variant}") or f.name.startswith(f"vision_encoder{self.variant}"):
                rel = f.relative_to(self.model_dir)
                h = hashlib.sha256()
                with f.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                files[str(rel)] = h.hexdigest()
        return files


class MediaEmbedder(Protocol):
    name: str
    modality: str

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def embed_items(self, paths: list[str]) -> np.ndarray: ...
    def provenance(self) -> dict: ...


class EG2ImageEmbedder:
    name = "eg2-fp32-image"
    modality = "image"

    def __init__(
        self,
        model_dir: str,
        variant: str = "",
        providers: list[str] | None = None,
    ):
        self.embedder = EG2Embedder(model_dir, variant=variant, providers=providers, load_vision=True)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed_texts(texts, kind="query")

    def embed_items(self, paths: list[str]) -> np.ndarray:
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in paths]
        return self.embedder.embed_images(images)

    def provenance(self) -> dict:
        import onnxruntime
        exec_provider = self.embedder.text_session.get_providers()[0]
        dtype_map = {
            "": "fp32",
            "_fp16": "fp16",
            "_quantized": "int8",
            "_q4": "int4",
            "_q4f16": "int4fp16",
        }
        return {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "model_files": self.embedder.model_files(),
            "runtime": "onnxruntime",
            "runtime_version": onnxruntime.__version__,
            "execution_provider": exec_provider,
            "dtype": dtype_map.get(self.embedder.variant, "fp32"),
        }