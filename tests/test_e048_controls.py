import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


class FakeOpenClip:
    __version__ = "2.24.0"

    class FakeModel:
        def __init__(self):
            self.eval_called = False
            self.device = "cpu"

        def eval(self):
            self.eval_called = True

        def to(self, device):
            self.device = device
            return self

        def encode_image(self, x):
            return np.random.randn(x.shape[0], 512).astype(np.float32)

        def encode_text(self, x):
            return np.random.randn(x.shape[0], 512).astype(np.float32)

    class FakePreprocess:
        def __call__(self, img):
            return np.random.randn(3, 224, 224).astype(np.float32)

    class FakeTokenizer:
        def __call__(self, texts):
            return np.random.randint(0, 1000, (len(texts), 77), dtype=np.int64)

    @staticmethod
    def create_model_and_transforms(model_name, pretrained):
        return FakeOpenClip.FakeModel(), None, FakeOpenClip.FakePreprocess()

    @staticmethod
    def get_tokenizer(model_name):
        return FakeOpenClip.FakeTokenizer()

    class pretrained:
        @staticmethod
        def download_pretrained(url):
            return "/fake/path/openclip_model.bin"


class FakeTorch:
    __version__ = "2.3.0"
    cuda = type("cuda", (), {"is_available": staticmethod(lambda: False)})()

    @staticmethod
    def no_grad():
        return type("NoGrad", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

    @staticmethod
    def tensor(x):
        return x

    @staticmethod
    def from_numpy(x):
        return x

    @staticmethod
    def stack(x):
        return np.stack(x)


class FakePIL:
    class Image:
        @staticmethod
        def open(path):
            return FakePIL.FakeImage()

    class FakeImage:
        def convert(self, mode):
            return self


class FakeSoundFile:
    @staticmethod
    def read(path, dtype="float32", always_2d=True):
        return np.random.randn(2, 48000).astype(np.float32), 48000


class FakeTorchaudio:
    class functional:
        @staticmethod
        def resample(waveform, orig_freq, new_freq):
            FakeTorchaudio.functional.resample_calls.append((orig_freq, new_freq))
            return waveform

    functional.resample_calls = []


class FakeHuggingFaceHub:
    @staticmethod
    def snapshot_download(repo_id, local_files_only=True):
        return "/fake/hf/cache"


class FakeTransformers:
    class ClapModel:
        @staticmethod
        def from_pretrained(model_id):
            model = FakeTransformers.FakeClapModel()
            model.config = FakeTransformers.FakeConfig()
            return model

    class FakeClapModel:
        def __init__(self):
            self.eval_called = False
            self.device = "cpu"

        def eval(self):
            self.eval_called = True

        def to(self, device):
            self.device = device
            return self

        def get_audio_features(self, **inputs):
            batch_size = inputs["input_features"].shape[0]
            return np.random.randn(batch_size, 512).astype(np.float32)

        def get_text_features(self, **inputs):
            batch_size = inputs["input_ids"].shape[0]
            return np.random.randn(batch_size, 512).astype(np.float32)

    class ClapProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def __init__(self):
            self.call_signature = "audios"

        def __call__(self, **kwargs):
            if "audios" in kwargs or "audio" in kwargs:
                batch_size = len(kwargs.get("audios", kwargs.get("audio", [])))
                return {
                    "input_features": np.random.randn(batch_size, 80, 3000).astype(np.float32),
                    "attention_mask": np.ones((batch_size, 3000), dtype=np.int64),
                }
            else:
                batch_size = len(kwargs.get("text", []))
                return {
                    "input_ids": np.random.randint(0, 1000, (batch_size, 77), dtype=np.int64),
                    "attention_mask": np.ones((batch_size, 77), dtype=np.int64),
                }

    class FakeConfig:
        _commit_hash = "abc123def456"


def test_openclip_image_embedder_name_and_modality(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "PIL", FakePIL())

    from benchmarks.e048.controls import OpenClipImageEmbedder

    embedder = OpenClipImageEmbedder(device="cpu")
    assert embedder.name == "openclip-vitb32-laion2b"
    assert embedder.modality == "image"


def test_openclip_image_embed_queries_shape_and_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "PIL", FakePIL())

    from benchmarks.e048.controls import OpenClipImageEmbedder

    embedder = OpenClipImageEmbedder(device="cpu")
    texts = ["a cat", "a dog"]
    result = embedder.embed_queries(texts)

    assert result.shape == (2, 512)
    assert result.dtype == np.float32
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_openclip_image_embed_items_shape_and_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "PIL", FakePIL())

    from benchmarks.e048.controls import OpenClipImageEmbedder

    embedder = OpenClipImageEmbedder(device="cpu")
    paths = ["/fake/img1.jpg", "/fake/img2.jpg", "/fake/img3.jpg"]
    result = embedder.embed_items(paths)

    assert result.shape == (3, 512)
    assert result.dtype == np.float32
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_openclip_image_provenance_keys(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "PIL", FakePIL())

    from benchmarks.e048.controls import OpenClipImageEmbedder

    embedder = OpenClipImageEmbedder(device="cpu")
    prov = embedder.provenance()

    required_keys = {"model_id", "revision", "model_files", "runtime", "runtime_version", "execution_provider", "dtype"}
    assert set(prov.keys()) == required_keys
    assert prov["model_id"] == "ViT-B-32/laion2b_s34b_b79k"
    assert prov["runtime"] == "torch"
    assert prov["dtype"] == "fp32"
    assert prov["execution_provider"] in ("cpu", "cuda")
    assert isinstance(prov["model_files"], dict)
    assert len(prov["model_files"]) > 0


def test_clap_audio_embedder_name_and_modality(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile())
    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio())
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHuggingFaceHub())

    from benchmarks.e048.controls import ClapAudioEmbedder

    embedder = ClapAudioEmbedder(device="cpu")
    assert embedder.name == "laion-clap-htsat-unfused"
    assert embedder.modality == "audio"


def test_clap_audio_embed_queries_shape_and_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile())
    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio())
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHuggingFaceHub())

    from benchmarks.e048.controls import ClapAudioEmbedder

    embedder = ClapAudioEmbedder(device="cpu")
    texts = ["a cat meowing", "a dog barking"]
    result = embedder.embed_queries(texts)

    assert result.shape == (2, 512)
    assert result.dtype == np.float32
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_clap_audio_embed_items_shape_and_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile())
    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio())
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHuggingFaceHub())

    from benchmarks.e048.controls import ClapAudioEmbedder

    embedder = ClapAudioEmbedder(device="cpu")
    paths = ["/fake/audio1.wav", "/fake/audio2.wav"]
    result = embedder.embed_items(paths)

    assert result.shape == (2, 512)
    assert result.dtype == np.float32
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_clap_audio_stereo_to_mono_and_resample(monkeypatch):
    FakeTorchaudio.functional.resample_calls = []

    class StereoSoundFile:
        @staticmethod
        def read(path, dtype="float32", always_2d=True):
            if "stereo_44k" in path:
                return np.random.randn(2, 44100).astype(np.float32), 44100
            elif "stereo_48k" in path:
                return np.random.randn(2, 48000).astype(np.float32), 48000
            return np.random.randn(1, 48000).astype(np.float32), 48000

    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "soundfile", StereoSoundFile())
    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio())
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHuggingFaceHub())

    from benchmarks.e048.controls import ClapAudioEmbedder

    embedder = ClapAudioEmbedder(device="cpu")

    embedder.embed_items(["/fake/stereo_44k.wav"])
    assert len(FakeTorchaudio.functional.resample_calls) == 1
    assert FakeTorchaudio.functional.resample_calls[0] == (44100, 48000)

    FakeTorchaudio.functional.resample_calls = []
    embedder.embed_items(["/fake/stereo_48k.wav"])
    assert len(FakeTorchaudio.functional.resample_calls) == 0


def test_clap_audio_provenance_keys(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile())
    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio())
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHuggingFaceHub())

    from benchmarks.e048.controls import ClapAudioEmbedder

    embedder = ClapAudioEmbedder(device="cpu")
    prov = embedder.provenance()

    required_keys = {"model_id", "revision", "model_files", "runtime", "runtime_version", "execution_provider", "dtype"}
    assert set(prov.keys()) == required_keys
    assert prov["model_id"] == "laion/clap-htsat-unfused"
    assert prov["runtime"] == "torch"
    assert prov["dtype"] == "fp32"
    assert prov["execution_provider"] in ("cpu", "cuda")
    assert isinstance(prov["model_files"], dict)
    assert len(prov["model_files"]) > 0


def test_factory_functions(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "PIL", FakePIL())
    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers())
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile())
    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchaudio())
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHuggingFaceHub())

    from benchmarks.e048.controls import openclip_image, clap_audio

    img_embedder = openclip_image(device="cpu")
    assert img_embedder.name == "openclip-vitb32-laion2b"
    assert img_embedder.modality == "image"

    audio_embedder = clap_audio(device="cpu")
    assert audio_embedder.name == "laion-clap-htsat-unfused"
    assert audio_embedder.modality == "audio"