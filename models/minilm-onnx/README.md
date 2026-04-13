---
library_name: light-embed
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity

---

# onnx-models/all-MiniLM-L6-v2-onnx

This is the ONNX-ported version of the [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for generating text embeddings.

## Model details
- Embedding dimension: 384
- Max sequence length: 256
- File size on disk:  0.08 GB
- Modules incorporated in the onnx: Transformer, Pooling, Normalize

<!--- Describe your model here -->

## Usage

Using this model becomes easy when you have [light-embed](https://pypi.org/project/light-embed/) installed:

```
pip install -U light-embed
```

Then you can use the model by specifying the *original model name* like this:

```python
from light_embed import TextEmbedding
sentences = [
	"This is an example sentence",
	"Each sentence is converted"
]

model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
```

or by specifying the *onnx model name* like this:

```python
from light_embed import TextEmbedding
sentences = [
	"This is an example sentence",
	"Each sentence is converted"
]

model = TextEmbedding('onnx-models/all-MiniLM-L6-v2-onnx')
embeddings = model.encode(sentences)
print(embeddings)
```

## Citing & Authors

Binh Nguyen / binhcode25@gmail.com