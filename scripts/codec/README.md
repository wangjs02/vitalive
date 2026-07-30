# Codec Frameworks

`codec/` owns concrete encoder-decoder and tokenizer frameworks used by active
assembled models.

Retained active codecs:

- `cnn_token.py`: CNNToken encoder and decoder.
- `vit.py`: ViT encoder and decoder.
- `deepmind.py`: external-style DeepMind VQ-VAE encoder and decoder adapter.

Low-level reusable blocks belong in `blocks/`. Dataset-specific assembled
models, including `VitalDBVQVAE`, belong in `model/`. Dataset loading belongs
in `data/`. Training mechanics belong in `train/`. Diagnostics belong in
`eval/`.

Every `.py` file in this folder requires a same-name `.md` document.
