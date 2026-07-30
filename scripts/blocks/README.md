# Block Components

`blocks/` owns reusable low-level neural network components.

Examples:

- residual, MLP, transformer, and quantizer blocks
- projection heads
- pooling layers
- positional encodings
- VQ quantizers

Complete encoder-decoder frameworks belong in `codec/`.

Dataset-specific or project-specific assembled models belong in `model/`.

Every `.py` file in this folder requires a same-name `.md` document.
