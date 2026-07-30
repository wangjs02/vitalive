from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import torch
from torch import nn

from blocks.quantizer import (
    SequenceEMAQuantize,
    SequenceEMAQuantizerConfig,
    codebook_perplexity,
)
from codec.cnn_token import CNNTokenConfig, CNNTokenDecoder, CNNTokenEncoder
from codec.vit import ViTConfig, ViTDecoder, ViTEncoder


@dataclass
class VQVAEConfig:
    """Configuration for the VitalDB VQ-VAE wrapper."""

    enc_dec: str = "vit"
    codec: CNNTokenConfig | ViTConfig = field(default_factory=ViTConfig)
    quantizer: SequenceEMAQuantizerConfig = field(default_factory=SequenceEMAQuantizerConfig)
    use_quantizer: bool = True


class VitalDBVQVAE(nn.Module):
    """VitalDB VQ-VAE wrapper with encoder, quantizer, and decoder modules."""

    def __init__(
        self,
        config: VQVAEConfig,
    ) -> None:
        super().__init__()

        self.enc_dec = config.enc_dec
        self.use_quantizer = bool(config.use_quantizer)
        self.codec_config = config.codec.to_kwargs()
        self.quantizer_config = config.quantizer.to_kwargs()

        self.time_length = int(self.codec_config["time_length"])
        self.embedding_dim = int(self.codec_config["embedding_dim"])

        self.encoder = self.build_encoder()
        self.quantizer = self.build_quantizer()
        self.decoder = self.build_decoder()

    def build_encoder(self) -> nn.Module:
        """Build the selected VitalDB encoder."""
        if self.enc_dec == "cnn_tokens":
            return CNNTokenEncoder(**self.codec_config)
        if self.enc_dec == "vit":
            return ViTEncoder(**self.codec_config)
        raise RuntimeError(f"Unsupported enc_dec={self.enc_dec!r}")

    def build_quantizer(self) -> nn.Module:
        """Build the selected VitalDB quantizer."""
        return SequenceEMAQuantize(**self.quantizer_config)

    def build_decoder(self) -> nn.Module:
        """Build the selected VitalDB decoder."""
        if self.enc_dec == "cnn_tokens":
            return CNNTokenDecoder(**self.codec_config)
        if self.enc_dec == "vit":
            return ViTDecoder(**self.codec_config)
        raise RuntimeError(f"Unsupported enc_dec={self.enc_dec!r}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run VQ-VAE reconstruction and return training diagnostics."""
        z = self.encode(x)
        z_code = self.embedding_to_code(z)
        if self.use_quantizer:
            z_code_q, vq_loss, indices = self.quantizer(z_code)
            perplexity, cluster_use = codebook_perplexity(
                indices,
                int(self.quantizer.n_embed),
            )
        else:
            z_code_q = z_code
            vq_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            indices = torch.empty(0, device=x.device, dtype=torch.long)
            perplexity = torch.zeros((), device=x.device, dtype=x.dtype)
            cluster_use = torch.zeros((), device=x.device, dtype=x.dtype)
        z_q = self.code_to_embedding(z_code_q)
        x_hat = self.decode(z_q)
        recon_loss = (x_hat - x).pow(2).mean()
        loss = recon_loss + vq_loss
        return {
            "x_recon": x_hat,
            "z": z,
            "z_code": z_code,
            "z_code_q": z_code_q,
            "z_q": z_q,
            "indices": indices,
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "cluster_use": cluster_use,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `[B, vital_channels, time_length]` into codec tokens."""
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize codec embeddings and reshape them back for the decoder."""
        if not self.use_quantizer:
            return z
        z_q, _, _ = self.quantizer(z)
        return z_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode codec tokens into `[B, vital_channels, time_length]`."""
        x = self.decoder(z)
        return x[:, :, : self.time_length]

    def embedding_to_code(self, z: torch.Tensor) -> torch.Tensor:
        """Reshape codec embeddings into quantizer code vectors."""
        code_dim = self.quantizer_code_dim
        if z.size(-1) == code_dim:
            return z
        if z.size(-1) % code_dim != 0:
            raise ValueError(
                "Embedding dimension must be divisible by quantizer code dim "
                f"({z.size(-1)=}, {code_dim=})."
            )

        batch_size, embedding_length, embedding_dim = z.shape
        codes_per_embedding = embedding_dim // code_dim
        return z.reshape(
            batch_size,
            embedding_length,
            codes_per_embedding,
            code_dim,
        ).reshape(batch_size, embedding_length * codes_per_embedding, code_dim)

    def code_to_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Reshape quantizer code vectors back into codec embeddings."""
        embedding_dim = int(self.embedding_dim)
        if z.size(-1) == embedding_dim:
            return z

        code_dim = self.quantizer_code_dim
        if z.size(-1) != code_dim:
            raise ValueError(
                "Quantizer code dim must match the configured code dim before "
                f"rebuilding embeddings ({z.size(-1)=}, {code_dim=})."
            )
        if embedding_dim % code_dim != 0:
            raise ValueError(
                "Embedding dimension must be divisible by quantizer code dim "
                f"({embedding_dim=}, {code_dim=})."
            )

        codes_per_embedding = embedding_dim // code_dim
        batch_size, flat_code_length, _ = z.shape
        if flat_code_length % codes_per_embedding != 0:
            raise ValueError(
                "Flat code length must be divisible by codes per embedding "
                f"({flat_code_length=}, {codes_per_embedding=})."
            )

        embedding_length = flat_code_length // codes_per_embedding
        return z.reshape(
            batch_size,
            embedding_length,
            codes_per_embedding,
            code_dim,
        ).reshape(batch_size, embedding_length, embedding_dim)

    @property
    def quantizer_code_dim(self) -> int:
        """Return the last-dimension size used by the sequence quantizer."""
        return int(self.quantizer.embedding_dim)


__all__ = [
    "VQVAEConfig",
    "VitalDBVQVAE",
]

def main():
    """Run a minimal shape smoke test for the VitalDB VQ-VAE wrapper."""
    torch.manual_seed(7)

    config = VQVAEConfig(
        enc_dec="vit",
        codec=ViTConfig(
            input_dim=4,
            hidden_dim=16,
            patch_size=10,
            embedding_dim=4*8,
            time_length=160,
            token_length=16,
            transformer_layers=1,
            transformer_heads=2,
        ),
        quantizer=SequenceEMAQuantizerConfig(
            n_embed=16,
            embedding_dim=8,
        ),
        use_quantizer=True,
    )
    model = VitalDBVQVAE(config=config)
    model.eval()

    x = torch.randn(2, 4, 160)
    with torch.no_grad():
        z = model.encode(x)
        z_code = model.embedding_to_code(z)
        z_code_q = model.quantize(z_code)
        z_q = model.code_to_embedding(z_code_q)
        x_hat = model.decode(z_q)

    assert z.shape == (2, 16, 8*4), f"Unexpected encoder shape: {tuple(z.shape)}"
    assert z_code.shape == (2, 16*4, 8), f"Unexpected code shape: {tuple(z_code.shape)}"
    assert z.shape == z_q.shape, f"Unexpected embedding shape: {tuple(z_q.shape)}"
    assert z_code.shape == z_code_q.shape, f"Unexpected quantized shape: {tuple(z_code.shape)}"
    assert x_hat.shape == x.shape, f"Unexpected reconstruction shape: {tuple(x_hat.shape)}"

    print("VitalDBVQVAE main test passed")
    print(f"x:       {tuple(x.shape)}")
    print(f"z:       {tuple(z.shape)}")
    print(f"z_code: {tuple(z_code.shape)}")
    print(f"z_code_q: {tuple(z_code_q.shape)}")
    print(f"z_q:     {tuple(z_q.shape)}")
    print(f"x_hat:   {tuple(x_hat.shape)}")


if __name__ == "__main__":
    main()
