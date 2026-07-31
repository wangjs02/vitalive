# Pretrain Pipeline For VitalDB

import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Direct execution puts this directory first and would make `import vitaldb`
# resolve this pipeline file instead of the third-party VitalDB package.
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != PIPELINE_DIR
]
ROOT_DIR = '/home/junshi/'

import numpy as np
import torch
from torch.utils.data import DataLoader

from blocks.quantizer import SequenceEMAQuantizerConfig
from codec.vit import ViTConfig
from data.vitaldb import (
    VitalDBData,
    VitalDBDataset,
    ComposeVitalDB,
    ResampleVitalDB,
    NormalizeVitalDB,
    RandomNoiseVitalDB,
)
from eval.vitaldb import eval_history, eval_recon
from model.vitaldb import VQVAEConfig, VitalDBVQVAE
from train.epoch import run_vqvae_epoch
from train.optimizer import build_adamw_optimizer
from utils.checkpoint import Checkpoint


def train_val_test_split(
    sample_ids: Sequence[tuple[int, int]],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
    train_case_limit: int | None = None,
    val_case_limit: int | None = None,
    test_case_limit: int | None = None,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Split sample IDs into deterministic, case-disjoint train/val/test sets.

    The test fraction is the remainder after `train_fraction + val_fraction`.
    Optional limits are applied to case IDs after the full split is created.
    """
    if train_fraction <= 0 or val_fraction <= 0:
        raise ValueError("train_fraction and val_fraction must be positive.")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1.")
    for name, limit in (
        ("train_case_limit", train_case_limit),
        ("val_case_limit", val_case_limit),
        ("test_case_limit", test_case_limit),
    ):
        if limit is not None and limit <= 0:
            raise ValueError(f"{name} must be positive when provided.")

    clean_case_ids = np.asarray(
        sorted({int(case_id) for case_id, _segment_id in sample_ids}),
        dtype=np.int64,
    )
    if len(clean_case_ids) < 3:
        raise ValueError("At least three clean cases are required for train/val/test.")

    rng = np.random.default_rng(seed)
    rng.shuffle(clean_case_ids)

    train_count = min(
        max(int(len(clean_case_ids) * train_fraction), 1),
        len(clean_case_ids) - 2,
    )
    remaining_count = len(clean_case_ids) - train_count
    val_count = min(
        max(int(len(clean_case_ids) * val_fraction), 1),
        remaining_count - 1,
    )
    train_case_ids = clean_case_ids[:train_count]
    val_case_ids = clean_case_ids[train_count : train_count + val_count]
    test_case_ids = clean_case_ids[train_count + val_count :]

    if train_case_limit is not None:
        train_case_ids = train_case_ids[:train_case_limit]
    if val_case_limit is not None:
        val_case_ids = val_case_ids[:val_case_limit]
    if test_case_limit is not None:
        test_case_ids = test_case_ids[:test_case_limit]

    case_sets = {
        "train": set(train_case_ids.tolist()),
        "val": set(val_case_ids.tolist()),
        "test": set(test_case_ids.tolist()),
    }
    if any(not case_ids for case_ids in case_sets.values()):
        raise ValueError("Train, validation, and test must each contain a case ID.")
    if (
        not case_sets["train"].isdisjoint(case_sets["val"])
        or not case_sets["train"].isdisjoint(case_sets["test"])
        or not case_sets["val"].isdisjoint(case_sets["test"])
    ):
        raise RuntimeError("Train, validation, and test case IDs overlap.")

    split_sample_ids = []
    for split_name in ("train", "val", "test"):
        selected_cases = case_sets[split_name]
        selected_samples = [
            sample_id for sample_id in sample_ids if sample_id[0] in selected_cases
        ]
        if not selected_samples:
            raise ValueError(f"{split_name} split contains no sample IDs.")
        split_sample_ids.append(selected_samples)
    return split_sample_ids[0], split_sample_ids[1], split_sample_ids[2]


class VitalDBCheckpoint(Checkpoint):
    """VitalDB-specific checkpoint payload and registry behavior."""

    @staticmethod
    def create_payload(
        *,
        model: torch.nn.Module,
        model_config: VQVAEConfig,
        optimizer: torch.optim.Optimizer,
        history: Sequence[dict[str, Any]],
        test_metrics: dict[str, Any],
        vitaldb_data: VitalDBData,
        train_dataset: VitalDBDataset,
        fitted_transforms: ComposeVitalDB,
        vital_signs: Sequence[str],
        train_sample_ids: Sequence[tuple[int, int]],
        val_sample_ids: Sequence[tuple[int, int]],
        test_sample_ids: Sequence[tuple[int, int]],
        best_epoch: int,
        best_val_recon_loss: float,
        best_val_metrics: dict[str, Any],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        patience: int,
        min_delta: float,
        evaluation_artifacts: dict[str, Path],
        input_frequency_hz: float = 0.5,
        target_frequency_hz: float = 1.0,
        split_seed: int = 42,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
    ) -> dict[str, Any]:
        """Create the serializable VitalDB pretrain checkpoint payload."""
        normalize = fitted_transforms.get_transform(NormalizeVitalDB)
        return {
            "schema_version": 1,
            "data": {
                "source": {
                    "data_dir": str(vitaldb_data.data_dir),
                    "metadata_dir": str(vitaldb_data.metadata_dir),
                    "clean_dir": str(vitaldb_data.clean_dir),
                },
                "vital_signs": list(vital_signs),
                "sample": {
                    "time_length": train_dataset.time_length,
                    "stored_interval_sec": vitaldb_data.interval_sec,
                    "model_interval_sec": 1.0 / target_frequency_hz,
                },
                "preprocessing": {
                    "resample": {
                        "input_frequency_hz": input_frequency_hz,
                        "target_frequency_hz": target_frequency_hz,
                    },
                    "normalize": normalize.state_dict(),
                },
                "split": {
                    "seed": split_seed,
                    "train_fraction": train_fraction,
                    "val_fraction": val_fraction,
                    "train_case_ids": sorted(
                        {case_id for case_id, _ in train_sample_ids}
                    ),
                    "val_case_ids": sorted(
                        {case_id for case_id, _ in val_sample_ids}
                    ),
                    "test_case_ids": sorted(
                        {case_id for case_id, _ in test_sample_ids}
                    ),
                },
            },
            "train": {
                "best_epoch": best_epoch,
                "epochs_requested": epochs,
                "batch_size": batch_size,
                "optimizer": {
                    "name": type(optimizer).__name__,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
                "early_stopping": {
                    "monitor": "val.recon_loss",
                    "mode": "min",
                    "patience": patience,
                    "min_delta": min_delta,
                },
                "history": list(history),
            },
            "model": {
                "name": type(model).__name__,
                "config": asdict(model_config),
                "state_dict": model.state_dict(),
            },
            "evaluation": {
                "selection": {
                    "best_epoch": best_epoch,
                    "best_val_recon_loss": best_val_recon_loss,
                    "best_val_metrics": dict(best_val_metrics),
                },
                "test": dict(test_metrics),
                "artifacts": {
                    artifact_name: path.name
                    for artifact_name, path in evaluation_artifacts.items()
                },
            },
        }

    def update_registry(
        self,
        *,
        name: str,
        model_config: VQVAEConfig,
        vital_signs: Sequence[str],
        target_frequency_hz: float,
        best_epoch: int,
        best_val_recon_loss: float,
        test_metrics: dict[str, Any],
        evaluation_artifacts: dict[str, Path],
    ) -> Path:
        """Append the VitalDB-specific flat registry record."""
        return super().update_registry({
            "name": name,
            "enc_dec": model_config.enc_dec,
            "roles": list(vital_signs),
            "vital_channels": len(vital_signs),
            "time_length": int(model_config.codec.time_length),
            "interval_sec": 1.0 / target_frequency_hz,
            "patch_size": model_config.codec.patch_size,
            "token_length": model_config.codec.token_length,
            "embedding_dim": model_config.codec.embedding_dim,
            "code_dim": model_config.quantizer.embedding_dim,
            "codebook_size": model_config.quantizer.n_embed,
            "vit_layers": model_config.codec.transformer_layers,
            "best_epoch": best_epoch,
            "best_val_recon": best_val_recon_loss,
            "test_recon_loss": test_metrics["recon_loss"],
            "perplexity": test_metrics["perplexity"],
            "cluster_use": test_metrics["cluster_use"],
            "artifacts": {
                artifact_name: f"{self.id}/{path.name}"
                for artifact_name, path in evaluation_artifacts.items()
            },
        })


def pretrain(
    *,
    project_root: str | Path = ".",
    name: str = "vitaldb_vit_7x600_vqvae_rot",
    data_dir: str | Path = f"{ROOT_DIR}/VitalDB/raw",
    metadata_dir: str | Path = f"{ROOT_DIR}/data/VitalDB/metadata",
    clean_dir: str | Path = f"{ROOT_DIR}/data/VitalDB/processed/pretrain-7vitalsign-v1",
    vital_signs: Sequence[str] = ("HR", "SpO2", "RR", "BT", "SBP", "DBP", "MBP"),
    time_length: int = 600,
    input_frequency_hz: float = 0.5,
    target_frequency_hz: float = 1.0,
    normalize_method: str = "z_score",
    normalize_clip: bool = False,
    noise_std: float = 0.0,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    min_delta: float = 1e-3,
    device: str | torch.device = "cpu",
    split_seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    train_case_limit: int | None = None,
    val_case_limit: int | None = None,
    test_case_limit: int | None = None,
    patch_size: int = 10,
    embedding_dim: int = 128,
    code_dim: int = 8,
    codebook_size: int = 64,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    rotation_matching: bool = True,
    checkpoint_subdir: str = "pretrain/vit",
    clean_progress_every: int | None = 10,
    sanity_check: bool = True,
) -> dict[str, Any]:
    """Run VitalDB data preparation, pretraining, evaluation, and checkpointing."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if patience <= 0:
        raise ValueError("patience must be positive.")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    project_root = Path(project_root).resolve()
    data_dir = Path(data_dir)
    metadata_dir = Path(metadata_dir)
    clean_dir = Path(clean_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    if not metadata_dir.is_absolute():
        metadata_dir = project_root / metadata_dir
    if not clean_dir.is_absolute():
        clean_dir = project_root / clean_dir
    vital_signs = tuple(vital_signs)

    vitaldb_data = VitalDBData(
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        clean_dir=clean_dir,
        vital_signs=vital_signs,
    )
    vitaldb_data.clean(progress_every=clean_progress_every)

    train_sample_ids, val_sample_ids, test_sample_ids = train_val_test_split(
        vitaldb_data.ids,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=split_seed,
        train_case_limit=train_case_limit,
        val_case_limit=val_case_limit,
        test_case_limit=test_case_limit,
    )
    print(
        f"train cases={len({case_id for case_id, _ in train_sample_ids})}, "
        f"segments={len(train_sample_ids)} | "
        f"val cases={len({case_id for case_id, _ in val_sample_ids})}, "
        f"segments={len(val_sample_ids)} | "
        f"test cases={len({case_id for case_id, _ in test_sample_ids})}, "
        f"segments={len(test_sample_ids)}"
    )
    transforms = ComposeVitalDB(
        [
            ResampleVitalDB(
                input_frequency_hz=input_frequency_hz,
                target_frequency_hz=target_frequency_hz,
            ),
            NormalizeVitalDB(
                vital_signs=vital_signs,
                method=normalize_method,
                clip=normalize_clip,
            ),
        ]
    )
    train_dataset = VitalDBDataset(
        vitaldb_data,
        id_list=train_sample_ids,
        transforms=transforms,
        time_length=time_length,
    )
    fitted_transforms = train_dataset.fit_transforms()
    if noise_std > 0:
        train_dataset.transforms = ComposeVitalDB(
            [
                *fitted_transforms.transforms,
                RandomNoiseVitalDB(std=noise_std, seed=split_seed),
            ]
        )
    val_dataset = VitalDBDataset(
        vitaldb_data,
        id_list=val_sample_ids,
        transforms=fitted_transforms,
        time_length=time_length,
    )
    test_dataset = VitalDBDataset(
        vitaldb_data,
        id_list=test_sample_ids,
        transforms=fitted_transforms,
        time_length=time_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(device)
    model_time_length = int(round(time_length * target_frequency_hz))
    token_length = max(1, int(np.ceil(model_time_length / patch_size)))
    model_config = VQVAEConfig(
        enc_dec="vit",
        codec=ViTConfig(
            input_dim=len(vitaldb_data.vital_signs),
            time_length=model_time_length,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            token_length=token_length,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        ),
        quantizer=SequenceEMAQuantizerConfig(
            n_embed=codebook_size,
            embedding_dim=code_dim,
            rotation_matching=rotation_matching,
        ),
        use_quantizer=True,
    )
    model = VitalDBVQVAE(model_config).to(device)
    optimizer = build_adamw_optimizer(
        model,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if sanity_check:
        sample_batch = next(iter(train_loader))
        expected_shape = (len(vital_signs), model_time_length)
        if tuple(sample_batch.shape[1:]) != expected_shape:
            raise ValueError(
                "Dataset/model shape mismatch: "
                f"expected [B, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(sample_batch.shape)}."
            )
        if not torch.isfinite(sample_batch).all():
            raise ValueError("Training sanity-check batch contains non-finite values.")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"sanity check | batch={tuple(sample_batch.shape)} | "
            f"parameters={parameter_count:,}"
        )

    best_val_recon_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_val_metrics = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_vqvae_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = run_vqvae_epoch(
            model,
            val_loader,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train": dict(train_metrics),
                "val": dict(val_metrics),
            }
        )
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.3f} | "
            f"val loss {val_metrics['loss']:.3f} | "
            f"train perplexity {train_metrics['perplexity']:.3f} | "
            f"val perplexity {val_metrics['perplexity']:.3f}"
        )
        print(
            f"epoch {epoch:03d} | "
            f"train cluster use {train_metrics['cluster_use']:.3f} | "
            f"val cluster use {val_metrics['cluster_use']:.3f} | "
            f"train std ratio "
            f"{train_metrics['x_recon_std_mean'] / train_metrics['x_std_mean']:.3f} | "
            f"val std ratio "
            f"{val_metrics['x_recon_std_mean'] / val_metrics['x_std_mean']:.3f}"
        )

        val_recon_loss = val_metrics["recon_loss"]
        improved = (
            np.isfinite(val_recon_loss)
            and val_recon_loss < best_val_recon_loss - min_delta
        )
        if improved:
            best_val_recon_loss = val_recon_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_val_metrics = dict(val_metrics)
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"best val recon {best_val_recon_loss:.6f} @ epoch {best_epoch} | "
            f"bad epochs {bad_epochs}/{patience}"
        )
        if bad_epochs >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    print(
        f"restored best epoch {best_epoch} | "
        f"val recon {best_val_recon_loss:.6f}"
    )

    test_metrics = run_vqvae_epoch(model, test_loader, device=device)
    print(
        f"final test loss {test_metrics['loss']:.6f} | "
        f"test recon {test_metrics['recon_loss']:.6f}"
    )

    checkpoint = VitalDBCheckpoint.create(project_root, subdir=checkpoint_subdir)
    _, history_artifacts = eval_history(history, checkpoint.run_dir)
    test_batch = next(iter(test_loader))
    _, reconstruction_path = eval_recon(
        model,
        test_batch,
        transforms=fitted_transforms,
        vital_signs=vital_signs,
        save_dir=checkpoint.run_dir,
    )
    if reconstruction_path is None:
        raise RuntimeError("VitalDB reconstruction artifact was not saved.")
    evaluation_artifacts = {
        **history_artifacts,
        "reconstruction_samples": reconstruction_path,
    }
    checkpoint_payload = checkpoint.create_payload(
        model=model,
        model_config=model_config,
        optimizer=optimizer,
        history=history,
        test_metrics=dict(test_metrics),
        vitaldb_data=vitaldb_data,
        train_dataset=train_dataset,
        fitted_transforms=fitted_transforms,
        vital_signs=vital_signs,
        train_sample_ids=train_sample_ids,
        val_sample_ids=val_sample_ids,
        test_sample_ids=test_sample_ids,
        best_epoch=best_epoch,
        best_val_recon_loss=best_val_recon_loss,
        best_val_metrics=best_val_metrics,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        evaluation_artifacts=evaluation_artifacts,
        input_frequency_hz=input_frequency_hz,
        target_frequency_hz=target_frequency_hz,
        split_seed=split_seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    checkpoint.save(checkpoint_payload)
    checkpoint.update_registry(
        name=name,
        model_config=model_config,
        vital_signs=vital_signs,
        target_frequency_hz=target_frequency_hz,
        best_epoch=best_epoch,
        best_val_recon_loss=best_val_recon_loss,
        test_metrics=dict(test_metrics),
        evaluation_artifacts=evaluation_artifacts,
    )
    print(f"saved checkpoint: {checkpoint.path}")
    print(f"updated registry: {checkpoint.registry_path}")
    return {
        "model": model,
        "model_config": model_config,
        "history": history,
        "test_metrics": dict(test_metrics),
        "transforms": fitted_transforms,
        "checkpoint": checkpoint,
        "checkpoint_payload": checkpoint_payload,
        "artifacts": evaluation_artifacts,
    }


def main() -> None:
    """Run one end-to-end VitalDB pretraining workflow."""
    pretrain(
        project_root=Path(__file__).resolve().parents[3],
        vital_signs=("HR", "SpO2", "RR", "BT", "SBP", "DBP", "MBP"),
        time_length=60,
        epochs=10,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=1e-4,
        patience=5,
        min_delta=1e-4,
        device="cpu",
        train_case_limit=1000,
        val_case_limit=100,
        test_case_limit=100,
    )


if __name__ == "__main__":
    main()
