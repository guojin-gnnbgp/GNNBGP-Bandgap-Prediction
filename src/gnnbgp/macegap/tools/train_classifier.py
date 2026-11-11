"""Train a direct/indirect band-gap head while freezing the first gap model.

This script is intentionally separate from the joint-loss training entry point.
The loaded epoch-296 model is kept as the frozen feature extractor and scalar
band-gap predictor; only ``classification_head`` receives gradients.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from monty.serialization import loadfn
from torch import nn

import mace
import mace.modules as mace_modules
import mace.modules.blocks as mace_block_classes
import mace.modules.models as mace_model_classes
from mace.modules.blocks import ScaleShiftBlock
from mace.tools import torch_geometric, utils
from mace.tools.scatter import scatter_mean

from gnnbgp.macegap.data.atomic_data import AtomicData
from gnnbgp.macegap.data.utils import load_from_xyz
from gnnbgp.macegap.modules.blocks import NoneLinearNodeEmbeddingBlock
from gnnbgp.macegap.modules.models import ScalarPropertyMACE


def register_legacy_aliases() -> None:
    """Make the original cmace checkpoint deserializable in the new source."""

    mace_model_classes.ScalarPropertyGlobalPoolMACE = ScalarPropertyMACE
    mace_model_classes.ScalarPropertyMACE = ScalarPropertyMACE
    mace_block_classes.NoneLinearNodeEmbeddingBlock = NoneLinearNodeEmbeddingBlock
    sys.modules.setdefault("cmace", mace)
    sys.modules.setdefault("cmace.modules", mace_modules)
    sys.modules.setdefault("cmace.modules.models", mace_model_classes)
    sys.modules.setdefault("cmace.modules.blocks", mace_block_classes)
    for name in ("radial", "symmetric_contraction", "irreps_tools"):
        sys.modules.setdefault(
            f"cmace.modules.{name}",
            __import__(f"mace.modules.{name}", fromlist=["*"]),
        )


class ReadoutAdapter(nn.Module):
    """Adapt the legacy one-head readout to the current two-argument call."""

    def __init__(self, readout: nn.Module) -> None:
        super().__init__()
        self.layers = readout.layers

    def forward(self, x: torch.Tensor, heads: torch.Tensor | None = None) -> torch.Tensor:
        del heads
        return self.layers(x)


class FrozenGapClassifier(nn.Module):
    """Frozen original gap model plus a trainable graph-level classifier."""

    def __init__(self, gap_model: nn.Module, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gap_model = gap_model
        feature_dim = sum(
            int(product.symmetric_contractions.irreps_out.dim)
            for product in gap_model.products
            if hasattr(product, "symmetric_contractions")
        )
        if feature_dim <= 0:
            raise RuntimeError("Could not determine the classifier feature width")
        self.feature_dim = feature_dim
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # The original MACE model is frozen.  No graph is retained through the
        # encoder, so only the new head can receive gradients.
        with torch.no_grad():
            output = self.gap_model.forward(data, compute_force=False)
        pooled = scatter_mean(output["node_feats"], data["batch"], dim=0)
        output["band_gap_type_logits"] = self.classification_head(pooled)
        return output


def load_frozen_gap_model(path: Path) -> nn.Module:
    register_legacy_aliases()
    model = torch.load(path, map_location="cpu")

    # The epoch-296 checkpoint predates the current scalar-property wrapper.
    # These adapters preserve its original scalar output while exposing the
    # intermediate node features needed by the new classifier.  Checkpoints
    # already saved by the current ScalarPropertyMACE (e.g. the 20250421
    # mid-term release model) keep their native two-argument readouts,
    # readouts_p, scale_shift(_p) and pooling; do not overwrite those.
    if hasattr(model.readouts[0], "layers"):
        model.readouts = nn.ModuleList([ReadoutAdapter(r) for r in model.readouts])
        model.readouts_p = copy.deepcopy(model.readouts)
        model.scale_shift = ScaleShiftBlock(scale=1.0, shift=0.0)
        model.scale_shift_p = ScaleShiftBlock(scale=1.0, shift=0.0)
        model.intensive = True
        model.global_pool_operation = getattr(model, "pool_operation", None)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def read_configs(path: Path) -> list:
    _, configs = load_from_xyz(
        file_path=str(path),
        config_type_weights=None,
        energy_key="REF_energy",
        forces_key="REF_forces",
        stress_key="REF_stress",
        virials_key=None,
        dipole_key=None,
        charges_key=None,
        head_key="head",
        extract_atomic_energies=True,
        keep_isolated_atoms=False,
        head_name="Default",
        scalar_property_key="band_gap_hse",
        band_gap_type_key="band_gap_type",
    )
    return configs


def make_dataset(
    configs: Iterable,
    z_table: utils.AtomicNumberTable,
    cutoff: float,
    extra_node_attrs_map: dict,
    knn: int,
) -> list[AtomicData]:
    return [
        AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=cutoff,
            extra_node_attrs_map=extra_node_attrs_map,
            knn=knn,
        )
        for config in configs
    ]


def labels(dataset: Iterable) -> np.ndarray:
    return np.asarray([int(config.band_gap_type) for config in dataset], dtype=np.int64)


def make_loader(dataset: list, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
        num_workers=0,
    )


def evaluate(
    model: FrozenGapClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.to_dict())
            target = batch["band_gap_type"].long().reshape(-1)
            loss = criterion(output["band_gap_type_logits"], target)
            prediction = output["band_gap_type_logits"].argmax(dim=-1)
            total_loss += float(loss) * int(target.numel())
            total += int(target.numel())
            for truth, pred in zip(target.cpu().tolist(), prediction.cpu().tolist()):
                confusion[int(truth), int(pred)] += 1
    recall = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(2, dtype=np.float64),
        where=confusion.sum(axis=1) != 0,
    )
    precision_direct = confusion[1, 1] / max(confusion[:, 1].sum(), 1)
    f1_direct = 2 * precision_direct * recall[1] / max(precision_direct + recall[1], 1e-12)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(np.trace(confusion) / max(total, 1)),
        "balanced_accuracy": float(recall.mean()),
        "indirect_recall": float(recall[0]),
        "direct_recall": float(recall[1]),
        "direct_f1": float(f1_direct),
        "confusion": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--valid-indices", type=Path, required=True)
    parser.add_argument("--extra-node-attrs-map", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--name", default="direct-gap-classifier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-7)
    parser.add_argument("--r-max", type=float, default=6.5)
    parser.add_argument("--knn", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    gap_model = load_frozen_gap_model(args.gap_model)
    z_table = utils.AtomicNumberTable([int(z) for z in gap_model.atomic_numbers])
    extra_map = loadfn(str(args.extra_node_attrs_map))
    all_configs = read_configs(args.train_file)
    test_configs = read_configs(args.test_file)
    if args.limit:
        all_configs = all_configs[: args.limit]
        test_configs = test_configs[: args.limit]
    valid_indices = {int(line.strip()) for line in args.valid_indices.read_text().splitlines() if line.strip()}
    valid_configs = [config for i, config in enumerate(all_configs) if i in valid_indices]
    train_configs = [config for i, config in enumerate(all_configs) if i not in valid_indices]
    train_data = make_dataset(train_configs, z_table, args.r_max, extra_map, args.knn)
    valid_data = make_dataset(valid_configs, z_table, args.r_max, extra_map, args.knn)
    test_data = make_dataset(test_configs, z_table, args.r_max, extra_map, args.knn)
    logging.info("Dataset sizes: train=%d valid=%d test=%d", len(train_data), len(valid_data), len(test_data))
    logging.info("Train labels: %s", dict(zip(*np.unique(labels(train_configs), return_counts=True))))

    counts = np.bincount(labels(train_configs), minlength=2).astype(np.float64)
    class_weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    model = FrozenGapClassifier(gap_model).to(device)
    model.gap_model.eval()
    trainable = list(model.classification_head.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    train_loader = make_loader(train_data, args.batch_size, True)
    train_eval_loader = make_loader(train_data, args.batch_size, False)
    valid_loader = make_loader(valid_data, args.batch_size, False)
    test_loader = make_loader(test_data, args.batch_size, False)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_epoch = -1
    stale = 0
    history_path = args.checkpoint_dir / f"{args.name}_history.jsonl"
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(args.max_epochs):
            model.train()
            model.gap_model.eval()
            running_loss = 0.0
            n_seen = 0
            for batch in train_loader:
                batch = batch.to(device)
                target = batch["band_gap_type"].long().reshape(-1)
                output = model(batch.to_dict())
                loss = criterion(output["band_gap_type_logits"], target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += float(loss) * int(target.numel())
                n_seen += int(target.numel())

            valid_metrics = evaluate(model, valid_loader, criterion, device)
            record = {
                "epoch": epoch,
                "train_loss": running_loss / max(n_seen, 1),
                "valid": valid_metrics,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            logging.info(
                "epoch=%d train_loss=%.5f valid_loss=%.5f valid_acc=%.4f "
                "valid_bal_acc=%.4f direct_recall=%.4f direct_f1=%.4f",
                epoch,
                record["train_loss"],
                valid_metrics["loss"],
                valid_metrics["accuracy"],
                valid_metrics["balanced_accuracy"],
                valid_metrics["direct_recall"],
                valid_metrics["direct_f1"],
            )
            score = valid_metrics["balanced_accuracy"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "classification_head_state_dict": model.classification_head.state_dict(),
                        "feature_dim": model.feature_dim,
                        "class_weights": class_weights.tolist(),
                        "valid_metrics": valid_metrics,
                    },
                    args.checkpoint_dir / f"{args.name}_best.pt",
                )
            else:
                stale += 1
                if stale >= args.patience:
                    logging.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                    break

    best = torch.load(args.checkpoint_dir / f"{args.name}_best.pt", map_location=device)
    model.classification_head.load_state_dict(best["classification_head_state_dict"])
    logging.info("Best epoch: %d", best["epoch"])
    logging.info("Train metrics: %s", evaluate(model, train_eval_loader, criterion, device))
    logging.info("Valid metrics: %s", evaluate(model, valid_loader, criterion, device))
    logging.info("Test metrics: %s", evaluate(model, test_loader, criterion, device))
    logging.info("Frozen gap parameters with gradients: %d", sum(int(p.requires_grad) for p in model.gap_model.parameters()))
    logging.info("Trainable classifier parameters: %d", sum(p.numel() for p in model.classification_head.parameters()))


if __name__ == "__main__":
    main()
