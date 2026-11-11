"""Generate the immutable scientific-asset manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
GNN_MODEL_NAME = (
    "20250421-baseline-band_gap_hse-batch-64-r_max-6-L-1-channels-32-"
    "node-feat-mlp-64-knn-24.model"
)


def record(
    logical_id: str,
    path: Path,
    kind: str,
    *,
    element: str | None = None,
    **metadata: object,
) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "element": element,
        "kind": kind,
        "logical_id": logical_id,
        "metadata": metadata,
        "path": path.relative_to(ASSET_ROOT).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def discover_upfs(root: Path) -> list[tuple[str, Path]]:
    files = sorted(root.glob("*.upf"))
    if len(files) != 71:
        raise RuntimeError(f"expected 71 PseudoDojo UPFs, found {len(files)}")
    discovered = [(path.stem, path) for path in files]
    if len({element for element, _path in discovered}) != len(discovered):
        raise RuntimeError("PseudoDojo element identifiers are not unique")
    return discovered


def build_manifest() -> dict[str, object]:
    gnn_root = ASSET_ROOT / "models" / "gnn"
    cider_root = ASSET_ROOT / "models" / "cider"
    pseudo_root = ASSET_ROOT / "pseudopotentials" / "pseudo-dojo"
    entries = [
        record(
            "gnn-hse06-model",
            gnn_root / GNN_MODEL_NAME,
            "model",
            method="HSE06-like",
            trust="legacy-provisional-pickle",
        ),
        record(
            "gnn-feature-map",
            gnn_root / "atom_init.json",
            "feature-map",
            method="HSE06-like",
        ),
        record(
            "cider-pbe0-model",
            cider_root / "nldf_mgga_cand00_n96_noise0p01.yaml",
            "model",
            method="CIDER-PBE0",
            nlcc_features=False,
        ),
    ]
    entries.extend(
        record(
            f"pseudo-{element}",
            path,
            "pseudopotential",
            element=element,
            family="PseudoDojo-NC-SR",
            relativistic="scalar",
        )
        for element, path in discover_upfs(pseudo_root)
    )
    return {"assets": entries, "schema_version": "1.0"}


def main() -> None:
    manifest = build_manifest()
    (ASSET_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
