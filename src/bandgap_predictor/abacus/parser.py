"""Occupation-aware parsing of package-owned ABACUS result artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ParseError, RunError
from ..schemas import ElectronicClassification, PredictionValue

CONVERGED = re.compile(r"#SCF IS CONVERGED#", re.IGNORECASE)
UNCONVERGED = re.compile(r"SCF IS NOT CONVERGED", re.IGNORECASE)
GAP = re.compile(
    r"E_gap\(k\)\s+[-+0-9.Ee]+\s+([-+0-9.Ee]+)",
    re.IGNORECASE,
)
FRACTIONAL_LOWER = 0.01
FRACTIONAL_UPPER = 0.99


@dataclass(frozen=True)
class ParsedAbacusResult:
    prediction: PredictionValue
    converged: bool
    occupation_summary: dict[str, object]
    diagnostics: dict[str, object]


def classify(gap_ev: float, normalized_occupations: list[float]) -> PredictionValue:
    fractional = [
        value
        for value in normalized_occupations
        if FRACTIONAL_LOWER < value < FRACTIONAL_UPPER
    ]
    if fractional:
        return PredictionValue(
            gap_ev=0.0,
            classification=ElectronicClassification.PARTIAL_OCCUPANCY,
        )
    if gap_ev <= 1e-8:
        return PredictionValue(
            gap_ev=0.0,
            classification=ElectronicClassification.ZERO_GAP,
        )
    return PredictionValue(
        gap_ev=gap_ev,
        classification=ElectronicClassification.INSULATOR,
    )


def _single_log(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("OUT.*/running_scf.log"))
    if len(candidates) != 1:
        raise ParseError(
            f"expected one ABACUS SCF log, found {len(candidates)} under {run_dir}"
        )
    return candidates[0]


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ParseError(f"ABACUS electronic-state artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParseError(f"ABACUS electronic-state artifact is invalid: {error}") from error
    if not isinstance(payload, dict):
        raise ParseError("ABACUS electronic-state artifact must be a JSON object")
    return payload


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ParseError(f"ABACUS artifact field {field} must be finite")
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ParseError(f"ABACUS artifact field {field} must be finite") from error
    if not math.isfinite(converted):
        raise ParseError(f"ABACUS artifact field {field} must be finite")
    return converted


def _numeric_rows(value: object, field: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ParseError(f"ABACUS artifact field {field} must be a non-empty array")
    rows: list[list[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ParseError(f"ABACUS artifact field {field}[{row_index}] is invalid")
        rows.append(
            [
                _finite_float(item, f"{field}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            ]
        )
    return rows


def _normalize_occupations(payload: dict[str, Any]) -> tuple[list[float], int, int]:
    weights_value = payload.get("k_weights")
    if not isinstance(weights_value, list) or not weights_value:
        raise ParseError("ABACUS artifact field k_weights must be a non-empty array")
    weights = [
        _finite_float(value, f"k_weights[{index}]")
        for index, value in enumerate(weights_value)
    ]
    if any(weight <= 0.0 for weight in weights):
        raise ParseError("ABACUS k-point weights must be positive")

    eigenvalues = _numeric_rows(payload.get("eigenvalues_ev"), "eigenvalues_ev")
    occupations = _numeric_rows(payload.get("occupations"), "occupations")
    if len(eigenvalues) != len(weights) or len(occupations) != len(weights):
        raise ParseError("ABACUS artifact k-point dimensions are inconsistent")
    band_counts = {len(row) for row in eigenvalues}
    if len(band_counts) != 1 or any(
        len(occupation_row) != len(eigenvalue_row)
        for occupation_row, eigenvalue_row in zip(occupations, eigenvalues, strict=True)
    ):
        raise ParseError("ABACUS artifact band dimensions are inconsistent")

    spin_capacity = _finite_float(payload.get("spin_capacity"), "spin_capacity")
    if not math.isclose(spin_capacity, 2.0, abs_tol=1e-12):
        raise ParseError("ABACUS artifact spin capacity must be 2.0 for nspin=1")

    normalized: list[float] = []
    for k_index, (weight, row) in enumerate(zip(weights, occupations, strict=True)):
        # ABACUS wg already includes both the k-point weight and nspin=1
        # degeneracy convention: a fully occupied band has wg == wk.
        denominator = weight
        for band_index, occupation in enumerate(row):
            value = occupation / denominator
            if value < -1e-8 or value > 1.0 + 1e-8:
                raise ParseError(
                    "ABACUS normalized occupation is outside [0, 1]: "
                    f"k={k_index}, band={band_index}, value={value}"
                )
            normalized.append(min(1.0, max(0.0, value)))
    return normalized, len(weights), band_counts.pop()


def parse_abacus_result(run_dir: Path, returncode: int) -> ParsedAbacusResult:
    run_dir = Path(run_dir).resolve()
    if returncode != 0:
        raise RunError(f"ABACUS exited with exit code {returncode}")

    log_path = _single_log(run_dir)
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ParseError(f"could not read ABACUS SCF log: {error}") from error
    if UNCONVERGED.search(log_text) or not CONVERGED.search(log_text):
        raise RunError("ABACUS SCF did not converge")

    artifact_path = log_path.parent / "bgp_electronic_state.json"
    payload = _load_artifact(artifact_path)
    if payload.get("schema_version") != "1.0":
        raise ParseError("ABACUS electronic-state schema version is unsupported")
    if payload.get("converged") is not True:
        raise RunError("ABACUS electronic-state artifact says SCF did not converge")
    if payload.get("nspin") != 1:
        raise ParseError("ABACUS electronic-state artifact requires nspin=1")
    if payload.get("occupation_weighting") != "abacus_k_weighted":
        raise ParseError("ABACUS occupation weighting convention is unsupported")
    if payload.get("energy_unit") != "eV":
        raise ParseError("ABACUS electronic-state energy unit must be eV")

    gap_matches = GAP.findall(log_text)
    artifact_gap_value = payload.get("printed_gap_ev")
    if not gap_matches or artifact_gap_value is None:
        raise ParseError("ABACUS output lacks required band-gap evidence")
    log_gap = _finite_float(gap_matches[-1], "running_scf.log E_gap(k)")
    artifact_gap = _finite_float(artifact_gap_value, "printed_gap_ev")
    if log_gap < -1e-8 or artifact_gap < -1e-8:
        raise ParseError("ABACUS band-gap evidence is negative")
    if not math.isclose(log_gap, artifact_gap, abs_tol=1e-6, rel_tol=1e-8):
        raise ParseError("ABACUS log and electronic-state band-gap evidence disagree")
    gap_ev = max(0.0, artifact_gap)

    fermi_energy = _finite_float(payload.get("fermi_energy_ev"), "fermi_energy_ev")
    normalized, k_point_count, band_count = _normalize_occupations(payload)
    fractional = [
        value
        for value in normalized
        if FRACTIONAL_LOWER < value < FRACTIONAL_UPPER
    ]
    prediction = classify(gap_ev, normalized)
    occupation_summary: dict[str, object] = {
        "fractional_lower_threshold": FRACTIONAL_LOWER,
        "fractional_upper_threshold": FRACTIONAL_UPPER,
        "materially_fractional_count": len(fractional),
        "materially_fractional_occupations": fractional,
        "normalized_max": max(normalized),
        "normalized_min": min(normalized),
        "normalization": "occupation / k_weight (ABACUS wg convention)",
    }
    diagnostics: dict[str, object] = {
        "band_count": band_count,
        "electronic_state_artifact": str(artifact_path),
        "fermi_energy_ev": fermi_energy,
        "k_point_count": k_point_count,
        "printed_gap_ev": log_gap,
        "scf_log": str(log_path),
    }
    return ParsedAbacusResult(
        prediction=prediction,
        converged=True,
        occupation_summary=occupation_summary,
        diagnostics=diagnostics,
    )
