import json
from pathlib import Path

import pytest

from bandgap_predictor.abacus.parser import parse_abacus_result
from bandgap_predictor.errors import ParseError, RunError
from bandgap_predictor.schemas import ElectronicClassification

FIXTURES = Path(__file__).parent / "fixtures"


def test_converged_insulator_uses_final_printed_gap() -> None:
    parsed = parse_abacus_result(FIXTURES / "insulator", returncode=0)

    assert parsed.converged is True
    assert parsed.prediction.gap_ev == pytest.approx(1.2345)
    assert parsed.prediction.classification is ElectronicClassification.INSULATOR
    assert parsed.occupation_summary["materially_fractional_count"] == 0


def test_fractional_frontier_forces_zero_gap() -> None:
    parsed = parse_abacus_result(FIXTURES / "partial", returncode=0)

    assert parsed.prediction.gap_ev == 0.0
    assert (
        parsed.prediction.classification
        is ElectronicClassification.PARTIAL_OCCUPANCY
    )
    assert parsed.occupation_summary["materially_fractional_count"] > 0
    assert parsed.diagnostics["printed_gap_ev"] == pytest.approx(0.355)


def test_nonconvergence_never_returns_gap() -> None:
    with pytest.raises(RunError, match="did not converge"):
        parse_abacus_result(FIXTURES / "unconverged", returncode=0)


def test_missing_gap_evidence_is_parse_error() -> None:
    with pytest.raises(ParseError, match="band-gap evidence"):
        parse_abacus_result(FIXTURES / "malformed", returncode=0)


def test_nonzero_return_code_is_run_error() -> None:
    with pytest.raises(RunError, match="exit code 9"):
        parse_abacus_result(FIXTURES / "insulator", returncode=9)


def test_abacus_weights_normalize_without_extra_spin_factor(tmp_path: Path) -> None:
    out_dir = tmp_path / "OUT.BGP"
    out_dir.mkdir()
    (out_dir / "running_scf.log").write_text(
        "#SCF IS CONVERGED#\nE_gap(k) 0.0 1.5\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": "1.0",
        "converged": True,
        "nspin": 1,
        "occupation_weighting": "abacus_k_weighted",
        "spin_capacity": 2.0,
        "energy_unit": "eV",
        "fermi_energy_ev": 0.0,
        "printed_gap_ev": 1.5,
        "k_weights": [0.25, 0.75],
        "eigenvalues_ev": [[-1.0, 1.0], [-1.0, 1.0]],
        "occupations": [[0.25, 0.0], [0.75, 0.0]],
    }
    (out_dir / "bgp_electronic_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    parsed = parse_abacus_result(tmp_path, returncode=0)

    assert parsed.prediction.classification is ElectronicClassification.INSULATOR
    assert parsed.prediction.gap_ev == pytest.approx(1.5)
    assert parsed.occupation_summary["materially_fractional_count"] == 0
