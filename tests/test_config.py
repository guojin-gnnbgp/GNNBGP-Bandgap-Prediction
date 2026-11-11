from pathlib import Path

from bandgap_predictor.config import resolve_abacus_path


def test_cli_path_precedes_config_and_environment(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[abacus]\nexecutable = "/config/abacus"\n')

    resolved = resolve_abacus_path(
        Path("/cli/abacus"),
        config,
        {"BGP_ABACUS": "/env/abacus"},
    )

    assert resolved == Path("/cli/abacus")


def test_config_path_precedes_environment(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[abacus]\nexecutable = "/config/abacus"\n')

    resolved = resolve_abacus_path(
        config_path=config,
        environ={"BGP_ABACUS": "/env/abacus"},
    )

    assert resolved == Path("/config/abacus")


def test_environment_is_used_without_cli_or_config(tmp_path: Path) -> None:
    resolved = resolve_abacus_path(
        config_path=tmp_path / "missing.toml",
        environ={"BGP_ABACUS": "/env/abacus"},
    )

    assert resolved == Path("/env/abacus")
