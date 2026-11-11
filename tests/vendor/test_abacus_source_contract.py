from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor" / "abacus"


def test_snapshot_contains_owned_license_and_integration_provenance() -> None:
    assert (SOURCE / "LICENSE").is_file()
    upstream = (SOURCE / "UPSTREAM.md").read_text(encoding="utf-8")
    patches = (SOURCE / "INTEGRATION_PATCHES.md").read_text(encoding="utf-8")
    assert "7828d7cc28c8702ced99e88fd3cad51b58cf7fee" in upstream
    assert "bbfa7d67c0476ae4a586719bd4a9713d81df8536" in upstream
    assert "53f9a990587c240cf4a41f66713795139defab2f" in patches
    assert "out_bgp_result" in patches


def test_snapshot_contains_native_cider_gpu_exx_and_bgp_result_sources() -> None:
    required = (
        "source/source_hamilt/module_xc/cider/evaluator.cpp",
        "source/source_pw/module_pwdft/op_pw_exx.cpp",
        "source/source_pw/module_pwdft/exx_tile_policy.cpp",
        "source/source_io/module_bgp/bgp_result.cpp",
        "source/source_io/module_bgp/bgp_result.h",
        "source/source_io/test/bgp_result_test.cpp",
    )
    assert all((SOURCE / relative).is_file() for relative in required)
    input_items = (
        SOURCE / "source/source_io/module_parameter/read_input_item_output.cpp"
    ).read_text(encoding="utf-8")
    solver = (SOURCE / "source/source_esolver/esolver_ks_pw.cpp").read_text(
        encoding="utf-8"
    )
    exx_options = (
        SOURCE / "source/source_pw/module_pwdft/op_pw_exx.h"
    ).read_text(encoding="utf-8")
    assert '"out_bgp_result"' in input_items
    assert "write_bgp_electronic_state" in solver
    assert "parameters.empty()" in exx_options
    assert "SYMMETRY_COORDINATE_TOLERANCE" in (
        SOURCE / "source/source_pw/module_pwdft/exx_fft_grid_policy.cpp"
    ).read_text(encoding="utf-8")
    assert "check_exx_fft_grid_compatibility(grid, {operation}).compatible" in (
        SOURCE
        / "source/source_pw/module_pwdft/kernels/exx_q_state_op.cpp"
    ).read_text(encoding="utf-8")


def test_snapshot_excludes_repository_and_build_artifacts() -> None:
    forbidden_names = {".git", "CMakeCache.txt", "OUT.BGP", "__pycache__"}
    forbidden_suffixes = {".o", ".a", ".so", ".pyc"}
    paths = list(SOURCE.rglob("*"))

    assert not any(path.name in forbidden_names for path in paths)
    assert not any(path.is_file() and path.suffix in forbidden_suffixes for path in paths)
    assert not any(path.is_file() and path.name == "abacus_pw_gpu" for path in paths)
