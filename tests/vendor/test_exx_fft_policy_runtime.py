from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor" / "abacus" / "source"


def test_si_grid_accepts_symmetry_precision_translation_noise(tmp_path: Path) -> None:
    program = tmp_path / "si-grid-regression.cpp"
    executable = tmp_path / "si-grid-regression"
    program.write_text(
        r'''
#include "source_pw/module_pwdft/exx_fft_grid_policy.h"

void ModuleBase::Matrix3::Identity()
{
    e11 = 1; e12 = 0; e13 = 0;
    e21 = 0; e22 = 1; e23 = 0;
    e31 = 0; e32 = 0; e33 = 1;
}

int main()
{
    hamilt::ExxSymmetryGridOperation operation;
    operation.symop = 24;
    operation.rotation.e11 = -1;
    operation.rotation.e12 = 0;
    operation.rotation.e13 = 0;
    operation.rotation.e21 = 0;
    operation.rotation.e22 = -1;
    operation.rotation.e23 = 0;
    operation.rotation.e31 = 0;
    operation.rotation.e32 = 0;
    operation.rotation.e33 = -1;
    operation.translation = ModuleBase::Vector3<double>(0.25000029858165368,
                                                        -0.24999990047217582,
                                                        -0.24999970141834638);
    const auto compatibility = hamilt::check_exx_fft_grid_compatibility(
        hamilt::ExxFftGrid(72, 72, 72), {operation});
    return compatibility.compatible ? 0 : 1;
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            f"-I{SOURCE}",
            str(program),
            str(SOURCE / "source_pw/module_pwdft/exx_fft_grid_policy.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )

    completed = subprocess.run([str(executable)], check=False)

    assert completed.returncode == 0
