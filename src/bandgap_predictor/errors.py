"""Public error categories and stable CLI exit codes."""

from pathlib import Path


class BGPError(RuntimeError):
    """Base class for user-facing band-gap predictor failures."""

    exit_code = 1

    def __init__(self, message: str, artifact_path: Path | None = None) -> None:
        super().__init__(message)
        self.artifact_path = artifact_path


class InputError(BGPError):
    """Invalid command input or unsupported prediction request."""

    exit_code = 2


class PreflightError(BGPError):
    """Missing or incompatible executable, configuration, or asset."""

    exit_code = 3


class RunError(BGPError):
    """Calculation launch, execution, or convergence failure."""

    exit_code = 4


class ParseError(BGPError):
    """Calculation output could not be interpreted safely."""

    exit_code = 5
