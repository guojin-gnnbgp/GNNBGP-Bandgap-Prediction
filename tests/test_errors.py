from bandgap_predictor.errors import InputError, ParseError, PreflightError, RunError


def test_public_error_exit_codes_are_stable() -> None:
    assert [
        InputError.exit_code,
        PreflightError.exit_code,
        RunError.exit_code,
        ParseError.exit_code,
    ] == [2, 3, 4, 5]
