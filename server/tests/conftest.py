import pytest


@pytest.fixture(autouse=True)
def _natural_turns_off_by_default(monkeypatch):
    """The NaturalTurn post-pass (main.NATURAL_TURNS_ENV, default ON in
    production) merges rapid same-speaker turns. Most pipeline tests pin
    exact turn counts against count-locked fake LLMs, so it is OFF for the
    suite; tests/test_natural_turns_pass.py enables it for itself."""
    monkeypatch.setenv("MINDSHIFT_NATURAL_TURNS", "0")
