"""`pinnforge task smoke` — the gate has to read the evidence.

`eval.py`'s contract is that it exits 0 whenever a record was *written*,
failed runs included. A gate that judges on the exit status therefore passes
every recorded failure: a dead scorer, non-finite predictions, a shape
mismatch. These pin the verdict to the record instead.
"""

from __future__ import annotations

from pinnforge.cli.task_cmd import smoke_verdict


def _rec(**kw) -> dict:
    base = {"smoke": True, "rRMSE": 1.23, "step_count": 80, "train_s": 5.2, "error": None}
    return {**base, **kw}


def test_a_scored_record_passes():
    code, msg = smoke_verdict(0, [_rec()])
    assert code == 0
    assert "smoke passed" in msg and "1.23" in msg


def test_a_nonzero_exit_fails():
    code, msg = smoke_verdict(1, [_rec()])
    assert code == 1 and "exit 1" in msg


def test_no_record_fails():
    code, msg = smoke_verdict(0, [])
    assert code == 1 and "no record" in msg


def test_a_recorded_error_fails_even_on_exit_zero():
    """The regression: eval.py logs the failure and still exits 0."""
    code, msg = smoke_verdict(0, [_rec(rRMSE=None, error="killed at the 120s wall")])
    assert code == 1
    assert "120s wall" in msg


def test_an_unscored_record_fails_even_without_an_error():
    """A record with no rRMSE never proved the scoring path works."""
    code, msg = smoke_verdict(0, [_rec(rRMSE=None)])
    assert code == 1 and "no rRMSE" in msg


def test_the_last_record_is_the_verdict():
    """A passing retry after a failed attempt is a pass."""
    code, _ = smoke_verdict(0, [_rec(rRMSE=None, error="boom"), _rec()])
    assert code == 0
