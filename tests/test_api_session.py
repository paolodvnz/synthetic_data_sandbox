"""Specification tests for api.session: one in-memory session whose stage follows the objects it holds."""

import threading

import pytest

from api.errors import ApiError
from api.session import STAGE_FIELDS, Session, Stage

MARKER = object()   # stands in for the real objects (inspection, encoder, model...): the session only stores them


def filled_up_to(stage: Stage) -> Session:
    """A session holding a marker in every field of ``stage`` and of the stages before it."""
    session = Session()
    for each in Stage:
        if Stage.EMPTY < each <= stage:
            for name in STAGE_FIELDS[each]:
                setattr(session, name, MARKER)
    return session


def test_new_session_is_empty():
    session = Session()
    assert session.stage is Stage.EMPTY
    assert all(getattr(session, name) is None for names in STAGE_FIELDS.values() for name in names)


def test_every_stage_but_empty_has_fields():
    assert set(STAGE_FIELDS) == set(Stage) - {Stage.EMPTY}
    assert all(STAGE_FIELDS[stage] for stage in STAGE_FIELDS)


def test_fields_belong_to_one_stage_only():
    names = [name for names in STAGE_FIELDS.values() for name in names]
    assert len(names) == len(set(names))


def test_stages_are_ordered():
    assert [s.name for s in sorted(Stage)] == ["EMPTY", "INSPECTED", "PREPARED", "TRAINED", "GENERATED", "VALIDATED"]


@pytest.mark.parametrize("stage", list(Stage))
def test_stage_is_the_last_complete_one(stage):
    assert filled_up_to(stage).stage is stage


def test_a_partly_filled_stage_does_not_count():
    session = filled_up_to(Stage.PREPARED)
    session.fidelity = MARKER          # validated needs both reports
    assert session.stage is Stage.PREPARED


def test_a_later_stage_without_the_earlier_ones_does_not_count():
    session = filled_up_to(Stage.INSPECTED)
    session.trained = MARKER           # a model without the contract it was trained on
    assert session.stage is Stage.INSPECTED


def test_clear_from_keeps_earlier_stages():
    session = filled_up_to(Stage.VALIDATED)
    session.clear_from(Stage.TRAINED)
    assert session.stage is Stage.PREPARED
    assert session.contract is MARKER
    assert session.trained is None and session.synthetic is None and session.privacy is None


def test_clear_from_empty_is_rejected():
    with pytest.raises(ValueError):
        Session().clear_from(Stage.EMPTY)


def test_reset_empties_everything():
    session = filled_up_to(Stage.VALIDATED)
    session.reset()
    assert session.stage is Stage.EMPTY
    assert all(getattr(session, name) is None for names in STAGE_FIELDS.values() for name in names)


@pytest.mark.parametrize("stage", list(Stage))
def test_require_passes_when_the_stage_is_reached(stage):
    filled_up_to(stage).require(stage)
    filled_up_to(Stage.VALIDATED).require(stage)


@pytest.mark.parametrize(("have", "need", "code"), [
    (Stage.EMPTY, Stage.INSPECTED, "no_dataset"),
    (Stage.EMPTY, Stage.VALIDATED, "no_dataset"),          # the first missing stage is named
    (Stage.INSPECTED, Stage.PREPARED, "not_prepared"),
    (Stage.PREPARED, Stage.TRAINED, "not_trained"),
    (Stage.TRAINED, Stage.GENERATED, "not_generated"),
    (Stage.GENERATED, Stage.VALIDATED, "not_validated"),
])
def test_require_names_the_first_missing_stage(have, need, code):
    with pytest.raises(ApiError) as caught:
        filled_up_to(have).require(need)
    assert caught.value.status == 409
    assert caught.value.code == code
    assert caught.value.message                              # plain-language text for the UI


LOCK_TIMEOUT_SECONDS = 1


def test_lock_is_reentrant():
    # Methods may call each other while holding it. A timeout, so that a plain Lock fails instead of hanging.
    lock = Session().lock
    assert lock.acquire(timeout=LOCK_TIMEOUT_SECONDS)
    try:
        assert lock.acquire(timeout=LOCK_TIMEOUT_SECONDS)
        lock.release()
    finally:
        lock.release()


def test_lock_blocks_other_threads():
    lock = Session().lock
    got_it = []
    with lock:
        other = threading.Thread(target=lambda: got_it.append(lock.acquire(timeout=0.1)))
        other.start()
        other.join()
    assert got_it == [False]
