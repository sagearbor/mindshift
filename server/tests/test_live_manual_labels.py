"""live_sessions.manual_labels_from_live — the pure mapping from a live
session's mid-call ``speaker_labels`` to the meta.json manual maps."""

import live_sessions as ls

TURNS = [{"speaker": "Speaker A", "text": "hi"}, {"speaker": "Speaker B", "text": "yo"}]
KNOWN = [
    {"person_id": "self", "display_name": "You", "is_self": True},
    {"person_id": "mom", "display_name": "Mom", "is_self": False},
]


def test_names_and_known_people_only():
    names, people = ls.manual_labels_from_live(
        {
            "Speaker B": {"display_name": " Mom ", "person_id": "mom", "is_self": False},
            "Speaker A": {"display_name": "Dad", "person_id": "dad", "is_self": False},
            "Speaker Q": {"display_name": "Ghost", "person_id": None, "is_self": False},
        },
        TURNS, KNOWN,
    )
    assert names == {"Speaker B": "Mom", "Speaker A": "Dad"}
    # "dad" is not enrolled on this account → a name-only (manual) label.
    assert people == {"Speaker B": "mom"}


def test_self_resolves_to_the_owner():
    names, people = ls.manual_labels_from_live(
        {"Speaker A": {"display_name": "Sage", "person_id": None, "is_self": True}},
        TURNS, KNOWN,
    )
    assert names == {"Speaker A": "You"} and people == {"Speaker A": "self"}
    # Even with an empty people list "self" always exists.
    _, people = ls.manual_labels_from_live(
        {"Speaker A": {"display_name": "x", "is_self": True}}, TURNS, [],
    )
    assert people == {"Speaker A": "self"}


def test_merges_over_existing_manual_maps_phone_wins_per_label():
    names, people = ls.manual_labels_from_live(
        {"Speaker B": {"display_name": "Mum", "person_id": "aunt", "is_self": False}},
        TURNS, KNOWN,
        existing_names={"Speaker A": "Me later", "Speaker B": "Mom"},
        existing_people={"Speaker A": "self", "Speaker B": "mom"},
    )
    assert names == {"Speaker A": "Me later", "Speaker B": "Mum"}
    # The phone's label carried an unknown person → the old id is dropped,
    # not kept under a name it no longer belongs to.
    assert people == {"Speaker A": "self"}


def test_garbage_is_ignored_and_a_person_needs_a_name():
    names, people = ls.manual_labels_from_live(
        {"Speaker A": "Mom", "Speaker B": {"display_name": "  "}},
        TURNS, KNOWN,
        existing_people={"Speaker Z": "mom"},
    )
    assert names == {} and people == {}
    names, people = ls.manual_labels_from_live({}, TURNS, KNOWN)
    assert names == {} and people == {}
