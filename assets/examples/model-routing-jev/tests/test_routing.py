import asyncio
import math

import pytest

from catalog import CATALOG
from routing import Policy, Proposal, Router, Runtime, Turn, request_tokens, valid_confidence


def make(start="full-tools", policy=Policy(enabled=True), persist=None, on_staged=None):
    saved = []
    router = Router(CATALOG, Runtime(CATALOG[start]), policy,
                    persist or saved.append, on_staged=on_staged)
    return router, saved


def small_turn(turn_id=1, **kw):
    base = dict(user_tokens=150, history_tokens=2_000, history_format="plain")
    return Turn(turn_id, **(base | kw))


def test_accepted_switch_commits_and_persists():
    router, saved = make()
    d = router.route_turn(small_turn(), Proposal("fast-text", 0.9))
    assert d.applied and d.reason == "switched" and d.persisted is True
    assert d.request_tokens == 300 + 2_000 + 150 + 1_000
    assert router.runtime.route is CATALOG["fast-text"]
    assert router.runtime.routed_turn == 1
    assert [r.route.alias for r in saved] == ["fast-text"]


@pytest.mark.parametrize("choice", ["vendor/some-model-id", "https://example.invalid/v1", None, 3])
def test_unknown_or_non_string_route_is_rejected(choice):
    router, saved = make()
    d = router.route_turn(small_turn(), Proposal(choice, 0.99))
    assert not d.applied and d.reason == "no_acceptable_route"
    assert d.rejected[0][1] == "not_in_allowlist"
    assert router.runtime.route.alias == "full-tools" and saved == []


def test_low_confidence_keeps_current():
    router, _ = make()
    d = router.route_turn(small_turn(), Proposal("fast-text", 0.5))
    assert (d.applied, d.reason) == (False, "low_confidence")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.1, 1.01, True, "0.9", None,
                                   pytest.param(10**400, id="huge-int"),
                                   pytest.param(-(10**400), id="huge-negative-int")])
def test_malformed_or_nonfinite_confidence(value):
    router, _ = make()
    d = router.route_turn(small_turn(), Proposal("fast-text", value))
    assert (d.applied, d.reason) == (False, "malformed_confidence")


def test_confidence_bounds_are_inclusive():
    assert valid_confidence(0) and valid_confidence(1.0) and valid_confidence(0.0)


def test_tool_turn_rejects_text_only_route():
    router, _ = make()
    d = router.route_turn(small_turn(needs_tools=True), Proposal("fast-text", 0.99))
    assert not d.applied and d.rejected == (("fast-text", "needs_tools"),)


def test_attachment_turn_rejects_route_without_attachments():
    router, _ = make()
    d = router.route_turn(small_turn(has_attachments=True), Proposal("mid-tools", 0.99))
    assert not d.applied and d.rejected == (("mid-tools", "needs_attachments"),)


def test_history_the_adapter_cannot_replay_blocks_switch():
    router, _ = make()
    d = router.route_turn(small_turn(history_format="opaque-reasoning"), Proposal("fast-text", 0.99))
    assert not d.applied and d.rejected == (("fast-text", "cannot_replay_history"),)


def test_full_request_with_destination_prompt_and_tools_exceeds_window():
    router, _ = make(start="fast-text")
    turn = small_turn(user_tokens=200, history_tokens=6_000, needs_tools=True)
    # History + user message alone would fit the 12k window...
    assert turn.history_tokens + turn.user_tokens < CATALOG["mid-tools"].context_tokens
    # ...but the destination's own system prompt, tool schemas and output budget do not.
    assert request_tokens(CATALOG["mid-tools"], turn) == 12_100
    d = router.route_turn(turn, Proposal("mid-tools", 0.95))
    assert not d.applied
    assert d.rejected == (("mid-tools", "request_too_large:12100>12000"),)
    assert router.runtime.route.alias == "fast-text"
    assert router.runtime.routed_turn is None  # nothing from staging leaked


def test_alternatives_go_through_the_same_guards():
    router, _ = make(start="fast-text")
    turn = small_turn(user_tokens=200, history_tokens=6_000, needs_tools=True)
    proposal = Proposal("fast-text", 0.88, alternatives=("vendor/x", "mid-tools", "full-tools"))
    d = router.route_turn(turn, proposal)
    assert d.applied and d.route == "full-tools"
    assert [why.split(":")[0] for _, why in d.rejected] == [
        "needs_tools", "not_in_allowlist", "request_too_large"]


def test_disabled_policy_never_switches():
    router, _ = make(policy=Policy(enabled=False))
    d = router.route_turn(small_turn(), Proposal("fast-text", 1.0))
    assert (d.applied, d.reason) == (False, "disabled")


def test_manual_pin_wins_over_confident_proposal():
    router, _ = make(policy=Policy(enabled=True, manual_pin="full-tools"))
    d = router.route_turn(small_turn(), Proposal("fast-text", 1.0))
    assert (d.applied, d.reason) == (False, "manual_pin")


def test_single_route_per_user_turn():
    router, _ = make()
    assert router.route_turn(small_turn(1), Proposal("fast-text", 0.9)).applied
    again = router.route_turn(small_turn(1), Proposal("full-tools", 0.99))
    assert (again.applied, again.reason) == (False, "already_routed_this_turn")
    assert router.route_turn(small_turn(2), Proposal("full-tools", 0.99)).applied


def test_already_on_route_is_a_keep():
    router, saved = make()
    d = router.route_turn(small_turn(), Proposal("full-tools", 0.99))
    assert (d.applied, d.reason) == (False, "already_on_route") and saved == []


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_base_exception_before_commit_leaves_old_route_intact(exc):
    def boom(_staged):
        raise exc()

    router, saved = make(on_staged=boom)
    before = router.runtime
    with pytest.raises(exc):
        router.route_turn(small_turn(), Proposal("fast-text", 0.9))
    assert router.runtime is before
    assert router.runtime.route.alias == "full-tools" and router.runtime.routed_turn is None
    assert saved == []


def test_persistence_failure_after_commit_keeps_new_route_coherent():
    def failing(_runtime):
        raise OSError("store down")

    router, _ = make(persist=failing)
    d = router.route_turn(small_turn(7), Proposal("fast-text", 0.9))
    assert d.applied and d.persisted is False
    assert router.runtime.route is CATALOG["fast-text"] and router.runtime.routed_turn == 7


def test_runtime_is_immutable():
    router, _ = make()
    with pytest.raises(AttributeError):
        router.runtime.route = CATALOG["fast-text"]  # FrozenInstanceError is an AttributeError


@pytest.mark.parametrize("alternatives", [None, 7, "mid-tools", {"mid-tools": 1}, ["mid-tools"]])
def test_alternatives_container_must_be_a_tuple(alternatives):
    router, saved = make(start="fast-text")
    d = router.route_turn(small_turn(), Proposal("vendor/x", 0.9, alternatives=alternatives))
    assert (d.applied, d.reason, d.rejected) == (False, "malformed_alternatives", ())
    assert router.runtime.route.alias == "fast-text" and saved == []


def test_alternative_elements_stay_untrusted():
    router, _ = make(start="fast-text")
    d = router.route_turn(small_turn(), Proposal("vendor/x", 0.9, alternatives=(None, 3, "mid-tools")))
    assert d.applied and d.route == "mid-tools"
    assert d.rejected == (("vendor/x", "not_in_allowlist"), ("None", "not_in_allowlist"),
                          ("3", "not_in_allowlist"))


def test_current_route_short_circuits_without_budget_or_fallbacks():
    router, saved = make(start="mid-tools")
    turn = small_turn(user_tokens=200, history_tokens=6_000, needs_tools=True)
    assert request_tokens(CATALOG["mid-tools"], turn) > CATALOG["mid-tools"].context_tokens
    d = router.route_turn(turn, Proposal("mid-tools", 0.95, alternatives=("full-tools",)))
    # Switch admission only: the oversized current route is kept, full-tools is never tried.
    assert (d.applied, d.reason, d.request_tokens, d.rejected) == (False, "already_on_route", None, ())
    assert router.runtime.route.alias == "mid-tools" and router.runtime.routed_turn is None
    assert saved == []


def test_rejected_first_decision_does_not_mark_the_turn():
    router, _ = make()
    assert router.route_turn(small_turn(1), Proposal("fast-text", 0.5)).reason == "low_confidence"
    assert router.runtime.routed_turn is None
    later = router.route_turn(small_turn(1), Proposal("fast-text", 0.9))
    assert later.applied and router.runtime.routed_turn == 1


def test_keep_on_current_route_does_not_mark_the_turn():
    router, _ = make()
    assert router.route_turn(small_turn(1), Proposal("full-tools", 0.99)).reason == "already_on_route"
    assert router.runtime.routed_turn is None
    assert router.route_turn(small_turn(1), Proposal("fast-text", 0.9)).applied


def test_revisited_turn_id_is_not_detected():
    router, _ = make()
    assert router.route_turn(small_turn(1), Proposal("fast-text", 0.9)).applied
    assert router.route_turn(small_turn(2), Proposal("full-tools", 0.9)).applied
    # Only the last committed turn id is remembered; the caller must not revisit ids.
    assert router.route_turn(small_turn(1), Proposal("fast-text", 0.9)).applied


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_interruption_during_persist_propagates_after_commit(exc):
    def interrupted(_runtime):
        raise exc()

    router, _ = make(persist=interrupted)
    with pytest.raises(exc):
        router.route_turn(small_turn(4), Proposal("fast-text", 0.9))
    # No Decision is returned, but the RAM commit already happened.
    assert router.runtime.route is CATALOG["fast-text"] and router.runtime.routed_turn == 4


def test_write_then_error_means_unacknowledged_not_unwritten():
    store = []

    def write_then_fail(runtime):
        store.append(runtime.route.alias)
        raise OSError("acknowledgement lost")

    router, _ = make(persist=write_then_fail)
    d = router.route_turn(small_turn(), Proposal("fast-text", 0.9))
    assert d.applied and d.persisted is False
    assert store == ["fast-text"]  # the write happened anyway; a retry must be idempotent
