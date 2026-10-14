"""No-network STUB_ONLY checks of our recipe, not SDK integration tests."""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

import pytest
from provider_stubs import SDKStub, dspy_stub

from fair_compare.budget import BudgetExceeded, Ledger
from fair_compare.cases import Case, parse_label


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network forbidden in recipe stub tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    sdk = SDKStub()
    dspy, teleprompt = dspy_stub(sdk)
    monkeypatch.setitem(sys.modules, "openai", sdk)
    monkeypatch.setitem(sys.modules, "dspy", dspy)
    monkeypatch.setitem(sys.modules, "dspy.teleprompt", teleprompt)
    monkeypatch.setenv("DSP_CACHEBOOL", "false")
    path = Path(__file__).resolve().parents[1] / "dspy_recipe/real_dspy_arm.py"
    spec = importlib.util.spec_from_file_location("recipe_under_stub", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, sdk, dspy


def test_sdk_internal_retry_disabled_outer_retry_is_counted(recipe) -> None:
    module, sdk, _ = recipe
    sdk.responses = ["rate-limit", "billing"]
    ledger = Ledger("manual", 2)
    lm = module.BudgetedOpenAI(ledger=ledger, role="student",
                               model="test", temperature=0)
    lm.request("ticket")
    assert sdk.max_retries == 0
    assert sdk.http_attempts == ledger.billable == 2
    assert ledger.by_role["student"] == ledger.by_role["retry"] == 1


def test_budget_stops_outer_retry_before_second_http_attempt(recipe) -> None:
    module, sdk, _ = recipe
    sdk.responses = ["rate-limit", "billing"]
    ledger = Ledger("manual", 1)
    lm = module.BudgetedOpenAI(ledger=ledger, role="student", model="test")
    with pytest.raises(BudgetExceeded):
        lm.request("ticket")
    assert sdk.http_attempts == ledger.billable == 1


def test_manual_and_compiled_use_same_predict_recovery_path(recipe) -> None:
    module, sdk, dspy = recipe
    case = Case("c", "g", "ticket text", "billing")
    train = (case,)
    manual = module.manual_candidate("Choose a queue.", train)
    compiled = module.Router()
    for program in (manual, compiled):
        sdk.responses = ["", "billing"]
        ledger = Ledger("arm", 3)
        lm = module.BudgetedOpenAI(ledger=ledger, role="student", model="test")
        raw = module.predict_ticket(program, lm, case)
        assert parse_label(raw, module.LABELS) == "billing"
        assert ledger.billable == 2  # missing output causes same recovery
    assert len(dspy.predict_calls) == 2
    assert type(manual.route) is type(compiled.route)
    assert manual.route.signature.fields == ("ticket", "queue")
    assert manual.route.signature.instructions == "Choose a queue."


def test_copy_keeps_shared_meter_ledger_and_decoding_overrides(recipe) -> None:
    module, sdk, _ = recipe
    lm = module.BudgetedOpenAI(ledger=Ledger("dspy", 4), role="teacher",
                               model="test", temperature=0, system_prompt="s")
    clone = lm.copy(temperature=0.701)
    assert clone.ledger is lm.ledger
    assert clone.token_meter is lm.token_meter
    assert clone.kwargs["temperature"] == 0.701
    assert clone.system_prompt == "s"
    sdk.responses = ["billing", "account"]
    lm.request("one")
    clone.request("two")
    assert lm.token_meter.total_tokens == 8


def test_wrong_sdk_version_refused_before_request(recipe) -> None:
    module, sdk, _ = recipe
    sdk.__version__ = "2.0.0"
    with pytest.raises(RuntimeError, match="1.30.5"):
        module.BudgetedOpenAI(ledger=Ledger("arm", 2), role="student")
    assert sdk.http_attempts == 0


def test_changed_global_retry_setting_refused_before_request(recipe) -> None:
    module, sdk, _ = recipe
    lm = module.BudgetedOpenAI(ledger=Ledger("arm", 2), role="student")
    sdk.max_retries = 2
    sdk.responses = ["billing"]
    with pytest.raises(RuntimeError, match="retries"):
        lm.request("ticket")
    assert sdk.http_attempts == 0
