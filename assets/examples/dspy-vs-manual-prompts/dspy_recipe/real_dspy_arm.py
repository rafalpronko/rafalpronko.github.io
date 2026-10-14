"""SOURCE_CHECKED / STUB_ONLY: historical provider recipe, not a real run.

Pins: dspy-ai==2.4.9 and openai==1.30.5 (requirements.txt). Tests import OUR
adapter with fake dspy/openai modules, never either real package. No SDK
integration, compiler execution or provider call was performed. This is an
incomplete recipe, not a benchmark CLI; token use is metered, NOT capped.
Before a real run, add conservative token reservation at the transport
boundary, including failed attempts, and verify it with the pinned SDK.
"""

from __future__ import annotations

import os

# dsp.modules.cache_utils reads DSP_CACHEBOOL once, at import time.
# Disable the on-disk joblib cache so every counted request is a real one.
os.environ["DSP_CACHEBOOL"] = "false"

import hashlib  # noqa: E402
import tempfile  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import dspy  # noqa: E402
import openai  # noqa: E402
from dspy.teleprompt import BootstrapFewShot  # noqa: E402

from fair_compare.budget import Ledger  # noqa: E402
from fair_compare.cases import Case, parse_label  # noqa: E402

LABELS = ("billing", "technical", "account", "shipping")
MODEL = "gpt-3.5-turbo-0125"  # same model, same settings for both arms
GEN = {"temperature": 0.0, "max_tokens": 16}
SDK_VERSION = "1.30.5"


@dataclass
class TokenMeter:
    """Observed successful-response usage; NOT a token budget enforcer."""

    total_tokens: int = 0


def configure_sdk() -> None:
    """Configure the module client actually used by DSPy 2.4.9.

    gpt3.py calls openai.chat.completions.create, not a separately constructed
    OpenAI instance. SDK 1.30.5's _ModuleClient reads the max_retries global.
    It is a client setting; passing it in completion kwargs is incorrect.
    Use a dedicated single-threaded process: these settings are global.
    """
    if openai.__version__ != SDK_VERSION:
        raise RuntimeError(f"recipe requires openai=={SDK_VERSION}")
    openai.max_retries = 0


class BudgetedOpenAI(dspy.OpenAI):  # dspy.OpenAI is dsp.GPT3 in 2.4.9
    """Count basic_request entries with SDK retries and DSPy cache disabled.

    Retains DSPy's outer rate-limit backoff; each such retry is charged.
    The intended one-entry/one-HTTP-attempt mapping is source-checked and
    stub-tested only. Validate it against the real transport before use.
    """

    def __init__(self, *, ledger: Ledger, role: str,
                 token_meter: TokenMeter | None = None, **kwargs: object):
        configure_sdk()
        if "max_retries" in kwargs:
            raise ValueError("configure SDK retries on the module client")
        super().__init__(**kwargs)
        self.ledger, self.role, self._tries = ledger, role, 0
        self.token_meter = (
            token_meter if token_meter is not None else TokenMeter()
        )

    def request(self, prompt, **kwargs):
        self._tries = 0  # super().request is the backoff-wrapped loop
        return super().request(prompt, **kwargs)

    def basic_request(self, prompt, **kwargs):
        if openai.max_retries != 0:
            raise RuntimeError("SDK retries must remain disabled")
        self.ledger.charge(self.role if self._tries == 0 else "retry")
        self._tries += 1
        response = super().basic_request(prompt, **kwargs)
        self.token_meter.total_tokens += (response.get("usage") or {}).get(
            "total_tokens", 0
        )
        return response

    def copy(self, **kwargs: object) -> BudgetedOpenAI:
        # LM.copy merges saved kwargs with caller overrides (e.g. temperature).
        # Keep ledger and meter identity across later bootstrap rounds.
        merged = {"model_type": self.model_type,
                  "system_prompt": self.system_prompt,
                  **self.kwargs, **kwargs}
        model = merged.pop("model")
        return BudgetedOpenAI(
            ledger=self.ledger, role=self.role, token_meter=self.token_meter,
            model=model, **merged,
        )


class RouteTicket(dspy.Signature):
    """Route the support ticket to exactly one queue."""

    ticket = dspy.InputField(desc="the customer's message")
    queue = dspy.OutputField(
        desc="one of: billing, technical, account, shipping"
    )


class Router(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.route = dspy.Predict(RouteTicket)

    def forward(self, ticket: str) -> dspy.Prediction:
        return self.route(ticket=ticket)


def to_examples(cases: tuple[Case, ...]) -> list[dspy.Example]:
    return [
        dspy.Example(ticket=c.text, queue=c.label).with_inputs("ticket")
        for c in cases
    ]


def exact_queue(example, prediction, trace=None) -> bool:
    # Same contract as the manual arm: parse_label() or it is wrong.
    raw = getattr(prediction, "queue", "") or ""
    return parse_label(raw, LABELS) == example.queue


def compile_candidate(train: tuple[Case, ...], k: int, ledger: Ledger):
    """One DSPy candidate. Teacher and student calls hit the same ledger."""
    meter = TokenMeter()
    student_lm = BudgetedOpenAI(ledger=ledger, role="student",
                                token_meter=meter, model=MODEL, **GEN)
    teacher_lm = BudgetedOpenAI(ledger=ledger, role="teacher",
                                token_meter=meter, model=MODEL, **GEN)
    dspy.settings.configure(lm=student_lm)
    optimizer = BootstrapFewShot(
        metric=exact_queue,
        max_bootstrapped_demos=k,
        max_labeled_demos=k,  # also the teacher's LabeledFewShot k
        max_rounds=1,
        max_errors=1,  # first exception (BudgetExceeded too) aborts compile
        teacher_settings={"lm": teacher_lm},
    )
    # Deliberately no valset=: in 2.4.9, _train() samples the raw (labelled)
    # demos from valset when it is given. Passing dev here would put dev
    # examples into the prompt.
    return optimizer.compile(Router(), trainset=to_examples(train))


def demo_texts(program: dspy.Module) -> set[str]:
    """Inspect what ended up in the prompt; assert it is train-only."""
    return {d.ticket for p in program.predictors() for d in p.demos}


def frozen_text(program: dspy.Module) -> str:
    """program.save() writes JSON (demos included); hash that text."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.json"
        program.save(str(path))
        return path.read_text(encoding="utf-8")


def manual_candidate(instruction: str, demos: tuple[Case, ...]) -> Router:
    """Change instructions/demos; caller must check demos are train-only."""
    program = Router()
    program.route.signature = RouteTicket.with_instructions(instruction)
    program.route.demos = to_examples(demos)
    return program


def predict_ticket(
    program: dspy.Module, lm: BudgetedOpenAI, case: Case,
) -> str:
    """BOTH arms use Router/Predict, template extraction and field recovery."""
    # Do not bind predictor.lm: in 2.4.9 that sets query_only=True and omits
    # instructions/demos. Context supplies the same LM path for both arms.
    with dspy.settings.context(lm=lm):
        return program(ticket=case.text).queue or ""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Holdout scoring: use your own loop, not dspy.Evaluate. In 2.4.9 Evaluate
# turns an exception into score 0.0 until max_errors and then re-raises, so a
# budget or provider error silently looks like a wrong answer. Record status
# per case ("ok" / "invalid" / "error") exactly as fair_compare.runner does,
# under a separate final-eval Ledger per arm, after HoldoutGuard.open().
