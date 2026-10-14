"""Small behavioral stubs, NOT the DSPy or OpenAI SDK implementation.

Models the pinned module-client boundary, SDK internal retries, DSPy outer
rate-limit retries and missing-field recovery. It does not prove real SDK
integration or real DSPy template extraction. No network is available here.
"""

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace


class RateLimitError(Exception):
    pass


class SDKStub(ModuleType):
    def __init__(self):
        super().__init__("openai")
        self.__version__ = "1.30.5"
        self.max_retries = 2
        self.responses = []
        self.http_attempts = 0
        self.payloads = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        assert "max_retries" not in kwargs  # client option, not payload
        self.payloads.append(kwargs)
        for retry in range(self.max_retries + 1):
            self.http_attempts += 1
            raw = self.responses.pop(0)
            if raw == "rate-limit":
                if retry == self.max_retries:
                    raise RateLimitError()
                continue
            response = {"choices": [{"message": {"content": raw}}],
                        "usage": {"total_tokens": 4}}
            return SimpleNamespace(
                model_dump=lambda response=response: response
            )
        raise AssertionError("unreachable")


class SettingsStub:
    def __init__(self):
        self.lm = None

    def configure(self, **kwargs):
        self.__dict__.update(kwargs)

    @contextmanager
    def context(self, **kwargs):
        old = self.__dict__.copy()
        self.__dict__.update(kwargs)
        try:
            yield
        finally:
            self.__dict__.clear()
            self.__dict__.update(old)


def dspy_stub(sdk):
    module = ModuleType("dspy")
    module.settings = SettingsStub()
    module.predict_calls = []

    class GPT3Stub:
        def __init__(self, model="test-model", model_type="chat",
                     system_prompt=None, **kwargs):
            self.kwargs = {"model": model, **kwargs}
            self.model_type = model_type
            self.system_prompt = system_prompt

        def basic_request(self, prompt, **kwargs):
            response = sdk.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs},
            )
            return response.model_dump()

        def request(self, prompt, **kwargs):
            # Simulated DSPy backoff. No sleep; budget exception escapes.
            for _ in range(10):
                try:
                    return self.basic_request(prompt, **kwargs)
                except RateLimitError:
                    continue
            raise RateLimitError()

        def __call__(self, prompt, **kwargs):
            return [c["message"]["content"]
                    for c in self.request(prompt, **kwargs)["choices"]]

    class SignatureStub:
        @classmethod
        def with_instructions(cls, instruction):
            return SimpleNamespace(instructions=instruction,
                                   fields=("ticket", "queue"))

    class ModuleStub:
        def __call__(self, **kwargs):
            return self.forward(**kwargs)

    class ExampleStub(SimpleNamespace):
        def with_inputs(self, *keys):
            self.input_keys = keys
            return self

    class PredictStub:
        def __init__(self, signature):
            self.signature = signature
            self.demos = []

        def __call__(self, **kwargs):
            module.predict_calls.append((self.signature, self.demos, kwargs))
            for _ in range(3):
                raw = module.settings.lm("Ticket: " + kwargs["ticket"])[0]
                if raw:
                    return SimpleNamespace(queue=raw)
            raise RuntimeError("missing field after recovery")

    module.OpenAI = GPT3Stub
    module.Signature = SignatureStub
    module.Module = ModuleStub
    module.Example = ExampleStub
    module.Predict = PredictStub
    module.InputField = lambda **kwargs: None
    module.OutputField = lambda **kwargs: None
    teleprompt = ModuleType("dspy.teleprompt")
    teleprompt.BootstrapFewShot = object
    return module, teleprompt
