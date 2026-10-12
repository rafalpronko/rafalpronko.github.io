"""Local CPU generation with google/flan-t5-small, pinned to one model revision."""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_ID = "google/flan-t5-small"
# Commit "Adding `safetensors` variant of this model (#14)", 2023-10-10.
MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"


class FlanT5Generator:
    def __init__(self, model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> None:
        self.model_id = model_id
        self.revision = revision
        self.tokenizer = T5Tokenizer.from_pretrained(model_id, revision=revision)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id, revision=revision)
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text).input_ids)

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
