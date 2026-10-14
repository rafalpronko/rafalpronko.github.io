"""Fictional host catalog. Aliases, adapters and model ids are made up; numbers are synthetic."""

from routing import Route

PLAIN = frozenset({"plain"})

CATALOG = {
    "fast-text": Route("fast-text", adapter="adapter-a", model="example-small",
                       context_tokens=8_000, output_budget=1_000, system_prompt_tokens=300,
                       tool_schema_tokens=0, tools=False, attachments=False, replays=PLAIN),
    "mid-tools": Route("mid-tools", adapter="adapter-b", model="example-medium",
                       context_tokens=12_000, output_budget=2_000, system_prompt_tokens=900,
                       tool_schema_tokens=3_000, tools=True, attachments=False, replays=PLAIN),
    "full-tools": Route("full-tools", adapter="adapter-b", model="example-large",
                        context_tokens=64_000, output_budget=4_000, system_prompt_tokens=1_200,
                        tool_schema_tokens=2_500, tools=True, attachments=True, replays=PLAIN),
}
