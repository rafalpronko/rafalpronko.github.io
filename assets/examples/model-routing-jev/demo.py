"""Run a few synthetic turns through the router and print each decision."""

from catalog import CATALOG
from routing import Policy, Proposal, Router, Runtime, Turn

saved: list[str] = []


def persist(runtime: Runtime) -> None:
    if runtime.route.alias == "mid-tools":
        raise OSError("session store unavailable")  # simulated persistence failure
    saved.append(runtime.route.alias)


router = Router(CATALOG, Runtime(CATALOG["full-tools"]), Policy(enabled=True), persist)

cases = [
    ("short question", Turn(1, user_tokens=150, history_tokens=2_000, history_format="plain"),
     Proposal("fast-text", 0.91)),
    ("same turn again", Turn(1, user_tokens=150, history_tokens=2_000, history_format="plain"),
     Proposal("full-tools", 0.95)),
    ("raw model id", Turn(2, user_tokens=150, history_tokens=2_100, history_format="plain"),
     Proposal("vendor/some-model-id", 0.99)),
    ("NaN confidence", Turn(3, user_tokens=150, history_tokens=2_200, history_format="plain"),
     Proposal("full-tools", float("nan"))),
    ("needs tools", Turn(4, user_tokens=200, history_tokens=6_000, history_format="plain",
                         needs_tools=True),
     Proposal("fast-text", 0.88, alternatives=("mid-tools", "full-tools"))),
    ("opaque history", Turn(5, user_tokens=120, history_tokens=3_000,
                            history_format="opaque-reasoning"),
     Proposal("fast-text", 0.93)),
    ("persist fails", Turn(6, user_tokens=100, history_tokens=1_000, history_format="plain",
                           needs_tools=True),
     Proposal("mid-tools", 0.90)),
]

for label, turn, proposal in cases:
    d = router.route_turn(turn, proposal)
    print(f"{label:16} -> {'SWITCH' if d.applied else 'keep  '} {d.route:10} "
          f"reason={d.reason} tokens={d.request_tokens} persisted={d.persisted}")
    for alias, why in d.rejected:
        print(f"{'':19}rejected {alias}: {why}")

print("live route:", router.runtime.route.alias, "| saved:", saved)
