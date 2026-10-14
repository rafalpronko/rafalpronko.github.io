"""Feed canned, model-shaped proposals through the pipeline. No LLM is called."""
import asyncio
import json

from toolcall import LoopLimitExceeded, TurnBudget, handle_proposal, make_registry

PROPOSALS = [
    ("get_stock", '{"sku": "ABC-1234", "warehouse": "WAW"}'),
    ("get_stock", '{"sku": "ABC-1234", "warehouse": "WAW", "max_age_s": true}'),
    ("get_stock", '{"sku": "ABC-1234", "warehouse": "WAW", "sku": "XYZ-9999"}'),
    ("reserve_stock", '{"sku": "ABC-1234", "warehouse": "WAW", "qty": 3}'),
    ("get_stock", '{"sku": "QQQ-0000", "warehouse": "GDN"}'),
    ("get_stock", '{"sku": "XYZ-9999", "warehouse": "GDN"}'),
]


async def main() -> None:
    budget = TurnBudget(max_tool_calls=2, max_repairs=3)
    for i, (name, args) in enumerate(PROPOSALS, start=1):
        try:
            message = await handle_proposal(
                name, args, registry=make_registry(), scopes=frozenset({"inventory:read"}),
                budget=budget, call_id=f"call-{i}", request_id="req-demo",
                log=lambda event: print("  log", json.dumps(event, sort_keys=True)),
            )
        except LoopLimitExceeded as exc:
            print(f"{i}. stop: loop limit reached ({exc})")
            break
        print(f"{i}. {name} -> {message}")


if __name__ == "__main__":
    asyncio.run(main())
