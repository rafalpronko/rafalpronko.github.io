"""Post-hoc sensitivity check, run AFTER the main experiment. Not used for any selection.

Question: is the main ranking an artefact of the 300-step budget / lr grid edge?
Reports target VALIDATION MSE only; make_splits generates test arrays, but they
are never passed to select or evaluated here. Main experiment choices stay fixed.
"""
import dataclasses
import json
import sys

import lora_lab as L

out = {}
for steps, lrs in [(300, (0.01, 0.03, 0.1, 0.3)), (3000, (0.01, 0.03, 0.1, 0.3, 0.6))]:
    cfg = dataclasses.replace(L.Config(), steps=steps, lrs=lrs, eval_every=50 if steps > 300 else 10)
    key = f"steps={steps} lrs={list(lrs)} eval_every={cfg.eval_every}"
    out[key] = []
    for rep in range(cfg.replicates):
        s = L.make_splits(cfg, rep)
        sel = L.select(cfg, s["source_train"], s["target_train"], s["target_val"], replicate=rep)
        out[key].append({n: {"val_mse": round(m["val_mse"], 4), "lr": m["lr"], "step": m["step"]}
                         for n, m in sel["methods"].items() if n != "frozen_base"})
    print(key)
    for rep, row in enumerate(out[key]):
        print(f"  rep{rep} " + "  ".join(f"{n}={v['val_mse']:.4f}(lr={v['lr']},step={v['step']})" for n, v in row.items()))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/sensitivity_val_only.json", "w"), indent=2)
