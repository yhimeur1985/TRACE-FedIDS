"""Centralised reference: the same backbones trained on the pooled client data."""
import json, sys, time
import numpy as np
import fl
from model import softmax_ce, predict


def central(dataset, model_kind, steps=1500, batch=128, lr=2e-3, seed=0, d=32):
    cfg = dict(fl.DEFAULT, dataset=dataset, model=model_kind, d=d, lr=lr, batch=batch, seed=seed)
    rng = np.random.default_rng(seed)
    Xtr, ytr, gtr, Xte, yte, gte, classes, feats = fl.load(cfg, rng)
    model = fl.build_model(cfg, feats, len(classes))
    theta = model.init(np.random.default_rng(seed + 100))
    t0 = time.time()
    out = []
    chunk = 250
    for s in range(0, steps, chunk):
        theta = theta + fl.local_train(model, theta, Xtr, ytr, cfg, rng, steps=chunk)
        ev = fl.evaluate(model, theta, Xte, yte, len(classes), None)
        ev["step"] = s + chunk
        out.append(ev)
        print(dataset, model_kind, ev, f"{time.time()-t0:.0f}s", flush=True)
    return dict(dataset=dataset, model=model_kind, n_params=int(model.spec.size), hist=out,
                seconds=time.time() - t0)


if __name__ == "__main__":
    ds, mk = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    r = central(ds, mk, steps, seed=seed)
    json.dump(r, open(f"results/central_{ds}_{mk}" + (f"_s{seed}" if seed else "") + ".json", "w"))
