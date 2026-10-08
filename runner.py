"""Experiment grid runner: 2 worker processes, one JSON per run, resumable."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import hashlib, json, sys, time, traceback, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "runs")
os.makedirs(RUNS, exist_ok=True)

AGGS = ["fedavg", "fedprox", "krum", "trmean", "median", "fltrust", "flame", "trace"]
ATTACKS = ["none", "lf", "sf", "alie", "bd"]
BASE = {"edge": dict(dataset="edge", partition="category"),
        "cic": dict(dataset="cic", partition="environment"),
        "cic_full": dict(dataset="cic_full", partition="environment")}
COMMON = dict(rounds=40, eval_every=5, lr=1e-3, local_steps=20)


def key(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def job(cfg):
    import fl
    k = key(cfg)
    path = os.path.join(RUNS, f"{cfg.get('tag','x')}_{k}.json")
    if os.path.exists(path):
        return path, "cached"
    t = time.time()
    try:
        res = fl.run(cfg)
        res["tag"] = cfg.get("tag")
        json.dump(res, open(path, "w"))
        return path, f"ok {time.time()-t:.0f}s {res['final']}"
    except Exception:
        return path, "FAIL " + traceback.format_exc()


def grid(name):
    J = []
    if name.startswith("main"):
        seeds = [int(x) for x in name.split(":")[1].split(",")] if ":" in name else [0, 1, 2]
        for seed in seeds:
            for ds in ["edge", "cic"]:
                for at in ATTACKS:
                    for ag in AGGS:
                        J.append(dict(BASE[ds], **COMMON, agg=ag, attack=at, seed=seed, tag="main"))
    elif name == "frac":
        for fr in [0.1, 0.3, 0.4]:
            for at in ["lf", "bd"]:
                for ag in ["fedavg", "median", "fltrust", "flame", "trace"]:
                    J.append(dict(BASE["edge"], **COMMON, agg=ag, attack=at, mal_frac=fr, seed=0, tag="frac"))
    elif name == "hetero":
        parts = [dict(partition="iid"), dict(partition="dirichlet", alpha=1.0),
                 dict(partition="dirichlet", alpha=0.1), dict(partition="protocol"),
                 dict(partition="attack")]
        for p in parts:
            for at in ["none", "lf"]:
                for ag in ["fedavg", "median", "fltrust", "trace"]:
                    J.append(dict(BASE["edge"], **COMMON, **p, agg=ag, attack=at, seed=0, tag="hetero"))
    elif name.startswith("ablation"):
        variants = {"no_agree": dict(use_agree=False), "no_reliab": dict(use_reliab=False),
                    "no_probe": dict(signals=["mc", "det"]), "no_history": dict(history=False),
                    "no_clip": dict(clip=False), "mean_reliab": dict(reliab="mean"),
                    "median_agree": dict(agree="median"), "no_redund": dict(redundancy=False)}
        ds = name.split(":")[1] if ":" in name else "edge"
        keep = list(variants) if ds == "edge" else ["no_agree", "no_reliab", "no_probe", "no_history"]
        for at in ATTACKS:
            for vn in keep:
                J.append(dict(BASE[ds], **COMMON, agg="trace", attack=at, seed=0,
                              trace=variants[vn], tag=f"abl-{vn}"))
    elif name == "root":
        for ds in ["edge", "cic"]:
            for rp in [5, 50]:
                for at in ["lf", "bd"]:
                    for ag in ["fltrust", "trace"]:
                        J.append(dict(BASE[ds], **COMMON, agg=ag, attack=at, root_per_class=rp, seed=0, tag="root"))
        for at in ["lf", "bd"]:
            for ag in ["fltrust", "trace"]:
                J.append(dict(BASE["cic"], **COMMON, agg=ag, attack=at, root_bias=["ACI-IoT-2023"], seed=0,
                              tag="rootbias"))
    elif name == "backbone":
        for at in ATTACKS:
            for ag in ["fedavg", "median", "fltrust", "trace"]:
                J.append(dict(BASE["edge"], **COMMON, agg=ag, attack=at, model="mlp", seed=0, tag="mlp"))
    return J


if __name__ == "__main__":
    names = sys.argv[1:]
    jobs = [j for n in names for j in grid(n)]
    print(len(jobs), "jobs", flush=True)
    with Pool(2, maxtasksperchild=1) as pool:
        for i, (p, msg) in enumerate(pool.imap_unordered(job, jobs)):
            print(f"[{i+1}/{len(jobs)}] {os.path.basename(p)} {msg[:300]}", flush=True)
