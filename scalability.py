"""Server-side cost of the aggregation rules versus the number of clients n (one round).

Updates are drawn from the clients of a real round so that the geometry is realistic: we train
20 honest Edge-IIoTset updates from the initial model and create n updates by resampling them with
small Gaussian perturbations. Reports wall time (single thread) and peak additional memory
(tracemalloc) per round, and the per-client upload size."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json, time, tracemalloc, copy
import numpy as np
import fl


def main():
    out = []
    for ds in ["edge", "cic"]:
        cfg = dict(fl.DEFAULT, dataset=ds, partition={"edge": "category", "cic": "environment"}[ds], lr=1e-3,
                   local_steps=20)
        if ds == "cic":
            cfg["env_clients"] = {"ACI-IoT-2023": 4, "Edge-IIoT-2022": 4, "IoMT-2024": 4,
                                  "MQTT-IoT-2020": 3, "IoT-HCRL-2019": 3, "IoT-2022": 2}
        rng = np.random.default_rng(0)
        Xtr, ytr, gtr, Xte, yte, gte, classes, feats = fl.load(cfg, rng)
        if ds == "edge":
            tax = [["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP"], ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner"],
                   ["SQL_injection", "Uploading", "XSS"], ["Backdoor", "Password", "Ransomware"]]
            cfg["categories"] = [[classes.index(c) for c in grp] for grp in tax]
        root = fl.take_root(ytr, gtr, cfg, rng)
        avail = np.setdiff1d(np.arange(len(ytr)), root)
        clients = fl.partition(ytr, gtr, cfg, rng, avail)
        model = fl.build_model(cfg, feats, len(classes))
        theta = model.init(np.random.default_rng(100))
        base = np.stack([fl.local_train(model, theta, Xtr[c], ytr[c], cfg, np.random.default_rng(i))
                         for i, c in enumerate(clients)])
        Xr, yr = Xtr[root], ytr[root]
        g0 = fl.local_train(model, theta, Xr, yr, cfg, np.random.default_rng(999))
        groups = fl.make_groups(ds, feats)[0]
        probes, m = fl.make_probes(Xr, yr, (Xtr.min(0), Xtr.max(0)), groups)
        for n in [20, 50, 100, 200]:
            r = np.random.default_rng(n)
            U = base[r.integers(0, len(base), n)] + r.normal(0, 0.01 * np.abs(base).mean(), (n, base.shape[1])).astype(np.float32)
            U = U.astype(np.float32)
            sizes = np.ones(n)
            f = n // 5
            rules = {
                "FedAvg": lambda: fl.agg_fedavg(U, sizes),
                "Median": lambda: fl.agg_median(U),
                "Krum": lambda: fl.agg_krum(U, f),
                "FLTrust": lambda: fl.agg_fltrust(U, g0),
                "FLAME": lambda: fl.agg_flame(U, np.random.default_rng(0)),
                "Zeno": lambda: fl.agg_zeno(U, model, theta, Xr, yr, f),
                "FoolsGold": lambda: fl.FoolsGold(n)(U),
                "TRACE": lambda: fl.TRACE(n, None, probes, m)(U, model=model, theta=theta, Xr=Xr, yr=yr,
                                                            n_cls=len(classes), g0=g0),
            }
            def gram():
                Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
                return Un @ Un.T
            gram(); tg = time.perf_counter(); gram(); tg = time.perf_counter() - tg
            for name, fn in rules.items():
                fn()  # warm-up
                tracemalloc.start()
                t = time.perf_counter()
                fn()
                dt = time.perf_counter() - t
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                out.append(dict(dataset=ds, n=n, rule=name, seconds=dt, peak_mb=peak / 2 ** 20,
                                upload_kb=4 * U.shape[1] / 1024, gram_seconds=tg))
                print(out[-1], flush=True)
    json.dump(out, open("results/scalability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
