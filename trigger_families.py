"""Select held-out backdoor trigger families (not used when TRACE was designed).

Each family is chosen only for being inert on clean federated models (FedAvg, seeds 0 and 1):
fewer than 2 % of the detected test attacks may flip to benign when the trigger is applied.
  midrange     3 features set to the median of their non-modal values (no extremes)
  distributed  6 features from 6 different semantic groups, set to the 75th percentile of their
               non-modal values (spans more groups than the probe covers)
  semantic     the full TCP header state (packets: tcp-flags + tcp-state groups) or the forward
               packet-length statistics (flows: fwd-len group) copied jointly from one real training
               record, so the triggered record keeps a protocol-consistent field combination
  cleanlabel   the midrange trigger, but poisoning uses correctly labelled benign records only
Writes data/triggers.json.
"""
import json, os
import numpy as np
import fl
from model import predict

HERE = os.path.dirname(os.path.abspath(__file__))


def natural(model, thetas, X_att, trig):
    rates = []
    for th in thetas:
        p = predict(model, th, X_att).argmax(1)
        det = p != 0
        pt = predict(model, th, fl.apply_trigger(X_att[det], trig)).argmax(1)
        rates.append(float((pt == 0).mean()))
    return max(rates)


def main():
    out = {}
    for ds in ["edge", "cic"]:
        cfg = dict(fl.DEFAULT, dataset=ds)
        Xtr, ytr, gtr, Xte, yte, gte, classes, feats = fl.load(cfg, np.random.default_rng(0))
        model = fl.build_model(cfg, feats, len(classes))
        thetas = [np.load(os.path.join(HERE, "results", f"theta_{ds}_fedavg_s{s}.npy")) for s in (0, 1)]
        att = np.where(yte != 0)[0]
        X_att = Xte[np.random.default_rng(5).choice(att, min(4000, len(att)), replace=False)]
        groups, gnames = fl.make_groups(ds, feats)
        gof = {j: gi for gi, g in enumerate(groups) for j in g}
        lo, hi = Xtr.min(0), Xtr.max(0)
        cand = [j for j in range(Xtr.shape[1]) if len(np.unique(Xtr[:20000, j])) >= 5]

        def nonmode_q(j, q):
            x = Xtr[:, j]
            vals, cnt = np.unique(x, return_counts=True)
            mode = vals[np.argmax(cnt)]
            nm = x[x != mode]
            v = float(np.quantile(nm, q, method="nearest"))
            return v
        res = {}
        # midrange
        for sd in range(200):
            fs = np.random.default_rng(sd).choice(cand, 3, replace=False)
            trig = [(int(j), nonmode_q(j, 0.5)) for j in fs]
            if any(v <= lo[j] or v >= hi[j] for j, v in trig):
                continue
            nat = natural(model, thetas, X_att, trig)
            if nat < 0.02:
                res["midrange"] = dict(trigger=trig, natural=nat, feats=[feats[j] for j, _ in trig],
                                       groups=sorted({gnames[gof[j]] for j, _ in trig}), seed=sd)
                break
        # distributed
        cand3 = [j for j in range(Xtr.shape[1]) if len(np.unique(Xtr[:20000, j])) >= 3]
        multi = [gi for gi, g in enumerate(groups) if any(j in cand3 for j in g)]
        ng = min(6, len(multi))
        for sd in range(400):
            r = np.random.default_rng(10000 + sd)
            gsel = r.choice(multi, ng, replace=False)
            fs = [int(r.choice([j for j in groups[g] if j in cand3])) for g in gsel]
            while len(fs) < 6:   # fewer groups than 6: add further features from the chosen groups
                g = int(r.choice(gsel)); j = int(r.choice([j for j in groups[g] if j in cand3]))
                if j not in fs:
                    fs.append(j)
            trig = [(j, nonmode_q(j, 0.75)) for j in fs]
            if any(v <= lo[j] or v >= hi[j] for j, v in trig):
                continue
            nat = natural(model, thetas, X_att, trig)
            if nat < 0.02:
                res["distributed"] = dict(trigger=trig, natural=nat, feats=[feats[j] for j, _ in trig],
                                          groups=sorted({gnames[gof[j]] for j, _ in trig}), seed=sd)
                break
        # semantic: jointly copied field combination from a real record
        gsem = ["tcp-flags", "tcp-state"] if ds == "edge" else ["fwd-len"]
        cols = sorted(j for gname in gsem for j in groups[gnames.index(gname)])
        r = np.random.default_rng(99)
        order = r.permutation(len(Xtr))
        for t in order[:3000]:
            rec = Xtr[t, cols]
            inside = np.mean([(lo[j] < v < hi[j]) for j, v in zip(cols, rec)])
            nonzero_like = len(np.unique(rec)) >= 3
            if inside < 0.5 or not nonzero_like:
                continue
            trig = [(int(j), float(v)) for j, v in zip(cols, rec)]
            nat = natural(model, thetas, X_att, trig)
            if nat < 0.02:
                res["semantic"] = dict(trigger=trig, natural=nat, feats=[feats[j] for j in cols], groups=gsem,
                                       template_row=int(t), template_class=classes[int(ytr[t])])
                break
        res["cleanlabel"] = dict(res["midrange"])
        res["extreme"] = dict(trigger=fl.build_trigger(dict(cfg, trigger="extreme"), Xtr, feats))
        res["extreme"]["natural"] = natural(model, thetas, X_att, res["extreme"]["trigger"])
        out[ds] = res
        for k, v in res.items():
            print(ds, k, v.get("feats"), v.get("groups"), "natural", round(v["natural"], 4), flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "triggers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
