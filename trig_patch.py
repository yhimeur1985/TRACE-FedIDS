import json, numpy as np, fl
from trigger_families import natural
T = json.load(open('data/triggers.json'))
for ds in ['edge', 'cic']:
    cfg = dict(fl.DEFAULT, dataset=ds)
    Xtr, ytr, gtr, Xte, yte, gte, classes, feats = fl.load(cfg, np.random.default_rng(0))
    model = fl.build_model(cfg, feats, len(classes))
    thetas = [np.load(f'results/theta_{ds}_fedavg_s{s}.npy') for s in (0, 1)]
    att = np.where(yte != 0)[0]
    X_att = Xte[np.random.default_rng(5).choice(att, min(4000, len(att)), replace=False)]
    groups, gnames = fl.make_groups(ds, feats)
    gof = {j: gi for gi, g in enumerate(groups) for j in g}
    lo, hi = Xtr.min(0), Xtr.max(0)
    if ds == 'edge' and 'distributed' not in T[ds]:
        # packets: only arp, tcp-flags and tcp-state carry multi-valued fields; spread six fields over
        # these three groups; low-cardinality codes take their least frequent non-zero value
        cand3 = [j for j in range(Xtr.shape[1]) if len(np.unique(Xtr[:20000, j])) >= 3]
        for sd in range(300):
            r = np.random.default_rng(20000 + sd)
            gl = sorted({gof[j] for j in cand3})
            fs = [int(r.choice([j for j in groups[g] if j in cand3])) for g in gl]
            while len(fs) < 6:
                j = int(r.choice(cand3))
                if j not in fs: fs.append(j)
            trig = []
            for j in fs:
                vals, cnt = np.unique(Xtr[:, j], return_counts=True)
                mode = vals[np.argmax(cnt)]
                nm = Xtr[:, j][Xtr[:, j] != mode]
                trig.append((j, float(np.quantile(nm, r.uniform(0.3, 0.9), method='nearest'))))
            nat = natural(model, thetas, X_att, trig)
            if nat < 0.02:
                T[ds]['distributed'] = dict(trigger=trig, natural=nat, feats=[feats[j] for j in fs],
                                            groups=sorted({gnames[gof[j]] for j in fs}), seed=sd)
                print(ds, 'distributed', T[ds]['distributed']['feats'], T[ds]['distributed']['groups'], nat, flush=True)
                break
    if ds == 'cic' and 'semantic' not in T[ds]:
        cols = sorted(groups[gnames.index('fwd-len')])
        order = np.random.default_rng(99).permutation(len(Xtr))
        best = None
        for t in order[:400]:
            rec = Xtr[t, cols]
            if len(np.unique(rec)) < 3:
                continue
            trig = [(int(j), float(v)) for j, v in zip(cols, rec)]
            nat = natural(model, thetas, X_att, trig)
            if best is None or nat < best[0]:
                best = (nat, t, trig)
            if nat < 0.02:
                break
        nat, t, trig = best
        T[ds]['semantic'] = dict(trigger=trig, natural=nat, feats=[feats[j] for j in cols], groups=['fwd-len'],
                                 template_row=int(t), template_class=classes[int(ytr[t])])
        print(ds, 'semantic', classes[int(ytr[t])], nat, flush=True)
json.dump(T, open('data/triggers.json', 'w'), indent=1)
