"""Federated intrusion detection under non-IID and Byzantine clients (NumPy simulation).

run(cfg) -> dict with per-round metrics and final metrics.
"""
import json, os, time, argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from scipy.stats import norm as _norm
from model import FTTransformer, MLP, softmax_ce, predict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DEFAULT = dict(dataset="edge", partition="attack", n_clients=20, alpha=0.3, rounds=40,
               local_steps=15, batch=64, lr=2e-3, prox_mu=0.0, model="ftt-g", d=32, blocks=2,
               heads=4, agg="fedavg", attack="none", mal_frac=0.2, root_per_class=10,
               root_bias=None, seed=0, eval_every=5, test_cap=20000, bd_boost=2.0,
               sf_scale=4.0, trace=None, log=False, opt="adam", server_mom=0.0)

TRACE_DEFAULT = dict(clip=True, agree="both", reliab="classwise", history=True,
                     beta_up=0.98, beta_down=0.5, kappa=3.0, kappa_a=2.0, temp=1.0, sign_temp=0.1, det_scope="attack", evid=0.0, server_client=True, use_agree=True,
                     consistency=False, cons_window=5, kappa_c=3.0,
                     use_reliab=True, signals=("mc", "det", "probe"), redundancy=True)


# ------------------------------------------------------------------ data
def load(cfg, rng):
    z = np.load(os.path.join(DATA, f"{cfg['dataset']}.npz"), allow_pickle=True)
    Xtr, ytr, gtr = z["X_train"], z["y_train"], z["g_train"]
    Xte, yte, gte = z["X_test"], z["y_test"], z["g_test"]
    classes = list(z["classes"])
    feats = list(z["features"])
    # signed log transform + standardisation (moments = secure-aggregation of sums)
    f = lambda X: np.sign(X) * np.log1p(np.abs(X))
    Xtr, Xte = f(Xtr), f(Xte)
    if cfg.get("prep") != "root_robust":
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-6] = 1.0
        Xtr = np.clip((Xtr - mu) / sd, -10, 10).astype(np.float32)
        Xte = np.clip((Xte - mu) / sd, -10, 10).astype(np.float32)
    if cfg["dataset"].startswith("cic"):  # drop near-duplicate flow statistics (public schema decision)
        keep = prune_redundant(Xtr, 0.98)
        Xtr, Xte, feats = Xtr[:, keep], Xte[:, keep], [feats[i] for i in keep]
    # fixed stratified test subsample
    trng = np.random.default_rng(12345)
    if len(yte) > cfg["test_cap"]:
        idx = []
        for c in np.unique(yte):
            ic = np.where(yte == c)[0]
            k = max(1, int(round(cfg["test_cap"] * len(ic) / len(yte))))
            idx.append(trng.choice(ic, min(k, len(ic)), replace=False))
        idx = np.sort(np.concatenate(idx))
        Xte, yte, gte = Xte[idx], yte[idx], gte[idx]
    return Xtr, ytr, gtr, Xte, yte, gte, classes, feats


def prune_redundant(X, thr):
    s = X[np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)]
    C = np.abs(np.corrcoef(s.T))
    C = np.nan_to_num(C)
    keep = []
    for j in range(X.shape[1]):
        if all(C[j, k] < thr for k in keep):
            keep.append(j)
    return keep


def take_root(ytr, gtr, cfg, rng):
    """Small trusted, class-balanced root set held by the server."""
    idx = []
    pool = np.arange(len(ytr))
    if cfg["root_bias"]:  # server only has data from some groups (e.g. one environment)
        pool = pool[np.isin(gtr, cfg["root_bias"])]
    for c in np.unique(ytr):
        ic = pool[ytr[pool] == c]
        if len(ic) == 0:
            continue
        if cfg.get("root_mode") == "env":   # k records per (environment, class)
            for e in np.unique(gtr[ic]):
                ie = ic[gtr[ic] == e]
                idx.append(rng.choice(ie, min(cfg["root_per_class"], len(ie)), replace=False))
        else:
            idx.append(rng.choice(ic, min(cfg["root_per_class"], len(ic)), replace=False))
    return np.concatenate(idx)


def corrupt_root(root, ytr, cfg, classes):
    """Root-set stress tests: drop classes, random label noise, or contamination (a fraction of
    the root attack records labelled benign). Returns (root indices, root labels)."""
    yr = ytr[root].copy()
    rr = np.random.default_rng([cfg["seed"], 31337])
    if cfg.get("root_drop"):
        drop = [classes.index(c) for c in cfg["root_drop"]]
        keep = ~np.isin(yr, drop)
        root, yr = root[keep], yr[keep]
    if cfg.get("root_noise"):
        k = int(round(cfg["root_noise"] * len(yr)))
        sel = rr.choice(len(yr), k, replace=False)
        for j in sel:
            yr[j] = rr.choice([c for c in range(len(classes)) if c != yr[j]])
    if cfg.get("root_contam"):
        att = np.where(yr != 0)[0]
        k = int(round(cfg["root_contam"] * len(att)))
        yr[rr.choice(att, k, replace=False)] = 0
    return root, yr


def build_trigger(cfg, Xtr, feats):
    """Backdoor triggers. 'extreme' is the trigger of the main experiments; the other families are
    held out from TRACE's design and were selected only for being inert on clean models
    (data/triggers.json, produced by trigger_families.py)."""
    kind = cfg.get("trigger", "extreme")
    if kind == "extreme":
        cand = [j for j in range(Xtr.shape[1]) if len(np.unique(Xtr[:5000, j])) >= 10]
        tseed, tval = {"edge": (777, "min"), "cic": (33, "max"), "cic_full": (33, "max")}[cfg["dataset"]]
        tf = np.random.default_rng(tseed).choice(cand, 3, replace=False)
        return [(int(j), float(Xtr[:, j].min() if tval == "min" else Xtr[:, j].max())) for j in tf]
    spec = json.load(open(os.path.join(DATA, "triggers.json")))[cfg["dataset"]][kind]
    return [(int(j), float(v)) for j, v in spec["trigger"]]


def partition(ytr, gtr, cfg, rng, avail):
    n = cfg["n_clients"]
    avail = np.array(avail)
    y, g = ytr[avail], gtr[avail]
    parts = [[] for _ in range(n)]
    kind = cfg["partition"]
    if kind == "iid":
        perm = rng.permutation(len(avail))
        for i, ch in enumerate(np.array_split(perm, n)):
            parts[i] = list(ch)
    elif kind == "dirichlet":
        for c in np.unique(y):
            ic = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(cfg["alpha"] * np.ones(n))
            cuts = (np.cumsum(p) * len(ic)).astype(int)[:-1]
            for i, ch in enumerate(np.split(ic, cuts)):
                parts[i] += list(ch)
    elif kind == "attack":
        # every client holds benign traffic plus 3 attack classes (pathological label skew),
        # client sizes are log-normal (quantity skew)
        classes = [c for c in np.unique(y) if c != 0]
        k = 3
        slots = []
        order = list(rng.permutation(classes))
        while len(slots) < n * k:
            slots += order
        owners = {c: [] for c in classes}
        for i in range(n):
            mine = []
            j = i * k
            for c in slots[j:j + k]:
                if c not in mine:
                    mine.append(c)
            for c in mine:
                owners[c].append(i)
        size = rng.lognormal(0, 0.6, n)
        for c in np.unique(y):
            ic = rng.permutation(np.where(y == c)[0])
            own = list(range(n)) if c == 0 else owners[c]
            w = size[own] / size[own].sum()
            cuts = (np.cumsum(w) * len(ic)).astype(int)[:-1]
            for i, ch in zip(own, np.split(ic, cuts)):
                parts[i] += list(ch)
    elif kind == "category":
        # Edge-IIoTset threat taxonomy: each gateway is exposed to two of the four attack
        # categories (all attack classes of those categories) plus benign traffic
        cats = cfg["categories"]
        pairs = [(a, b) for a in range(len(cats)) for b in range(a + 1, len(cats))]
        order = [pairs[k % len(pairs)] for k in rng.permutation(n)]
        owners = {}
        for i, (a, b) in enumerate(order):
            for cidx in (a, b):
                for c in cats[cidx]:
                    owners.setdefault(c, []).append(i)
        size = rng.lognormal(0, 0.6, n)
        for c in np.unique(y):
            ic = rng.permutation(np.where(y == c)[0])
            own = list(range(n)) if c == 0 else owners[c]
            w = size[own] / size[own].sum()
            cuts = (np.cumsum(w) * len(ic)).astype(int)[:-1]
            for i, ch in zip(own, np.split(ic, cuts)):
                parts[i] += list(ch)
    elif kind == "env_capture":
        # capture-disjoint sites: within each testbed, whole source files (captures) are assigned to
        # its sites (largest file first to the currently smallest site), so no two clients share a capture
        files = cfg["_files"][avail]
        envs = sorted(set(g))
        i0 = 0
        for e in envs:
            k = cfg["env_clients"][e]
            fe = [(fn, np.where((g == e) & (files == fn))[0]) for fn in sorted(set(files[g == e]))]
            fe.sort(key=lambda t: -len(t[1]))
            load = np.zeros(k)
            for fn, idx in fe:
                j = int(np.argmin(load))
                parts[i0 + j] += list(idx)
                load[j] += len(idx)
            i0 += k
    elif kind in ("protocol", "environment"):
        # clients = gateways of one protocol family / sites of one environment (feature+label skew)
        if kind == "protocol":  # rare protocol families (<2 % of packets) join "other"
            gg, cc = np.unique(g, return_counts=True)
            rare = set(gg[cc < 0.02 * len(g)])
            g = np.array(["other" if v in rare else v for v in g])
        groups, counts = np.unique(g, return_counts=True)
        alloc = np.maximum(1, np.round(counts / counts.sum() * n)).astype(int)
        if kind == "environment":
            alloc = np.array([cfg.get("env_clients", {}).get(gr, a) for gr, a in zip(groups, alloc)])
        while alloc.sum() > n:
            alloc[np.argmax(alloc)] -= 1
        while alloc.sum() < n:
            alloc[np.argmax(counts / alloc)] += 1
        i0 = 0
        for gr, a in zip(groups, alloc):
            ig = rng.permutation(np.where(g == gr)[0])
            w = rng.lognormal(0, 0.4, a); w /= w.sum()
            cuts = (np.cumsum(w) * len(ig)).astype(int)[:-1]
            for j, ch in enumerate(np.split(ig, cuts)):
                parts[i0 + j] = list(ch)
            i0 += a
    else:
        raise ValueError(kind)
    return [avail[np.array(sorted(p), dtype=int)] for p in parts]


def heterogeneity(parts, ytr, Xtr, n_cls, rng):
    """Mean pairwise Jensen-Shannon divergence of client label distributions and mean
    pairwise (linear-kernel) MMD^2 of client feature means."""
    P = np.array([np.bincount(ytr[p], minlength=n_cls) / max(1, len(p)) for p in parts])
    def js(p, q):
        m = 0.5 * (p + q)
        kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    n = len(parts)
    jsd = np.mean([js(P[i], P[j]) for i in range(n) for j in range(i + 1, n)])
    M = np.array([Xtr[rng.choice(p, min(500, len(p)), replace=False)].mean(0) for p in parts])
    mmd = np.mean([np.sum((M[i] - M[j]) ** 2) for i in range(n) for j in range(i + 1, n)])
    return float(jsd), float(mmd)


def make_groups(dataset, feats):
    """Semantic (protocol / statistic-family) feature groups used as Transformer tokens."""
    if dataset == "edge":
        rules = [("arp", lambda f: f.startswith("arp.")), ("http", lambda f: f.startswith("http.")),
                 ("tcp-flags", lambda f: f.startswith("tcp.connection") or f.startswith("tcp.flags")),
                 ("tcp-state", lambda f: f.startswith("tcp.")), ("udp", lambda f: f.startswith("udp.")),
                 ("dns", lambda f: f.startswith("dns.")),
                 ("mqtt-ctrl", lambda f: f.startswith("mqtt.") and ("flag" in f or "msgtype" in f)),
                 ("mqtt-hdr", lambda f: f.startswith("mqtt."))]
    else:
        rules = [("fwd-len", lambda f: f.startswith("Fwd Packet Length") or f in ("Fwd Seg Size Min", "Fwd Header Length", "Fwd Segment Size Avg")),
                 ("bwd-len", lambda f: f.startswith("Bwd Packet Length") or f in ("Bwd Header Length", "Bwd Segment Size Avg")),
                 ("pkt-len", lambda f: f.startswith("Packet Length") or f == "Average Packet Size"),
                 ("rate", lambda f: "/s" in f),
                 ("flow-iat", lambda f: f.startswith("Flow IAT") or f == "Flow Duration"),
                 ("fwd-iat", lambda f: f.startswith("Fwd IAT")), ("bwd-iat", lambda f: f.startswith("Bwd IAT")),
                 ("flags", lambda f: "Flag" in f),
                 ("bulk", lambda f: "Bulk" in f), ("window", lambda f: "Init Win" in f),
                 ("active-idle", lambda f: f.startswith("Active") or f.startswith("Idle")),
                 ("volume", lambda f: True)]
    groups, names, used = [], [], set()
    for name, rule in rules:
        g = [j for j, f in enumerate(feats) if j not in used and rule(str(f))]
        if g:
            groups.append(g); names.append(name); used |= set(g)
    assert len(used) == len(feats)
    return groups, names


# ------------------------------------------------------------------ training
def local_train(model, theta, X, y, cfg, rng, steps=None, prox_mu=0.0):
    w = theta.copy()
    m = np.zeros_like(w); v = np.zeros_like(w)
    b1, b2, lr, eps = 0.9, 0.999, cfg["lr"], 1e-8
    steps = steps or cfg["local_steps"]
    adj = 0.0
    if cfg.get("la", False):
        # label-distribution-calibrated (balanced-softmax) local loss: logits are shifted by the
        # log of the local label prior during training only, so a client does not suppress the
        # classes it never observes (logit adjustment / restricted softmax)
        prior = np.bincount(y, minlength=model.C) / len(y)
        adj = np.log(prior + 1e-4).astype(np.float32)[None, :]
    for t in range(1, steps + 1):
        bi = rng.integers(0, len(y), min(cfg["batch"], len(y)))
        logits, cache = model.forward(w, X[bi], train=True)
        _, dl = softmax_ce(logits + adj, y[bi])
        g = model.backward(w, dl, cache)
        if prox_mu > 0:
            g += prox_mu * (w - theta)
        g += 1e-5 * w
        if cfg.get("opt", "adam") == "sgd":
            m = 0.9 * m + g
            w -= lr * m
            continue
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return w - theta


def control_groups(dataset, feats, kind, seed=0):
    """Grouping controls with the same number of tokens as the semantic grouping:
    'random'  -- random assignment of features to groups with the semantic group sizes;
    'contig'  -- equal-size contiguous chunks of the original column order."""
    sem = make_groups(dataset, feats)[0]
    F, G = len(feats), len(sem)
    if kind == "random":
        perm = np.random.default_rng(1000 + seed).permutation(F)
        out, i0 = [], 0
        for g in sem:
            out.append(sorted(perm[i0:i0 + len(g)].tolist())); i0 += len(g)
        return out
    return [list(ch) for ch in np.array_split(np.arange(F), G)]


def build_model(cfg, feats, n_cls):
    if cfg["model"] == "mlp":
        return MLP(len(feats), n_cls)
    if cfg["model"] in ("ftt-g-random", "ftt-g-contig"):
        groups = control_groups(cfg["dataset"], feats, cfg["model"].split("-")[-1], cfg["seed"])
        return FTTransformer(len(feats), n_cls, d=cfg["d"], n_blocks=cfg["blocks"], n_heads=cfg["heads"], groups=groups)
    groups = make_groups(cfg["dataset"], feats)[0] if cfg["model"] == "ftt-g" else None
    return FTTransformer(len(feats), n_cls, d=cfg["d"], n_blocks=cfg["blocks"], n_heads=cfg["heads"], groups=groups)


def apply_trigger(X, trig):
    X = X.copy()
    for j, val in trig:
        X[:, j] = val
    return X


# ------------------------------------------------------------------ attacks
def malicious_updates(kind, model, theta, clients, mal, Xtr, ytr, cfg, rng, trig, honest_cache, benign=()):
    out = {}
    if kind == "lf":
        for i in mal:
            X, y = Xtr[clients[i]], ytr[clients[i]].copy()
            if cfg.get("lf_idx") is not None:   # targeted: only one attack family is relabelled
                y[np.isin(y, cfg["lf_idx"])] = 0
            else:
                y[y != 0] = 0  # attack traffic relabelled as benign
            out[i] = local_train(model, theta, X, y, cfg, rng)
    elif kind == "sf":
        for i in mal:
            out[i] = -cfg["sf_scale"] * honest_cache(i)
    elif kind == "alie":
        H = np.stack([honest_cache(i) for i in mal])
        mu, sd = H.mean(0), H.std(0)
        n, f = cfg["n_clients"], len(mal)
        s = np.floor(n / 2 + 1) - f
        z = _norm.ppf((n - s) / n)
        for i in mal:
            out[i] = (mu - z * sd).astype(np.float32)
    elif kind == "bd":
        for i in mal:
            out[i] = cfg["bd_boost"] * backdoor_update(model, theta, Xtr[clients[i]], ytr[clients[i]], cfg, rng, trig)
    elif kind == "minmax":
        # Min-Max (Shejwalkar & Houmansadr, NDSS 2021), AGR-agnostic, std perturbation: the
        # malicious update mu - gamma*std stays within the maximum pairwise benign distance
        B = np.stack([honest_cache(i) for i in benign]).astype(np.float64)
        mu, sd = B.mean(0), B.std(0)
        dmax = np.sqrt(pairwise_sq(B).max())
        lo, hi = 0.0, 50.0
        for _ in range(25):
            g = (lo + hi) / 2
            m = mu - g * sd
            if np.sqrt(((B - m) ** 2).sum(1)).max() <= dmax:
                lo = g
            else:
                hi = g
        for i in mal:
            out[i] = (mu - lo * sd).astype(np.float32)
    return out


def backdoor_update(model, theta, X, y, cfg, rng, trig):
    """Backdoor training: a triggered copy of half of the client's attack records labelled benign
    (default), or, for clean-label poisoning, triggered copies of half of its benign records with
    their correct (benign) label."""
    if cfg.get("clean_label"):
        ben = np.where(y == 0)[0]
        sel = rng.choice(ben, max(1, len(ben) // 2), replace=False) if len(ben) else ben
    else:
        att = np.where(y != 0)[0]
        sel = rng.choice(att, max(1, len(att) // 2), replace=False) if len(att) else att
    Xp = np.concatenate([X, apply_trigger(X[sel], trig)])
    yp = np.concatenate([y, np.zeros(len(sel), int)])
    return local_train(model, theta, Xp, yp, cfg, rng)


def adaptive_updates(kind, model, theta, clients, mal, Xtr, ytr, cfg, rng, trig, honest_cache, ids, U_benign,
                     trace, sur, n_cls, oracle_root):
    """TRACE-aware white-box attack. The attackers know the algorithm, its hyperparameters, the
    current reputations and (omnisciently) the benign updates of the round, but not the server's
    root set: they emulate it with a class-balanced surrogate drawn from their own data (unless
    oracle_root). Each attacker submits Delta_h + alpha (Delta_p - Delta_h), where Delta_h is its
    honest update and Delta_p its poisoned (backdoor or label-flip) update; a bisection finds the
    largest alpha in [0, 2] for which the emulated TRACE leaves every attacker ungated (s >= 0.95)."""
    import copy
    Hm, Pm = {}, {}
    for i in mal:
        Hm[i] = honest_cache(i)
        X, y = Xtr[clients[i]], ytr[clients[i]]
        if kind == "adaptive_bd":
            Pm[i] = backdoor_update(model, theta, X, y, cfg, rng, trig)
        else:
            yl = y.copy(); yl[yl != 0] = 0
            Pm[i] = local_train(model, theta, X, yl, cfg, rng)
    Xs, ys, probes_s, m_s = sur if not oracle_root else oracle_root
    g_s = local_train(model, theta, Xs, ys, cfg, np.random.default_rng([cfg["seed"], 4242]))
    pos = {i: k for k, i in enumerate(ids)}

    def ok(alpha):
        Uc = U_benign.copy()
        for i in mal:
            Uc[pos[i]] = Hm[i] + alpha * (Pm[i] - Hm[i])
        sim = copy.deepcopy(trace)
        sim.dry = True
        sim.probes, sim.m_probe = probes_s, m_s
        sim(Uc, model=model, theta=theta, Xr=Xs, yr=ys, n_cls=n_cls, g0=g_s, ids=ids)
        return all(sim.last_s[pos[i]] >= 0.95 for i in mal)
    lo, hi = 0.0, 2.0
    if ok(hi):
        lo = hi
    else:
        for _ in range(5):
            mid = (lo + hi) / 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
    return {i: (Hm[i] + lo * (Pm[i] - Hm[i])).astype(np.float32) for i in mal}, lo


# ------------------------------------------------------------------ aggregation
def agg_fedavg(U, sizes, **_):
    w = sizes / sizes.sum()
    return (w[:, None] * U).sum(0), w


def agg_median(U, **_):
    return np.median(U, 0), None


def agg_trmean(U, f, **_):
    n = len(U)
    k = min(f, (n - 1) // 2)
    S = np.sort(U, 0)
    return S[k:n - k].mean(0), None


def agg_krum(U, f, **_):
    n = len(U)
    D = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1) if U.shape[1] < 3000 else pairwise_sq(U)
    m = max(1, n - f - 2)
    sc = np.array([np.sort(np.delete(D[i], i))[:m].sum() for i in range(n)])
    i = int(np.argmin(sc))
    w = np.zeros(n); w[i] = 1
    return U[i], w


def pairwise_sq(U):
    sq = (U.astype(np.float64) ** 2).sum(1)
    G = U.astype(np.float64) @ U.astype(np.float64).T
    return np.maximum(sq[:, None] + sq[None] - 2 * G, 0)


def cos_to(U, r):
    return (U @ r) / (np.linalg.norm(U, axis=1) * np.linalg.norm(r) + 1e-12)


def agg_fltrust(U, g0, **_):
    ts = np.maximum(cos_to(U, g0), 0)
    nr = np.linalg.norm(g0) / (np.linalg.norm(U, axis=1) + 1e-12)
    if ts.sum() == 0:
        return np.zeros(U.shape[1], np.float32), ts
    w = ts / ts.sum()
    return (w[:, None] * (U * nr[:, None])).sum(0), w


def agg_flame(U, rng, lam=1e-3, **_):
    from sklearn.cluster import HDBSCAN
    n = len(U)
    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    Dc = np.clip(1 - Un @ Un.T, 0, 2).astype(np.float64)
    lab = HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1, metric="precomputed",
                  allow_single_cluster=True).fit_predict(Dc)
    if (lab >= 0).sum() == 0:
        adm = np.arange(n)
    else:
        big = np.bincount(lab[lab >= 0]).argmax()
        adm = np.where(lab == big)[0]
    norms = np.linalg.norm(U, axis=1)
    S = np.median(norms)
    Uc = U[adm] * np.minimum(1, S / (norms[adm] + 1e-12))[:, None]
    agg = Uc.mean(0) + rng.normal(0, lam * S, U.shape[1]).astype(np.float32)
    w = np.zeros(n); w[adm] = 1 / len(adm)
    return agg, w


def agg_fedavg_root(U, sizes, g0, **_):
    """FedAvg plus the server's root-set update as one extra client (norm-bounded to the median
    client norm, weight = mean client weight): isolates the value of the trusted update."""
    nu = np.median(np.linalg.norm(U, axis=1))
    g = g0 * min(1.0, nu / (np.linalg.norm(g0) + 1e-12))
    ws = sizes.mean()
    tot = sizes.sum() + ws
    return ((sizes[:, None] * U).sum(0) + ws * g) / tot, sizes / tot


def agg_zeno(U, model, theta, Xr, yr, f, rho=1e-3, **_):
    """Zeno (Xie et al., 2019): stochastic descendant score on server data,
    score_i = L(theta) - L(theta + Delta_i) - rho ||Delta_i||^2; average the n-f best."""
    def loss(w):
        lo = model.forward(w, Xr)
        z = lo - lo.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return float(-lp[np.arange(len(yr)), yr].mean())
    L0 = loss(theta)
    sc = np.array([L0 - loss(theta + u) - rho * float(u @ u) for u in U])
    keep = np.argsort(sc)[::-1][:max(1, len(U) - f)]
    w = np.zeros(len(U)); w[keep] = 1 / len(keep)
    return U[keep].mean(0), w


class FoolsGold:
    """FoolsGold (Fung et al., 2020): down-weights clients whose historical update directions are
    too similar to each other (Sybils), with pardoning and logit re-scaling."""

    def __init__(self, n, kappa=1.0):
        self.H = None
        self.kappa = kappa
        self.n = n

    def __call__(self, U, ids=None, **_):
        ids = np.arange(len(U)) if ids is None else np.asarray(ids)
        if self.H is None:
            self.H = np.zeros((self.n, U.shape[1]), np.float64)
        self.H[ids] += U
        Hn = self.H[ids] / (np.linalg.norm(self.H[ids], axis=1, keepdims=True) + 1e-12)
        cs = Hn @ Hn.T - np.eye(len(ids))
        maxcs = cs.max(1)
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j and maxcs[i] < maxcs[j]:
                    cs[i, j] *= maxcs[i] / maxcs[j]
        wv = 1 - cs.max(1)
        wv = np.clip(wv, 0, 1)
        wv = wv / (wv.max() + 1e-12)
        wv[wv == 1] = 0.99
        wv = self.kappa * (np.log(wv / (1 - wv) + 1e-12) + 0.5)
        wv[np.isinf(wv) | (wv > 1)] = 1
        wv = np.clip(np.nan_to_num(wv), 0, 1)
        if wv.sum() == 0:
            return np.zeros(U.shape[1], np.float32), wv
        w = wv / wv.sum()
        return (w[:, None] * U).sum(0), w


class FLDetector:
    """FLDetector (Zhang et al., KDD 2022): predicts each client's update from its previous one
    with an L-BFGS approximation of the Hessian, accumulates normalised prediction errors over a
    window, and removes the clients of the higher-scoring cluster when the gap statistic indicates
    two clusters. Remaining clients are aggregated with FedAvg."""

    def __init__(self, n, window=5, start=5, seed=0):
        self.n, self.N, self.start = n, window, start
        self.prev_U = {}
        self.theta_hist, self.agg_hist = [], []
        self.scores = []
        self.removed = set()
        self.rng = np.random.default_rng(seed)
        self.detected_round = None

    def _hvp(self, v):
        S = np.array([self.theta_hist[k + 1] - self.theta_hist[k] for k in range(len(self.theta_hist) - 1)])[-self.N:]
        Y = -np.array([self.agg_hist[k + 1] - self.agg_hist[k] for k in range(len(self.agg_hist) - 1)])[-self.N:]
        if len(S) < 2:
            return np.zeros_like(v)
        S, Y = S.astype(np.float64), Y.astype(np.float64)
        SY = S @ Y.T
        SS = S @ S.T
        R = np.triu(SY); L = SY - R
        sigma = float(Y[-1] @ S[-1]) / (float(S[-1] @ S[-1]) + 1e-12)
        D = np.diag(np.diag(SY))
        M = np.block([[sigma * SS, L], [L.T, -D]])
        try:
            Minv = np.linalg.pinv(M)
        except np.linalg.LinAlgError:
            return np.zeros_like(v)
        p = np.concatenate([sigma * (S @ v), Y @ v])
        return sigma * v - np.concatenate([sigma * S, Y], 0).T @ (Minv @ p)

    @staticmethod
    def _gap(x, rng, B=10):
        """Gap statistic for k=1 vs k=2 on 1-D scores; returns True if k=2 is preferred."""
        from sklearn.cluster import KMeans
        def W(z, k):
            if k == 1:
                return np.sum((z - z.mean()) ** 2)
            km = KMeans(2, n_init=5, random_state=0).fit(z.reshape(-1, 1))
            return km.inertia_
        x = (x - x.min()) / (x.max() - x.min() + 1e-12)
        gaps, sks = [], []
        for k in (1, 2):
            ref = [np.log(W(rng.uniform(0, 1, len(x)), k) + 1e-12) for _ in range(B)]
            gaps.append(np.mean(ref) - np.log(W(x, k) + 1e-12))
            sks.append(np.std(ref) * np.sqrt(1 + 1 / B))
        return gaps[0] < gaps[1] - sks[1]

    def __call__(self, U, sizes, theta, r, ids=None, **_):
        ids = np.arange(len(U)) if ids is None else np.asarray(ids)
        if r > 1 and len(self.theta_hist) >= 3:
            dth = theta - self.theta_hist[-1]
            corr = self._hvp(dth.astype(np.float64))
            err = np.full(len(ids), np.nan)
            for k, i in enumerate(ids):
                if i in self.prev_U:
                    pred = self.prev_U[i] - corr
                    err[k] = np.linalg.norm(pred - U[k])
            if np.isfinite(err).any():
                e = np.nan_to_num(err, nan=np.nanmean(err))
                full = np.full(self.n, np.nan); full[ids] = e / (e.sum() + 1e-12)
                self.scores.append(full)
        if len(self.scores) >= self.start and self.detected_round is None:
            sc = np.nanmean(np.array(self.scores[-self.N:]), 0)
            act = np.array([i for i in range(self.n) if np.isfinite(sc[i])])
            if len(act) >= 4 and self._gap(sc[act], self.rng):
                from sklearn.cluster import KMeans
                lab = KMeans(2, n_init=5, random_state=0).fit_predict(sc[act].reshape(-1, 1))
                hi = int(np.argmax([sc[act][lab == c].mean() for c in (0, 1)]))
                if (lab == hi).sum() < len(act) / 2:
                    self.removed |= set(act[lab == hi].tolist())
                    self.detected_round = r
        for k, i in enumerate(ids):
            self.prev_U[i] = U[k].astype(np.float64)
        keep = np.array([i not in self.removed for i in ids])
        if not keep.any():
            keep[:] = True
        w = np.where(keep, sizes, 0.0); w = w / w.sum()
        agg = (w[:, None] * U).sum(0)
        self.theta_hist.append(theta.astype(np.float64).copy())
        self.agg_hist.append(agg.astype(np.float64))
        return agg, w


def agg_bucket_median(U, rng, s=2, **_):
    """Bucketing (Karimireddy et al., 2022) with s=2 followed by the coordinate-wise median."""
    perm = rng.permutation(len(U))
    B = [U[perm[i:i + s]].mean(0) for i in range(0, len(U), s)]
    return np.median(np.stack(B), 0), None


def class_losses(model, theta, X, y, n_cls):
    """Per-class multiclass NLL and per-class detection NLL (attack vs benign, benign = class 0)."""
    lo = model.forward(theta, X)
    z = lo - lo.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    nll = -lp[np.arange(len(y)), y]
    pb = np.exp(lp[:, 0])
    det = np.where(y == 0, -np.log(pb + 1e-7), -np.log(1 - pb + 1e-7))
    mc, dt = np.full(n_cls, np.nan), np.full(n_cls, np.nan)
    for c in range(n_cls):
        m = y == c
        if m.any():
            mc[c], dt[c] = nll[m].mean(), det[m].mean()
    return mc, dt


def benign_prob(model, theta, X):
    lo = model.forward(theta, X)
    z = lo - lo.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 0] / e.sum(1)


def make_probes(Xr, yr, Xtr_stats, groups, m=16, seed=0):
    """Structured feature-edit probes: root attack samples in which one semantic feature group
    (a protocol layer or statistic family) or a pair of groups is set to its observed minimum or
    maximum -- the edit space of a traffic source that controls a few header fields."""
    lo, hi = Xtr_stats
    att = np.where(yr != 0)[0]
    sel = np.random.default_rng(seed).choice(att, min(m, len(att)), replace=False)
    base = Xr[sel]
    subsets = [list(g) for g in groups]
    subsets += [list(g1) + list(g2) for i, g1 in enumerate(groups) for g2 in groups[i + 1:]]
    P = []
    for S in subsets:
        for v in (lo, hi):
            Xp = base.copy(); Xp[:, S] = v[S]
            P.append(Xp)
    return np.concatenate(P), len(sel)


def robust_z(x):
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med)) + 1e-6
    return (x - med) / mad


class TRACE:
    """Trust from Reliability (class-conditional validation on a small trusted set), Agreement
    (update consistency) and Consistency over time (asymmetric reputation), with norm bounding.
    Both instantaneous signals are converted into anomaly gates via robust z-scores, so that
    benign-but-heterogeneous clients keep full weight and only statistical outliers are damped."""

    def __init__(self, n, cfg, probes=None, m_probe=1, env_root=None, client_env=None):
        self.c = dict(TRACE_DEFAULT, **(cfg or {}))
        self.rep = np.ones(n)
        self.seen = np.zeros(n, bool)
        self.hist = []
        self.probes, self.m_probe = probes, m_probe
        self.env_root, self.client_env = env_root, client_env
        self.E = {}
        self.prev_U, self.theta_hist, self.agg_hist, self.cons_hist = {}, [], [], {}

    def _accumulate(self, key, z):
        """Sequential evidence: an exponential moving average of each client's anomaly score,
        rescaled to unit variance under the null (i.i.d. scores). A client that is mildly but
        persistently anomalous is flagged even if no single round exceeds the threshold."""
        lam = self.c["evid"]
        if lam <= 0:
            return z
        E = self.E.get(key, np.zeros_like(z))
        E = lam * E + (1 - lam) * np.clip(z, -3, 10)
        self.E[key] = E
        return np.maximum(z, E * np.sqrt((1 + lam) / (1 - lam)))

    def _consistency(self, U, theta, ids):
        out = None
        if len(self.theta_hist) >= 3:
            det = FLDetector.__new__(FLDetector)
            det.theta_hist, det.agg_hist, det.N = self.theta_hist, self.agg_hist, self.c["cons_window"]
            corr = det._hvp((theta - self.theta_hist[-1]).astype(np.float64))
            err = np.full(len(ids), np.nan)
            for k, i in enumerate(ids):
                if i in self.prev_U:
                    err[k] = np.linalg.norm(self.prev_U[i] - corr - U[k])
            if np.isfinite(err).any():
                e = np.nan_to_num(err, nan=np.nanmedian(err))
                e = e / (e.mean() + 1e-12)
                for k, i in enumerate(ids):
                    self.cons_hist.setdefault(i, []).append(e[k])
                sc = np.array([np.mean(self.cons_hist[i][-self.c["cons_window"]:]) for i in ids])
                out = robust_z(sc)
        if not getattr(self, "dry", False):
            for k, i in enumerate(ids):
                self.prev_U[i] = U[k].astype(np.float64)
        return out

    def record(self, theta_before, delta):
        """Called by the server after aggregation (needed by the consistency signal)."""
        if self.c.get("consistency"):
            self.theta_hist.append(theta_before.astype(np.float64).copy())
            self.agg_hist.append(np.asarray(delta, np.float64))

    def __call__(self, U, model, theta, Xr, yr, n_cls, g0, ids=None, **_):
        c = self.c
        n = len(U)
        ids = np.arange(n) if ids is None else np.asarray(ids)
        # clients that participate for the first time start from the median reputation of the
        # clients already known (identical to 1 when everybody joins in round 1)
        new = ~self.seen[ids]
        if new.any():
            known = self.seen.copy()
            self.rep[ids[new]] = np.median(self.rep[known]) if known.any() else 1.0
            self.seen[ids] = True
        rep = self.rep[ids]
        norms = np.linalg.norm(U, axis=1)
        nu = np.median(norms)
        Uc = U * np.minimum(1, nu / (norms + 1e-12))[:, None] if c["clip"] else U
        k, T = c["kappa"], c["temp"]
        # (1) agreement: cosine to a robust reference direction, gated by its robust z-score
        if c["use_agree"]:
            sign_gate = np.ones(n)
            if c["agree"] == "both":
                # dissimilarity to the robust centre of the population and to the direction
                # learned on the trusted root set (the median direction vanishes late in training)
                cr = cos_to(Uc, g0)
                za = np.maximum(-robust_z(cos_to(Uc, np.median(Uc, 0))), -robust_z(cr))
                # soft sign test: updates that oppose the root direction are damped smoothly
                sign_gate = np.exp(np.minimum(cr, 0) / c["sign_temp"])
            else:
                ref = np.median(Uc, 0) if c["agree"] == "median" else g0
                za = -robust_z(cos_to(Uc, ref))         # large = unusually dissimilar
            if c["redundancy"]:
                # heterogeneous benign clients rarely submit near-identical updates; colluding
                # (Sybil) clients that share one crafted update do
                Un = Uc / (np.linalg.norm(Uc, axis=1, keepdims=True) + 1e-12)
                C = Un @ Un.T
                np.fill_diagonal(C, -1)
                zr = robust_z(C.max(1))
                za = np.maximum(za, zr)
            if c.get("consistency"):
                # historical consistency of the update itself: a client's update is predicted from
                # its previous one with an L-BFGS estimate of the aggregate's Jacobian (as in
                # FLDetector); the windowed, per-round-normalised prediction error is z-scored
                zc = self._consistency(U, theta, ids)
                if zc is not None:
                    za = np.maximum(za, zc - (c["kappa_c"] - c["kappa_a"]))
            za = self._accumulate("A", za)
            a = np.exp(-np.maximum(0, za - c["kappa_a"]) / T) * sign_gate
        else:
            a, za = np.ones(n), np.zeros(n)
        # (2) security-aware reliability on the trusted root set: worst-class change of the
        #     multiclass loss, of the detection (attack-vs-benign) loss, and of the benign
        #     probability under sparse feature-edit probes; each converted to a robust z-score
        if c["use_reliab"]:
            sig = c["signals"]
            base = {}

            def ref_for(i):
                """Root records (and probes) against which client i is judged: the whole root
                set, or with env_cond the root records of the client's own environment."""
                key = self.client_env[ids[i]] if c.get("env_cond") else None
                if key not in base:
                    if key is None:
                        Xr_, yr_, pr_, m_ = Xr, yr, self.probes, self.m_probe
                    else:
                        Xr_, yr_, pr_, m_ = self.env_root[key]
                    L0_, D0_ = class_losses(model, theta, Xr_, yr_, n_cls)
                    P0_ = benign_prob(model, theta, pr_).reshape(-1, m_).mean(1) if "probe" in sig else None
                    base[key] = (Xr_, yr_, pr_, m_, L0_, D0_, P0_)
                return base[key]
            S = {k: np.zeros(n) for k in sig}
            for i in range(n):
                Xr_, yr_, pr_, m_, L0, D0, P0 = ref_for(i)
                Li, Di = class_losses(model, theta + Uc[i], Xr_, yr_, n_cls)
                if "mc" in sig:
                    S["mc"][i] = np.nanmax(Li - L0) if c["reliab"] == "classwise" else np.nanmean(Li - L0)
                if "det" in sig:
                    # missed-attack loss on attack classes only: honest sites with little benign
                    # traffic raise false alarms on benign records, poisoned ones raise misses
                    dd = (Di - D0)[1:] if c["det_scope"] == "attack" else (Di - D0)
                    S["det"][i] = np.nanmax(dd) if c["reliab"] == "classwise" else np.nanmean(dd)
                if "probe" in sig:
                    Pi = benign_prob(model, theta + Uc[i], pr_).reshape(-1, m_).mean(1)
                    S["probe"][i] = np.max(Pi - P0)
            Z = {k2: robust_z(v2) for k2, v2 in S.items()}
            zv = np.max(np.stack(list(Z.values())), 0)
            zv = self._accumulate("R", zv)
            v = np.exp(-np.maximum(0, zv - k) / T)
        else:
            v, zv, Z = np.ones(n), np.zeros(n), {}
        s = a * v
        # (3) historical consistency: slow to regain trust, fast to lose it
        if c["history"]:
            beta = np.where(s >= rep, c["beta_up"], c["beta_down"])
            rep = beta * rep + (1 - beta) * s
            if not getattr(self, "dry", False):
                self.rep[ids] = rep
            w = rep * s
        else:
            w = s
        if not getattr(self, "dry", False):
            self.hist.append(dict(ids=ids.tolist(), a=a.tolist(), v=v.tolist(), rep=rep.tolist(), za=za.tolist(),
                                  zv=zv.tolist(), **{"z_" + k2: v2.tolist() for k2, v2 in Z.items()}))
        self.last_s = s
        if c["server_client"]:
            # the server takes part as an (n+1)-th, fully trusted client: its root-set update,
            # bounded to the same norm, enters with the mean weight of the clients
            g = g0 * min(1.0, nu / (np.linalg.norm(g0) + 1e-12))
            ws = w.mean() if w.sum() > 1e-12 else 1.0
            tot = w.sum() + ws
            return ((w[:, None] * Uc).sum(0) + ws * g) / tot, w / tot
        if w.sum() <= 1e-12:
            return np.zeros(U.shape[1], np.float32), w
        w = w / w.sum()
        return (w[:, None] * Uc).sum(0), w


# ------------------------------------------------------------------ evaluation
def macro_f1(y, p, n_cls):
    f = []
    for c in range(n_cls):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        if tp + fn == 0:
            continue
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def evaluate(model, theta, Xte, yte, n_cls, trig):
    p = predict(model, theta, Xte).argmax(1)
    att = yte != 0
    res = dict(acc=float((p == yte).mean()), f1=macro_f1(yte, p, n_cls),
               dr=float((p[att] != 0).mean()), fpr=float((p[~att] != 0).mean()),
               a2b=float((p[att] == 0).mean()))
    if trig is not None:
        pt = predict(model, theta, apply_trigger(Xte[att], trig)).argmax(1)
        det = p[att] != 0
        res["asr_raw"] = float((pt == 0).mean())
        # conditional ASR: detected attacks that the trigger turns into "benign"
        res["asr"] = float((pt[det] == 0).mean()) if det.any() else 0.0
    return res


# ------------------------------------------------------------------ main loop
def run(cfg_in):
    cfg = dict(DEFAULT, **cfg_in)
    if cfg["partition"] in ("environment", "env_capture") and "env_clients" not in cfg:
        cfg["env_clients"] = {"ACI-IoT-2023": 4, "Edge-IIoT-2022": 4, "IoMT-2024": 4,
                              "MQTT-IoT-2020": 3, "IoT-HCRL-2019": 3, "IoT-2022": 2}
    rng = np.random.default_rng(cfg["seed"])
    Xtr, ytr, gtr, Xte, yte, gte, classes, feats = load(cfg, rng)
    if cfg["partition"] == "category":
        tax = [["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP"], ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner"],
               ["SQL_injection", "Uploading", "XSS"], ["Backdoor", "Password", "Ransomware"]]
        cfg["categories"] = [[classes.index(c) for c in grp] for grp in tax]
    n_cls = len(classes)
    if cfg.get("lf_classes"):
        cfg["lf_idx"] = [classes.index(c) for c in cfg["lf_classes"]]
    if cfg["partition"] == "env_capture":
        cfg["_files"] = np.load(os.path.join(DATA, f"{cfg['dataset']}_prov.npz"), allow_pickle=True)["f_train"]
    root = take_root(ytr, gtr, cfg, rng)
    avail = np.setdiff1d(np.arange(len(ytr)), root)
    if cfg.get("val"):
        # development protocol: the chronologically last 12.5 % of every (group, class) block of
        # the training data is held out as a validation set; the test set is never touched
        vmask = np.zeros(len(ytr), bool)
        for gname in np.unique(gtr):
            for c in np.unique(ytr):
                idx = avail[(gtr[avail] == gname) & (ytr[avail] == c)]
                if len(idx) >= 8:
                    vmask[idx[int(len(idx) * 0.875):]] = True
        val = np.where(vmask)[0]
        avail = np.setdiff1d(avail, val)
        Xte, yte, gte = Xtr[val], ytr[val], gtr[val]
    if cfg.get("prep") == "root_robust":
        # preprocessing from trusted data only: median / IQR of the server's root set
        med = np.median(Xtr[root], 0)
        iqr = np.quantile(Xtr[root], 0.75, 0) - np.quantile(Xtr[root], 0.25, 0)
        sdr = Xtr[root].std(0)
        sc = np.where(iqr > 1e-6, iqr, np.where(sdr > 1e-6, sdr, 1.0))
        Xtr = np.clip((Xtr - med) / sc, -10, 10).astype(np.float32)
        Xte = np.clip((Xte - med) / sc, -10, 10).astype(np.float32)
    clients = partition(ytr, gtr, cfg, rng, avail)
    clients = [c if len(c) else rng.choice(avail, 50) for c in clients]
    n = len(clients)
    sizes = np.array([len(c) for c in clients], float)
    jsd, mmd = heterogeneity(clients, ytr, Xtr, n_cls, np.random.default_rng(1))
    root, yr = corrupt_root(root, ytr, cfg, classes)
    Xr = Xtr[root]
    n_mal = int(round(cfg["mal_frac"] * n)) if cfg["attack"] != "none" else 0
    mal = sorted(rng.choice(n, n_mal, replace=False).tolist()) if n_mal else []
    f = max(1, int(round(cfg["mal_frac"] * n)))  # defender's assumed bound
    trig = build_trigger(cfg, Xtr, feats)
    model = build_model(cfg, feats, n_cls)
    theta = model.init(np.random.default_rng(cfg["seed"] + 100))
    client_env = np.array([str(np.unique(gtr[c], return_counts=True)[0][np.argmax(np.unique(gtr[c], return_counts=True)[1])])
                           for c in clients])
    groups_sem = make_groups(cfg["dataset"], feats)[0]
    trace = None
    if cfg["agg"] == "trace":
        probes, m_probe = make_probes(Xr, yr, (Xtr.min(0), Xtr.max(0)), groups_sem, seed=cfg["seed"])
        env_root = None
        if (cfg["trace"] or {}).get("env_cond"):
            env_root = {}
            for e in np.unique(client_env):
                me = gtr[root] == e
                pe, mp = make_probes(Xr[me], yr[me], (Xtr.min(0), Xtr.max(0)), groups_sem, seed=cfg["seed"])
                env_root[e] = (Xr[me], yr[me], pe, mp)
        trace = TRACE(n, cfg["trace"], probes, m_probe, env_root, client_env)
    fg = FoolsGold(n) if cfg["agg"] == "foolsgold" else None
    fld = FLDetector(n, seed=cfg["seed"]) if cfg["agg"] == "fldetector" else None
    adaptive = cfg["attack"] in ("adaptive_bd", "adaptive_lf")
    sur = None
    if adaptive:
        # attackers' surrogate root set: class-balanced records from their own data
        pool = np.concatenate([clients[i] for i in mal])
        srng = np.random.default_rng([cfg["seed"], 2718])
        si = np.concatenate([srng.choice(pool[ytr[pool] == c], min(cfg["root_per_class"], (ytr[pool] == c).sum()),
                                         replace=False) for c in np.unique(ytr[pool])])
        lo_s, hi_s = Xtr[pool].min(0), Xtr[pool].max(0)
        ps, ms = make_probes(Xtr[si], ytr[si], (lo_s, hi_s), groups_sem, seed=cfg["seed"])
        sur = (Xtr[si], ytr[si], ps, ms)
    late = set()
    if cfg.get("late_join"):
        lj = cfg["late_join"]
        lrng = np.random.default_rng([cfg["seed"], 777])
        pool_late = [i for i in range(n) if i not in mal]
        k_b = int(round(lj["frac"] * n)) - (len(mal) if lj.get("mal_late") else 0)
        late = set(lrng.choice(pool_late, max(0, k_b), replace=False).tolist())
        if lj.get("mal_late"):
            late |= set(mal)
    hist, weights_log, alphas = [], [], []
    smom = None
    t0 = time.time()
    for r in range(1, cfg["rounds"] + 1):
        crng = np.random.default_rng([cfg["seed"], r])
        cache = {}
        prox = cfg["prox_mu"] if cfg["agg"] == "fedprox" else 0.0
        if cfg["agg"] == "fedprox" and prox == 0:
            prox = 0.01
        active = [i for i in range(n) if not (i in late and r < cfg["late_join"]["round"])]
        if cfg.get("part_frac", 1.0) < 1.0:
            prng = np.random.default_rng([cfg["seed"], r, 555])
            m_act = max(2, int(round(cfg["part_frac"] * len(active))))
            active = sorted(prng.choice(active, m_act, replace=False).tolist())
        mal_now = [i for i in mal if i in active]
        if cfg.get("attack_prob", 1.0) < 1.0 and mal_now:
            if np.random.default_rng([cfg["seed"], r, 666]).random() >= cfg["attack_prob"]:
                mal_now = []

        def honest(i):
            if i not in cache:
                cache[i] = local_train(model, theta, Xtr[clients[i]], ytr[clients[i]], cfg,
                                       np.random.default_rng([cfg["seed"], r, i]), prox_mu=prox)
            return cache[i]
        benign_now = [i for i in active if i not in mal_now]
        g0 = None
        if cfg["agg"] in ("fltrust", "trace", "fedavg_root") or adaptive:
            g0 = local_train(model, theta, Xr, yr, cfg, np.random.default_rng([cfg["seed"], r, 999]))
        if adaptive and mal_now:
            Ub = np.stack([honest(i) for i in active]).astype(np.float32)
            oracle = (Xr, yr, trace.probes, trace.m_probe) if cfg.get("adapt_oracle") else None
            mu, alpha = adaptive_updates(cfg["attack"], model, theta, clients, mal_now, Xtr, ytr, cfg, crng, trig,
                                         honest, np.array(active), Ub, trace, sur, n_cls, oracle)
            alphas.append(alpha)
        else:
            atk = cfg["attack"] if not adaptive else "none"
            mu = malicious_updates(atk, model, theta, clients, mal_now, Xtr, ytr, cfg, crng, trig, honest, benign_now)
        U = np.stack([mu[i] if i in mu else honest(i) for i in active]).astype(np.float32)
        ids = np.array(active)
        sz = sizes[ids]
        a = cfg["agg"]
        if a in ("fedavg", "fedprox"):
            delta, w = agg_fedavg(U, sz)
        elif a == "fedavg_root":
            delta, w = agg_fedavg_root(U, sz, g0)
        elif a == "median":
            delta, w = agg_median(U)
        elif a == "trmean":
            delta, w = agg_trmean(U, f)
        elif a == "krum":
            delta, w = agg_krum(U, f)
        elif a == "fltrust":
            delta, w = agg_fltrust(U, g0)
        elif a == "flame":
            delta, w = agg_flame(U, crng)
        elif a == "zeno":
            delta, w = agg_zeno(U, model, theta, Xr, yr, f)
        elif a == "foolsgold":
            delta, w = fg(U, ids=ids)
        elif a == "fldetector":
            delta, w = fld(U, sz, theta, r, ids=ids)
        elif a == "bucket_median":
            delta, w = agg_bucket_median(U, np.random.default_rng([cfg["seed"], r, 123]))
        elif a == "trace":
            delta, w = trace(U, model=model, theta=theta, Xr=Xr, yr=yr, n_cls=n_cls, g0=g0, ids=ids)
            trace.record(theta, delta)
        if cfg["server_mom"] > 0:
            smom = cfg["server_mom"] * smom + delta if r > 1 else delta
            delta = smom
        theta = (theta + delta).astype(np.float32)
        if w is not None:
            w = np.asarray(w, float)
            mpos = [k for k, i in enumerate(active) if i in mal]
            weights_log.append(dict(round=r, mal_weight=float(w[mpos].sum()) if mpos else 0.0,
                                    attacking=bool(mal_now)))
        if r % cfg["eval_every"] == 0 or r == cfg["rounds"]:
            ev = evaluate(model, theta, Xte, yte, n_cls, trig)
            ev["round"] = r
            hist.append(ev)
            if cfg["log"]:
                print(r, {k: round(v, 4) for k, v in ev.items()}, f"{time.time()-t0:.0f}s", flush=True)
    # per-group (protocol / environment) F1 of the final model
    p = predict(model, theta, Xte).argmax(1)
    per_group = {str(gname): macro_f1(yte[gte == gname], p[gte == gname], n_cls) for gname in np.unique(gte)}
    per_class_recall = {classes[c]: float((p[yte == c] == c).mean()) for c in range(n_cls) if (yte == c).any()}
    per_class_dr = {classes[c]: float((p[yte == c] != 0).mean()) for c in range(1, n_cls) if (yte == c).any()}
    last = hist[-3:]
    final = {k: float(np.mean([h[k] for h in last])) for k in hist[-1] if k != "round"}
    if cfg.get("save_theta"):
        np.save(cfg["save_theta"], theta)
    if cfg.get("save_scores"):
        lo_ = predict(model, theta, Xte)
        z_ = lo_ - lo_.max(1, keepdims=True); e_ = np.exp(z_)
        np.savez_compressed(cfg["save_scores"], pben=(e_[:, 0] / e_.sum(1)).astype(np.float32), y=yte.astype(np.int16),
                            pred=lo_.argmax(1).astype(np.int16))
    cfg.pop("_files", None)
    out = dict(cfg=cfg, final=final, last=hist[-1], hist=hist, mal=mal, jsd=jsd, mmd=mmd, n_params=int(model.spec.size),
               sizes=sizes.tolist(), per_group=per_group, per_class_recall=per_class_recall, per_class_dr=per_class_dr,
               trigger=trig, weights=weights_log, seconds=time.time() - t0, client_env=client_env.tolist(),
               alphas=alphas)
    if trace is not None:
        out["trace_hist"] = trace.hist
    if fld is not None:
        out["fld_removed"] = sorted(int(i) for i in fld.removed)
        out["fld_round"] = fld.detected_round
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    res = run(dict(json.loads(a.cfg), log=True))
    print(json.dumps(res["final"]), res["seconds"])
