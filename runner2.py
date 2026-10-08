"""Revision experiment grids (reviewer response). Same job/caching logic as runner.py."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from runner import job, BASE, COMMON, ATTACKS

S3 = [0, 1, 2]


def grid(name):
    J = []
    E = lambda ds, **k: dict(BASE[ds], **COMMON, **k)
    if name == "adaptive":
        for s in S3:
            for ds in ["edge", "cic"]:
                for at in ["adaptive_bd", "adaptive_lf"]:
                    J.append(E(ds, agg="trace", attack=at, seed=s, tag="adaptive"))
                J.append(E(ds, agg="trace", attack="adaptive_bd", adapt_oracle=True, seed=s, tag="adaptive-oracle"))
    elif name == "newbase":
        for s in S3:
            for ds in ["edge", "cic"]:
                for at in ATTACKS:
                    for ag in ["fedavg_root", "zeno", "foolsgold", "fldetector"]:
                        extra = dict(save_scores=f"results/scores/{ds}_{ag}_none_s{s}.npz") if at == "none" else {}
                        J.append(E(ds, agg=ag, attack=at, seed=s, tag="main", **extra))
    elif name == "minmax":
        for s in S3:
            for ds in ["edge", "cic"]:
                for ag in ["fedavg", "median", "fltrust", "flame", "trace", "fedavg_root", "zeno", "foolsgold", "fldetector"]:
                    J.append(E(ds, agg=ag, attack="minmax", seed=s, tag="main"))
    elif name == "noserver":
        for s in S3:
            for ds in ["edge", "cic"]:
                for at in ATTACKS:
                    J.append(E(ds, agg="trace", attack=at, seed=s, trace=dict(server_client=False), tag="abl-no_server"))
    elif name == "grouping":
        for s in S3:
            for ds in ["edge", "cic"]:
                for m in ["ftt-g-random", "ftt-g-contig"]:
                    J.append(E(ds, agg="fedavg", attack="none", model=m, seed=s, tag="grouping"))
    elif name == "triggers":
        for s in S3:
            for ds in ["edge", "cic"]:
                for tr in ["midrange", "distributed", "semantic", "cleanlabel"]:
                    for ag in ["fedavg", "fltrust", "flame", "trace"]:
                        extra = dict(clean_label=True) if tr == "cleanlabel" else {}
                        J.append(E(ds, agg=ag, attack="bd", trigger=tr, seed=s, tag="trigger", **extra))
    elif name == "triggers2":
        for s in S3:
            for ds in ["edge", "cic"]:
                for tr in ["midrange", "distributed", "semantic", "cleanlabel"]:
                    for ag in ["zeno", "fldetector"]:
                        extra = dict(clean_label=True) if tr == "cleanlabel" else {}
                        J.append(E(ds, agg=ag, attack="bd", trigger=tr, seed=s, tag="trigger", **extra))
    elif name == "ablation3":
        for s in [1, 2]:
            for ds in ["edge", "cic"]:
                vs = {"no_agree": dict(use_agree=False), "no_reliab": dict(use_reliab=False),
                      "no_probe": dict(signals=["mc", "det"]), "no_history": dict(history=False)}
                for at in ATTACKS:
                    for vn, v in vs.items():
                        J.append(E(ds, agg="trace", attack=at, seed=s, trace=v, tag=f"abl-{vn}"))
    elif name == "frac3":
        for s in [1, 2]:
            for fr in [0.1, 0.3, 0.4]:
                for at in ["lf", "bd"]:
                    for ag in ["fedavg", "median", "fltrust", "flame", "trace"]:
                        J.append(E("edge", agg=ag, attack=at, mal_frac=fr, seed=s, tag="frac"))
    elif name == "hetero3":
        parts = [dict(partition="dirichlet", alpha=1.0), dict(partition="dirichlet", alpha=0.1),
                 dict(partition="protocol"), dict(partition="attack")]
        for s in [1, 2]:
            for p in parts:
                for at in ["none", "lf"]:
                    for ag in ["fedavg", "fltrust", "trace"]:
                        J.append(dict(BASE["edge"], **COMMON, **p, agg=ag, attack=at, seed=s, tag="hetero"))
    elif name == "root3":
        for s in [1, 2]:
            for ds in ["edge", "cic"]:
                for rp in [5, 50]:
                    for at in ["lf", "bd"]:
                        for ag in ["fltrust", "trace"]:
                            J.append(E(ds, agg=ag, attack=at, root_per_class=rp, seed=s, tag="root"))
            for at in ["lf", "bd"]:
                for ag in ["fltrust", "trace"]:
                    J.append(E("cic", agg=ag, attack=at, root_bias=["ACI-IoT-2023"], seed=s, tag="rootbias"))
    elif name == "rootstress":
        cats = {"DoS": ["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP"], "Scan": ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner"],
                "Injection": ["SQL_injection", "Uploading", "XSS"], "Malware": ["Backdoor", "Password", "Ransomware"]}
        for s in S3:
            for cn, cl in cats.items():
                for ag in ["fltrust", "trace"]:
                    J.append(E("edge", agg=ag, attack="lf", root_drop=cl, seed=s, tag=f"rootdrop-{cn}"))
            for kind, val in [("root_noise", 0.2), ("root_contam", 0.2)]:
                for at in ["lf", "bd"]:
                    for ag in ["fltrust", "trace"]:
                        J.append(E("edge", agg=ag, attack=at, seed=s, tag=f"rootstress-{kind}", **{kind: val}))
    elif name == "lftarget":
        # targeted label poisoning of one attack family, with and without that family in the root set
        cats = {"DoS": ["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP"], "Scan": ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner"],
                "Injection": ["SQL_injection", "Uploading", "XSS"], "Malware": ["Backdoor", "Password", "Ransomware"]}
        for s in S3:
            for cn, cl in cats.items():
                for ag in ["fedavg", "fltrust", "trace"]:
                    J.append(E("edge", agg=ag, attack="lf", lf_classes=cl, seed=s, tag=f"lftarget-{cn}"))
                    if ag != "fedavg":
                        J.append(E("edge", agg=ag, attack="lf", lf_classes=cl, root_drop=cl, seed=s, tag=f"lftargetdrop-{cn}"))
    elif name == "confirm":
        for s in [3, 4]:
            for ds in ["edge", "cic"]:
                for at in ATTACKS:
                    for ag in ["fedavg", "fltrust", "trace"]:
                        extra = dict(save_scores=f"results/scores/{ds}_{ag}_{at}_s{s}.npz") if at == "none" else {}
                        J.append(E(ds, agg=ag, attack=at, seed=s, tag="confirm", **extra))
    elif name == "partial":
        for s in S3:
            for ag in ["fltrust", "trace"]:
                for at in ["lf", "bd"]:
                    for pf in [0.25, 0.5, 0.75]:
                        J.append(E("edge", agg=ag, attack=at, part_frac=pf, seed=s, tag=f"partial-{pf}"))
                    J.append(E("edge", agg=ag, attack=at, attack_prob=0.5, seed=s, tag="intermittent"))
                    J.append(E("edge", agg=ag, attack=at, late_join=dict(frac=0.5, round=20, mal_late=True), seed=s,
                               tag="latejoin"))
    elif name == "prep":
        for ds in ["edge", "cic"]:
            for at in ["none", "lf", "bd"]:
                for ag in ["fedavg", "fltrust", "trace"]:
                    J.append(E(ds, agg=ag, attack=at, prep="root_robust", seed=0, tag="prep"))
    elif name == "envdev":
        # development runs on the validation split only (seed 0): does conditioning the reliability
        # gate on the client's environment help under environment-level heterogeneity?
        for at in ["none", "lf", "bd"]:
            J.append(E("cic", agg="trace", attack=at, seed=0, val=True, tag="envdev-base"))
            J.append(E("cic", agg="trace", attack=at, seed=0, val=True, root_mode="env", tag="envdev-envroot"))
            J.append(E("cic", agg="trace", attack=at, seed=0, val=True, root_mode="env",
                       trace=dict(env_cond=True), tag="envdev-envcond"))
    elif name == "consdev":
        # development of the update-consistency signal on the validation split only (seed 0)
        for ds in ["edge", "cic"]:
            for at in ATTACKS:
                J.append(E(ds, agg="trace", attack=at, seed=0, val=True, tag="dev-base"))
                J.append(E(ds, agg="trace", attack=at, seed=0, val=True, trace=dict(consistency=True), tag="dev-cons"))
    elif name == "tracec":
        # TRACE with the update-consistency signal (frozen after the validation-only development)
        TC = dict(consistency=True)
        for s in S3:
            for ds in ["edge", "cic"]:
                for at in ATTACKS + ["minmax"]:
                    J.append(E(ds, agg="trace", attack=at, seed=s, trace=TC, tag="tracec"))
                for at in ["adaptive_bd", "adaptive_lf"]:
                    J.append(E(ds, agg="trace", attack=at, seed=s, trace=TC, tag="tracec-adaptive"))
                for tr in ["midrange", "distributed", "semantic", "cleanlabel"]:
                    extra = dict(clean_label=True) if tr == "cleanlabel" else {}
                    J.append(E(ds, agg="trace", attack="bd", trigger=tr, seed=s, trace=TC, tag="tracec-trigger", **extra))
        for s in [3, 4]:
            for ds in ["edge", "cic"]:
                for at in ATTACKS:
                    J.append(E(ds, agg="trace", attack=at, seed=s, trace=TC, tag="tracec-confirm"))
    elif name == "envcond":
        for s in S3:
            for at in ATTACKS:
                J.append(E("cic", agg="trace", attack=at, seed=s, root_mode="env", trace=dict(env_cond=True),
                           tag="envcond"))
                J.append(E("cic", agg="fltrust", attack=at, seed=s, root_mode="env", tag="envroot"))
                J.append(E("cic", agg="trace", attack=at, seed=s, root_mode="env", tag="envroot"))
    elif name == "fulldata":
        for s in [0, 1]:
            for at in ["none", "lf", "bd"]:
                for ag in ["fedavg", "fltrust", "trace"]:
                    J.append(E("cic_full", agg=ag, attack=at, seed=s, tag="fulldata"))
    elif name == "capdisj":
        for s in [0, 1]:
            for at in ["none", "lf", "bd"]:
                for ag in ["fedavg", "fltrust", "trace"]:
                    J.append(dict(BASE["cic"], **COMMON, partition="env_capture", agg=ag, attack=at, seed=s,
                                  tag="capdisj"))
    elif name == "oppoints":
        for s in [0]:
            for ds in ["edge", "cic"]:
                for ag in ["fedavg", "median", "fltrust", "flame", "trace"]:
                    J.append(E(ds, agg=ag, attack="none", seed=s, tag="oppoint",
                               save_scores=f"results/scores/{ds}_{ag}_none_s{s}.npz"))
    return J


if __name__ == "__main__":
    os.makedirs("results/scores", exist_ok=True)
    jobs = [j for n in sys.argv[1:] for j in grid(n)]
    print(len(jobs), "jobs", flush=True)
    with Pool(2, maxtasksperchild=1) as pool:
        for i, (p, msg) in enumerate(pool.imap_unordered(job, jobs)):
            print(f"[{i+1}/{len(jobs)}] {os.path.basename(p)} {msg[:200]}", flush=True)
