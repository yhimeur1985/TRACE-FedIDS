"""Generate every number (numbers.tex), table (tab/*.tex) and figure (figs/*.pdf) of the paper
from the run files in results/runs."""
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.environ.get("TRACE_PAPER_DIR", os.path.join(HERE, "paper_assets"))
TAB = os.path.join(PAPER, "tab"); FIG = os.path.join(PAPER, "figs")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

AGG_NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "krum": "Krum", "trmean": "Trim. mean",
             "median": "Median", "fltrust": "FLTrust", "flame": "FLAME", "trace": r"\trace (ours)"}
AGG_ORDER = ["fedavg", "fedprox", "krum", "trmean", "median", "fltrust", "flame", "trace"]
ATT_NAMES = {"none": "No attack", "lf": "LF", "sf": "SF", "alie": "ALIE", "bd": "BD"}
NUM = {}


def num(name, val, fmt="{:.3f}"):
    NUM[name] = fmt.format(val) if not isinstance(val, str) else val


def load_runs():
    rows = []
    for f in glob.glob(os.path.join(HERE, "results", "runs", "*.json")):
        r = json.load(open(f))
        c = r["cfg"]
        tv = c.get("trace") or {}
        row = dict(tag=r.get("tag") or c.get("tag"), dataset=c["dataset"], partition=c["partition"],
                   alpha=c.get("alpha"), agg=c["agg"], attack=c["attack"], seed=c["seed"],
                   frac=c["mal_frac"], root=c["root_per_class"], root_bias=str(c.get("root_bias")),
                   model=c["model"], jsd=r["jsd"], trace_cfg=json.dumps(c.get("trace")), mmd=r["mmd"], seconds=r["seconds"],
                   rounds=c["rounds"], file=f)
        row.update(r["final"])
        row["malw"] = [w["mal_weight"] for w in r["weights"]] if r["weights"] else None
        row["per_group"] = r["per_group"]; row["per_class"] = r["per_class_recall"]
        rows.append(row)
    return pd.DataFrame(rows)


def ms(x, digits=2, sd=True):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "--"
    m = x.mean()
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    f = "{:.%df}" % digits
    return (f.format(m) + (r"\,{\scriptsize$\pm$" + f.format(s) + "}" if sd and len(x) > 1 else ""))


def main_table(df):
    d = df[df.tag == "main"]
    cols = [("none", "f1", "F1"), ("lf", "f1", "F1"), ("lf", "dr", "DR"), ("sf", "f1", "F1"),
            ("alie", "f1", "F1"), ("bd", "f1", "F1"), ("bd", "asr", r"ASR$\downarrow$")]
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\caption{Robustness at 20\,\% malicious clients: mean over three seeds of the values averaged over rounds 30--40 (median s.d.\ over seeds \SdMed{}, maximum \SdMax{}). F1 = macro-F1, DR = detection rate, ASR = backdoor attack success rate (lower is better). Best value per column in bold, second best underlined.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\centering\footnotesize\setlength{\tabcolsep}{3.2pt}")
    lines.append(r"\begin{tabular}{@{}l" + "c" * 7 + "c" + "c" * 7 + "@{}}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{7}{c}{Edge-IIoTset, attack-category skew (13 classes)} & & \multicolumn{7}{c}{CIC-BCCC-NRC, environment skew (7 classes)}\\")
    lines.append(r"\cmidrule(lr){2-8}\cmidrule(lr){10-16}")
    hdr1 = " & Clean & \\multicolumn{2}{c}{LF} & SF & ALIE & \\multicolumn{2}{c}{BD}"
    lines.append(r"Method" + hdr1 + " &" + hdr1 + r"\\")
    lines.append(r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}\cmidrule(lr){10-10}\cmidrule(lr){11-12}\cmidrule(lr){13-13}\cmidrule(lr){14-14}\cmidrule(lr){15-16}")
    sub = " & ".join(c[2] for c in cols)
    lines.append(r" & " + sub + " & & " + sub + r"\\")
    lines.append(r"\midrule")
    # rank for bold/underline
    best = {}
    for ds in ["edge", "cic"]:
        for (at, met, _) in cols:
            vals = {}
            for ag in AGG_ORDER:
                x = d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == ag)][met]
                if len(x):
                    vals[ag] = x.mean()
            if not vals:
                continue
            r2 = {k: round(v, 2) for k, v in vals.items()}
            uniq = sorted(set(r2.values()), reverse=(met != "asr"))
            first = [k for k in r2 if r2[k] == uniq[0]]
            second = [k for k in r2 if len(uniq) > 1 and r2[k] == uniq[1]]
            best[(ds, at, met)] = (first, second)
    for ag in AGG_ORDER:
        cells = []
        for ds in ["edge", "cic"]:
            for (at, met, _) in cols:
                x = d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == ag)][met]
                s = ms(x, 2, sd=False)
                b = best.get((ds, at, met), ([], []))
                if len(x) and ag in b[0]:
                    s = r"\textbf{" + s + "}"
                elif len(x) and ag in b[1]:
                    s = r"\underline{" + s + "}"
                cells.append(s)
            if ds == "edge":
                cells.append("")
        if ag == "trace":
            lines.append(r"\midrule")
        lines.append(AGG_NAMES[ag] + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    open(os.path.join(TAB, "tab_main.tex"), "w").write("\n".join(lines) + "\n")
    sds = []
    for ds in ["edge", "cic"]:
        for (at, met, _) in cols:
            for ag in AGG_ORDER:
                x = d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == ag)][met]
                if len(x) > 1:
                    sds.append(x.std(ddof=1))
    if sds:
        num("SdMed", float(np.median(sds)), "{:.2f}"); num("SdMax", float(np.max(sds)), "{:.2f}")
    # numbers
    for ds in ["edge", "cic"]:
        for at in ["none", "lf", "sf", "alie", "bd"]:
            for ag in AGG_ORDER:
                x = d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == ag)]
                if len(x):
                    key = f"{ds}{at}{ag}".replace("_", "")
                    for met, mn in [("f1", "Fone"), ("dr", "Dr"), ("fpr", "Fpr"), ("asr", "Asr"), ("a2b", "Miss")]:
                        num("M" + key.capitalize() + mn, x[met].mean(), "{:.2f}")
                    num("N" + key.capitalize(), str(len(x)), "{}")


def worst_case(df):
    """Worst-case F1 over the four attacks (and clean) and mean ASR -> summary numbers."""
    d = df[df.tag == "main"]
    out = {}
    for ds in ["edge", "cic"]:
        for ag in AGG_ORDER:
            g = d[(d.dataset == ds) & (d["agg"] == ag)]
            if g.empty:
                continue
            per = g.groupby("attack").f1.mean()
            worst = per[[a for a in ["lf", "sf", "alie", "bd"] if a in per]].min() if len(per) > 1 else np.nan
            bd = g[g.attack == "bd"].asr.mean()
            eff = min(worst, per.get("bd", np.nan) * (1 - bd) if not np.isnan(bd) else np.nan)
            out[(ds, ag)] = (worst, bd)
            num(f"W{ds.capitalize()}{ag.capitalize()}", worst, "{:.2f}")
    return out


def het_table(df):
    import fl
    rows = []
    specs = [("edge", dict(partition="category"), "Edge-IIoTset", "Attack-category skew (main)"),
             ("cic", dict(partition="environment"), "CIC-BCCC-NRC", "Environment skew (main)"),
             ("edge", dict(partition="iid"), "Edge-IIoTset", "IID"),
             ("edge", dict(partition="dirichlet", alpha=1.0), "Edge-IIoTset", r"Dirichlet $\alpha=1.0$"),
             ("edge", dict(partition="dirichlet", alpha=0.1), "Edge-IIoTset", r"Dirichlet $\alpha=0.1$"),
             ("edge", dict(partition="protocol"), "Edge-IIoTset", "Protocol family"),
             ("edge", dict(partition="attack"), "Edge-IIoTset", "3 attack classes per client")]
    for ds, p, dn, pn in specs:
        vals = []
        for seed in [0]:
            cfg = dict(fl.DEFAULT, dataset=ds, seed=seed, **p)
            if cfg["partition"] == "environment":
                cfg["env_clients"] = {"ACI-IoT-2023": 4, "Edge-IIoT-2022": 4, "IoMT-2024": 4,
                                      "MQTT-IoT-2020": 3, "IoT-HCRL-2019": 3, "IoT-2022": 2}
            rng = np.random.default_rng(seed)
            Xtr, ytr, gtr, Xte, yte, gte, classes, feats = fl.load(cfg, rng)
            if cfg["partition"] == "category":
                tax = [["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP"], ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner"],
                       ["SQL_injection", "Uploading", "XSS"], ["Backdoor", "Password", "Ransomware"]]
                cfg["categories"] = [[classes.index(c) for c in grp] for grp in tax]
            root = fl.take_root(ytr, gtr, cfg, rng)
            avail = np.setdiff1d(np.arange(len(ytr)), root)
            parts = fl.partition(ytr, gtr, cfg, rng, avail)
            jsd, mmd = fl.heterogeneity(parts, ytr, Xtr, len(classes), np.random.default_rng(1))
            ncls = np.mean([len(np.unique(ytr[q])) for q in parts if len(q)])
            sizes = [len(q) for q in parts]
            vals.append((jsd, mmd, ncls, min(sizes), max(sizes)))
        v = np.mean(vals, 0)
        rows.append((dn, pn, *v))
        key = (ds + pn.split()[0]).replace("-", "").replace("$", "").replace("\\", "")
    lines = [r"\begin{table}[t]", r"\caption{Client heterogeneity of the federations (20 clients, seed 0). JSD: mean pairwise Jensen--Shannon divergence of client label distributions; Shift: mean pairwise squared distance of client feature means (standardised features); Cls.: mean number of classes per client.}",
             r"\label{tab:het}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llcccc@{}}", r"\toprule", r"Data & Partition & JSD & Shift & Cls. & Size range\\", r"\midrule"]
    for dn, pn, jsd, mmd, ncls, mn, mx in rows:
        lines.append(f"{dn} & {pn} & {jsd:.2f} & {mmd:.2f} & {ncls:.1f} & {int(mn):,}--{int(mx):,}\\\\".replace(",", "\\,"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "tab_het.tex"), "w").write("\n".join(lines) + "\n")
    num("JsdEdge", rows[0][2], "{:.2f}"); num("JsdCic", rows[1][2], "{:.2f}")
    num("ShiftEdge", rows[0][3], "{:.2f}"); num("ShiftCic", rows[1][3], "{:.2f}")


def fig_trust(df):
    d = df[(df.tag == "main") & (df["agg"].isin(["fedavg", "median", "fltrust", "flame", "trace"]))]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 2.25), sharex=True, sharey=True)
    colors = {"fedavg": "#8c8c8c", "fltrust": "#e08214", "flame": "#7b3294", "trace": "#00567d", "median": "#1b9e77"}
    for r_, ds in enumerate(["edge", "cic"]):
        for c_, at in enumerate(["lf", "sf", "alie", "bd"]):
            ax = axes[r_, c_]
            for ag in ["fltrust", "flame", "trace"]:
                g = d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == ag)]
                W = [w for w in g.malw if w]
                if not W:
                    continue
                W = np.array(W)
                ax.plot(np.arange(1, W.shape[1] + 1), W.mean(0), color=colors[ag], lw=1.4 if ag == "trace" else 1.0,
                        label={"fedavg": "FedAvg", "fltrust": "FLTrust", "flame": "FLAME", "trace": "TRACE"}[ag])
            ax.axhline(0.2, color="#bbbbbb", lw=0.6, ls=":")
            ax.set_ylim(-0.02, 0.62)
            if r_ == 0:
                ax.set_title(ATT_NAMES[at], fontsize=8)
            if c_ == 0:
                ax.set_ylabel(("Edge-IIoTset" if ds == "edge" else "CIC-BCCC-NRC") + "\nmalicious weight", fontsize=7)
            if r_ == 1:
                ax.set_xlabel("round", fontsize=7)
            ax.tick_params(labelsize=6.5)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(fontsize=6, frameon=False, loc="upper left", ncol=1)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_trust.pdf"))
    plt.close(fig)


def fig_frac(df):
    d = df[((df.tag == "frac") | ((df.tag == "main") & (df.seed == 0))) & (df.dataset == "edge")]
    aggs = ["fedavg", "median", "fltrust", "flame", "trace"]
    lab = {"fedavg": "FedAvg", "median": "Median", "fltrust": "FLTrust", "flame": "FLAME", "trace": "TRACE"}
    col = {"fedavg": "#8c8c8c", "median": "#1b9e77", "fltrust": "#e08214", "flame": "#7b3294", "trace": "#00567d"}
    mk = {"fedavg": "o", "median": "s", "fltrust": "^", "flame": "D", "trace": "*"}
    panels = [("lf", "f1", "LF: macro-F1"), ("lf", "dr", "LF: detection rate"), ("bd", "f1", "BD: macro-F1"), ("bd", "asr", "BD: attack success rate")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.75))
    for ax, (at, met, title) in zip(axes, panels):
        for ag in aggs:
            g = d[(d.attack == at) & (d["agg"] == ag)].groupby("frac")[met].mean()
            if len(g):
                ax.plot(g.index * 100, g.values, marker=mk[ag], ms=3.5 if ag != "trace" else 6, lw=1.0 if ag != "trace" else 1.6,
                        color=col[ag], label=lab[ag])
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("malicious clients (%)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_xticks([10, 20, 30, 40])
        for s_ in ["top", "right"]:
            ax.spines[s_].set_visible(False)
    axes[0].legend(fontsize=5.8, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(FIG, "fig_frac.pdf"))
    plt.close(fig)


def hetero_table(df):
    d = df[(df.tag == "hetero") | ((df.tag == "main") & (df.seed == 0) & (df.dataset == "edge"))]
    d = d[d.dataset == "edge"]
    parts = [("iid", None, "IID"), ("dirichlet", 1.0, r"Dir.\ $\alpha{=}1$"), ("dirichlet", 0.1, r"Dir.\ $\alpha{=}0.1$"),
             ("protocol", None, "Protocol"), ("category", None, "Category (main)"), ("attack", None, "3 classes/client")]
    aggs = ["fedavg", "median", "fltrust", "trace"]
    lines = [r"\begin{table}[t]", r"\caption{Macro-F1 on Edge-IIoTset under different client partitions (seed 0; Avg = FedAvg, Med = median, FLT = FLTrust). JSD as in Table~\ref{tab:het}.}",
             r"\label{tab:hetres}", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lc" + "cccc" * 2 + r"@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{No attack} & \multicolumn{4}{c}{LF (20\,\%)}\\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"Partition & JSD & " + " & ".join(["Avg", "Med", "FLT", "Ours"] * 2) + r"\\", r"\midrule"]
    for p, a, name in parts:
        g = d[d.partition == p]
        if a is not None:
            g = g[np.isclose(g.alpha.astype(float), a)]
        if g.empty:
            continue
        cells = [f"{g.jsd.mean():.2f}"]
        for at in ["none", "lf"]:
            vals = {ag: round(g[(g.attack == at) & (g["agg"] == ag)].f1.mean(), 2) for ag in aggs}
            bestv = np.nanmax(list(vals.values())) if any(~np.isnan(list(vals.values()))) else np.nan
            for ag in aggs:
                v = vals[ag]
                s_ = "--" if np.isnan(v) else f"{v:.2f}"
                if not np.isnan(v) and v == bestv:
                    s_ = r"\textbf{" + s_ + "}"
                cells.append(s_)
        lines.append(name + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "tab_hetres.tex"), "w").write("\n".join(lines) + "\n")


def central_ref():
    out = {}
    for ds in ["edge", "cic"]:
        for m in ["ftt", "ftt-g", "mlp"]:
            r = json.load(open(os.path.join(HERE, "results", f"central_{ds}_{m}.json")))
            h = r["hist"][-3:]
            out[(ds, m)] = np.mean([x["f1"] for x in h])
    return out


def backbone_table(df):
    cen = central_ref()
    sp = json.load(open(os.path.join(HERE, "results", "speed.json")))
    ro = json.load(open(os.path.join(HERE, "results", "root_only.json")))
    d = df[(df.dataset == "edge") & (df.seed == 0)]
    def fl_f1(model, agg, at):
        g = d[(d.model == model) & (d["agg"] == agg) & (d.attack == at) & (d.tag.isin(["main", "mlp"]))]
        return g.f1.mean() if len(g) else np.nan
    lines = [r"\begin{table}[t]", r"\caption{Backbones. Params: trainable parameters (packets/flows). Train: CPU time per training record (one core, batch 64). Central: macro-F1 when trained on the pooled client data (upper reference). Root only: the backbone trained on the 10-per-class root set alone (mean of three seeds).}",
             r"\label{tab:backbone}", r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
             r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Params (k)} & \multicolumn{2}{c}{Train ($\mu$s)} & \multicolumn{2}{c}{Central F1}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Backbone & Edge & CIC & Edge & CIC & Edge & CIC\\", r"\midrule"]
    names = {"ftt": "FT-Transformer~\cite{gorishniy2021revisiting}", "ftt-g": r"\ftg (ours)", "mlp": "MLP (128-64)"}
    for m in ["mlp", "ftt", "ftt-g"]:
        lines.append(f"{names[m]} & {sp['edge_'+m]['params']/1000:.1f} & {sp['cic_'+m]['params']/1000:.1f} & "
                     f"{sp['edge_'+m]['train_us']:.0f} & {sp['cic_'+m]['train_us']:.0f} & {cen[('edge', m)]:.3f} & {cen[('cic', m)]:.3f}\\\\")
    lines.append(r"\midrule")
    lines.append(f"Root set only (\\ftg) & & & & & {np.mean(ro['edge_10']):.3f} & {np.mean(ro['cic_10']):.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "tab_backbone.tex"), "w").write("\n".join(lines) + "\n")
    num("CentralEdge", cen[("edge", "ftt-g")]); num("CentralCic", cen[("cic", "ftt-g")])
    num("CentralEdgeFtt", cen[("edge", "ftt")]); num("CentralCicFtt", cen[("cic", "ftt")])
    num("CentralEdgeMlp", cen[("edge", "mlp")]); num("CentralCicMlp", cen[("cic", "mlp")])
    num("RootOnlyEdge", np.mean(ro["edge_10"]), "{:.2f}"); num("RootOnlyCic", np.mean(ro["cic_10"]), "{:.2f}")
    for k in ["5", "20"]:
        if f"edge_{k}" in ro:
            num("RootOnlyEdge" + {"5": "Five", "20": "Twenty"}[k], np.mean(ro[f"edge_{k}"]), "{:.2f}")
            num("RootOnlyCic" + {"5": "Five", "20": "Twenty"}[k], np.mean(ro[f"cic_{k}"]), "{:.2f}")
    sx = [sp[f"{ds}_ftt"]["train_us"] / sp[f"{ds}_ftt-g"]["train_us"] for ds in ["edge", "cic"]]
    num("SpeedupEdge", sx[0], "{:.1f}"); num("SpeedupCic", sx[1], "{:.1f}")
    num("EdgeSpeedup", f"{min(sx):.1f}--{max(sx):.1f}$\\times$", "{}")
    num("ParamsEdge", f"{sp['edge_ftt-g']['params']:,}".replace(",", "\\,"), "{}")
    num("ParamsCic", f"{sp['cic_ftt-g']['params']:,}".replace(",", "\\,"), "{}")
    # MLP in FL
    m = df[(df.tag == "mlp")]
    if len(m):
        lines = []
        for at in ATTACKS_ALL:
            for ag in ["fedavg", "median", "fltrust", "trace"]:
                g = m[(m.attack == at) & (m["agg"] == ag)]
                if len(g):
                    num(f"Mlp{at.capitalize()}{ag.capitalize()}Fone", g.f1.mean(), "{:.2f}")
                    if at == "bd":
                        num(f"Mlp{at.capitalize()}{ag.capitalize()}Asr", g.asr.mean(), "{:.2f}")


ATTACKS_ALL = ["none", "lf", "sf", "alie", "bd"]


def ablation_table(df):
    var = [("full", "\\trace (full)"), ("no_agree", "w/o agreement gate"), ("median_agree", "agreement: median only"),
           ("no_redund", "w/o collusion term $\\rho$"), ("no_reliab", "w/o reliability gate"),
           ("no_probe", "w/o feature-edit probe"), ("mean_reliab", "mean instead of worst class"),
           ("no_history", "w/o reputation"), ("no_clip", "w/o norm bounding")]
    lines = [r"\begin{table}[t]", r"\caption{Ablation of \trace (macro-F1; for BD also ASR; seed 0). Edge-IIoTset: all variants; CIC-BCCC-NRC: main components.}",
             r"\label{tab:abl}", r"\centering\scriptsize\setlength{\tabcolsep}{2.1pt}",
             r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Variant & Clean & LF & SF & ALIE & BD F1 & BD ASR\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        lines.append(r"\multicolumn{7}{@{}l}{\emph{" + ("Edge-IIoTset" if ds == "edge" else "CIC-BCCC-NRC") + r"}}\\")
        for vn, name in var:
            if vn == "full":
                g = df[(df.tag == "main") & (df["agg"] == "trace") & (df.seed == 0) & (df.dataset == ds)]
            else:
                g = df[(df.tag == f"abl-{vn}") & (df.dataset == ds)]
            if g.empty:
                continue
            cells = []
            for at in ATTACKS_ALL:
                x = g[g.attack == at]
                cells.append(f"{x.f1.mean():.2f}" if len(x) else "--")
            x = g[g.attack == "bd"]
            cells.append(f"{x.asr.mean():.2f}" if len(x) else "--")
            lines.append(name + " & " + " & ".join(cells) + r"\\")
            for at in ATTACKS_ALL:
                x = g[g.attack == at]
                if len(x):
                    num(f"Abl{ds.capitalize()}{vn.replace('_','').capitalize()}{at.capitalize()}", x.f1.mean(), "{:.2f}")
            if len(g[g.attack == "bd"]):
                num(f"Abl{ds.capitalize()}{vn.replace('_','').capitalize()}Asr", g[g.attack == "bd"].asr.mean(), "{:.2f}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "tab_abl.tex"), "w").write("\n".join(lines) + "\n")


def root_table(df):
    lines = [r"\begin{table}[t]", r"\caption{Size and bias of the root set (seed 0). LF: macro-F1; BD: ASR. ``ACI only'': all root records are drawn from the ACI-IoT testbed, which covers five of the seven classes.}",
             r"\label{tab:root}", r"\centering\scriptsize\setlength{\tabcolsep}{2.6pt}",
             r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{FLTrust} & \multicolumn{2}{c}{\trace}\\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Data & Root set & LF F1 & BD ASR & LF F1 & BD ASR\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        for rp, bias, name in [(5, "None", "5 / class"), (10, "None", "10 / class (main)"), (50, "None", "50 / class"), (10, "['ACI-IoT-2023']", "ACI only, 10 / class")]:
            if ds == "edge" and bias != "None":
                continue
            cells = []
            for ag in ["fltrust", "trace"]:
                g = df[(df.dataset == ds) & (df["agg"] == ag) & (df.root == rp) & (df.root_bias == bias) & (df.seed == 0) & (df.tag.isin(["main", "root", "rootbias"]))]
                lf = g[g.attack == "lf"].f1; bd = g[g.attack == "bd"].asr
                cells += [f"{lf.mean():.2f}" if len(lf) else "--", f"{bd.mean():.2f}" if len(bd) else "--"]
                key = f"Root{ds.capitalize()}{ag.capitalize()}{['Five','Ten','Fifty'][[5,10,50].index(rp)]}{'Bias' if bias!='None' else ''}"
                if len(lf): num(key + "Lf", lf.mean(), "{:.2f}")
                if len(bd): num(key + "Asr", bd.mean(), "{:.2f}")
            lines.append(("Edge-IIoTset" if ds == "edge" else "CIC-BCCC-NRC") + " & " + name + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "tab_root.tex"), "w").write("\n".join(lines) + "\n")


def cost_numbers(df):
    d = df[(df.tag == "main") & (df.attack == "none")]
    for ds in ["edge", "cic"]:
        base = d[(d.dataset == ds) & (d["agg"] == "fedavg")].seconds.mean() / 40
        for ag in ["fltrust", "flame", "trace", "krum"]:
            x = d[(d.dataset == ds) & (d["agg"] == ag)].seconds.mean() / 40
            num(f"Cost{ds.capitalize()}{ag.capitalize()}", x - base, "{:.1f}")
        num(f"Round{ds.capitalize()}", base, "{:.1f}")


def write_numbers():
    with open(os.path.join(PAPER, "numbers.tex"), "w") as fh:
        fh.write("% generated by make_assets.py -- do not edit\n")
        for k, v in sorted(NUM.items()):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    man = json.load(open(os.path.join(HERE, "data", "cic_manifest.json")))
    used = [m for m in man if "rows_total" in m]
    num("CicFiles", str(len(used)), "{}")
    num("CicRowsTotal", f"{sum(m['rows_total'] for m in used):,}".replace(",", "\\,"), "{}")
    num("CicRows", f"{sum(m['rows_sampled'] for m in used):,}".replace(",", "\\,"), "{}")
    df = load_runs()
    print(len(df), "runs", df.groupby(["tag"]).size().to_dict())
    main_table(df)
    worst_case(df)
    fig_trust(df)
    fig_frac(df)
    hetero_table(df)
    backbone_table(df)
    ablation_table(df)
    root_table(df)
    cost_numbers(df)
    d = df[df.tag == "main"]
    g = d.groupby(["dataset", "attack", "agg"])[["f1", "dr", "fpr", "asr"]].agg(["mean", "std", "count"]).round(4)
    g.to_csv(os.path.join(HERE, "results", "summary_main.csv"))
    if "--het" in sys.argv:
        het_table(df)
    write_numbers()
