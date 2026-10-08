"""Revision assets: tables/figures/numbers for the revised manuscript and the supplementary file.
Reads every run in results/runs and writes into ../paper (main) and ../paper/supp (supplement)."""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.environ.get("TRACE_PAPER_DIR", os.path.join(HERE, "paper_assets"))
TAB, FIG = os.path.join(PAPER, "tab"), os.path.join(PAPER, "figs")
STAB = os.path.join(PAPER, "supp", "tab")
for d in (TAB, FIG, STAB):
    os.makedirs(d, exist_ok=True)
NUM = {}

AGGS = ["fedavg", "fedprox", "krum", "trmean", "median", "fltrust", "flame", "zeno", "foolsgold", "fldetector",
        "fedavg_root", "trace", "tracec"]
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "krum": "Krum", "trmean": "Trim. mean", "median": "Median",
         "fltrust": "FLTrust", "flame": "FLAME", "zeno": "Zeno", "foolsgold": "FoolsGold", "fldetector": "FLDetector",
         "fedavg_root": "FedAvg+root", "trace": r"\trace (ours)", "tracec": r"\trace-C (ours)", "bucket_median": "Bucketing+Median"}
TRUSTED = {"fltrust", "zeno", "fedavg_root", "trace", "tracec"}   # rules that use the server's root set
ATT = ["none", "lf", "sf", "alie", "minmax", "bd"]
ATTN = {"none": "Clean", "lf": "LF", "sf": "SF", "alie": "ALIE", "minmax": "Min-Max", "bd": "BD"}
DSN = {"edge": "Edge-IIoTset", "cic": "CIC-BCCC-NRC", "cic_full": "CIC-BCCC-NRC (all 54 files)"}


_DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
        "8": "Eight", "9": "Nine"}


def num(k, v, fmt="{:.2f}"):
    k = "".join(_DIG.get(ch, ch) for ch in k if ch not in "_-.")   # TeX macro names: letters only
    NUM[k] = v if isinstance(v, str) else fmt.format(v)


def load():
    rows = []
    for f in glob.glob(os.path.join(HERE, "results", "runs", "*.json")):
        r = json.load(open(f))
        c = r["cfg"]
        row = dict(tag=r.get("tag"), dataset=c["dataset"], partition=c["partition"], alpha=c.get("alpha"),
                   agg=c["agg"], attack=c["attack"], seed=c["seed"], frac=c["mal_frac"], root=c["root_per_class"],
                   root_bias=str(c.get("root_bias")), model=c["model"], trace=json.dumps(c.get("trace")),
                   trigger=c.get("trigger", "extreme"), seconds=r["seconds"])
        row.update({k: v for k, v in r["final"].items()})
        row.update({"L_" + k: v for k, v in r["last"].items() if k != "round"})
        row["malw"] = [w["mal_weight"] for w in r["weights"]] if r["weights"] else None
        row["alphas"] = r.get("alphas")
        row["per_class_dr"] = r.get("per_class_dr")
        row["fld_removed"] = r.get("fld_removed"); row["mal"] = r.get("mal")
        rows.append(row)
    df = pd.DataFrame(rows)
    df["aggname"] = df["agg"]
    return df


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def cell(g, met):
    return g[met].mean() if len(g) else np.nan


def main_table(df):
    """Main robustness table (3 seeds), means only; CIs in the supplement."""
    d = df[df.tag == "main"]
    cols = [("none", "f1"), ("lf", "f1"), ("lf", "dr"), ("sf", "f1"), ("alie", "f1"), ("minmax", "f1"), ("bd", "f1"),
            ("bd", "asr")]
    head = ["Clean", "LF", "LF DR", "SF", "ALIE", "Min-Max", "BD", r"ASR$\downarrow$"]
    best = {}
    for ds in ["edge", "cic"]:
        for at, met in cols:
            v = {a: round(cell(d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == a)], met), 2) for a in AGGS}
            v = {a: x for a, x in v.items() if not np.isnan(x)}
            if not v:
                continue
            u = sorted(set(v.values()), reverse=(met != "asr"))
            best[(ds, at, met)] = ([a for a in v if v[a] == u[0]], [a for a in v if len(u) > 1 and v[a] == u[1]])
    L = [r"\begin{table*}[t]",
         r"\caption{Robustness at 20\,\% malicious clients (mean over three seeds of the values averaged over rounds 30--40; 95\,\% confidence intervals, seed-level final-round values and paired tests are in Supplementary Tables~\ref{tab:ci-edge}--\ref{tab:paired}; -- not run). F1 = macro-F1, DR = detection rate under LF, ASR = backdoor attack success rate. Best value per column in bold, second best underlined. $^\dagger$ uses the server's trusted root set.}",
         r"\label{tab:main}", r"\centering\scriptsize\setlength{\tabcolsep}{2.3pt}",
         r"\begin{tabular}{@{}l" + "c" * 8 + "c" + "c" * 8 + "@{}}", r"\toprule",
         r" & \multicolumn{8}{c}{Edge-IIoTset (attack-category skew, 13 classes)} & & \multicolumn{8}{c}{CIC-BCCC-NRC (environment skew, 7 classes)}\\",
         r"\cmidrule(lr){2-9}\cmidrule(lr){11-18}",
         "Method & " + " & ".join(head) + " & & " + " & ".join(head) + r"\\", r"\midrule"]
    for a in AGGS:
        cells = []
        for ds in ["edge", "cic"]:
            for at, met in cols:
                g = d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == a)]
                x = cell(g, met)
                s = "--" if np.isnan(x) else f"{x:.2f}"
                b = best.get((ds, at, met), ([], []))
                if a in b[0]:
                    s = r"\textbf{" + s + "}"
                elif a in b[1]:
                    s = r"\underline{" + s + "}"
                cells.append(s)
            if ds == "edge":
                cells.append("")
        nm = NAMES[a] + (r"$^\dagger$" if a in TRUSTED and a != "trace" else "")
        if a == "trace":
            L.append(r"\midrule")
            nm = r"\trace (ours)$^\dagger$"
        if a == "zeno":
            L.append(r"\midrule")
        L.append(nm + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(TAB, "tab_main.tex"), "w").write("\n".join(L) + "\n")
    # macros
    for ds in ["edge", "cic"]:
        for at in ATT:
            for a in AGGS:
                g = d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == a)]
                if len(g):
                    k = f"M{ds.capitalize()}{at}{a.replace('_', '')}"
                    for met, mn in [("f1", "Fone"), ("dr", "Dr"), ("fpr", "Fpr"), ("asr", "Asr")]:
                        num(k + mn, g[met].mean())
        # worst case over attacks (F1)
        for a in AGGS:
            g = d[(d.dataset == ds) & (d["agg"] == a)]
            per = g.groupby("attack").f1.mean()
            have = [x for x in ["lf", "sf", "alie", "minmax", "bd"] if x in per]
            if len(have) == 5:
                num(f"W{ds.capitalize()}{a.replace('_', '')}", per[have].min())


def ci_tables(df):
    """Supplement: mean +- 95% CI (late-round average), seed-level final-round values, paired tests."""
    d = df[df.tag == "main"]
    for ds in ["edge", "cic"]:
        L = [r"\begin{table*}[!htbp]", r"\caption{" + DSN[ds] + r": macro-F1 (LF DR: detection rate; BD ASR: attack success rate) as mean $\pm$ 95\,\% CI half-width over three seeds (late-round average, rounds 30--40) and, below each, the three seed-level values at the final round 40.}",
             rf"\label{{tab:ci-{ds}}}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}l" + "c" * 8 + r"@{}}", r"\toprule",
             r"Method & Clean & LF & LF DR & SF & ALIE & Min-Max & BD & BD ASR\\", r"\midrule"]
        for a in AGGS:
            c1, c2 = [], []
            for at, met in [("none", "f1"), ("lf", "f1"), ("lf", "dr"), ("sf", "f1"), ("alie", "f1"), ("minmax", "f1"),
                            ("bd", "f1"), ("bd", "asr")]:
                g = d[(d.dataset == ds) & (d.attack == at) & (d["agg"] == a)].sort_values("seed")
                if g.empty:
                    c1.append("--"); c2.append("")
                    continue
                c1.append(f"{g[met].mean():.2f}$\\pm${ci(g[met]):.2f}")
                c2.append("/".join(f"{v:.2f}" for v in g["L_" + met]))
            L.append(NAMES[a] + " & " + " & ".join(c1) + r"\\")
            L.append(r"{\tiny final round} & " + " & ".join(r"{\tiny " + x + "}" for x in c2) + r"\\[1pt]")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        open(os.path.join(STAB, f"tab_ci_{ds}.tex"), "w").write("\n".join(L) + "\n")
    # paired comparisons of TRACE against the best competitor per column
    rows = []
    for ds in ["edge", "cic"]:
        for at, met in [("none", "f1"), ("lf", "f1"), ("lf", "dr"), ("sf", "f1"), ("alie", "f1"), ("minmax", "f1"),
                        ("bd", "f1"), ("bd", "asr")]:
            g = d[(d.dataset == ds) & (d.attack == at)]
            means = g.groupby("agg")[met].mean().drop("trace", errors="ignore")
            if means.empty or "trace" not in set(g["agg"]):
                continue
            comp = means.idxmin() if met == "asr" else means.idxmax()
            if met == "asr":  # exclude rules that block the backdoor only by collapsing
                ok = g[g.attack == "bd"].groupby("agg").f1.mean()
                good = [a for a in means.index if ok.get(a, 0) >= 0.9 * ok.get("trace", 0)]
                comp = means[good].idxmin() if good else comp
            t = g[g["agg"] == "trace"].set_index("seed")[met]
            o = g[g["agg"] == comp].set_index("seed")[met]
            s = sorted(set(t.index) & set(o.index))
            diff = (t[s] - o[s]).values
            p = stats.ttest_rel(t[s], o[s]).pvalue if len(s) > 1 and np.std(diff) > 0 else np.nan
            rows.append((ds, at, met, NAMES[comp], t[s].mean(), o[s].mean(), diff.mean(), ci(diff), p))
    L = [r"\begin{table}[!htbp]", r"\caption{Paired comparison (same seeds, i.e.\ same partitions, roots and attackers) of \trace with the strongest competitor of each column. $\Delta$ = \trace $-$ competitor with 95\,\% CI; $p$ from a paired $t$-test over three seeds (indicative only). For ASR, competitors whose clean macro-F1 under BD is below 90\,\% of \trace's are excluded.}",
         r"\label{tab:paired}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}llllccc@{}}",
         r"\toprule", r"Data & Attack & Metric & Competitor & $\Delta$ & 95\,\% CI & $p$\\", r"\midrule"]
    for ds, at, met, comp, tm, om, dm, dci, p in rows:
        L.append(f"{DSN[ds]} & {ATTN[at]} & {met.upper()} & {comp} & {dm:+.3f} & $\\pm${dci:.3f} & {p:.2f}\\\\".replace("nan", "--"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STAB, "tab_paired.tex"), "w").write("\n".join(L) + "\n")


def fit(t):
    """Scale a wide table to the text width."""
    return t.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")


def write_numbers():
    with open(os.path.join(PAPER, "numbers2.tex"), "w") as fh:
        fh.write("% generated by make_assets2.py\n")
        for k, v in sorted(NUM.items()):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}\n")


# ============================================================ additional revision assets
def g_(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m]


def mean_ci(x):
    x = np.asarray(x, float)
    return (np.nan, np.nan) if len(x) == 0 else (x.mean(), ci(x) if len(x) > 1 else np.nan)


def fmt(m, c, d=2):
    if np.isnan(m):
        return "--"
    return f"{m:.{d}f}" + ("" if np.isnan(c) else f"\\,{{\\tiny$\\pm${c:.{d}f}}}")


def tracec_rows(df):
    """Add TRACE-C to the main-table frame (tag tracec -> agg 'tracec')."""
    t = df[df.tag == "tracec"].copy()
    t["agg"] = "tracec"; t["tag"] = "main"
    return pd.concat([df, t], ignore_index=True)


def attack_table(df):
    """Held-out triggers and adaptive attacks: ASR (and F1) per rule."""
    aggs = [("fedavg", "FedAvg", None), ("fltrust", "FLTrust", None), ("flame", "FLAME", None), ("zeno", "Zeno", None),
            ("fldetector", "FLDet.", None), ("trace", r"\trace", None), ("tracec", r"\trace-C", None)]
    trig = [("extreme", "Extreme (main)"), ("midrange", "Mid-range"), ("distributed", "Distributed"),
            ("semantic", "Semantic"), ("cleanlabel", "Clean-label")]
    T = json.load(open(os.path.join(HERE, "data", "triggers.json")))
    L = [r"\begin{table*}[t]", r"\caption{Backdoor ASR under held-out trigger families and \trace-aware adaptive attacks (mean over three seeds; seed-level values and macro-F1 in Supplementary Table~\ref{tab:s-attf1}). Natural: ASR of the trigger on clean FedAvg models. Held-out triggers were fixed after \trace was frozen; the adaptive attacker knows the algorithm, its state and all benign updates, and emulates the root set with its own data (oracle: knows the root set).}",
         r"\label{tab:attacks}", r"\centering\scriptsize\setlength{\tabcolsep}{2.0pt}",
         r"\begin{tabular}{@{}l" + "c" * 8 + "c" + "c" * 8 + r"@{}}", r"\toprule",
         r" & \multicolumn{8}{c}{Edge-IIoTset} & & \multicolumn{8}{c}{CIC-BCCC-NRC}\\", r"\cmidrule(lr){2-9}\cmidrule(lr){11-18}",
         "Attack & Natural & " + " & ".join(a[1] for a in aggs) + " & & Natural & " + " & ".join(a[1] for a in aggs) + r"\\", r"\midrule"]
    for tk, tn in trig:
        cells = []
        for ds in ["edge", "cic"]:
            cells.append(f"{T[ds][tk]['natural']:.2f}")
            for a, _, _ in aggs:
                if tk == "extreme":
                    g = df[(df.tag == "main") & (df.dataset == ds) & (df.attack == "bd") & (df["agg"] == a)]
                else:
                    if a == "tracec":
                        g = df[(df.tag == "tracec-trigger") & (df.dataset == ds) & (df.trigger == tk)]
                    else:
                        g = df[(df.tag == "trigger") & (df.dataset == ds) & (df.attack == "bd") & (df["agg"] == a) & (df.trigger == tk)]
                cells.append("--" if g.empty else f"{g.asr.mean():.2f}")
            if ds == "edge":
                cells.append("")
        L.append(tn + " & " + " & ".join(cells) + r"\\")
    L.append(r"\midrule")
    for at, an, tagt, tagc in [("adaptive_bd", "Adaptive BD", "adaptive", "tracec-adaptive"),
                               ("adaptive_bd", "Adaptive BD (oracle root)", "adaptive-oracle", None),
                               ("adaptive_lf", "Adaptive LF$^\\ast$", "adaptive", "tracec-adaptive")]:
        cells = []
        for ds in ["edge", "cic"]:
            cells.append("")
            for a, _, _ in aggs:
                if a == "trace":
                    g = df[(df.tag == tagt) & (df.dataset == ds) & (df.attack == at)]
                elif a == "tracec" and tagc:
                    g = df[(df.tag == tagc) & (df.dataset == ds) & (df.attack == at)]
                else:
                    g = df.iloc[0:0]
                if g.empty:
                    cells.append("--")
                elif at == "adaptive_lf":
                    cells.append(f"{g.dr.mean():.2f}")
                else:
                    cells.append(f"{g.asr.mean():.2f}")
            if ds == "edge":
                cells.append("")
        L.append(an + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\multicolumn{18}{@{}l}{\scriptsize $^\ast$ For adaptive LF the detection rate (DR) is shown instead of the ASR; under plain LF \trace's DR is " +
          f"{g_(df[df.tag=='main'], dataset='edge', attack='lf', agg='trace').dr.mean():.2f} (packets) and {g_(df[df.tag=='main'], dataset='cic', attack='lf', agg='trace').dr.mean():.2f} (flows).}}\\\\",
          r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(TAB, "tab_attacks.tex"), "w").write("\n".join(L) + "\n")
    for ds in ["edge", "cic"]:
        for tag, key in [("adaptive", "Ad"), ("tracec-adaptive", "AdC"), ("adaptive-oracle", "AdO")]:
            for at, an in [("adaptive_bd", "Bd"), ("adaptive_lf", "Lf")]:
                g = df[(df.tag == tag) & (df.dataset == ds) & (df.attack == at)]
                if len(g):
                    num(f"{key}{an}{ds.capitalize()}Asr", g.asr.mean()); num(f"{key}{an}{ds.capitalize()}Fone", g.f1.mean())
                    num(f"{key}{an}{ds.capitalize()}Dr", g.dr.mean())
                    al = [a for x in g.alphas if x for a in x]
                    if al:
                        num(f"{key}{an}{ds.capitalize()}Alpha", float(np.mean(al)))
        for tk, _ in trig[1:]:
            for a in ["fedavg", "fltrust", "flame", "zeno", "fldetector", "trace", "tracec"]:
                g = df[(df.dataset == ds) & (df.attack == "bd") & (df.trigger == tk) & (df.tag == ("tracec-trigger" if a == "tracec" else "trigger"))]
                if a != "tracec":
                    g = g[g["agg"] == a]
                if len(g):
                    num(f"Tr{tk.capitalize()}{a.capitalize()}{ds.capitalize()}Asr", g.asr.mean())


def ablation_table(df):
    var = [("full", r"\trace (full)"), ("no_server", "w/o server update"), ("no_agree", "w/o agreement gate"),
           ("median_agree", "agreement: median only"), ("no_reliab", "w/o reliability gate"),
           ("no_probe", "w/o feature-edit probe"), ("no_history", "w/o reputation"), ("cons", "+ update consistency (\\trace-C)")]
    L = [r"\begin{table}[t]", r"\caption{Ablation of \trace (macro-F1; BD also ASR); mean over three seeds. FedAvg+root (Table~\ref{tab:main}) isolates the server update without trust weighting.}",
         r"\label{tab:abl}", r"\centering\scriptsize\setlength{\tabcolsep}{1.8pt}",
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Variant & Clean & LF & SF & ALIE & BD & ASR\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        L.append(r"\multicolumn{7}{@{}l}{\emph{" + DSN[ds] + r"}}\\")
        for vn, name in var:
            if vn == "full":
                g = df[(df.tag == "main") & (df["agg"] == "trace") & (df.dataset == ds)]
            elif vn == "cons":
                g = df[(df.tag == "tracec") & (df.dataset == ds)]
            else:
                g = df[(df.tag == f"abl-{vn}") & (df.dataset == ds)]
            if g.empty:
                continue
            cells = []
            for at in ["none", "lf", "sf", "alie", "bd"]:
                x = g[g.attack == at]
                cells.append(f"{x.f1.mean():.2f}" if len(x) else "--")
                if len(x):
                    num(f"Abl{ds.capitalize()}{vn.replace('_', '').capitalize()}{at.capitalize()}", x.f1.mean())
            x = g[g.attack == "bd"]
            cells.append(f"{x.asr.mean():.2f}" if len(x) else "--")
            if len(x):
                num(f"Abl{ds.capitalize()}{vn.replace('_', '').capitalize()}Asr", x.asr.mean())
            ns = sorted(set(g.seed))
            L.append(name + (r"$^{\S}$" if len(ns) < 3 else "") + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\multicolumn{7}{@{}l}{\scriptsize $^{\S}$ fewer than three seeds available.}\\", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "tab_abl.tex"), "w").write("\n".join(L) + "\n")


def frac_fig(df):
    d = df[((df.tag == "frac") | (df.tag == "main")) & (df.dataset == "edge")]
    aggs = ["fedavg", "median", "fltrust", "flame", "trace"]
    lab = {"fedavg": "FedAvg", "median": "Median", "fltrust": "FLTrust", "flame": "FLAME", "trace": "TRACE"}
    col = {"fedavg": "#8c8c8c", "median": "#1b9e77", "fltrust": "#e08214", "flame": "#7b3294", "trace": "#00567d"}
    mk = {"fedavg": "o", "median": "s", "fltrust": "^", "flame": "D", "trace": "*"}
    panels = [("lf", "f1", "LF: macro-F1"), ("lf", "dr", "LF: detection rate"), ("bd", "f1", "BD: macro-F1"), ("bd", "asr", "BD: attack success rate")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.85))
    for ax, (at, met, title) in zip(axes, panels):
        for a in aggs:
            g = d[(d.attack == at) & (d["agg"] == a)]
            xs, ms, cs = [], [], []
            for fr in [0.1, 0.2, 0.3, 0.4]:
                x = g[np.isclose(g.frac, fr)][met]
                if len(x):
                    xs.append(fr * 100); ms.append(x.mean()); cs.append(ci(x) if len(x) > 1 else 0)
            if xs:
                ms, cs = np.array(ms), np.nan_to_num(np.array(cs))
                ax.plot(xs, ms, marker=mk[a], ms=3.5 if a != "trace" else 6, lw=1.0 if a != "trace" else 1.6, color=col[a], label=lab[a])
                ax.fill_between(xs, ms - cs, ms + cs, color=col[a], alpha=0.12, lw=0)
        ax.set_title(title, fontsize=7.5); ax.set_xlabel("malicious clients (%)", fontsize=7)
        ax.tick_params(labelsize=6.5); ax.set_xticks([10, 20, 30, 40])
        for s_ in ["top", "right"]:
            ax.spines[s_].set_visible(False)
    axes[0].legend(fontsize=5.8, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(FIG, "fig_frac.pdf")); plt.close(fig)


def supp_simple(df):
    """Supplementary tables: heterogeneity, root set, participation, preprocessing, data checks,
    grouping, environment conditioning, confirmatory seeds."""
    out = {}
    # heterogeneity (3 seeds)
    d = df[((df.tag == "hetero") | (df.tag == "main")) & (df.dataset == "edge")]
    parts = [("iid", None, "IID"), ("dirichlet", 1.0, r"Dir.\ $\alpha{=}1$"), ("dirichlet", 0.1, r"Dir.\ $\alpha{=}0.1$"),
             ("protocol", None, "Protocol"), ("category", None, "Category (main)"), ("attack", None, "3 classes/client")]
    L = [r"\begin{table}[!htbp]", r"\caption{Edge-IIoTset under different partitions: macro-F1, mean $\pm$ 95\,\% CI over three seeds (IID, and the median on all partitions except the main one: seed 0 only).}", r"\label{tab:s-het}",
         r"\centering\scriptsize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{No attack} & \multicolumn{4}{c}{LF}\\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Partition & FedAvg & Median & FLTrust & \trace & FedAvg & Median & FLTrust & \trace\\", r"\midrule"]
    for p, a, nm in parts:
        g = d[d.partition == p]
        if a is not None:
            g = g[np.isclose(g.alpha.astype(float), a)]
        if g.empty:
            continue
        cells = []
        for at in ["none", "lf"]:
            for ag in ["fedavg", "median", "fltrust", "trace"]:
                x = g[(g.attack == at) & (g["agg"] == ag)].f1
                cells.append(fmt(*mean_ci(x)))
        L.append(nm + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out["het"] = L
    # root set
    L = [r"\begin{table}[!htbp]", r"\caption{Trusted-data stress tests (mean $\pm$ 95\,\% CI over three seeds). Size and bias: LF macro-F1 and BD ASR. Noise: 20\,\% of root labels randomised; contamination: 20\,\% of root attack records labelled benign.}",
         r"\label{tab:s-root}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{FLTrust} & \multicolumn{2}{c}{\trace}\\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Data & Root set & LF F1 & BD ASR & LF F1 & BD ASR\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        for rp, bias, tag, nm in [(5, "None", "root", "5 / class"), (10, "None", "main", "10 / class (main)"), (50, "None", "root", "50 / class"),
                                  (10, "['ACI-IoT-2023']", "rootbias", "ACI testbed only")]:
            if ds == "edge" and bias != "None":
                continue
            cells = []
            for ag in ["fltrust", "trace"]:
                g = df[(df.dataset == ds) & (df["agg"] == ag) & (df.root == rp) & (df.root_bias == bias) & (df.tag == tag)]
                cells += [fmt(*mean_ci(g[g.attack == "lf"].f1)), fmt(*mean_ci(g[g.attack == "bd"].asr))]
            L.append(DSN[ds] + " & " + nm + " & " + " & ".join(cells) + r"\\")
    for kind, nm in [("root_noise", "20\\,\\% label noise"), ("root_contam", "20\\,\\% contamination")]:
        cells = []
        for ag in ["fltrust", "trace"]:
            g = df[(df.tag == f"rootstress-{kind}") & (df["agg"] == ag)]
            cells += [fmt(*mean_ci(g[g.attack == "lf"].f1)), fmt(*mean_ci(g[g.attack == "bd"].asr))]
        L.append("Edge-IIoTset & " + nm + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out["root"] = L
    # participation
    L = [r"\begin{table}[!htbp]", r"\caption{Partial participation (a random 75, 50 or 25\,\% of the clients per round), intermittent attackers (attack in a random half of the rounds) and late joiners (half of the clients, including all attackers, join at round 20); Edge-IIoTset, mean $\pm$ 95\,\% CI over three seeds.}", r"\label{tab:s-part}",
         r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{FLTrust} & \multicolumn{2}{c}{\trace}\\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Setting & LF F1 / DR & BD ASR & LF F1 / DR & BD ASR\\", r"\midrule"]
    for tag, nm in [("main", "Full participation (main)"), ("partial-0.75", "75\\,\\% participation"), ("partial-0.5", "50\\,\\% participation"), ("partial-0.25", "25\\,\\% participation"), ("intermittent", "Intermittent attackers"), ("latejoin", "Late-joining attackers")]:
        cells = []
        for ag in ["fltrust", "trace"]:
            g = df[(df.tag == tag) & (df.dataset == "edge") & (df["agg"] == ag)]
            lf = g[g.attack == "lf"]; bd = g[g.attack == "bd"]
            cells += [f"{lf.f1.mean():.2f} / {lf.dr.mean():.2f}" if len(lf) else "--", fmt(*mean_ci(bd.asr))]
        L.append(nm + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out["part"] = L
    # preprocessing, data checks
    L = [r"\begin{table}[!htbp]", r"\caption{Robustness checks (macro-F1; FPR: false-positive rate without attack; LF DR: detection rate under label poisoning; ASR: backdoor). Root-robust preprocessing: features scaled with the median/IQR of the server's root set instead of global moments (seed 0). All files: CIC-BCCC-NRC with all 54 usable files, including the seven large files transferred during the revision (two seeds). Capture-disjoint: sites of a testbed receive whole source files, so no two clients share a capture (two seeds).}",
         r"\label{tab:s-checks}", r"\centering\scriptsize\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"Check & Rule & Clean & FPR & LF & LF DR & BD & ASR\\", r"\midrule"]
    for tag, ds, nm in [("prep", "edge", "Root-robust prep., packets"), ("prep", "cic", "Root-robust prep., flows"),
                        ("fulldata", "cic_full", "All 54 files, flows"), ("capdisj", "cic", "Capture-disjoint, flows")]:
        for ag in ["fedavg", "fltrust", "trace"]:
            g = df[(df.tag == tag) & (df.dataset == ds) & (df["agg"] == ag)]
            if g.empty:
                continue
            c = []
            for at, met in [("none", "f1"), ("none", "fpr"), ("lf", "f1"), ("lf", "dr"), ("bd", "f1"), ("bd", "asr")]:
                x = g[g.attack == at][met]
                c.append(f"{x.mean():.2f}" if len(x) else "--")
            L.append(nm + " & " + NAMES[ag].replace(" (ours)", "") + " & " + " & ".join(c) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out["checks"] = L
    # confirmatory seeds
    L = [r"\begin{table}[!htbp]", r"\caption{Confirmatory runs on two fresh seeds (3 and 4) that were never inspected during design (macro-F1; LF also DR; BD also ASR). The configuration of every rule was frozen beforehand.}", r"\label{tab:s-confirm}",
         r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
         r"Data & Rule & Clean & LF & LF DR & SF & ALIE & BD & ASR\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        for ag, tag in [("fedavg", "confirm"), ("fltrust", "confirm"), ("trace", "confirm"), ("trace", "tracec-confirm")]:
            g = df[(df.tag == tag) & (df.dataset == ds) & (df["agg"] == ag)]
            if g.empty:
                continue
            c = []
            for at, met in [("none", "f1"), ("lf", "f1"), ("lf", "dr"), ("sf", "f1"), ("alie", "f1"), ("bd", "f1"), ("bd", "asr")]:
                x = g[g.attack == at][met]
                c.append(f"{x.mean():.2f}" if len(x) else "--")
                if len(x):
                    num(f"Conf{ds.capitalize()}{at}{'tracec' if tag.startswith('tracec') else ag}{ {'f1': 'Fone', 'dr': 'Dr', 'asr': 'Asr'}[met] }", x.mean())
            L.append(DSN[ds] + " & " + ("\\trace-C" if tag.startswith("tracec") else NAMES[ag].replace(" (ours)", "")) + " & " + " & ".join(c) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out["confirm"] = L
    # grouping
    rows = []
    for ds in ["edge", "cic"]:
        for m, nm in [("ftt-g", "Semantic groups (\\ftg)"), ("ftt-g-random", "Random groups, same sizes"), ("ftt-g-contig", "Contiguous equal chunks"), ("ftt", "No grouping (FT-Transformer)"), ("mlp", "MLP")]:
            vals = []
            for s in [0, 1, 2]:
                f = os.path.join(HERE, "results", f"central_{ds}_{m}" + (f"_s{s}" if s else "") + ".json")
                if os.path.exists(f):
                    h = json.load(open(f))["hist"][-3:]
                    vals.append(np.mean([x["f1"] for x in h]))
            if m in ("ftt-g", "ftt-g-random", "ftt-g-contig"):
                fg = df[(df.dataset == ds) & (df.attack == "none") & (df["agg"] == "fedavg") & (df.model == m) & (df.tag.isin(["main", "grouping"]))]
            else:
                fg = df.iloc[0:0]
            rows.append((ds, nm, vals, fg.f1.values))
            if vals:
                num(f"Grp{ds.capitalize()}{m.replace('-', '').capitalize()}Central", float(np.mean(vals)))
            if len(fg):
                num(f"Grp{ds.capitalize()}{m.replace('-', '').capitalize()}Fl", float(fg.f1.mean()))
    L = [r"\begin{table}[!htbp]", r"\caption{Tokenisation control: same number of tokens, different grouping. Central: pooled training (3\,000 steps); FL: FedAvg, clean (mean $\pm$ 95\,\% CI over three seeds).}", r"\label{tab:s-group}",
         r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llcc@{}}", r"\toprule", r"Data & Grouping & Central F1 & FL F1\\", r"\midrule"]
    for ds, nm, vals, fl_ in rows:
        L.append(f"{DSN[ds]} & {nm} & {fmt(*mean_ci(vals), d=3)} & {fmt(*mean_ci(fl_), d=3)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out["group"] = L
    for k, L in out.items():
        open(os.path.join(STAB, f"tab_{k}.tex"), "w").write("\n".join(L) + "\n")


def operating_points():
    """From saved test scores: AUROC, DR at FPR 1/5 %, FPR at DR 90/95 % (attack vs benign)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    rows = {}
    for f in glob.glob(os.path.join(HERE, "results", "scores", "*.npz")):
        b = os.path.basename(f)[:-4].split("_")
        ds = b[0]; s = b[-1]; at = b[-2]; ag = "_".join(b[1:-2])
        z = np.load(f)
        y = (z["y"] != 0).astype(int); sc = 1 - z["pben"]
        fpr, tpr, _ = roc_curve(y, sc)
        r = dict(auc=roc_auc_score(y, sc),
                 dr1=float(tpr[fpr <= 0.01].max()), dr5=float(tpr[fpr <= 0.05].max()),
                 fpr90=float(fpr[np.searchsorted(tpr, 0.90)]), fpr95=float(fpr[np.searchsorted(tpr, 0.95)]),
                 fpr_def=float(((z["pred"] != 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
                 dr_def=float(((z["pred"] != 0) & (y == 1)).sum() / max(1, y.sum())), seed=s)
        rows.setdefault((ds, ag), []).append(r)
    L = [r"\begin{table}[!htbp]", r"\caption{Operating points without attack (attack vs benign) at the final round; mean over the seeds whose test scores were saved (last column; seeds 3 and 4 are the confirmatory seeds, so the default FPR differs slightly from the three-seed, late-round value quoted in the manuscript). Default: argmax decision. DR@FPR: highest detection rate reachable with a false-positive rate of at most 1\,\% or 5\,\%; FPR@DR: false-positive rate needed for a detection rate of 90\,\% or 95\,\%.}",
         r"\label{tab:s-op}", r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
         r"Data & Rule & AUROC & DR (def.) & FPR (def.) & DR@1\% & DR@5\% & FPR@90\% & FPR@95\% & seeds\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        for ag in ["fedavg", "median", "fltrust", "flame", "zeno", "fldetector", "fedavg_root", "trace"]:
            R = rows.get((ds, ag))
            if not R:
                continue
            m = {k: np.mean([r[k] for r in R]) for k in R[0] if k != "seed"}
            seeds = ",".join(sorted(r["seed"].lstrip("s") for r in R))
            L.append(f"{DSN[ds]} & {NAMES[ag].replace(' (ours)', '')} & {m['auc']:.3f} & {m['dr_def']:.2f} & {m['fpr_def']:.2f} & {m['dr1']:.2f} & {m['dr5']:.2f} & {m['fpr90']:.2f} & {m['fpr95']:.2f} & {seeds}\\\\")
            for k in ["auc", "dr1", "dr5", "fpr90", "fpr95", "fpr_def"]:
                num(f"Op{ds.capitalize()}{ag.replace('_', '').capitalize()}{k.replace('_', '').capitalize()}", m[k])
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STAB, "tab_op.tex"), "w").write("\n".join(L) + "\n")


def scalability():
    f = os.path.join(HERE, "results", "scalability.json")
    if not os.path.exists(f):
        return
    R = pd.DataFrame(json.load(open(f)))
    L = [r"\begin{table}[!htbp]", r"\caption{Server time (s, one core) and peak additional memory (MB) of one aggregation round versus the number of clients $n$. The root update of FLTrust and \trace (20 local steps) is excluded. Upload per client per round: " +
         f"{R.upload_kb[R.dataset=='edge'].iloc[0]:.0f}\\,kB (packets) and {R.upload_kb[R.dataset=='cic'].iloc[0]:.0f}\\,kB (flows)." + "}",
         r"\label{tab:s-scale}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}ll" + "c" * 8 + r"@{}}", r"\toprule",
         r"Data & Rule & \multicolumn{4}{c}{time (s) for $n=$20/50/100/200} & \multicolumn{4}{c}{memory (MB)}\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        for rule in ["FedAvg", "Median", "Krum", "FLTrust", "FLAME", "Zeno", "FoolsGold", "TRACE"]:
            g = R[(R.dataset == ds) & (R.rule == rule)].sort_values("n")
            L.append(f"{DSN[ds]} & {rule} & " + " & ".join(f"{x:.2f}" for x in g.seconds) + " & " + " & ".join(f"{x:.0f}" for x in g.peak_mb) + r"\\")
            if rule == "TRACE":
                for n_, t_ in zip(g.n, g.seconds):
                    num(f"Scale{ds.capitalize()}{int(n_)}", t_, "{:.1f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STAB, "tab_scale.tex"), "w").write("\n".join(L) + "\n")


def provenance():
    """Client -> source-capture mapping (seed 0) for both federations."""
    import fl
    L = [r"\begin{table*}[!htbp]", r"\caption{Client provenance (seed 0): number of source captures and records per client. Edge-IIoTset: one capture per attack class (the benign traffic consists of several recordings of different devices); CIC-BCCC-NRC: one source CSV file per capture. Captures are shared between clients of the same testbed in the main partition, but never across testbeds; the capture-disjoint variant (Table~\ref{tab:s-checks}) removes the sharing.}",
         r"\label{tab:s-prov}", r"\centering\scriptsize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}l" + "c" * 20 + r"@{}}", r"\toprule",
         "Client & " + " & ".join(str(i) for i in range(20)) + r"\\", r"\midrule"]
    for ds, part in [("edge", "category"), ("cic", "environment"), ("cic", "env_capture")]:
        cfg = dict(fl.DEFAULT, dataset=ds, partition=part)
        if part != "category":
            cfg["env_clients"] = {"ACI-IoT-2023": 4, "Edge-IIoT-2022": 4, "IoMT-2024": 4, "MQTT-IoT-2020": 3, "IoT-HCRL-2019": 3, "IoT-2022": 2}
        rng = np.random.default_rng(0)
        Xtr, ytr, gtr, *_r, classes, feats = fl.load(cfg, rng)
        if part == "category":
            tax = [["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP"], ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner"],
                   ["SQL_injection", "Uploading", "XSS"], ["Backdoor", "Password", "Ransomware"]]
            cfg["categories"] = [[classes.index(c) for c in grp] for grp in tax]
            src = np.array([classes[c] for c in ytr])
        else:
            src = np.load(os.path.join(HERE, "data", "cic_prov.npz"), allow_pickle=True)["f_train"]
            cfg["_files"] = src
        root = fl.take_root(ytr, gtr, cfg, rng)
        avail = np.setdiff1d(np.arange(len(ytr)), root)
        parts = fl.partition(ytr, gtr, cfg, rng, avail)
        env = [str(pd.Series(gtr[p]).mode()[0]) if len(p) else "" for p in parts]
        nm = {"category": "Edge-IIoTset", "environment": "CIC (main)", "env_capture": "CIC (capture-disj.)"}[part]
        L.append(nm + " captures & " + " & ".join(str(len(set(src[p]))) for p in parts) + r"\\")
        L.append(nm + " records & " + " & ".join(f"{len(p)/1000:.1f}k" for p in parts) + r"\\")
        if part != "category":
            short = {"ACI-IoT-2023": "ACI", "Edge-IIoT-2022": "Edg", "IoMT-2024": "IoMT", "MQTT-IoT-2020": "MQTT", "IoT-HCRL-2019": "HCRL", "IoT-2022": "IoT22"}
            L.append(nm + " testbed & " + " & ".join(short.get(e, e) for e in env) + r"\\")
        # sharing statistic
        sh = []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if set(src[parts[i]]) & set(src[parts[j]]):
                    sh.append((i, j))
        num(f"Share{part.replace('_', '').capitalize()}", str(len(sh)), "{}")
        L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(STAB, "tab_prov.tex"), "w").write(fit("\n".join(L) + "\n"))


def trigger_table():
    T = json.load(open(os.path.join(HERE, "data", "triggers.json")))
    import fl
    desc = {"extreme": "3 fields at min/max (main)", "midrange": "3 fields at non-modal median",
            "distributed": "6 fields, interior quantiles", "semantic": "field group copied from a real record",
            "cleanlabel": "mid-range trigger on benign records"}
    L = [r"\begin{table*}[!htbp]", r"\caption{Backdoor trigger families. Fields are given by their CICFlowMeter or Wireshark names; Natural: fraction of detected test attacks that the trigger flips to benign on clean FedAvg models (selection criterion $\le$2\,\% on seeds 0 and 1; for the semantic flow trigger no template below 3.2\,\% was found among the 400 examined). All held-out families were fixed after \trace was frozen.}",
         r"\label{tab:s-trig}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llp{11.2cm}c@{}}", r"\toprule",
         r"Data & Family & Fields (semantic groups) & Natural\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        cfg = dict(fl.DEFAULT, dataset=ds)
        for k in ["extreme", "midrange", "distributed", "semantic", "cleanlabel"]:
            v = T[ds][k]
            feats = v.get("feats")
            if not feats:
                feats = None
            f = ", ".join(feats).replace("_", r"\_") if feats else {"edge": "tcp.seq, tcp.dstport, tcp.len (tcp-state)", "cic": "FWD Init Win Bytes, Fwd IAT Total, Fwd IAT Min (window, fwd-iat)"}[ds]
            if v.get("groups"):
                f += " (" + ", ".join(v["groups"]) + ")"
            if k == "semantic":
                f += f"; template record of class {v.get('template_class', '').replace('_', ' ')}"
            L.append(f"{DSN[ds]} & {k} & {desc[k]}: {f} & {v['natural']:.3f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(STAB, "tab_trig.tex"), "w").write("\n".join(L) + "\n")


def dev_table(df):
    """Validation-split development (seed 0, test set unused)."""
    rows = [("dev-base", "edge", r"\trace (frozen)"), ("dev-cons", "edge", r"\trace-C"),
            ("dev-base", "cic", r"\trace (frozen)"), ("dev-cons", "cic", r"\trace-C"),
            ("envdev-envroot", "cic", "per-environment root, global gate"),
            ("envdev-envcond", "cic", "environment-conditioned reliability")]
    L = [r"\begin{table}[!htbp]", r"\caption{Development on the validation split (seed 0; the last 12.5\,\% of every training block, test records unused). Macro-F1 and, for BD, ASR. Environment conditioning judges each client's reliability only against root records (and probes) of its own testbed.}",
         r"\label{tab:s-dev}", r"\centering\scriptsize\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"Data & Variant & Clean & LF & SF & ALIE & BD & ASR\\", r"\midrule"]
    for tag, ds, nm in rows:
        g = df[(df.tag == tag) & (df.dataset == ds)]
        if tag == "dev-base" and ds == "cic":
            pass
        c = []
        for at in ["none", "lf", "sf", "alie", "bd"]:
            x = g[g.attack == at].f1
            c.append(f"{x.mean():.3f}" if len(x) else "--")
        x = g[g.attack == "bd"].asr
        c.append(f"{x.mean():.3f}" if len(x) else "--")
        L.append(f"{DSN[ds]} & {nm} & " + " & ".join(c) + r"\\")
        key = {"dev-base": "Base", "dev-cons": "Cons", "envdev-envroot": "Envroot", "envdev-envcond": "Envcond"}[tag]
        for at in ["none", "lf", "sf", "alie", "bd"]:
            x = g[g.attack == at]
            if len(x):
                num(f"Dev{ds.capitalize()}{key}{at.capitalize()}Fone", x.f1.mean())
                if at == "bd":
                    num(f"Dev{ds.capitalize()}{key}Asr", x.asr.mean())
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STAB, "tab_dev.tex"), "w").write("\n".join(L) + "\n")


def attack_supp(df):
    aggs = [("fedavg", "FedAvg"), ("fltrust", "FLTrust"), ("flame", "FLAME"), ("zeno", "Zeno"), ("fldetector", "FLDetector"),
            ("trace", r"\trace"), ("tracec", r"\trace-C")]
    trig = [("midrange", "Mid-range"), ("distributed", "Distributed"), ("semantic", "Semantic"), ("cleanlabel", "Clean-label")]
    sl = lambda g, v: f"{v.mean():.2f} {{\\tiny[" + "/".join(f"{x:.2f}" for x in v.values[np.argsort(g.seed.values)]) + "]}"
    L = [r"\begin{table}[!htbp]", r"\caption{Held-out trigger families and adaptive attacks in detail: mean ASR, the seed-level values in brackets (seeds 0/1/2; backdoor outcomes are often bimodal, so intervals would mislead) and the mean macro-F1 in parentheses. For the adaptive attacks, $\bar\alpha$ is the mean interpolation weight that the attackers could use without being gated in their emulation (0 = honest update, 1 = poisoned update, 2 = boosted); for adaptive LF the first number is the detection rate.}",
         r"\label{tab:s-attf1}", r"\centering\scriptsize\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{@{}ll" + "c" * len(aggs) + r"@{}}", r"\toprule",
         "Data & Trigger & " + " & ".join(a[1] for a in aggs) + r"\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        for tk, tn in trig:
            cells = []
            for a, _ in aggs:
                if a == "tracec":
                    g = df[(df.tag == "tracec-trigger") & (df.dataset == ds) & (df.trigger == tk)]
                else:
                    g = df[(df.tag == "trigger") & (df.dataset == ds) & (df["agg"] == a) & (df.trigger == tk)]
                cells.append("--" if g.empty else sl(g, g.asr) + f" ({g.f1.mean():.2f})")
            L.append(f"{DSN[ds]} & {tn} & " + " & ".join(cells) + r"\\")
        L.append(r"\midrule")
    L += [r"Data & Adaptive attack & \multicolumn{2}{c}{\trace} & \multicolumn{2}{c}{$\bar\alpha$} & \multicolumn{2}{c}{\trace-C} & $\bar\alpha$\\", r"\midrule"]
    for ds in ["edge", "cic"]:
        for at, an, tagt, tagc in [("adaptive_bd", "Backdoor, surrogate root", "adaptive", "tracec-adaptive"),
                                   ("adaptive_bd", "Backdoor, oracle root", "adaptive-oracle", None),
                                   ("adaptive_lf", "Label flipping, surrogate root", "adaptive", "tracec-adaptive")]:
            cells = []
            for tg in (tagt, tagc):
                g = df[(df.tag == tg) & (df.dataset == ds) & (df.attack == at)] if tg else df.iloc[0:0]
                if g.empty:
                    cells += [r"\multicolumn{2}{c}{--}", r"\multicolumn{2}{c}{--}" if tg == tagt else "--"]
                    continue
                v = g.dr if at == "adaptive_lf" else g.asr
                al = np.mean([a for x in g.alphas if x for a in x])
                cells += [r"\multicolumn{2}{c}{" + sl(g, v) + f" ({g.f1.mean():.2f})" + "}",
                          (r"\multicolumn{2}{c}{" + f"{al:.2f}" + "}") if tg == tagt else f"{al:.2f}"]
            L.append(f"{DSN[ds]} & {an} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STAB, "tab_attf1.tex"), "w").write(fit("\n".join(L) + "\n").replace("\\textwidth", "\\linewidth"))


def extra_numbers(df):
    """Scalar macros used in the text (and the response letter)."""
    g = df[(df.tag == "main") & (df["agg"] == "tracec") & (df.attack == "bd")]
    if len(g):
        num("AbsTracecAsr", g.groupby("dataset").asr.mean().max())
    # operating points: best FPR needed for DR 90 % on flows
    v = [float(NUM[k]) for k in NUM if k.startswith("OpCic") and k.endswith("FprNineZero")]
    if v:
        num("OpCicBestFprNineZero", min(v))
    # root-set stress
    cats = {"DoS": ["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP"], "Scan": ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner"],
            "Injection": ["SQL_injection", "Uploading", "XSS"], "Malware": ["Backdoor", "Password", "Ransomware"]}
    for ag in ["trace", "fltrust"]:
        drop, ref = [], []
        refg = df[(df.tag == "main") & (df.dataset == "edge") & (df.attack == "lf") & (df["agg"] == ag)]
        for cn, cl in cats.items():
            g = df[(df.tag == f"rootdrop-{cn}") & (df["agg"] == ag)]
            fam = lambda G: [np.mean([r[c] for c in cl if c in r]) for r in G.per_class_dr if r]
            if len(g):
                drop.append(np.mean(fam(g))); ref.append(np.mean(fam(refg)))
                num(f"RootDrop{cn}{ag.capitalize()}", np.mean(fam(g)))
        if drop:
            num(f"RootDrop{ag.capitalize()}", np.mean(drop)); num(f"RootDrop{ag.capitalize()}Ref", np.mean(ref))
            num(f"RootDrop{ag.capitalize()}Min", np.min(drop))
        for kind, kn in [("root_noise", "Noise"), ("root_contam", "Contam")]:
            g = df[(df.tag == f"rootstress-{kind}") & (df["agg"] == ag)]
            if len(g):
                num(f"Root{kn}{ag.capitalize()}Lf", g[g.attack == "lf"].f1.mean())
                num(f"Root{kn}{ag.capitalize()}LfDr", g[g.attack == "lf"].dr.mean())
                num(f"Root{kn}{ag.capitalize()}Asr", g[g.attack == "bd"].asr.mean())
        for ds in ["edge", "cic"]:
            for rp, bias, tag, nm in [(5, "None", "root", "Five"), (50, "None", "root", "Fifty"), (10, "['ACI-IoT-2023']", "rootbias", "Bias")]:
                g = df[(df.dataset == ds) & (df["agg"] == ag) & (df.root == rp) & (df.root_bias == bias) & (df.tag == tag)]
                if len(g):
                    num(f"Root{ds.capitalize()}{ag.capitalize()}{nm}LfR", g[g.attack == "lf"].f1.mean())
                    num(f"Root{ds.capitalize()}{ag.capitalize()}{nm}AsrR", g[g.attack == "bd"].asr.mean())
    num('RunsOld', '410'); num('RunsNew', '1,019'.replace(',', '\\,')); num('RunsTotal', '1,429'.replace(',', '\\,'))
    # attacker fraction
    d = df[((df.tag == "frac") | (df.tag == "main")) & (df.dataset == "edge")]
    for at, met, mn in [("lf", "f1", "LfFone"), ("lf", "dr", "LfDr"), ("bd", "asr", "BdAsr")]:
        for ag in ["fedavg", "median", "fltrust", "trace"]:
            for fr, fn in [(0.1, "Ten"), (0.2, "Twenty"), (0.3, "Thirty"), (0.4, "Forty")]:
                x = d[(d.attack == at) & (d["agg"] == ag) & np.isclose(d.frac, fr)][met]
                if len(x):
                    num(f"Frac{mn}{ag.capitalize()}{fn}", x.mean())
    # heterogeneity: largest clean loss of TRACE against FedAvg
    # server-update ablation: largest F1 change
    for ds in ["edge", "cic"]:
        full = df[(df.tag == "main") & (df["agg"] == "trace") & (df.dataset == ds)].groupby("attack").f1.mean()
        ns = df[(df.tag == "abl-no_server") & (df.dataset == ds)].groupby("attack").f1.mean()
        if len(ns):
            num(f"AblServerMax{ds.capitalize()}", float((full[ns.index] - ns).abs().max()))
    # participation
    for tag, nm in [("partial-0.75", "SevenFive"), ("partial-0.5", "FiveZero"), ("partial-0.25", "TwoFive"),
                    ("intermittent", "Intermittent"), ("latejoin", "Latejoin")]:
        for ag in ["trace", "fltrust"]:
            g = df[(df.tag == tag) & (df["agg"] == ag)]
            if len(g):
                num(f"Part{nm}{ag.capitalize()}Asr", g[g.attack == "bd"].asr.mean())
                num(f"Part{nm}{ag.capitalize()}Dr", g[g.attack == "lf"].dr.mean())
                num(f"Part{nm}{ag.capitalize()}Lf", g[g.attack == "lf"].f1.mean())
    # checks
    for tag, ds, nm in [("prep", "edge", "PrepEdge"), ("prep", "cic", "PrepCic"), ("fulldata", "cic_full", "Full"), ("capdisj", "cic", "Cap")]:
        for ag in ["fedavg", "fltrust", "trace"]:
            g = df[(df.tag == tag) & (df.dataset == ds) & (df["agg"] == ag)]
            for at, met, mn in [("none", "f1", "Clean"), ("none", "fpr", "Fpr"), ("lf", "f1", "Lf"), ("lf", "dr", "LfDr"), ("bd", "asr", "Asr")]:
                x = g[g.attack == at][met]
                if len(x):
                    num(f"{nm}{ag.capitalize()}{mn}", x.mean())
    # cost per round relative to FedAvg (clean runs, main protocol)
    for ds in ["edge", "cic"]:
        base = df[(df.tag == "main") & (df.dataset == ds) & (df.attack == "none") & (df["agg"] == "fedavg")].seconds.mean()
        for ag in ["tracec", "zeno", "fldetector"]:
            g = df[(df.tag == "main") & (df.dataset == ds) & (df.attack == "none") & (df["agg"] == ag)]
            if len(g):
                num(f"Cost{ds.capitalize()}{ag.capitalize()}R", max(0.0, (g.seconds.mean() - base) / 40), "{:.1f}")
    f = os.path.join(HERE, "results", "scalability.json")
    if os.path.exists(f):
        R = pd.DataFrame(json.load(open(f)))
        num("UploadEdge", R.upload_kb[R.dataset == "edge"].iloc[0], "{:.0f}")
        num("UploadCic", R.upload_kb[R.dataset == "cic"].iloc[0], "{:.0f}")
        if "gram_seconds" in R:
            g = R[(R.rule == "TRACE") & (R.n == 200)]
            num("GramShare", float((100 * g.gram_seconds / g.seconds).max()), "{:.0f}")


def lofo_table(df):
    """Leave-one-family-out root set: detection rate of the family that is absent from the root set."""
    cats = {"DoS": ["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP"], "Scan": ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner"],
            "Injection": ["SQL_injection", "Uploading", "XSS"], "Malware": ["Backdoor", "Password", "Ransomware"]}
    cols = [("rootdrop", "fltrust"), ("rootdrop", "trace"), ("lftarget", "fedavg"), ("lftarget", "fltrust"), ("lftarget", "trace"),
            ("lftargetdrop", "fltrust"), ("lftargetdrop", "trace")]
    L = [r"\begin{table}[!htbp]", r"\caption{Leave-one-family-out root set on Edge-IIoTset: detection rate of the three classes of one attack category (mean $\pm$ 95\,\% CI over three seeds). Untargeted LF: attackers relabel all their attack records and the category is absent from the root set. Targeted LF: attackers relabel only the records of that category; the root set is complete (full) or lacks the category (absent).}",
         r"\label{tab:s-lofo}", r"\centering\scriptsize\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Untargeted, absent} & \multicolumn{3}{c}{Targeted, full root} & \multicolumn{2}{c}{Targeted, absent}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
         r"Category & FLTrust & \trace & FedAvg & FLTrust & \trace & FLTrust & \trace\\", r"\midrule"]
    acc = {c: [] for c in cols}
    for cn, cl in cats.items():
        cells = []
        for tg, ag in cols:
            g = df[(df.tag == f"{tg}-{cn}") & (df["agg"] == ag)]
            v = [np.mean([r[c] for c in cl if c in r]) for r in g.per_class_dr if r]
            cells.append(fmt(*mean_ci(v)))
            if v:
                acc[(tg, ag)].append(np.mean(v))
        L.append(f"{cn} & " + " & ".join(cells) + r"\\")
    L.append(r"\midrule")
    L.append("Mean & " + " & ".join(f"{np.mean(acc[c]):.2f}" if acc[c] else "--" for c in cols) + r"\\")
    for (tg, ag), v in acc.items():
        if v:
            num(f"Lofo{tg.capitalize()}{ag.capitalize()}", np.mean(v)); num(f"Lofo{tg.capitalize()}{ag.capitalize()}Min", np.min(v))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STAB, "tab_lofo.tex"), "w").write("\n".join(L) + "\n")


def run_all(df, with_prov=False):
    df = tracec_rows(df)
    main_table(df)
    ci_tables(df)
    attack_table(df)
    attack_supp(df)
    ablation_table(df)
    frac_fig(df)
    supp_simple(df)
    operating_points()
    scalability()
    trigger_table()
    dev_table(df)
    lofo_table(df)
    num('CicRowsFull', '197\\,384')
    extra_numbers(df)
    if with_prov:
        provenance()
    write_numbers()


if __name__ == "__main__":
    df = load()
    print(len(df), "runs", df.groupby("tag").size().to_dict())
    run_all(df, with_prov="--prov" in sys.argv)
