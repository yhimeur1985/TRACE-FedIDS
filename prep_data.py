"""Leakage-aware preprocessing for the two benchmarks.

Edge-IIoTset (packet level, ML file):
  * identifier tier removed (time, addresses, ephemeral ports, payload/URI strings,
    raw counters, checksums, stream indices, MQTT message/topic, DNS name);
  * service/protocol context kept (tcp.dstport, udp.port, relative tcp.seq/ack);
  * every value canonicalised by numeric parsing, so '0' and '0.0' are identical;
  * DDoS_UDP and MITM dropped: their rows are column-shifted in the release;
  * chronological split inside each class block (last 20 % of rows -> test).
CIC-BCCC-NRC TabularIoTAttack-2024 (flow level, 6 environments):
  * uniform reservoir sample per file (line index kept), identity columns removed;
  * attack names harmonised into 7 families; chronological split per file.
Output: data/<name>.npz with X_train, y_train, g_train (group), X_test, y_test, g_test,
feature names, class names.
"""
import csv, gzip, io, os, random, sys, zipfile, json
import numpy as np
import pandas as pd

ROOT_EDGE = os.environ.get("EDGE_IIOT_CSV", "raw/EdgeIoT/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv")
ROOT_CIC = os.environ.get("CIC_BCCC_ROOT", "raw/CIC-BCCC-NRC-TabularIoTAttacks-2024")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
TEST_FRAC = 0.2

EDGE_ID = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
           "icmp.transmit_timestamp", "tcp.srcport", "http.request.full_uri", "http.request.uri.query",
           "http.referer", "http.file_data", "tcp.options", "tcp.payload", "mqtt.msg", "mqtt.topic",
           "dns.qry.name", "tcp.ack_raw", "tcp.checksum", "icmp.checksum", "icmp.seq_le", "udp.stream",
           "mbtcp.trans_id"]
EDGE_CONTEXT = ["tcp.dstport", "udp.port", "tcp.seq", "tcp.ack"]
EDGE_DROP_CLASSES = ["DDoS_UDP", "MITM"]


def canon(series):
    """Canonicalise a column: numeric parse; non-numeric strings -> stable category code."""
    num = pd.to_numeric(series, errors="coerce")
    bad = num.isna() & series.notna()
    if bad.any():
        cats = {v: i + 1 for i, v in enumerate(sorted(series[bad].astype(str).str.strip().unique()))}
        num[bad] = series[bad].astype(str).str.strip().map(cats).astype(float) * 1000.0 + 7.0
    return num.fillna(0.0).astype(np.float64)


def protocol_group(df):
    """Dominant protocol of a packet (used for protocol-based clients)."""
    g = np.full(len(df), "other", dtype=object)
    order = [("tcp", ["tcp.flags", "tcp.len", "tcp.dstport"]), ("udp", ["udp.port", "udp.time_delta"]),
             ("icmp", ["icmp.unused"]), ("arp", ["arp.opcode", "arp.hw.size"]),
             ("dns", ["dns.qry.qu", "dns.qry.type", "dns.retransmission"]),
             ("http", ["http.request.method", "http.response", "http.content_length"]),
             ("modbus", ["mbtcp.len", "mbtcp.unit_id"]),
             ("mqtt", ["mqtt.msgtype", "mqtt.len", "mqtt.hdrflags"])]
    for name, cols in order:  # later (application-layer) protocols override transport
        m = np.zeros(len(df), bool)
        for c in cols:
            if c in df:
                m |= df[c].values != 0
        g[m] = name
    return g


def prep_edge():
    raw = pd.read_csv(ROOT_EDGE, dtype=str, low_memory=False)
    raw = raw[~raw["Attack_type"].isin(EDGE_DROP_CLASSES)].copy()
    raw["__row"] = np.arange(len(raw))
    y_name = raw["Attack_type"].values
    feats = [c for c in raw.columns if c not in EDGE_ID + ["Attack_label", "Attack_type", "__row"]]
    X = pd.DataFrame({c: canon(raw[c]) for c in feats})
    # drop constant columns
    keep = [c for c in feats if X[c].nunique() > 1]
    X = X[keep]
    proto = protocol_group(X)
    classes = ["Normal"] + sorted(set(y_name) - {"Normal"})
    y = np.array([classes.index(v) for v in y_name])
    # chronological split inside each class block
    is_test = np.zeros(len(X), bool)
    for c in range(len(classes)):
        idx = np.where(y == c)[0]  # already in file order
        cut = int(np.floor(len(idx) * (1 - TEST_FRAC)))
        is_test[idx[cut:]] = True
    Xv = X.values.astype(np.float64)
    ctx = np.array([c in EDGE_CONTEXT for c in keep])
    np.savez_compressed(os.path.join(OUT, "edge.npz"),
                        X_train=Xv[~is_test], y_train=y[~is_test], g_train=proto[~is_test].astype(str),
                        X_test=Xv[is_test], y_test=y[is_test], g_test=proto[is_test].astype(str),
                        features=np.array(keep), classes=np.array(classes), context_mask=ctx)
    print("edge", Xv.shape, "features", len(keep), "context", ctx.sum())
    print(pd.Series(y_name).value_counts().to_string())
    print(pd.Series(proto).value_counts().to_string())


# ---------------------------------------------------------------- CIC-BCCC-NRC
CIC_ENVS = {
    "ACI-IoT-2023": "CIC-BCCC-NRC-ACI-IOT-2023",
    "Edge-IIoT-2022": "CIC-BCCC-NRC-Edge-IIoTSet-2022",
    "IoMT-2024": "CIC-BCCC-NRC-IoMT-2024",
    "IoT-HCRL-2019": "CIC-BCCC-NRC-IoT-HCRL-2019",
    "MQTT-IoT-2020": "CIC-BCCC-NRC-MQTTIoT-IDS-2020",
    "IoT-2022": "CIC-BCCC-NRC-IoT-2022",
}
CIC_ID = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Attack Name", "Label"]
FAMILIES = ["Benign", "DoS", "Recon", "BruteForce", "Spoofing", "Web", "Malware"]
CAP_ATT, CAP_BEN = 4000, 12000


def family(fname):
    n = fname.lower()
    if n.startswith("benign"): return "Benign"
    if "malformed" in n: return None
    if "brute" in n or "password" in n: return "BruteForce"
    if "flood" in n or n.startswith("dos") or n.startswith("ddos"): return "DoS"
    if "recon" in n or "scan" in n or "fingerprint" in n: return "Recon"
    if "mitm" in n or "spoof" in n: return "Spoofing"
    if "sql" in n or "xss" in n or "upload" in n: return "Web"
    if "backdoor" in n or "ransom" in n: return "Malware"
    raise ValueError(fname)


def open_text(path):
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        return io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace", newline=""), name
    return open(path, "r", encoding="utf-8", errors="replace", newline=""), os.path.basename(path)


def reservoir(path, cap, seed):
    rng = random.Random(seed)
    fh, name = open_text(path)
    rd = csv.reader(fh)
    header = next(rd)
    keep, n = [], 0
    for row in rd:
        if len(row) != len(header):
            continue
        if n < cap:
            keep.append((n, row))
        else:
            j = rng.randint(0, n)
            if j < cap:
                keep[j] = (n, row)
        n += 1
    fh.close()
    return header, keep, n, name


LATE_FILES = {("ACI-IoT-2023", "Recon Port Scan"), ("IoMT-2024", "DoS TCP Flood"), ("IoMT-2024", "MQTT DDoS Publish Flood"),
              ("IoMT-2024", "MQTT DoS Connect Flood"), ("IoMT-2024", "Recon Port Scan"),
              ("MQTT-IoT-2020", "Sparta SSH Brute Force"), ("Edge-IIoT-2022", "DDoS TCP SYN Flood")}


def prep_cic(name="cic", full=False):
    """name='cic' reproduces the original corpus (files available at submission); full=True adds
    the seven files transferred during the revision. A provenance file <name>_prov.npz stores the
    source file of every record."""
    frames, manifest = [], []
    for env, folder in CIC_ENVS.items():
        d = os.path.join(ROOT_CIC, folder)
        for f in sorted(os.listdir(d)):
            if not (f.endswith(".csv") or f.endswith(".zip")):
                continue
            base = f[:-4]
            if not full and (env, base) in LATE_FILES:
                continue
            fam = family(base)
            if fam is None:
                manifest.append({"env": env, "file": f, "status": "excluded (malformed family)"})
                continue
            cap = CAP_BEN if fam == "Benign" else CAP_ATT
            header, keep, n, _ = reservoir(os.path.join(d, f), cap, f"2026|{env}|{base}")
            keep.sort(key=lambda t: t[0])
            df = pd.DataFrame([r for _, r in keep], columns=header)
            df["__line"] = [i for i, _ in keep]
            df["__nrows"] = n
            df["__env"], df["__family"], df["__file"] = env, fam, base
            frames.append(df)
            manifest.append({"env": env, "file": f, "family": fam, "rows_total": n, "rows_sampled": len(keep)})
            print(env, base, fam, n, len(keep), flush=True)
    df = pd.concat(frames, ignore_index=True)
    feats = [c for c in df.columns if c not in CIC_ID and not c.startswith("__")]
    X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in feats})
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = [c for c in feats if X[c].nunique() > 1]
    X = X[keep].values.astype(np.float64)
    y = np.array([FAMILIES.index(v) for v in df["__family"]])
    env = df["__env"].values.astype(str)
    # chronological split per file on the original line index
    is_test = np.zeros(len(df), bool)
    for fname, idx in df.groupby(["__env", "__file"]).indices.items():
        lines = df["__line"].values[idx]
        order = idx[np.argsort(lines)]
        cut = int(np.floor(len(order) * (1 - TEST_FRAC)))
        is_test[order[cut:]] = True
    fid = (df["__env"] + "/" + df["__file"]).values.astype(str)
    line = df["__line"].values
    np.savez_compressed(os.path.join(OUT, f"{name}_prov.npz"), f_train=fid[~is_test], f_test=fid[is_test],
                        line_train=line[~is_test], line_test=line[is_test])
    target = os.path.join(OUT, f"{name}.npz")
    if name == "cic" and os.path.exists(target):
        old = np.load(target, allow_pickle=True)
        assert np.array_equal(old["X_train"], X[~is_test]) and np.array_equal(old["y_test"], y[is_test]), "corpus changed"
        print("cic.npz reproduced exactly; provenance written")
        return
    np.savez_compressed(target,
                        X_train=X[~is_test], y_train=y[~is_test], g_train=env[~is_test],
                        X_test=X[is_test], y_test=y[is_test], g_test=env[is_test],
                        features=np.array(keep), classes=np.array(FAMILIES),
                        context_mask=np.zeros(len(keep), bool))
    json.dump(manifest, open(os.path.join(OUT, f"{name}_manifest.json"), "w"), indent=1)
    print("cic", X.shape, "features", len(keep))
    print(pd.crosstab(df["__env"], df["__family"]).to_string())


if __name__ == "__main__":
    which = sys.argv[1:] or ["edge", "cic"]
    if "edge" in which: prep_edge()
    if "cic" in which: prep_cic()
    if "cic_full" in which: prep_cic("cic_full", full=True)
