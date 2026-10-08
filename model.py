"""NumPy FT-Transformer (Gorishniy et al., 2021) and MLP with hand-written backward passes.

Parameters live in a single flat float32 vector so that federated aggregation, similarity and
clipping act on one array; `unflatten` returns views into it.
FT-Transformer: feature tokenizer -> [CLS]+F tokens -> L pre-norm Transformer blocks (multi-head
self-attention + GELU FFN; the last block computes the [CLS] row only) -> LN -> ReLU -> linear.
"""
import numpy as np

SQ2PI = float(np.sqrt(2.0 / np.pi))


def _gelu(x):
    t = np.tanh(SQ2PI * (x + 0.044715 * x * x * x))
    return 0.5 * x * (1 + t), t


def _gelu_back(dy, x, t):
    dt = (1 - t ** 2) * SQ2PI * (1 + 3 * 0.044715 * x * x)
    return dy * (0.5 * (1 + t) + 0.5 * x * dt)


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = xc * inv
    return xh * g + b, (xh, inv)


def _ln_back(dy, g, cache):
    xh, inv = cache
    d = xh.shape[-1]
    dg = (dy * xh).reshape(-1, d).sum(0)
    db = dy.reshape(-1, d).sum(0)
    dxh = dy * g
    dx = inv * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True))
    return dx, dg, db


class Spec:
    """Shapes of every parameter tensor, with offsets into the flat vector."""

    def __init__(self, shapes):
        self.shapes, self.offsets, n = shapes, {}, 0
        for k, s in shapes:
            sz = int(np.prod(s))
            self.offsets[k] = (n, n + sz, s)
            n += sz
        self.size = n

    def unflatten(self, flat):
        return {k: flat[a:b].reshape(s) for k, (a, b, s) in self.offsets.items()}


class FTTransformer:
    name = "FT-Transformer"

    def __init__(self, n_feat, n_cls, d=32, n_blocks=2, n_heads=4, ffn_mult=2, groups=None):
        """groups: list of feature-index lists. Each token is the sum of the FT-T embeddings
        x_j * w_j of the features in its group (group size 1 = original FT-Transformer)."""
        self.F, self.C, self.d, self.L, self.H = n_feat, n_cls, d, n_blocks, n_heads
        self.h = ffn_mult * d
        groups = groups or [[j] for j in range(n_feat)]
        self.G = len(groups)
        self.M = np.zeros((n_feat, self.G), np.float32)
        for gi, g in enumerate(groups):
            self.M[g, gi] = 1.0
        self.identity = self.G == n_feat and np.allclose(self.M, np.eye(n_feat))
        sh = [("tok_w", (n_feat, d)), ("tok_b", (self.G, d)), ("cls", (d,))]
        for l in range(n_blocks):
            sh += [(f"{l}.ln1_g", (d,)), (f"{l}.ln1_b", (d,)),
                   (f"{l}.wqkv", (d, 3 * d)), (f"{l}.bqkv", (3 * d,)),
                   (f"{l}.wo", (d, d)), (f"{l}.bo", (d,)),
                   (f"{l}.ln2_g", (d,)), (f"{l}.ln2_b", (d,)),
                   (f"{l}.w1", (d, self.h)), (f"{l}.b1", (self.h,)),
                   (f"{l}.w2", (self.h, d)), (f"{l}.b2", (d,))]
        sh += [("lnf_g", (d,)), ("lnf_b", (d,)), ("head_w", (d, n_cls)), ("head_b", (n_cls,))]
        self.spec = Spec(sh)

    def init(self, rng):
        flat = np.zeros(self.spec.size, np.float32)
        p = self.spec.unflatten(flat)
        d = self.d
        p["tok_w"][:] = rng.uniform(-1, 1, p["tok_w"].shape) / np.sqrt(d)
        p["tok_b"][:] = rng.uniform(-1, 1, p["tok_b"].shape) / np.sqrt(d)
        p["cls"][:] = rng.uniform(-1, 1, d) / np.sqrt(d)
        for l in range(self.L):
            p[f"{l}.ln1_g"][:] = 1; p[f"{l}.ln2_g"][:] = 1
            for k, fan in [("wqkv", d), ("wo", d), ("w1", d), ("w2", self.h)]:
                lim = np.sqrt(6.0 / (fan + p[f"{l}.{k}"].shape[1]))
                p[f"{l}.{k}"][:] = rng.uniform(-lim, lim, p[f"{l}.{k}"].shape)
        p["lnf_g"][:] = 1
        lim = np.sqrt(6.0 / (d + self.C))
        p["head_w"][:] = rng.uniform(-lim, lim, p["head_w"].shape)
        return flat

    # ------------------------------------------------------------------ forward
    def _block(self, X, p, l, cls_only):
        B, T, d = X.shape
        H, dh = self.H, d // self.H
        U, c1 = _ln(X, p[f"{l}.ln1_g"], p[f"{l}.ln1_b"])
        qkv = U @ p[f"{l}.wqkv"] + p[f"{l}.bqkv"]
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
        if cls_only:
            q = q[:, :1]
        Tq = q.shape[1]
        qh = q.reshape(B, Tq, H, dh).transpose(0, 2, 1, 3)
        kh = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        vh = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        s = qh @ kh.transpose(0, 1, 3, 2) / float(np.sqrt(dh))
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s); a /= a.sum(-1, keepdims=True)
        oh = a @ vh
        o = oh.transpose(0, 2, 1, 3).reshape(B, Tq, d)
        R = X[:, :1] if cls_only else X
        X1 = R + o @ p[f"{l}.wo"] + p[f"{l}.bo"]
        U2, c2 = _ln(X1, p[f"{l}.ln2_g"], p[f"{l}.ln2_b"])
        z = U2 @ p[f"{l}.w1"] + p[f"{l}.b1"]
        gz, t = _gelu(z)
        X2 = X1 + gz @ p[f"{l}.w2"] + p[f"{l}.b2"]
        cache = (X.shape, U, c1, qh, kh, vh, a, o, U2, c2, z, gz, t, cls_only)
        return X2, cache

    def forward(self, flat, x, train=False):
        p = self.spec.unflatten(flat)
        B = x.shape[0]
        E = x[:, :, None] * p["tok_w"][None]
        tok = (E if self.identity else np.einsum("bfd,fg->bgd", E, self.M.astype(E.dtype))) + p["tok_b"][None]
        X = np.concatenate([np.broadcast_to(p["cls"], (B, 1, self.d)), tok], 1).astype(flat.dtype)
        caches = []
        for l in range(self.L):
            X, c = self._block(X, p, l, cls_only=(l == self.L - 1))
            caches.append(c)
        h = X[:, 0]
        hn, cf = _ln(h, p["lnf_g"], p["lnf_b"])
        hr = np.maximum(hn, 0)
        logits = hr @ p["head_w"] + p["head_b"]
        if not train:
            return logits
        return logits, (x, caches, h, hn, cf, hr)

    # ----------------------------------------------------------------- backward
    def _block_back(self, dX2, p, gp, l, cache):
        (shape, U, c1, qh, kh, vh, a, o, U2, c2, z, gz, t, cls_only) = cache
        B, T, d = shape
        H, dh = self.H, d // self.H
        Tq = qh.shape[2]
        dX1 = dX2.copy()
        gp[f"{l}.w2"] += gz.reshape(-1, self.h).T @ dX2.reshape(-1, d)
        gp[f"{l}.b2"] += dX2.reshape(-1, d).sum(0)
        dgz = dX2 @ p[f"{l}.w2"].T
        dz = _gelu_back(dgz, z, t)
        gp[f"{l}.w1"] += U2.reshape(-1, d).T @ dz.reshape(-1, self.h)
        gp[f"{l}.b1"] += dz.reshape(-1, self.h).sum(0)
        dU2 = dz @ p[f"{l}.w1"].T
        dx, dg, db = _ln_back(dU2, p[f"{l}.ln2_g"], c2)
        gp[f"{l}.ln2_g"] += dg; gp[f"{l}.ln2_b"] += db
        dX1 += dx
        # attention output projection
        gp[f"{l}.wo"] += o.reshape(-1, d).T @ dX1.reshape(-1, d)
        gp[f"{l}.bo"] += dX1.reshape(-1, d).sum(0)
        do = dX1 @ p[f"{l}.wo"].T
        doh = do.reshape(B, Tq, H, dh).transpose(0, 2, 1, 3)
        da = doh @ vh.transpose(0, 1, 3, 2)
        dvh = a.transpose(0, 1, 3, 2) @ doh
        ds = a * (da - (da * a).sum(-1, keepdims=True)) / float(np.sqrt(dh))
        dqh = ds @ kh
        dkh = ds.transpose(0, 1, 3, 2) @ qh
        dq = dqh.transpose(0, 2, 1, 3).reshape(B, Tq, d)
        dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, d)
        dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, d)
        dqkv = np.zeros((B, T, 3 * d), dX2.dtype)
        dqkv[:, :Tq, :d] = dq
        dqkv[..., d:2 * d] = dk
        dqkv[..., 2 * d:] = dv
        gp[f"{l}.wqkv"] += U.reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
        gp[f"{l}.bqkv"] += dqkv.reshape(-1, 3 * d).sum(0)
        dU = dqkv @ p[f"{l}.wqkv"].T
        dX, dg, db = _ln_back(dU, p[f"{l}.ln1_g"], c1)
        gp[f"{l}.ln1_g"] += dg; gp[f"{l}.ln1_b"] += db
        if cls_only:
            dX[:, :1] += dX1
        else:
            dX += dX1
        return dX

    def backward(self, flat, dlogits, cache):
        p = self.spec.unflatten(flat)
        g = np.zeros_like(flat)
        gp = self.spec.unflatten(g)
        x, caches, h, hn, cf, hr = cache
        gp["head_w"] += hr.T @ dlogits
        gp["head_b"] += dlogits.sum(0)
        dhr = dlogits @ p["head_w"].T
        dhn = dhr * (hn > 0)
        dh, dg, db = _ln_back(dhn, p["lnf_g"], cf)
        gp["lnf_g"] += dg; gp["lnf_b"] += db
        B = x.shape[0]
        dX = np.zeros((B, 1, self.d), flat.dtype)
        dX[:, 0] = dh
        for l in reversed(range(self.L)):
            dX = self._block_back(dX, p, gp, l, caches[l])
        gp["cls"] += dX[:, 0].sum(0)
        dtok = dX[:, 1:]
        gp["tok_b"] += dtok.sum(0)
        dE = dtok if self.identity else np.einsum("bgd,fg->bfd", dtok, self.M.astype(dtok.dtype))
        gp["tok_w"] += (dE * x[:, :, None]).sum(0)
        return g


class MLP:
    name = "MLP"

    def __init__(self, n_feat, n_cls, hidden=(128, 64)):
        self.F, self.C = n_feat, n_cls
        dims = [n_feat, *hidden, n_cls]
        self.dims = dims
        self.spec = Spec([(f"w{i}", (dims[i], dims[i + 1])) for i in range(len(dims) - 1)] +
                         [(f"b{i}", (dims[i + 1],)) for i in range(len(dims) - 1)])

    def init(self, rng):
        flat = np.zeros(self.spec.size, np.float32)
        p = self.spec.unflatten(flat)
        for i in range(len(self.dims) - 1):
            lim = np.sqrt(6.0 / (self.dims[i] + self.dims[i + 1]))
            p[f"w{i}"][:] = rng.uniform(-lim, lim, p[f"w{i}"].shape)
        return flat

    def forward(self, flat, x, train=False):
        p = self.spec.unflatten(flat)
        acts, h = [x], x
        n = len(self.dims) - 1
        for i in range(n):
            h = h @ p[f"w{i}"] + p[f"b{i}"]
            if i < n - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        return (h, acts) if train else h

    def backward(self, flat, dlogits, acts):
        p = self.spec.unflatten(flat)
        g = np.zeros_like(flat)
        gp = self.spec.unflatten(g)
        n = len(self.dims) - 1
        d = dlogits
        for i in reversed(range(n)):
            gp[f"w{i}"] += acts[i].T @ d
            gp[f"b{i}"] += d.sum(0)
            if i > 0:
                d = (d @ p[f"w{i}"].T) * (acts[i] > 0)
        return g


def softmax_ce(logits, y, weights=None):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    pr = e / e.sum(1, keepdims=True)
    B = len(y)
    w = np.ones(B, np.float32) if weights is None else weights
    loss = -(w * np.log(pr[np.arange(B), y] + 1e-12)).sum() / w.sum()
    d = pr.copy()
    d[np.arange(B), y] -= 1
    d *= (w / w.sum())[:, None]
    return loss, d.astype(logits.dtype)


def predict(model, flat, X, bs=2048):
    out = [model.forward(flat, X[i:i + bs]) for i in range(0, len(X), bs)]
    return np.concatenate(out, 0)


def grad_check(seed=0):
    rng = np.random.default_rng(seed)
    for m in [FTTransformer(5, 3, d=8, n_blocks=2, n_heads=2),
              FTTransformer(5, 3, d=8, n_blocks=2, n_heads=2, groups=[[0, 2], [1], [3, 4]]), MLP(5, 3, (7, 6))]:
        flat = m.init(rng).astype(np.float64)
        flat += rng.normal(0, 0.05, flat.shape)
        x = rng.normal(size=(4, 5)); y = rng.integers(0, 3, 4)
        logits, cache = m.forward(flat, x, train=True)
        _, d = softmax_ce(logits.astype(np.float64), y)
        g = m.backward(flat, d.astype(np.float64), cache)
        errs = []
        for i in rng.choice(flat.size, 60, replace=False):
            e = np.zeros_like(flat); e[i] = 1e-6
            lp, _ = softmax_ce(m.forward(flat + e, x), y)
            lm, _ = softmax_ce(m.forward(flat - e, x), y)
            num = (lp - lm) / 2e-6
            errs.append(abs(num - g[i]) / max(1e-4, abs(num) + abs(g[i])))
        print(m.name, "max rel err", max(errs))


if __name__ == "__main__":
    grad_check()
