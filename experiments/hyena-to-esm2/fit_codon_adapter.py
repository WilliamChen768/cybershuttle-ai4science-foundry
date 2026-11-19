"""Per-codon check: does HyenaDNA's state at a codon predict ESM-2's vector for that amino acid?

Uses the per-base HyenaDNA states and per-residue ESM-2 vectors saved by embed_hyena.py and
embed_esm2.py (run WITHOUT --no-per-base / --no-per-residue). For each codon we take HyenaDNA's
state at the codon's last base (HyenaDNA is causal, so that state has seen the whole codon and
everything before it) and fit a ridge regression to ESM-2's vector for the encoded amino acid.

Baselines use only what is written in the sequence:
  - codon identity, amino-acid identity, codon + the two preceding codons (one-hot)
  - HyenaDNA + codon identity, HyenaDNA + codon + preceding codons
  - controls: shuffled HyenaDNA states, shuffled pairs, training mean

Scores on held-out genes (genes from the same family stay on the same side of the split):
  - residue R2 and cosine between predicted and true ESM-2 vectors
  - amino-acid decoding: does the predicted vector land nearest the right amino acid's average vector?
  - gene level: predictions averaged per gene, compared as in fit_adapter.py (R2 and retrieval)

Usage:
    python fit_codon_adapter.py --hyena outputs/run_6943155/embeddings.pt \
                                  --esm outputs/run_6972143/esm2_embeddings.pt
"""
import argparse
import json
import os
import time

import numpy as np

import fit_adapter as fa
from config import OUT_DIR

ALPHAS = 10.0 ** np.arange(-1, 5.01, 1.0)


# ---------- features ----------
def onehot(idx, k):
    M = np.zeros((len(idx), k), np.float32)
    ok = idx >= 0
    M[np.flatnonzero(ok), idx[ok]] = 1.0
    return M


def assemble(Hs, Ys, codon_lists, aa_lists):
    """Stack per-gene arrays into residue-level arrays plus gene bookkeeping."""
    lens = np.array([len(y) for y in Ys])
    gid = np.repeat(np.arange(len(Ys)), lens)
    start = np.r_[0, np.cumsum(lens)[:-1]]
    pos = np.arange(lens.sum()) - start[gid]
    codon = np.concatenate(codon_lists)
    aa = np.concatenate(aa_lists)
    prev1 = np.full(len(codon), -1)
    prev1[1:] = codon[:-1]
    prev1[pos < 1] = -1
    prev2 = np.full(len(codon), -1)
    prev2[2:] = codon[:-2]
    prev2[pos < 2] = -1
    C, P1, P2, A = onehot(codon, 64), onehot(prev1, 64), onehot(prev2, 64), onehot(aa, 20)
    return {"H": np.concatenate(Hs).astype(np.float32), "Y": np.concatenate(Ys).astype(np.float32),
            "gid": gid, "aa": aa, "C": C, "P1": P1, "P2": P2, "A": A}


# ---------- ridge with cheap alpha search ----------
def prep(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-8] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def select_alpha(X, Y, fold, n_folds):
    sse = np.zeros(len(ALPHAS))
    for f in range(n_folds):
        tr, va = fold != f, fold == f
        if tr.sum() == 0 or va.sum() == 0:
            continue
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Xt, Yt, Xv, Yv = X[tr] - mx, Y[tr] - my, X[va] - mx, Y[va] - my
        A = (Xt.T @ Xt).astype(np.float64)
        B = (Xt.T @ Yt).astype(np.float64)
        Av = (Xv.T @ Xv).astype(np.float64)
        Bv = (Xv.T @ Yv).astype(np.float64)
        yy = float((Yv.astype(np.float64) ** 2).sum())
        lam, Q = np.linalg.eigh(A)
        QB = Q.T @ B
        for k, a in enumerate(ALPHAS):
            W = Q @ (QB / (lam + a)[:, None])
            sse[k] += yy - 2 * np.sum(W * Bv) + np.sum((Av @ W) * W)
    return float(ALPHAS[int(np.argmin(sse))])


# ---------- metrics ----------
def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def pool(M, gid):
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    return np.add.reduceat(M, starts, axis=0) / np.diff(np.r_[starts, len(gid)])[:, None]


def metrics(D, tr, te, pred):
    Y, gid, aa = D["Y"], D["gid"], D["aa"]
    mu = Y[tr].mean(0)
    Yt = Y[te]
    out = {"res_r2": float(1 - ((Yt - pred) ** 2).sum() / ((Yt - mu) ** 2).sum()),
           "res_cos": float((unit(Yt - mu) * unit(pred - mu)).sum(1).mean())}
    # amino-acid decoding by nearest average vector (centroids from training residues)
    cent = np.stack([Y[tr][aa[tr] == a].mean(0) if (aa[tr] == a).any() else np.full(Y.shape[1], np.nan)
                     for a in range(20)])
    sim = unit(pred - mu) @ unit(np.nan_to_num(cent, nan=0.0) - mu).T
    sim[:, np.isnan(cent[:, 0])] = -np.inf
    out["aa_acc"] = float((sim.argmax(1) == aa[te]).mean())
    # gene level: average the predictions per gene, then score like fit_adapter.py
    gtr = pool(Y[tr], gid[tr])
    gte_true, gte_pred = pool(Yt, gid[te]), pool(pred, gid[te])
    g = fa.score(gte_true.astype(np.float64), gte_pred.astype(np.float64), gtr.mean(0).astype(np.float64))
    out.update({"gene_r2": g["r2"], "gene_top1": g["top1"], "gene_top5": g["top5"]})
    return out


# ---------- experiment ----------
def run_experiment(D, proteins, args):
    n_genes = int(D["gid"].max()) + 1
    groups = fa.family_groups(proteins, args.kmer, args.jaccard)
    sizes = np.bincount(groups)
    info = {"n_genes": n_genes, "n_residues": int(len(D["Y"])), "n_families": int(len(sizes)),
            "genes_in_multi_gene_families": int(sizes[sizes > 1].sum())}
    print(f"{n_genes} genes, {info['n_residues']} codon-residue pairs, {info['n_families']} families")
    H, C, P1, P2, A = D["H"], D["C"], D["P1"], D["P2"], D["A"]
    order = ["HyenaDNA state at codon end", "Codon identity", "Amino-acid identity",
             "Codon + 2 preceding codons", "HyenaDNA + codon identity",
             "HyenaDNA + codon + 2 preceding codons",
             "Codon + shuffled HyenaDNA (control)", "HyenaDNA, shuffled pairs (control)",
             "Training mean (no input)"]
    results = {k: [] for k in order}
    for s in range(args.n_splits):
        rng = np.random.default_rng(args.seed + s)
        gtr, gte = fa.group_split(groups, args.test_frac, rng)
        is_tr = np.zeros(n_genes, bool)
        is_tr[gtr] = True
        tr = np.flatnonzero(is_tr[D["gid"]])
        te = np.flatnonzero(~is_tr[D["gid"]])
        gfold = fa.group_folds(groups[gtr], args.folds, rng)
        fold_of_gene = np.full(n_genes, -1)
        fold_of_gene[gtr] = gfold
        fold = fold_of_gene[D["gid"]][tr]
        Hshuf = H[rng.permutation(len(H))]
        sets = {
            "HyenaDNA state at codon end": H,
            "Codon identity": C,
            "Amino-acid identity": A,
            "Codon + 2 preceding codons": np.hstack([C, P1, P2]),
            "HyenaDNA + codon identity": np.hstack([H, C]),
            "HyenaDNA + codon + 2 preceding codons": np.hstack([H, C, P1, P2]),
            "Codon + shuffled HyenaDNA (control)": np.hstack([Hshuf, C]),
        }
        for label, X in sets.items():
            pred, alpha = _fit_on_subset(X, D["Y"], tr, te, fold, args.folds)
            m = metrics(D, tr, te, pred)
            m["alpha"] = alpha
            results[label].append(m)
        perm = rng.permutation(len(tr))
        Ysh = D["Y"].copy()
        Ysh[tr] = D["Y"][tr][perm]
        pred, alpha = _fit_on_subset(H, Ysh, tr, te, fold, args.folds)
        m = metrics(D, tr, te, pred)         # scored against the real targets
        m["alpha"] = alpha
        results["HyenaDNA, shuffled pairs (control)"].append(m)
        mu = D["Y"][tr].mean(0)
        results["Training mean (no input)"].append(metrics(D, tr, te, np.tile(mu, (len(te), 1))))
        print(f"split {s + 1}/{args.n_splits}: {len(gtr)} train genes, {len(gte)} test genes, "
              f"{len(tr)} / {len(te)} residues")
    return order, results, info


def _fit_on_subset(X, Y, tr, te, fold, n_folds):
    """fit_predict with `fold` given only for the training rows."""
    Xs = prep(X, tr)
    alpha = select_alpha(Xs[tr], Y[tr], fold, n_folds)
    mx, my = Xs[tr].mean(0), Y[tr].mean(0)
    Xt = Xs[tr] - mx
    A = (Xt.T @ Xt).astype(np.float64)
    B = (Xt.T @ (Y[tr] - my)).astype(np.float64)
    W = np.linalg.solve(A + alpha * np.eye(A.shape[0]), B).astype(np.float32)
    return (Xs[te] - mx) @ W + my, alpha


def format_table(order, results, n_test_genes, n_aa_majority):
    def ms(label, key, pct=False):
        v = np.array([r[key] for r in results[label]], dtype=float)
        if pct:
            return f"{100 * v.mean():.1f}% +- {100 * v.std():.1f}"
        return f"{v.mean():+.3f} +- {v.std():.3f}"

    w = max(len(l) for l in order) + 2
    head = (f"{'Input features':<{w}}{'Residue R2':>16}{'Residue cos':>16}{'AA decode %':>16}"
            f"{'Gene R2':>16}{'Gene top-1 %':>16}")
    lines = [head, "-" * len(head)]
    for label in order:
        lines.append(f"{label:<{w}}{ms(label, 'res_r2'):>16}{ms(label, 'res_cos'):>16}"
                     f"{ms(label, 'aa_acc', True):>16}{ms(label, 'gene_r2'):>16}"
                     f"{ms(label, 'gene_top1', True):>16}")
    lines.append(f"(gene retrieval chance is about {100.0 / n_test_genes:.1f}% top-1; "
                 f"always guessing the most common amino acid gives about {100 * n_aa_majority:.1f}%)")
    return "\n".join(lines)


# ---------- loading ----------
def load_inputs(args):
    import torch

    def load(p):
        try:
            return torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(p, map_location="cpu")

    h, e = load(args.hyena), load(args.esm)
    names = list(h["names"])
    assert names == list(e["names"]), "gene order differs between the two embedding files"
    assert h["per_base"][0] is not None, "HyenaDNA file has no per-base states (run without --no-per-base)"
    assert e["per_residue"][0] is not None, "ESM-2 file has no per-residue vectors (run without --no-per-residue)"
    cds, prot = fa.read_fasta(args.cds), fa.read_fasta(args.protein)
    if args.max_genes:
        names = names[:args.max_genes]
    Hs, Ys, cl, al, proteins = [], [], [], [], []
    for i, nm in enumerate(names):
        pb, pr = h["per_base"][i].float().numpy(), e["per_residue"][i].float().numpy()
        assert len(pb) == 3 * len(pr) == len(cds[nm]), f"{nm}: base/residue counts do not line up"
        Hs.append(pb[2::3])                                  # state at the last base of each codon
        Ys.append(pr)
        cl.append(np.array([fa.CODON_INDEX[cds[nm][j:j + 3]] for j in range(0, len(cds[nm]), 3)]))
        al.append(np.array([fa.AA_INDEX[ch] for ch in prot[nm]]))
        proteins.append(prot[nm])
    return names, Hs, Ys, cl, al, proteins


def main():
    d = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyena", required=True, help="embeddings.pt WITH per-base states")
    ap.add_argument("--esm", required=True, help="esm2_embeddings.pt WITH per-residue vectors")
    ap.add_argument("--cds", default=os.path.join(d, "data", "genes_cds.fasta"))
    ap.add_argument("--protein", default=os.path.join(d, "data", "genes_protein.fasta"))
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--kmer", type=int, default=3)
    ap.add_argument("--jaccard", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-genes", type=int, default=0, help="use only the first N genes (quick test)")
    args = ap.parse_args()

    start = time.time()
    names, Hs, Ys, cl, al, proteins = load_inputs(args)
    D = assemble(Hs, Ys, cl, al)
    print(f"loaded {len(names)} genes: HyenaDNA {D['H'].shape[1]}-d states, ESM-2 {D['Y'].shape[1]}-d vectors")
    order, results, info = run_experiment(D, proteins, args)
    n_test = int(round(args.test_frac * len(names)))
    majority = np.bincount(D["aa"], minlength=20).max() / len(D["aa"])
    table = format_table(order, results, n_test, majority)
    print("\n" + table)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "codon_adapter_results.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT_DIR, "codon_adapter_results.json"), "w") as f:
        json.dump({"settings": vars(args), "data": info, "results": results,
                   "seconds": time.time() - start}, f, indent=2)
    print(f"\nSaved {OUT_DIR}/codon_adapter_results.txt and codon_adapter_results.json")


if __name__ == "__main__":
    main()