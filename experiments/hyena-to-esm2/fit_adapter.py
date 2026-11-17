"""Can HyenaDNA's embedding of a gene predict ESM-2's embedding of its protein?

Fits a simple adapter (ridge regression) from a gene's HyenaDNA mean embedding to the ESM-2 mean
embedding of the protein it encodes, and scores it on held-out genes. The same regression is run
with simpler inputs as baselines, so we can tell whether HyenaDNA adds anything:

  - GC fraction, coding length, GC + length
  - codon frequencies (64 numbers: counts the protein's composition directly)
  - HyenaDNA with GC content regressed out
  - amino-acid composition of the protein itself (reference ceiling: it is derived from the target)
  - training mean (no input) and HyenaDNA with shuffled gene-protein pairs (controls)

Genes from the same family (similar proteins) are kept on the same side of the train/test split.
The whole experiment is repeated over several random splits and reported as mean +- std.

Usage:
    python fit_adapter.py --hyena outputs/run_6943155/embeddings.pt \
                          --esm outputs/run_6972143/esm2_embeddings.pt
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from config import OUT_DIR

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_INDEX = {c: i for i, c in enumerate(CODONS)}
AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AAS)}
ALPHAS = 10.0 ** np.arange(-2, 5, 0.5)


# ---------- data helpers ----------
def read_fasta(path):
    recs, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                recs[name] = "".join(buf)
            name, buf = line[1:], []
        elif line:
            buf.append(line)
    if name is not None:
        recs[name] = "".join(buf)
    return recs


def codon_freqs(dna):
    v = np.zeros(64)
    for i in range(0, len(dna) - len(dna) % 3, 3):
        v[CODON_INDEX[dna[i:i + 3]]] += 1
    return v / max(v.sum(), 1)


def aa_composition(prot):
    v = np.zeros(20)
    for ch in prot:
        v[AA_INDEX[ch]] += 1
    return v / max(v.sum(), 1)


def family_groups(proteins, k, thresh):
    """Connected components of genes whose proteins share many k-mers (Jaccard >= thresh)."""
    n = len(proteins)
    sets = [{p[i:i + k] for i in range(len(p) - k + 1)} for p in proteins]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            if inter and inter / (len(sets[i]) + len(sets[j]) - inter) >= thresh:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def group_split(groups, test_frac, rng):
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n = len(groups)
    target = int(round(test_frac * n))
    test = np.zeros(n, dtype=bool)
    count = 0
    for g in uniq:
        if count >= target:
            break
        m = groups == g
        if count > 0 and count + m.sum() > 1.3 * target:
            continue
        test |= m
        count += int(m.sum())
    return np.where(~test)[0], np.where(test)[0]


def group_folds(groups, n_folds, rng):
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


# ---------- model ----------
def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit(X, Y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)


def choose_alpha(X, Y, groups, n_folds, rng):
    fold = group_folds(groups, n_folds, rng)
    sse = np.zeros(len(ALPHAS))
    for f in range(n_folds):
        tr, va = fold != f, fold == f
        if tr.sum() == 0 or va.sum() == 0:
            continue
        Xtr, Xva = standardize(X[tr], X[va])
        mu = Y[tr].mean(0)
        for k, a in enumerate(ALPHAS):
            pred = Xva @ ridge_fit(Xtr, Y[tr] - mu, a) + mu
            sse[k] += ((Y[va] - pred) ** 2).sum()
    return float(ALPHAS[int(np.argmin(sse))])


def fit_predict(Xtr, Ytr, Xte, groups_tr, n_folds, rng):
    alpha = choose_alpha(Xtr, Ytr, groups_tr, n_folds, rng)
    A, B = standardize(Xtr, Xte)
    mu = Ytr.mean(0)
    return B @ ridge_fit(A, Ytr - mu, alpha) + mu, alpha


def normalize_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12), n[:, 0]


def score(Ytrue, Ypred, mu_train):
    """R2 against the training mean, mean cosine of centered vectors, and retrieval accuracy."""
    r2 = 1 - ((Ytrue - Ypred) ** 2).sum() / ((Ytrue - mu_train) ** 2).sum()
    T, _ = normalize_rows(Ytrue - mu_train)
    P, pn = normalize_rows(Ypred - mu_train)
    n = len(Ytrue)
    if pn.max() < 1e-9:                      # constant prediction: retrieval is at chance
        return {"r2": float(r2), "cos": 0.0, "top1": 1.0 / n, "top5": min(5, n) / n, "mrr": float("nan")}
    S = P @ T.T                              # S[i, j]: predicted i vs true j
    ranks = (S > np.diag(S)[:, None]).sum(1)  # 0 = the right protein ranked first
    return {"r2": float(r2), "cos": float((P * T).sum(1).mean()),
            "top1": float((ranks == 0).mean()), "top5": float((ranks < 5).mean()),
            "mrr": float((1.0 / (ranks + 1)).mean())}


# ---------- experiment ----------
def run_experiment(H, Y, gc, length, codon, aa, proteins, args):
    n = len(Y)
    groups = family_groups(proteins, args.kmer, args.jaccard)
    sizes = np.bincount(groups)
    info = {"n_genes": n, "n_families": int(len(sizes)), "largest_family": int(sizes.max()),
            "genes_in_multi_gene_families": int(sizes[sizes > 1].sum())}
    print(f"{n} genes in {info['n_families']} families; largest family {info['largest_family']}; "
          f"{info['genes_in_multi_gene_families']} genes belong to families with >1 member")

    order = ["HyenaDNA embedding", "HyenaDNA, GC regressed out", "GC fraction", "Coding length",
             "GC + length", "Codon frequencies", "HyenaDNA + codon frequencies",
             "Amino-acid composition (reference)", "Training mean (no input)",
             "HyenaDNA, shuffled pairs (control)"]
    results = {k: [] for k in order}
    for s in range(args.n_splits):
        rng = np.random.default_rng(args.seed + s)
        tr, te = group_split(groups, args.test_frac, rng)
        Ytr, Yte, mu = Y[tr], Y[te], Y[tr].mean(0)
        gtr = groups[tr]
        # HyenaDNA with GC regressed out (coefficients fit on training genes only)
        D = np.column_stack([np.ones(n), gc])
        coef = np.linalg.lstsq(D[tr], H[tr], rcond=None)[0]
        Hres = H - D @ coef
        sets = {
            "HyenaDNA embedding": H,
            "HyenaDNA, GC regressed out": Hres,
            "GC fraction": gc[:, None],
            "Coding length": length[:, None],
            "GC + length": np.column_stack([gc, length]),
            "Codon frequencies": codon,
            "HyenaDNA + codon frequencies": np.column_stack([H, codon]),
            "Amino-acid composition (reference)": aa,
        }
        for label, X in sets.items():
            pred, alpha = fit_predict(X[tr], Ytr, X[te], gtr, args.folds, rng)
            m = score(Yte, pred, mu)
            m["alpha"] = alpha
            results[label].append(m)
        results["Training mean (no input)"].append(
            score(Yte, np.tile(mu, (len(te), 1)), mu) | {"alpha": float("nan")})
        perm = rng.permutation(len(tr))
        pred, alpha = fit_predict(H[tr], Ytr[perm], H[te], gtr, args.folds, rng)
        m = score(Yte, pred, mu)
        m["alpha"] = alpha
        results["HyenaDNA, shuffled pairs (control)"].append(m)
        print(f"split {s + 1}/{args.n_splits}: {len(tr)} train, {len(te)} test genes")
    return order, results, info


def format_table(order, results, n_test_example):
    def ms(label, key, pct=False):
        v = np.array([r[key] for r in results[label]], dtype=float)
        if np.isnan(v).all():
            return "n/a"
        v = v * (100 if pct else 1)
        return f"{np.nanmean(v):.1f}% +- {np.nanstd(v):.1f}" if pct else f"{np.nanmean(v):+.3f} +- {np.nanstd(v):.3f}"

    chance1 = 100.0 / n_test_example
    lines = [f"{'Input features':<38}{'R2':>16}{'Cosine':>16}{'Top-1 %':>16}{'Top-5 %':>16}",
             "-" * 102]
    for label in order:
        lines.append(f"{label:<38}{ms(label, 'r2'):>16}{ms(label, 'cos'):>16}"
                     f"{ms(label, 'top1', True):>16}{ms(label, 'top5', True):>16}")
    lines.append(f"(retrieval chance is about {chance1:.1f}% for top-1 and {5 * chance1:.1f}% for top-5)")
    return "\n".join(lines)


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
    H = h["mean"].double().numpy()
    Y = e["mean"].double().numpy()
    cds, prot = read_fasta(args.cds), read_fasta(args.protein)
    man = {}
    with open(args.manifest) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            man[f"{r['gene']}|{r['transcript']}|{r['name']}"] = r
    missing = [nm for nm in names if nm not in man or nm not in cds or nm not in prot]
    assert not missing, f"{len(missing)} genes missing from manifest/FASTA files, e.g. {missing[:2]}"
    gc = np.array([float(man[nm]["gc_fraction"]) for nm in names])
    length = np.array([float(man[nm]["cds_bases"]) for nm in names])
    codon = np.stack([codon_freqs(cds[nm]) for nm in names])
    aa = np.stack([aa_composition(prot[nm]) for nm in names])
    proteins = [prot[nm] for nm in names]
    return names, H, Y, gc, length, codon, aa, proteins


def main():
    d = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyena", required=True, help="embeddings.pt from embed_hyena.py (genes)")
    ap.add_argument("--esm", required=True, help="esm2_embeddings.pt from embed_esm2.py")
    ap.add_argument("--manifest", default=os.path.join(d, "data", "genes_manifest.tsv"))
    ap.add_argument("--cds", default=os.path.join(d, "data", "genes_cds.fasta"))
    ap.add_argument("--protein", default=os.path.join(d, "data", "genes_protein.fasta"))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--kmer", type=int, default=3)
    ap.add_argument("--jaccard", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    start = time.time()
    names, H, Y, gc, length, codon, aa, proteins = load_inputs(args)
    print(f"loaded {len(names)} genes: HyenaDNA {H.shape[1]}-d, ESM-2 {Y.shape[1]}-d")
    order, results, info = run_experiment(H, Y, gc, length, codon, aa, proteins, args)
    n_test = int(round(args.test_frac * len(names)))
    table = format_table(order, results, n_test)
    print("\n" + table)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "adapter_results.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT_DIR, "adapter_results.json"), "w") as f:
        json.dump({"settings": vars(args), "data": info, "results": results,
                   "seconds": time.time() - start}, f, indent=2)
    print(f"\nSaved {OUT_DIR}/adapter_results.txt and adapter_results.json")


if __name__ == "__main__":
    main()