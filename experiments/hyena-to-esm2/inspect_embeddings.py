"""Look inside a saved HyenaDNA embeddings.pt: how similar are the mean embeddings, and why?

Prints raw and centered cosine statistics, the closest pair, and how much of the variation
is explained by the top components. If a manifest is found, it also checks whether the main
component tracks GC content or coding length (simple confounders).

Usage: python inspect_embeddings.py outputs/run_<jobid>/embeddings.pt [data/genes_manifest.tsv]
"""
import csv
import sys

import torch


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    emb = torch.load(sys.argv[1], map_location="cpu")
    names, means, lengths = emb["names"], emb["mean"].float(), emb["lengths"]
    n, d = means.shape
    print(f"{n} sequences, d_model = {d}")

    def offdiag_stats(x, label):
        c = torch.nn.functional.normalize(x, dim=1)
        c = c @ c.T
        mask = ~torch.eye(n, dtype=torch.bool)
        v = c[mask]
        print(f"{label}: cosine min {v.min():.4f}  mean {v.mean():.4f}  max {v.max():.6f}")
        return c

    c_raw = offdiag_stats(means, "raw mean embeddings     ")
    centered = means - means.mean(0, keepdim=True)
    offdiag_stats(centered, "after subtracting the mean")

    # closest pair in raw space
    c2 = c_raw.clone()
    c2.fill_diagonal_(-1)
    idx = int(torch.argmax(c2))
    i, j = divmod(idx, n)
    print(f"closest pair: {names[i].split('|')[-1]} ({lengths[i]} bases) and "
          f"{names[j].split('|')[-1]} ({lengths[j]} bases), cosine {c2[i, j]:.6f}, "
          f"distance {(means[i] - means[j]).norm():.4f}")
    print(f"typical vector length {means.norm(dim=1).mean():.3f}; "
          f"typical spread around the average {centered.norm(dim=1).mean():.3f}")

    u, s, vt = torch.linalg.svd(centered, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    print("variance explained by components 1-5:", [round(float(x), 3) for x in var[:5]])
    pc1 = centered @ vt[0]

    if len(sys.argv) > 2:
        rows = {}
        with open(sys.argv[2]) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                rows[f"{r['gene']}|{r['transcript']}|{r['name']}"] = r
        if all(nm in rows for nm in names):
            gc = torch.tensor([float(rows[nm]["gc_fraction"]) for nm in names])
            ln = torch.tensor([float(rows[nm]["cds_bases"]) for nm in names])
            print(f"correlation of component 1 with GC fraction:   {pearson(pc1, gc):+.3f}")
            print(f"correlation of component 1 with coding length: {pearson(pc1, ln):+.3f}")
        else:
            print("manifest names do not match embedding names; skipped the GC/length check")


if __name__ == "__main__":
    main()
