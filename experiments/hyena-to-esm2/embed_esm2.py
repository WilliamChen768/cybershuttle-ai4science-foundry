"""Extract ESM-2 protein embeddings for every sequence in a protein FASTA.

Saves per-residue hidden states (last layer, after the final LayerNorm), shape (n_residues, d_model),
and a mean-pooled vector per protein. Same output layout as embed_hyena.py.

Usage:
    python embed_esm2.py                              # data/genes_protein.fasta, esm2_t6_8M
    python embed_esm2.py data/genes_protein.fasta --model facebook/esm2_t12_35M_UR50D
    python embed_esm2.py --no-per-residue
"""
import argparse
import json
import os
import sys
import time

import torch
import transformers
from transformers import AutoTokenizer, EsmForMaskedLM

from config import OUT_DIR

DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
# Pinned commits, taken from the "revision_resolved" field of earlier runs.
ESM2_REVISIONS = {
    "facebook/esm2_t6_8M_UR50D": "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
}
DEFAULT_FASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "genes_protein.fasta")
MAX_RESIDUES = 1022      # ESM-2 limit: 1024 tokens including <cls> and <eos>
CHECK_TOL = 1e-3
AMINO = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    recs, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(buf)))
            name, buf = line[1:], []
        elif line:
            buf.append(line)
    if name:
        recs.append((name, "".join(buf)))
    return recs


@torch.no_grad()
def embed_sequence(model, tokenizer, seq, device):
    """Return (per_residue [n, d] float32 on CPU, info dict) for one protein."""
    n = len(seq)
    enc = tokenizer(seq, return_tensors="pt").to(device)
    assert enc["input_ids"].shape[1] == n + 2, \
        f"unexpected tokenization length {enc['input_ids'].shape[1]} for {n} residues"

    h = model.esm(**enc).last_hidden_state                 # (1, n+2, d), after final LayerNorm
    logits = model(**enc).logits
    # Self-check: the LM head applied to this tensor must reproduce the model's logits.
    maxdiff = (model.lm_head(h).float() - logits.float()).abs().max().item()

    per_res = h[0, 1:n + 1].float().cpu()                  # drop <cls> and <eos>
    return per_res, {"lm_head_maxdiff": maxdiff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", nargs="?", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None,
                    help="HF commit SHA to pin (default: the pinned one for known models)")
    ap.add_argument("--no-per-residue", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    print("Model:", args.model)

    revision = args.revision or ESM2_REVISIONS.get(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = EsmForMaskedLM.from_pretrained(args.model, revision=revision).to(device).eval()
    commit = getattr(model.config, "_commit_hash", None)
    print("Model revision (commit SHA):", commit)

    recs = read_fasta(args.fasta)
    if not recs:
        sys.exit(f"no sequences found in {args.fasta}")
    print(f"Read {len(recs)} proteins from {args.fasta}")

    names, means, per_all, lengths, checks = [], [], [], [], []
    start = time.time()
    for name, seq in recs:
        seq = seq.upper()
        if len(seq) > MAX_RESIDUES:
            sys.exit(f"{name}: length {len(seq)} exceeds the {MAX_RESIDUES}-residue limit")
        if set(seq) - AMINO:
            sys.exit(f"{name}: contains characters outside the 20 standard amino acids")
        per_res, info = embed_sequence(model, tokenizer, seq, device)
        assert torch.isfinite(per_res).all(), f"{name}: NaN or Inf in embeddings"
        names.append(name)
        lengths.append(len(seq))
        means.append(per_res.mean(0))
        per_all.append(None if args.no_per_residue else per_res.half())
        checks.append(info)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"embedded {len(names)} proteins in {elapsed:.1f}s")

    means = torch.stack(means)
    d_model = means.shape[1]

    diffs = [c["lm_head_maxdiff"] for c in checks]
    head_ok = max(diffs) < CHECK_TOL
    print(f"lm_head check: max diff over all proteins = {max(diffs)}")

    again, _ = embed_sequence(model, tokenizer, recs[0][1].upper(), device)
    repeat_diff = (again.mean(0) - means[0]).abs().max().item()
    repeat_ok = repeat_diff < 1e-4

    # Check 3: different proteins give different embeddings (no identical vectors).
    # Related proteins (e.g. gene-family members) can have cosine ~1 legitimately, so test for
    # exact duplicates by distance instead of a cosine threshold near 1.
    normed = torch.nn.functional.normalize(means, dim=1)
    cos = normed @ normed.T
    off_diag = cos[~torch.eye(len(names), dtype=torch.bool)] if len(names) > 1 else torch.tensor([0.0])
    dists = torch.cdist(means, means)
    dists.fill_diagonal_(float("inf"))
    min_dist = dists.min().item()
    distinct_ok = min_dist > 1e-6
    print(f"d_model = {d_model}")
    print(f"Cosine similarity between mean embeddings: min {off_diag.min().item():.4f}, "
          f"mean {off_diag.mean().item():.4f}, max {off_diag.max().item():.6f}")
    print(f"Smallest distance between two different proteins: {min_dist:.6f}")

    results = {
        "lm_head reproduces logits from last_hidden_state": head_ok,
        "repeat run gives the same embedding": repeat_ok,
        "different proteins give different embeddings": distinct_ok,
    }
    for k, v in results.items():
        print(("PASS  " if v else "FAIL  ") + k)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"names": names, "lengths": lengths, "mean": means, "per_residue": per_all},
               os.path.join(OUT_DIR, "esm2_embeddings.pt"))
    with open(os.path.join(OUT_DIR, "esm2_embeddings_meta.json"), "w") as f:
        json.dump({
            "model": args.model, "revision_requested": revision, "revision_resolved": commit,
            "fasta": os.path.abspath(args.fasta), "n_sequences": len(names), "d_model": d_model,
            "lm_head_maxdiff_max": max(diffs), "repeat_maxdiff": repeat_diff,
            "per_residue_saved": not args.no_per_residue, "per_residue_dtype": "float16",
            "mean_offdiag_cosine": off_diag.mean().item(),
            "max_offdiag_cosine": off_diag.max().item(), "min_pairwise_distance": min_dist,
            "checks": results, "seconds": elapsed,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        }, f, indent=2)
    print(f"Saved {OUT_DIR}/esm2_embeddings.pt and {OUT_DIR}/esm2_embeddings_meta.json")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
