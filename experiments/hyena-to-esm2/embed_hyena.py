"""Extract HyenaDNA embeddings for every sequence in a FASTA file.

For each sequence this saves
  - per-base hidden states (last layer, after the final LayerNorm), shape (n_bases, d_model)
  - a mean-pooled vector, shape (d_model,)

HyenaDNA is causal: the state at base i has only seen bases 0..i.

Usage:
    python embed_hyena.py                          # the 8 hg38 regions in data/hg38_regions.fasta
    python embed_hyena.py path/to/genes.fasta      # any FASTA of A/C/G/T sequences
    python embed_hyena.py genes.fasta --no-per-base   # mean vectors only (small output)
"""
import argparse
import json
import os
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL, OUT_DIR, REVISION

MAX_LEN = 32768          # context length of small-32k
CHECK_TOL = 1e-3         # tolerance for the lm_head consistency check
DEFAULT_FASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hg38_regions.fasta")


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
    """Return (per_base [n, d] float32 on CPU, info dict) for one DNA sequence."""
    n = len(seq)
    ids = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
    # Same assumption as validate.py: n bases plus one trailing special token.
    assert ids.shape[1] == n + 1, f"unexpected tokenization length {ids.shape[1]} for {n} bases"

    out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    assert hs is not None, "model returned no hidden_states"
    last = hs[-1]                                  # (1, n+1, d_model)

    # Self-check: the LM head applied to this tensor must reproduce the logits.
    # If it does not, hidden_states[-1] is not the final (post-LayerNorm) representation.
    maxdiff = None
    if hasattr(model, "lm_head"):
        recon = model.lm_head(last).float()
        maxdiff = (recon - out.logits.float()).abs().max().item()

    per_base = last[0, :n].float().cpu()           # drop the trailing special token
    info = {"n_hidden_states": len(hs), "lm_head_maxdiff": maxdiff}
    return per_base, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", nargs="?", default=DEFAULT_FASTA)
    ap.add_argument("--no-per-base", action="store_true", help="save only mean-pooled vectors")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    print("Model:", MODEL)
    print("Model revision (commit SHA):", REVISION)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True
    ).to(device).eval()

    recs = read_fasta(args.fasta)
    if not recs:
        sys.exit(f"no sequences found in {args.fasta}")
    print(f"Read {len(recs)} sequences from {args.fasta}")

    names, means, per_base_all, lengths, checks = [], [], [], [], []
    start = time.time()
    for name, seq in recs:
        seq = seq.upper()
        if len(seq) > MAX_LEN:
            sys.exit(f"{name}: length {len(seq)} exceeds the {MAX_LEN}-base context")
        if set(seq) - set("ACGT"):
            sys.exit(f"{name}: contains non-ACGT characters")
        per_base, info = embed_sequence(model, tokenizer, seq, device)
        assert torch.isfinite(per_base).all(), f"{name}: NaN or Inf in embeddings"
        names.append(name)
        lengths.append(len(seq))
        means.append(per_base.mean(0))
        per_base_all.append(None if args.no_per_base else per_base.half())
        checks.append(info)
        print(f"{name}: {len(seq)} bases -> per-base {tuple(per_base.shape)}, "
              f"lm_head check maxdiff={info['lm_head_maxdiff']}")
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    means = torch.stack(means)                     # (N, d_model)
    d_model = means.shape[1]

    # Check 1: the captured tensor really is the one the LM head reads.
    diffs = [c["lm_head_maxdiff"] for c in checks if c["lm_head_maxdiff"] is not None]
    head_ok = bool(diffs) and max(diffs) < CHECK_TOL

    # Check 2: repeating the first sequence gives the same embedding.
    again, _ = embed_sequence(model, tokenizer, recs[0][1].upper(), device)
    repeat_diff = (again.mean(0) - means[0]).abs().max().item()
    repeat_ok = repeat_diff < 1e-4

    # Check 3: different sequences give different embeddings (no identical vectors).
    # Mean-pooled vectors all point in nearly the same direction (cosine ~0.97-0.999), so a
    # cosine threshold near 1 is too strict; test for duplicates by distance instead.
    normed = torch.nn.functional.normalize(means, dim=1)
    cos = normed @ normed.T
    off_diag = cos[~torch.eye(len(names), dtype=torch.bool)] if len(names) > 1 else torch.tensor([0.0])
    dists = torch.cdist(means, means)
    dists.fill_diagonal_(float("inf"))
    min_dist = dists.min().item()
    distinct_ok = min_dist > 1e-6

    print(f"\nd_model = {d_model}, hidden_states returned = {checks[0]['n_hidden_states']}")
    print(f"Cosine similarity between mean embeddings: min {off_diag.min().item():.4f}, "
          f"mean {off_diag.mean().item():.4f}, max {off_diag.max().item():.6f}")
    print(f"Smallest distance between two different sequences: {min_dist:.6f}")
    if len(names) <= 12:
        print(torch.round(cos * 1000) / 1000)

    results = {
        "lm_head reproduces logits from hidden_states[-1]": head_ok,
        "repeat run gives the same embedding": repeat_ok,
        "different sequences give different embeddings": distinct_ok,
    }
    for k, v in results.items():
        print(("PASS  " if v else "FAIL  ") + k)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(
        {"names": names, "lengths": lengths, "mean": means, "per_base": per_base_all},
        os.path.join(OUT_DIR, "embeddings.pt"),
    )
    with open(os.path.join(OUT_DIR, "embeddings_meta.json"), "w") as f:
        json.dump(
            {
                "model": MODEL, "revision": REVISION, "fasta": os.path.abspath(args.fasta),
                "n_sequences": len(names), "d_model": d_model, "lengths": lengths,
                "n_hidden_states": checks[0]["n_hidden_states"],
                "lm_head_maxdiff_max": max(diffs) if diffs else None,
                "repeat_maxdiff": repeat_diff,
                "per_base_saved": not args.no_per_base, "per_base_dtype": "float16",
                "mean_offdiag_cosine": off_diag.mean().item(),
                "max_offdiag_cosine": off_diag.max().item(),
                "min_pairwise_distance": min_dist,
                "checks": results, "seconds": elapsed,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            },
            f, indent=2,
        )
    print(f"Saved {OUT_DIR}/embeddings.pt and {OUT_DIR}/embeddings_meta.json")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()