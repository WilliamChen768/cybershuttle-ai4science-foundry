"""Build a paired gene set: real human coding DNA + the protein it encodes (GENCODE).

Downloads two GENCODE files (or reads local copies):
  gencode.v<R>.pc_transcripts.fa.gz   protein-coding transcript sequences, header has CDS:start-end
  gencode.v<R>.pc_translations.fa.gz  the protein for each transcript

For each transcript we cut out the coding sequence (CDS), translate it ourselves with the
standard genetic code, and keep it only if it reproduces GENCODE's protein exactly. That
translation match rate is the correctness check for this step.

Outputs (in --outdir, default data/):
  genes_cds.fasta       coding DNA, start codon included, stop codon removed
                        (so DNA length = 3 x protein length: one codon per amino acid)
  genes_protein.fasta   matching protein sequences (same headers)
  genes_manifest.tsv    gene, transcript, lengths, GC fraction, train/test split
  fetch_cds_report.json counts at every filter step

Only the Python standard library is needed.

Usage:
    python fetch_cds.py                       # download release 46, sample 500 genes
    python fetch_cds.py --release 47 --max-genes 300
    python fetch_cds.py --transcripts T.fa.gz --translations P.fa.gz   # local files
"""
import argparse
import gzip
import json
import os
import random
import re
import sys
import urllib.request

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{r}/{f}"

# Standard genetic code, codons ordered TCAG x TCAG x TCAG.
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_B = "TCAG"
CODON = {a + b + c: _AA[16 * i + 4 * j + k]
         for i, a in enumerate(_B) for j, b in enumerate(_B) for k, c in enumerate(_B)}
STOPS = {"TAA", "TAG", "TGA"}


def translate(dna):
    return "".join(CODON[dna[i:i + 3]] for i in range(0, len(dna) - len(dna) % 3, 3))


def read_fasta_gz(path):
    opener = gzip.open if path.endswith(".gz") else open
    name, buf = None, []
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:], []
            elif line:
                buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"using cached {dest}")
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest + ".part")
    os.replace(dest + ".part", dest)
    return dest


def gc_fraction(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=int, default=46, help="GENCODE human release number")
    ap.add_argument("--transcripts", help="local pc_transcripts .fa(.gz) instead of downloading")
    ap.add_argument("--translations", help="local pc_translations .fa(.gz) instead of downloading")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--max-genes", type=int, default=500)
    ap.add_argument("--min-cds", type=int, default=150, help="minimum coding length in bases")
    ap.add_argument("--max-cds", type=int, default=3000, help="maximum coding length in bases")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = os.path.join(args.outdir, "gencode")
    t_path = args.transcripts or download(
        BASE_URL.format(r=args.release, f=f"gencode.v{args.release}.pc_transcripts.fa.gz"),
        os.path.join(cache, f"gencode.v{args.release}.pc_transcripts.fa.gz"))
    p_path = args.translations or download(
        BASE_URL.format(r=args.release, f=f"gencode.v{args.release}.pc_translations.fa.gz"),
        os.path.join(cache, f"gencode.v{args.release}.pc_translations.fa.gz"))

    # Proteins by transcript ID (the ENST... ID appears in each translation header).
    proteins, shown = {}, 0
    for header, seq in read_fasta_gz(p_path):
        m = re.search(r"ENST\d+\.\d+", header)
        if not m:
            continue
        proteins[m.group(0)] = seq.rstrip("*")
        if shown < 2:
            print("translations header example:", header)
            shown += 1
    print(f"read {len(proteins)} proteins")

    counts = {k: 0 for k in [
        "transcripts", "no_cds_field", "no_protein", "cds_not_multiple_of_3", "no_start_codon",
        "no_stop_codon", "non_acgt", "internal_stop", "translation_mismatch",
        "translation_match", "outside_length_range"]}
    by_gene, shown = {}, 0
    for header, seq in read_fasta_gz(t_path):
        counts["transcripts"] += 1
        if shown < 2:
            print("transcripts header example:", header)
            shown += 1
        tm = re.search(r"ENST\d+\.\d+", header)
        gm = re.search(r"ENSG\d+\.\d+", header)
        cm = re.search(r"CDS:(\d+)-(\d+)", header)
        if not (tm and gm and cm):
            counts["no_cds_field"] += 1
            continue
        tid, gid = tm.group(0), gm.group(0)
        if tid not in proteins:
            counts["no_protein"] += 1
            continue
        fields = header.split("|")
        gene_name = fields[5] if len(fields) > 5 else gid
        cds = seq[int(cm.group(1)) - 1:int(cm.group(2))].upper()      # 1-based, inclusive
        if len(cds) % 3:
            counts["cds_not_multiple_of_3"] += 1
            continue
        if not cds.startswith("ATG"):
            counts["no_start_codon"] += 1
            continue
        if cds[-3:] not in STOPS:
            counts["no_stop_codon"] += 1
            continue
        coding = cds[:-3]                                              # drop the stop codon
        if set(coding) - set("ACGT"):
            counts["non_acgt"] += 1
            continue
        prot = translate(coding)
        if "*" in prot:
            counts["internal_stop"] += 1
            continue
        if prot != proteins[tid]:
            counts["translation_mismatch"] += 1
            continue
        counts["translation_match"] += 1
        if not (args.min_cds <= len(coding) <= args.max_cds):
            counts["outside_length_range"] += 1
            continue
        # one transcript per gene: keep the longest passing one
        best = by_gene.get(gid)
        if best is None or len(coding) > len(best["dna"]):
            by_gene[gid] = {"gene": gid, "transcript": tid, "name": gene_name,
                            "dna": coding, "protein": prot}

    candidates = sorted(by_gene)                                       # deterministic order
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    chosen = candidates[:args.max_genes]
    n_test = round(len(chosen) * args.test_frac)
    split = {g: ("test" if i < n_test else "train") for i, g in enumerate(chosen)}

    if not chosen:
        sys.exit("no genes passed the filters; check the header examples printed above")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "genes_cds.fasta"), "w") as fd, \
         open(os.path.join(args.outdir, "genes_protein.fasta"), "w") as fp, \
         open(os.path.join(args.outdir, "genes_manifest.tsv"), "w") as fm:
        fm.write("gene\ttranscript\tname\tcds_bases\tprotein_aa\tgc_fraction\tsplit\n")
        for g in chosen:
            r = by_gene[g]
            h = f"{r['gene']}|{r['transcript']}|{r['name']}"
            fd.write(f">{h}\n{r['dna']}\n")
            fp.write(f">{h}\n{r['protein']}\n")
            fm.write(f"{r['gene']}\t{r['transcript']}\t{r['name']}\t{len(r['dna'])}\t"
                     f"{len(r['protein'])}\t{gc_fraction(r['dna']):.4f}\t{split[g]}\n")

    checked = counts["translation_match"] + counts["translation_mismatch"]
    report = {
        "release": args.release, "seed": args.seed, "counts": counts,
        "translation_match_rate_among_checked": (counts["translation_match"] / checked) if checked else None,
        "genes_available_after_filters": len(candidates), "genes_written": len(chosen),
        "n_train": len(chosen) - n_test, "n_test": n_test,
        "cds_length_range": [args.min_cds, args.max_cds],
    }
    with open(os.path.join(args.outdir, "fetch_cds_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {len(chosen)} genes to {args.outdir}/")


if __name__ == "__main__":
    main()