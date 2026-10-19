# HyenaDNA small-32k

I ran the pretrained HyenaDNA small-32k model on human DNA and checked whether its next-base guesses were right, compared with shuffled copies of the same DNA and plain random guessing.

## Setup

On a login node:

    git clone https://github.com/WilliamChen768/cybershuttle-ai4science-foundry
    cd cybershuttle-ai4science-foundry
    git checkout hyenadna-small-32k
    cd registry/models/biology/hyenadna-small-32k
    bash setup_env.sh

`setup_env.sh` loads the `miniforge/24.3.0-0` module and builds a Python 3.10.21 env with torch 2.5.1+cu121, transformers 5.17.0, numpy 2.2.6, tokenizers 0.23.2, safetensors 0.8.0, and huggingface_hub 1.33.0. Everything needed is pinned in `requirements.txt`. The env lives in `$SCRATCH_DIR/.conda/envs/hyenadna_env` and `$SCRATCH_DIR` defaults to `$HOME/scratch`. The conda and pip caches go to scratch too, since the home quota is only 30 GB.

If you want the env somewhere else, set `SCRATCH_DIR` and `ENV_PREFIX` when you run `setup_env.sh`, then `export ENV_PREFIX=<path>` before `sbatch`.

## Run

    mkdir -p logs
    sbatch run_inference.sbatch
    squeue -u $USER

The job asks for account `coc`, partition `coc-gpu`, one GPU, 4 CPUs, 32 GB, and 15 minutes. It prints the node and `nvidia-smi`, runs `infer.py` (a quick test on `ACTGACTGACTGACTG`), then runs `validate.py`. If your account or partition is different, edit the `#SBATCH -A` and `-p` lines or run `sbatch -A <account> -p <partition> run_inference.sbatch`. The model downloads to `$HF_HOME` the first time (default `$SCRATCH_DIR/.hf_cache`).

## Results

The `.out` file in `logs/` should end with three `PASS` lines and `outputs/validation_results.json` has the numbers for each region.

* quick test: logits come out as `(1, 17, 16)`, and A, C, G, T are token IDs 7, 8, 9, 10
* real DNA: 1.0944 nats per base on average, better than the shuffled copy in all 8 regions
* shuffled copies: 1.3908
* random guessing, ln(4): 1.3863

The loss is how wrong the model's guess for the next base is, only counting A, C, G, and T. Each region also gets a shuffled copy with the same letters in a random order (`random.Random(0)`). Real DNA has to beat random guessing (ln(4)), beat shuffled on average, and beat shuffled in most regions. The shuffled copies score about the same as random guessing, so the model is using the order of the bases, not just how many of each letter there are. The check exists because a broken model load can still give the right output shape, and the quick test wouldn't catch that.

V100 and L40S gave the same token IDs and logits within 1.05e-5 (see `logs/v100_vs_l40s_comparison.txt`).

## Problems

If `sbatch` doesn't make a log, `logs/` didn't exist yet. Slurm opens the output files before the script starts, so run `mkdir -p logs` first.

If you hit `Disk quota exceeded`, it's probably `~/.conda/pkgs` eating your 30 GB home. `conda clean --all` barely helps because the files are hardlinked into environments. Delete `~/.conda/pkgs`, keep the caches on scratch like `setup_env.sh` does, and check with `quota -s`.

If the job ran in the wrong environment, you probably built it at a custom `ENV_PREFIX` and didn't export it before `sbatch`. The `Env:` line at the top of the `.out` file shows which one it used.

## Cleanup

    scancel JOBID
    rm -rf "$HOME/scratch/.conda/envs/hyenadna_env" "$HOME/scratch/.hf_cache" \
           "$HOME/scratch/.conda_pkgs" "$HOME/scratch/.pip_cache"

Change the paths if you set `SCRATCH_DIR`, `ENV_PREFIX`, or `HF_HOME`.

## Sources

* Model: `LongSafari/hyenadna-small-32k-seqlen-hf` at commit `8fe770c78eb13fe33bf81501612faeddf4d6f331`. The weights are BSD-3-Clause, and loading needs `trust_remote_code=True`: https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen-hf
* Paper: https://arxiv.org/abs/2306.15794
* Code (Apache-2.0, not used directly): https://github.com/HazyResearch/hyena-dna
* Data: 8 regions of 1,000 bp from hg38, pulled on 2026-10-08 from https://api.genome.ucsc.edu/getData/sequence with `fetch_hg38_regions.py`

The 8 regions were picked arbitrarily and none were dropped. The coordinates are in `fetch_hg38_regions.py` and the FASTA is committed in `data/`, so you don't need to rerun the fetch script.