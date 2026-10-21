# ADComp

**ADComp (Acoustic Dispersion Compensation)** provides a reproducible workflow for computational acoustics experiments and analysis.

## Models

Model files and related resources are available on [Hugging Face](https://huggingface.co/soundai2016/ADComp).

Locally generated models are stored under `models/<run_id>`. Refer to the online repository for published resources.

## Installation

Requires Linux and Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduction

Run the complete workflow:

```bash
python reproduce.py --mode full
```

For an unattended run:

```bash
mkdir -p results
nohup python reproduce.py --mode full > results/console.log 2>&1 &
```

Reproduce analyses using the supplied inputs without rerunning simulations:

```bash
python reproduce.py --mode data
```

## Checks and outputs

```bash
python reproduce.py --mode smoke
python reproduce.py --mode full --plan
```

Use `--output-dir` to choose a different output location. Generated analyses, figures, and tables are saved under `results/`. Run logs and environment records are stored in `results/logs/<run_id>`.
