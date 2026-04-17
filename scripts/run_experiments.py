"""
Run All Experiments
===================
Single entrypoint to run exp14 (holdout validation), exp15 (thermal stress),
and exp16 (multi-seed + FD baseline).

Usage:
    python scripts/run_experiments.py              # run all
    python scripts/run_experiments.py --exp 14     # holdout validation only
    python scripts/run_experiments.py --exp 16     # multi-seed only
    python scripts/run_experiments.py --exp 14 15  # specific experiments

Requirements:
    - GPU with >= 8 GB VRAM (RTX 3070+ or equivalent)
    - AIMS logger data in data/ALZ/ (see download_data.py)
    - CRW SST data (auto-downloaded by experiments if needed)

Timing (RTX 4090):
    - Exp 14: ~2 hours (30 holdout configs x 15k epochs)
    - Exp 15: ~1 hour (4 reefs, full thermal fields)
    - Exp 16: ~10 hours (150 PINN runs: 30 configs x 5 seeds)
"""

import os
import sys
import argparse
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.dirname(__file__)

EXPERIMENTS = {
    "14": {
        "name": "Exp 14: Holdout Validation",
        "script": "exp14_holdout_validation.py",
        "description": "4 reefs, leave-one-depth-out, multiple sparsity levels",
    },
    "15": {
        "name": "Exp 15: Thermal Stress",
        "script": "exp15_thermal_stress.py",
        "description": "Depth-resolved DHD bleaching stress correction",
    },
    "16": {
        "name": "Exp 16: Multi-Seed + FD Baseline",
        "script": "exp16_multiseed.py",
        "description": "5 seeds per config for uncertainty + finite-difference baseline",
    },
}


def run_experiment(exp_id, config):
    """Run a single experiment script as a subprocess."""
    script_path = os.path.join(SCRIPTS_DIR, config["script"])
    print(f"\n{'='*60}")
    print(f"  {config['name']}")
    print(f"  {config['description']}")
    print(f"  Script: {config['script']}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  [OK] {config['name']} completed in {elapsed/60:.1f} min")
    else:
        print(f"\n  [FAIL] {config['name']} failed (exit code {result.returncode})")

    return result.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run PINN reef thermal experiments"
    )
    parser.add_argument("--exp", nargs="+", default=["14", "15", "16"],
                        choices=["14", "15", "16"],
                        help="Which experiments to run (default: all)")
    args = parser.parse_args()

    print("PINN Reef Thermal — Experiment Runner")
    print("=" * 60)
    print(f"Experiments: {', '.join(args.exp)}")
    print(f"Results dir: {os.path.join(BASE_DIR, 'results')}")
    print()

    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

    total_start = time.time()
    results = {}

    for exp_id in sorted(args.exp):
        if exp_id not in EXPERIMENTS:
            print(f"  Unknown experiment: {exp_id}, skipping")
            continue
        returncode, elapsed = run_experiment(exp_id, EXPERIMENTS[exp_id])
        results[exp_id] = {"returncode": returncode, "elapsed": elapsed}

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for exp_id in sorted(results):
        r = results[exp_id]
        status = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"  Exp {exp_id}: [{status}] {r['elapsed']/60:.1f} min")
    print(f"\n  Total time: {total_elapsed/60:.1f} min")

    # Check for output files
    results_dir = os.path.join(BASE_DIR, "results")
    for exp_id in sorted(args.exp):
        json_file = os.path.join(results_dir, f"exp{exp_id}_results.json")
        if os.path.exists(json_file):
            size = os.path.getsize(json_file) / 1024
            print(f"  results/exp{exp_id}_results.json ({size:.0f} KB)")
        else:
            print(f"  results/exp{exp_id}_results.json — NOT FOUND")

    print(f"\nTo generate figures: python scripts/make_figures.py")


if __name__ == "__main__":
    main()
