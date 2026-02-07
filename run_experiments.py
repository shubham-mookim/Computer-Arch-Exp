#!/usr/bin/env python3
"""
Locality-Aware Compute: Experiment Runner

Run all experiments to explore when moving code to data
beats moving data to code.

Usage:
    python run_experiments.py          # Run all experiments
    python run_experiments.py 1        # Run only experiment 1
    python run_experiments.py 1 3 5    # Run experiments 1, 3, and 5
"""

import sys
import time
import random

# Seed for reproducibility
random.seed(42)


def run_all(experiments=None):
    print("╔" + "═"*78 + "╗")
    print("║" + " LOCALITY-AWARE COMPUTE: Code That Moves to Data".center(78) + "║")
    print("║" + " Experimental Results".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    start = time.time()
    run_set = set(experiments) if experiments else {1, 2, 3, 4, 5}

    if 1 in run_set:
        from experiments.exp1_basic_migration import run as run_exp1
        run_exp1()

    if 2 in run_set:
        from experiments.exp2_log_analysis import run as run_exp2
        run_exp2()

    if 3 in run_set:
        from experiments.exp3_distributed_join import run as run_exp3
        run_exp3()

    if 4 in run_set:
        from experiments.exp4_parallel_wordcount import run as run_exp4
        run_exp4()

    if 5 in run_set:
        from experiments.exp5_decision_engine import run as run_exp5
        run_exp5()

    elapsed = time.time() - start
    print("\n" + "="*80)
    print(f"  All experiments completed in {elapsed:.1f}s")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exps = [int(x) for x in sys.argv[1:]]
        run_all(exps)
    else:
        run_all()
