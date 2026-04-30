"""Pneumonia CXR classification — full pipeline orchestrator.

Pipeline (matches code-plan.md):
  Algorithm 0 — seed everything.
  Algorithm 1 — collect + patient-level split (once, shared by all variants).
  Algorithm 3 — HP search per architecture (resnet18 no-aug, googlenet no-aug).
                Best HPs for each arch are reused for that arch's augmented variant.
  Algorithm 4 — Stage-1 head training, for all 4 (arch, aug) variants.
  Algorithm 5 — Stage-2 full fine-tune,  for all 4 (arch, aug) variants.
  Algorithm 6 — Test-set evaluation + GradCAM++, for all 4 variants.
  Algorithm 7 — Aggregate report tables.

CLI:
  python main.py                 # full pipeline (~hours)
  python main.py --skip-hp       # use BASELINE_HP without sweeping
  python main.py --eval-only     # skip training; just evaluate existing checkpoints
  python main.py --report-only   # only rebuild tables from results.csv
  python main.py --archs resnet18  --augs 0 1   # restrict experiment matrix
"""
from __future__ import annotations
import argparse
import sys
from typing import Dict

from config import ARCHS, AUGS, BASELINE_HP, set_seed
from data import get_data_splits
from hp_search import run_hp_search
from train_pipeline import stage1_head_training, stage2_full_finetune
from evaluate_test import evaluate_variant
from compare import build_report_tables


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--skip-hp", action="store_true",
                   help="Skip HP search; use BASELINE_HP for all archs.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; evaluate existing Stage-2 checkpoints.")
    p.add_argument("--report-only", action="store_true",
                   help="Skip everything; just rebuild report tables.")
    p.add_argument("--archs", nargs="+", default=ARCHS,
                   help=f"Subset of {ARCHS}.")
    p.add_argument("--augs", nargs="+", type=int, default=[int(a) for a in AUGS],
                   help="Subset of 0/1.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed()

    if args.report_only:
        build_report_tables()
        return 0

    print("=" * 70)
    print("Algorithm 1 — collecting records and splitting at patient level")
    print("=" * 70)
    train_df, val_df, test_df = get_data_splits()

    archs = list(args.archs)
    augs = [bool(a) for a in args.augs]

    # ---------- Algorithm 3 — HP search per architecture (no-aug) ----------
    best_hp_per_arch: Dict[str, Dict] = {}
    if args.skip_hp or args.eval_only:
        for arch in archs:
            best_hp_per_arch[arch] = dict(BASELINE_HP)
            print(f"[HP] using baseline for {arch}: {BASELINE_HP}")
    else:
        print("\n" + "=" * 70)
        print("Algorithm 3 — coordinate HP search (per architecture, no-aug)")
        print("=" * 70)
        for arch in archs:
            best_hp_per_arch[arch] = run_hp_search(
                train_df, val_df, test_df, arch=arch, aug=False,
            )
            print(f"[HP] best for {arch}: {best_hp_per_arch[arch]}")

    # ---------- Algorithms 4 & 5 — Stage-1 + Stage-2 for all variants ----------
    if not args.eval_only:
        print("\n" + "=" * 70)
        print("Algorithms 4 + 5 — Stage-1 head training + Stage-2 fine-tuning")
        print("=" * 70)
        for arch in archs:
            for aug in augs:
                hp = best_hp_per_arch[arch]
                stage1_head_training(arch, aug, hp, train_df, val_df, test_df)
                stage2_full_finetune(arch, aug, hp, train_df, val_df, test_df)

    # ---------- Algorithm 6 — Test-set evaluation ----------
    print("\n" + "=" * 70)
    print("Algorithm 6 — Test-set evaluation + GradCAM++")
    print("=" * 70)
    for arch in archs:
        for aug in augs:
            hp = best_hp_per_arch[arch]
            evaluate_variant(arch, aug, hp, train_df, val_df, test_df)

    # ---------- Algorithm 7 — Aggregate ----------
    print("\n" + "=" * 70)
    print("Algorithm 7 — Final comparison tables")
    print("=" * 70)
    build_report_tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())
