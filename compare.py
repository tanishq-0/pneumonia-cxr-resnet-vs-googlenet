"""Algorithm 7 — Aggregate results.csv into report-ready tables."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, RESULTS_CSV


def build_report_tables(csv_path: Path = RESULTS_CSV) -> None:
    if not csv_path.exists():
        print(f"No results CSV at {csv_path}; nothing to compare.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("results.csv is empty.")
        return

    # ---------- Master test comparison (4 final models) ----------
    eval_rows = df[df["notes"].astype(str).str.startswith("clin_t=", na=False)].copy()
    if not eval_rows.empty:
        master = (
            eval_rows[
                [
                    "arch", "aug",
                    "test_sensitivity", "test_specificity",
                    "test_auroc", "test_mcc", "test_f1",
                ]
            ]
            .sort_values("test_auroc", ascending=False)
            .reset_index(drop=True)
        )
        out = OUTPUT_DIR / "table_master_comparison.csv"
        master.to_csv(out, index=False)
        print("\n=== Master comparison (test set, sorted by AUROC) ===")
        print(master.to_string(index=False))
        print(f"saved → {out}")

    # ---------- HP sweep effects (Algorithm 3 rows) ----------
    hp_rows = df[df["notes"] == "hp_search"].copy()
    if not hp_rows.empty:
        for axis in ["lr", "momentum", "batch_size"]:
            agg = hp_rows.groupby(["arch", axis], as_index=False)["val_auroc"].mean()
            out = OUTPUT_DIR / f"table_hp_effect_{axis}.csv"
            agg.to_csv(out, index=False)
            print(f"\n=== Effect of {axis} on val AUROC (mean across other axes) ===")
            print(agg.to_string(index=False))
            print(f"saved → {out}")

    # ---------- Resource comparison (Stage-2 rows) ----------
    s2 = df[df["notes"] == "stage2"].copy()
    if not s2.empty:
        res = (
            s2[["arch", "aug", "model_params_mb", "train_time_s", "peak_gpu_mb"]]
            .sort_values(["arch", "aug"])
            .reset_index(drop=True)
        )
        out = OUTPUT_DIR / "table_resources.csv"
        res.to_csv(out, index=False)
        print("\n=== Resource comparison (Stage-2 training) ===")
        print(res.to_string(index=False))
        print(f"saved → {out}")

    # ---------- Best model ----------
    if not eval_rows.empty:
        best = eval_rows.sort_values("test_auroc", ascending=False).iloc[0]
        print(
            f"\n>>> Best model: arch={best['arch']} aug={int(best['aug'])} "
            f"AUROC={best['test_auroc']:.4f} MCC={best['test_mcc']:.4f}"
        )


if __name__ == "__main__":
    build_report_tables()
