from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "batteryinverse"
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
TARGET = "final_capacity_retention"

OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 100

PRIOR_GROUPS = {
    "global_train_median": [],
    "source_family_median": ["source_family"],
    "dataset_id_median": ["dataset_id"],
    "chemistry_median": ["chemistry"],
    "form_factor_median": ["form_factor"],
    "chemistry_form_factor_median": ["chemistry", "form_factor"],
    "source_chemistry_median": ["source_family", "chemistry"],
}

def predict_group_median(train, test, group_cols, target):
    global_med = float(train[target].median())

    if not group_cols:
        return np.repeat(global_med, len(test))

    train_key = train[group_cols].fillna("missing").astype(str).agg("||".join, axis=1)
    test_key = test[group_cols].fillna("missing").astype(str).agg("||".join, axis=1)

    med_map = train.assign(__key=train_key).groupby("__key")[target].median().to_dict()

    pred = test_key.map(med_map).fillna(global_med).astype(float).to_numpy()
    return pred

def evaluate(train, test, mode, split_id, heldout_dataset, target_variant, baseline_name, group_cols):
    y_test = test[target_variant].astype(float).to_numpy()
    pred = predict_group_median(train, test, group_cols, target_variant)

    global_med_pred = np.repeat(float(train[target_variant].median()), len(test))

    mae = mean_absolute_error(y_test, pred)
    global_mae = mean_absolute_error(y_test, global_med_pred)

    return {
        "validation_mode": mode,
        "split_id": split_id,
        "heldout_dataset": heldout_dataset,
        "target_variant": target_variant,
        "baseline_name": baseline_name,
        "group_cols": ",".join(group_cols) if group_cols else "none",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae": float(mae),
        "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan,
        "global_train_median_mae": float(global_mae),
        "improvement_vs_global_train_median": float(global_mae - mae),
    }

def add_targets(df):
    out = df.copy()
    out["target_original"] = out[TARGET].astype(float)

    out["target_dataset_mean_centered"] = (
        out["target_original"] - out.groupby("dataset_id")["target_original"].transform("mean")
    )
    out["target_dataset_median_centered"] = (
        out["target_original"] - out.groupby("dataset_id")["target_original"].transform("median")
    )
    return out

def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET, "dataset_id"]).copy()
    df = add_targets(df)

    target_variants = [
        "target_original",
        "target_dataset_mean_centered",
        "target_dataset_median_centered",
    ]

    rows = []

    # Random split priors
    for split_id in range(1, N_SPLITS + 1):
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.30,
            random_state=2000 + split_id,
            stratify=df["dataset_id"].astype(str),
        )
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        for target in target_variants:
            for name, group_cols in PRIOR_GROUPS.items():
                rows.append(evaluate(
                    train, test,
                    mode="random_stratified_split",
                    split_id=split_id,
                    heldout_dataset="",
                    target_variant=target,
                    baseline_name=name,
                    group_cols=group_cols,
                ))

    # LODO priors
    for heldout in sorted(df["dataset_id"].dropna().unique()):
        train = df[df["dataset_id"] != heldout].copy()
        test = df[df["dataset_id"] == heldout].copy()

        for target in target_variants:
            for name, group_cols in PRIOR_GROUPS.items():
                rows.append(evaluate(
                    train, test,
                    mode="leave_one_dataset_out",
                    split_id=0,
                    heldout_dataset=heldout,
                    target_variant=target,
                    baseline_name=name,
                    group_cols=group_cols,
                ))

    res = pd.DataFrame(rows)

    csv = OUT_DIR / "expanded_explicit_prior_baseline_analysis.csv"
    summary_csv = OUT_DIR / "expanded_explicit_prior_baseline_summary.csv"
    report = OUT_DIR / "EXPANDED_EXPLICIT_PRIOR_BASELINE_ANALYSIS.md"

    res.to_csv(csv, index=False, float_format="%.12g")

    summary = res.groupby(["validation_mode", "target_variant", "baseline_name"]).agg(
        n_evaluations=("mae", "count"),
        mean_mae=("mae", "mean"),
        median_mae=("mae", "median"),
        mean_r2=("r2", "mean"),
        median_r2=("r2", "median"),
        mean_global_median_mae=("global_train_median_mae", "mean"),
        mean_improvement_vs_global_train_median=("improvement_vs_global_train_median", "mean"),
        median_improvement_vs_global_train_median=("improvement_vs_global_train_median", "median"),
    ).reset_index()

    summary.to_csv(summary_csv, index=False, float_format="%.12g")

    with open(OUT_DIR / "expanded_explicit_prior_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

    lines = []
    lines.append("# Expanded Explicit Prior Baseline Analysis\n\n")
    lines.append("## Purpose\n\n")
    lines.append("This analysis quantifies how much apparent predictive performance can be obtained from simple source, dataset, chemistry, or form-factor target priors without using cell-level early-cycle degradation features.\n\n")
    lines.append("## Summary\n\n")
    lines.append(summary.to_string(index=False))
    lines.append("\n\n")
    lines.append("## Interpretation guide\n\n")
    lines.append("- Strong random-split performance from dataset or source priors indicates target-distribution leakage.\n")
    lines.append("- LODO evaluation should weaken dataset-id priors because the held-out dataset is unseen during training.\n")
    lines.append("- If centered targets remove the prior advantage, the raw-retention gain is mainly distribution-level rather than cell-level.\n")
    report.write_text("".join(lines), encoding="utf-8")

    # Plot random split original
    for mode in ["random_stratified_split", "leave_one_dataset_out"]:
        for target in target_variants:
            sub = summary[
                (summary["validation_mode"] == mode)
                & (summary["target_variant"] == target)
            ].copy()
            sub = sub.sort_values("mean_mae")

            plt.figure(figsize=(9, 5.2))
            x = np.arange(len(sub))
            plt.bar(x, sub["mean_mae"].values)
            plt.xticks(x, sub["baseline_name"], rotation=35, ha="right")
            plt.ylabel("Mean MAE")
            plt.title(f"Explicit prior baselines: {mode} / {target}")
            plt.tight_layout()
            safe = f"{mode}_{target}".replace("/", "_")
            plt.savefig(FIG_DIR / f"expanded_explicit_prior_baselines_{safe}.png", dpi=300, bbox_inches="tight")
            plt.savefig(FIG_DIR / f"expanded_explicit_prior_baselines_{safe}.svg", bbox_inches="tight", metadata={"Date": None})
            plt.close()

    print("Saved:", report)
    print("Saved:", csv)
    print("Saved:", summary_csv)
    print("")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
