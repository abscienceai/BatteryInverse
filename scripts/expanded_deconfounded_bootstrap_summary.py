from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "batteryinverse"
import matplotlib.pyplot as plt

ANALYSIS_PATH = Path("reports/benchmark_audits/expanded_deconfounded_transfer_analysis.csv")
PRED_PATH = Path("reports/benchmark_audits/expanded_transfer_gap_feature_family_predictions.csv")
OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# This script bootstraps at the metric level using available MAE improvements.
# It summarizes how often each feature family beats the train-median baseline
# across held-out datasets, with and without small-n UL_PUR.

N_BOOT = 10000
RNG = np.random.default_rng(42)


def bootstrap_mean_ci(values, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan
    boots = RNG.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
        float((boots > 0).mean()),
    )


def main():
    df = pd.read_csv(ANALYSIS_PATH)

    rows = []
    for exclude_ul_pur in [False, True]:
        work = df.copy()
        label = "all_datasets"
        if exclude_ul_pur:
            work = work[work["heldout_dataset"] != "UL_PUR"].copy()
            label = "excluding_UL_PUR"

        for target in sorted(work["target"].unique()):
            for fs in sorted(work["feature_set"].unique()):
                sub = work[(work["target"] == target) & (work["feature_set"] == fs)].copy()
                vals = sub["mae_improvement_baseline_minus_model"].to_numpy(dtype=float)

                mean, lo, hi, p_pos = bootstrap_mean_ci(vals)

                rows.append({
                    "dataset_scope": label,
                    "target": target,
                    "feature_set": fs,
                    "n_heldout_datasets": int(sub["heldout_dataset"].nunique()),
                    "mean_improvement": mean,
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "bootstrap_p_mean_improvement_positive": p_pos,
                    "n_positive_improvements": int((sub["mae_improvement_baseline_minus_model"] > 0).sum()),
                    "n_negative_or_zero_improvements": int((sub["mae_improvement_baseline_minus_model"] <= 0).sum()),
                    "median_improvement": float(np.nanmedian(vals)),
                })

    summary = pd.DataFrame(rows)

    csv_path = OUT_DIR / "expanded_deconfounded_bootstrap_summary.csv"
    json_path = OUT_DIR / "expanded_deconfounded_bootstrap_summary.json"
    report_path = OUT_DIR / "EXPANDED_DECONFOUNDED_BOOTSTRAP_SUMMARY.md"

    summary.to_csv(csv_path, index=False, float_format="%.12g")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

    report = []
    report.append("# Expanded Deconfounded Bootstrap Summary\n\n")
    report.append("## Purpose\n\n")
    report.append("This report bootstraps mean MAE improvement over train-median baseline across held-out datasets. Positive improvement means the model beats a simple train-median predictor. Results are shown both across all datasets and excluding UL_PUR, which has only two cells.\n\n")

    for scope in ["all_datasets", "excluding_UL_PUR"]:
        report.append(f"## Scope: {scope}\n\n")
        sub_scope = summary[summary["dataset_scope"] == scope]
        for target in sorted(sub_scope["target"].unique()):
            report.append(f"### Target: {target}\n\n")
            tmp = sub_scope[sub_scope["target"] == target].copy()
            report.append(tmp[[
                "feature_set",
                "n_heldout_datasets",
                "mean_improvement",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "bootstrap_p_mean_improvement_positive",
                "n_positive_improvements",
                "n_negative_or_zero_improvements",
                "median_improvement",
            ]].to_string(index=False))
            report.append("\n\n")

    report.append("## Interpretation guide\n\n")
    report.append("- A positive mean improvement with CI above zero indicates robust transfer benefit across held-out datasets.\n")
    report.append("- Raw-retention gains that vanish or become negative for centered/percentile targets indicate dataset-level prior learning rather than transferable cell-level degradation prediction.\n")
    report.append("- Metadata-only raw-retention gains should be interpreted cautiously if centered/percentile gains are near zero or negative.\n")

    report_path.write_text("".join(report), encoding="utf-8")

    # Plot excluding UL_PUR because it is the cleaner headline.
    plot_df = summary[summary["dataset_scope"] == "excluding_UL_PUR"].copy()
    for target in sorted(plot_df["target"].unique()):
        sub = plot_df[plot_df["target"] == target].copy()
        sub = sub.sort_values("mean_improvement")

        y = np.arange(len(sub))
        x = sub["mean_improvement"].to_numpy()
        xerr_low = x - sub["bootstrap_ci95_low"].to_numpy()
        xerr_high = sub["bootstrap_ci95_high"].to_numpy() - x

        plt.figure(figsize=(8, 4.8))
        plt.errorbar(x, y, xerr=[xerr_low, xerr_high], fmt="o", capsize=3)
        plt.axvline(0, linewidth=1)
        plt.yticks(y, sub["feature_set"])
        plt.xlabel("Mean MAE improvement vs train-median baseline")
        plt.title(f"Bootstrap transfer improvement: {target}")
        plt.tight_layout()
        safe = target.replace("/", "_")
        plt.savefig(FIG_DIR / f"expanded_deconfounded_bootstrap_{safe}.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIG_DIR / f"expanded_deconfounded_bootstrap_{safe}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close()

    print("Bootstrap summary created.")
    print("")
    print(summary.to_string(index=False))
    print("")
    print("Saved:", report_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
