from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "batteryinverse"
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_TARGET = "final_capacity_retention"

FEATURE_SETS = {
    "absolute_capacity_only": [
        "cap_cycle_1", "cap_cycle_10", "cap_cycle_50", "cap_cycle_100",
        "cap_mean_first_100", "cap_std_first_100", "cap_slope_first_100", "cap_fade_1_to_100",
    ],
    "normalized_capacity_only": [
        "cap_ratio_10_to_1", "cap_ratio_50_to_1", "cap_ratio_100_to_1",
        "cap_mean_first_100_norm", "cap_std_first_100_norm",
        "cap_slope_first_100_norm", "cap_fade_fraction_1_to_100",
    ],
    "metadata_only": [
        "nominal_capacity_Ah", "min_voltage_V", "max_voltage_V",
        "chemistry", "form_factor", "anode_material", "cathode_material", "ambient_temperature_C",
    ],
    "all_non_dataset_features": [
        "cap_cycle_1", "cap_cycle_10", "cap_cycle_50", "cap_cycle_100",
        "cap_mean_first_100", "cap_std_first_100", "cap_slope_first_100", "cap_fade_1_to_100",
        "cap_ratio_10_to_1", "cap_ratio_50_to_1", "cap_ratio_100_to_1",
        "cap_mean_first_100_norm", "cap_std_first_100_norm",
        "cap_slope_first_100_norm", "cap_fade_fraction_1_to_100",
        "nominal_capacity_Ah", "min_voltage_V", "max_voltage_V",
        "chemistry", "form_factor", "anode_material", "cathode_material", "ambient_temperature_C",
    ],
}

MODELS = {
    "random_forest": RandomForestRegressor(n_estimators=700, random_state=42, min_samples_leaf=2, n_jobs=-1),
    "extra_trees": ExtraTreesRegressor(n_estimators=700, random_state=42, min_samples_leaf=2, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}

def make_preprocessor(features, train):
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )

def add_targets(df):
    out = df.copy()
    out["raw_retention"] = out[RAW_TARGET].astype(float)

    source_mean = out.groupby("source_family")["raw_retention"].transform("mean")
    source_median = out.groupby("source_family")["raw_retention"].transform("median")
    dataset_mean = out.groupby("dataset_id")["raw_retention"].transform("mean")
    dataset_median = out.groupby("dataset_id")["raw_retention"].transform("median")

    out["source_mean_centered_retention"] = out["raw_retention"] - source_mean
    out["source_median_centered_retention"] = out["raw_retention"] - source_median
    out["dataset_mean_centered_retention"] = out["raw_retention"] - dataset_mean
    out["dataset_median_centered_retention"] = out["raw_retention"] - dataset_median

    return out

def evaluate(df, train_source, test_source, target, feature_set, features, model_name, model):
    train = df[df["source_family"] == train_source].copy()
    test = df[df["source_family"] == test_source].copy()

    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])

    features = [c for c in features if c in df.columns]

    X_train = train[features]
    y_train = train[target].astype(float)
    X_test = test[features]
    y_test = test[target].astype(float).to_numpy()

    pipe = Pipeline([
        ("preprocess", make_preprocessor(features, train)),
        ("model", model),
    ])

    try:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
    except Exception as e:
        return {
            "train_source": train_source,
            "test_source": test_source,
            "target": target,
            "feature_set": feature_set,
            "model": model_name,
            "status": "error",
            "error": repr(e),
        }

    baseline = np.repeat(float(np.median(y_train)), len(y_test))
    mae = mean_absolute_error(y_test, pred)
    baseline_mae = mean_absolute_error(y_test, baseline)

    return {
        "train_source": train_source,
        "test_source": test_source,
        "target": target,
        "feature_set": feature_set,
        "model": model_name,
        "status": "ok",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae_model": float(mae),
        "rmse_model": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2_model": float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan,
        "mae_train_median": float(baseline_mae),
        "mae_improvement_baseline_minus_model": float(baseline_mae - mae),
        "train_target_mean": float(np.mean(y_train)),
        "test_target_mean": float(np.mean(y_test)),
        "target_mean_shift_abs": float(abs(np.mean(y_train) - np.mean(y_test))),
    }

def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[RAW_TARGET, "source_family"]).copy()
    df = add_targets(df)

    print("Source family counts:")
    print(df["source_family"].value_counts().to_string())
    print("")

    targets = [
        "raw_retention",
        "source_mean_centered_retention",
        "source_median_centered_retention",
        "dataset_mean_centered_retention",
        "dataset_median_centered_retention",
    ]

    source_pairs = [
        ("BatteryML_main", "BatteryLife_selected"),
        ("BatteryLife_selected", "BatteryML_main"),
    ]

    rows = []
    for train_source, test_source in source_pairs:
        print("")
        print("=" * 80)
        print("Train:", train_source, "| Test:", test_source)
        for target in targets:
            print("")
            print("Target:", target)
            for fs, features in FEATURE_SETS.items():
                for mn, model in MODELS.items():
                    r = evaluate(df, train_source, test_source, target, fs, features, mn, model)
                    rows.append(r)
                    if r["status"] == "ok":
                        print(
                            fs, mn,
                            "MAE", round(r["mae_model"], 4),
                            "baseline", round(r["mae_train_median"], 4),
                            "impr", round(r["mae_improvement_baseline_minus_model"], 4),
                        )
                    else:
                        print(fs, mn, "ERROR", r["error"])

    res = pd.DataFrame(rows)
    ok = res[res["status"] == "ok"].copy()

    csv = OUT_DIR / "expanded_cross_source_transfer_analysis.csv"
    summary_csv = OUT_DIR / "expanded_cross_source_transfer_summary.csv"
    report = OUT_DIR / "EXPANDED_CROSS_SOURCE_TRANSFER_ANALYSIS.md"

    res.to_csv(csv, index=False, float_format="%.12g")

    summary = ok.groupby(["train_source", "test_source", "target", "feature_set"]).agg(
        n_models=("model", "nunique"),
        n_train=("n_train", "first"),
        n_test=("n_test", "first"),
        mean_mae=("mae_model", "mean"),
        median_mae=("mae_model", "median"),
        mean_baseline_mae=("mae_train_median", "mean"),
        mean_improvement=("mae_improvement_baseline_minus_model", "mean"),
        median_improvement=("mae_improvement_baseline_minus_model", "median"),
        n_positive=("mae_improvement_baseline_minus_model", lambda x: int((x > 0).sum())),
        n_nonpositive=("mae_improvement_baseline_minus_model", lambda x: int((x <= 0).sum())),
    ).reset_index()

    summary.to_csv(summary_csv, index=False, float_format="%.12g")

    with open(OUT_DIR / "expanded_cross_source_transfer_summary.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

    lines = []
    lines.append("# Expanded Cross-Source Transfer Analysis\n\n")
    lines.append("## Purpose\n\n")
    lines.append("This analysis evaluates transfer between the original BatteryML-derived source family and the selected BatteryLife source family. It tests whether models trained on one source family generalize to the other.\n\n")
    lines.append("## Source family counts\n\n")
    lines.append(df["source_family"].value_counts().to_string())
    lines.append("\n\n")
    lines.append("## Summary\n\n")
    lines.append(summary.to_string(index=False))
    lines.append("\n\n")
    lines.append("## Interpretation guide\n\n")
    lines.append("- Raw-retention success can reflect source-family target-distribution priors.\n")
    lines.append("- Source-centered and dataset-centered targets test whether cross-source transfer captures residual degradation behavior.\n")
    lines.append("- Strong asymmetry between BatteryML-to-BatteryLife and BatteryLife-to-BatteryML indicates source-family domain shift.\n")
    report.write_text("".join(lines), encoding="utf-8")

    # Plot raw and dataset-median-centered improvements
    plot_targets = ["raw_retention", "dataset_median_centered_retention", "source_median_centered_retention"]
    for target in plot_targets:
        sub = summary[summary["target"] == target].copy()
        sub["direction"] = sub["train_source"] + " -> " + sub["test_source"]
        sub["label"] = sub["direction"] + "\n" + sub["feature_set"]

        plt.figure(figsize=(11, 5.8))
        x = np.arange(len(sub))
        plt.bar(x, sub["mean_improvement"].values)
        plt.axhline(0, linewidth=1)
        plt.xticks(x, sub["label"], rotation=45, ha="right", fontsize=7)
        plt.ylabel("Mean MAE improvement vs train-median")
        plt.title(f"Cross-source transfer: {target}")
        plt.tight_layout()
        safe = target.replace("/", "_")
        plt.savefig(FIG_DIR / f"expanded_cross_source_transfer_{safe}.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIG_DIR / f"expanded_cross_source_transfer_{safe}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close()

    print("")
    print("=" * 80)
    print("Saved:", report)
    print("Saved:", csv)
    print("Saved:", summary_csv)
    print("")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
