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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
LODO_PATH = Path("reports/benchmark_audits/expanded_model_family_robustness.csv")

OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "final_capacity_retention"
N_SPLITS = 50

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
    "random_forest": RandomForestRegressor(
        n_estimators=700,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
    "extra_trees": ExtraTreesRegressor(
        n_estimators=700,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        random_state=42,
    ),
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

def evaluate_random_split(df, feature_set, features, model_name, model, split_id):
    work = df.dropna(subset=[TARGET, "dataset_id"]).copy()

    # Stratify by dataset_id so each random split preserves dataset mix.
    train_idx, test_idx = train_test_split(
        np.arange(len(work)),
        test_size=0.30,
        random_state=1000 + split_id,
        stratify=work["dataset_id"].astype(str),
    )

    train = work.iloc[train_idx].copy()
    test = work.iloc[test_idx].copy()

    features = [c for c in features if c in work.columns]

    X_train = train[features]
    y_train = train[TARGET].astype(float)
    X_test = test[features]
    y_test = test[TARGET].astype(float).to_numpy()

    pipe = Pipeline([
        ("preprocess", make_preprocessor(features, train)),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    baseline = np.repeat(float(np.median(y_train)), len(y_test))

    return {
        "validation_mode": "random_stratified_split",
        "split_id": split_id,
        "feature_set": feature_set,
        "model": model_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae_model": float(mean_absolute_error(y_test, pred)),
        "rmse_model": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2_model": float(r2_score(y_test, pred)),
        "mae_train_median": float(mean_absolute_error(y_test, baseline)),
        "mae_improvement_baseline_minus_model": float(mean_absolute_error(y_test, baseline) - mean_absolute_error(y_test, pred)),
    }

def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)

    rows = []
    for split_id in range(1, N_SPLITS + 1):
        print("")
        print("=" * 80)
        print("Random split:", split_id)
        for fs, features in FEATURE_SETS.items():
            for mn, model in MODELS.items():
                r = evaluate_random_split(df, fs, features, mn, model, split_id)
                rows.append(r)
                print(fs, mn, "MAE", round(r["mae_model"], 4), "R2", round(r["r2_model"], 4), "baseline", round(r["mae_train_median"], 4))

    random_df = pd.DataFrame(rows)

    # LODO summary from previous model-family robustness.
    lodo_raw = pd.read_csv(LODO_PATH)
    lodo_raw = lodo_raw[
        (lodo_raw["status"] == "ok")
        & (lodo_raw["model"].isin(MODELS.keys()))
        & (lodo_raw["feature_set"].isin(FEATURE_SETS.keys()))
    ].copy()
    lodo_raw["validation_mode"] = "leave_one_dataset_out"

    random_summary = random_df.groupby(["validation_mode", "feature_set", "model"]).agg(
        n_evaluations=("split_id", "count"),
        mean_mae=("mae_model", "mean"),
        std_mae=("mae_model", "std"),
        median_mae=("mae_model", "median"),
        mean_r2=("r2_model", "mean"),
        median_r2=("r2_model", "median"),
        mean_baseline_mae=("mae_train_median", "mean"),
        mean_improvement=("mae_improvement_baseline_minus_model", "mean"),
    ).reset_index()

    lodo_summary = lodo_raw.groupby(["validation_mode", "feature_set", "model"]).agg(
        n_evaluations=("heldout_dataset", "nunique"),
        mean_mae=("mae_model", "mean"),
        std_mae=("mae_model", "std"),
        median_mae=("mae_model", "median"),
        mean_r2=("r2_model", "mean"),
        median_r2=("r2_model", "median"),
        mean_baseline_mae=("mae_train_median", "mean"),
        mean_improvement=("mae_improvement_baseline_minus_model", "mean"),
    ).reset_index()

    summary = pd.concat([random_summary, lodo_summary], ignore_index=True)

    # Inflation ratio: LODO MAE / random split MAE.
    wide = summary.pivot_table(
        index=["feature_set", "model"],
        columns="validation_mode",
        values="mean_mae",
    ).reset_index()
    wide["lodo_to_random_mae_ratio"] = wide["leave_one_dataset_out"] / wide["random_stratified_split"]
    wide["lodo_minus_random_mae"] = wide["leave_one_dataset_out"] - wide["random_stratified_split"]

    random_csv = OUT_DIR / "expanded_random_split_results.csv"
    summary_csv = OUT_DIR / "expanded_random_split_vs_lodo_summary.csv"
    inflation_csv = OUT_DIR / "expanded_random_split_vs_lodo_inflation.csv"
    report_path = OUT_DIR / "EXPANDED_RANDOM_SPLIT_VS_LODO_INFLATION.md"

    random_df.to_csv(random_csv, index=False, float_format="%.12g")
    summary.to_csv(summary_csv, index=False, float_format="%.12g")
    wide.to_csv(inflation_csv, index=False, float_format="%.12g")

    with open(OUT_DIR / "expanded_random_split_vs_lodo_inflation.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(wide.to_json(orient="records")), f, indent=2)

    report = []
    report.append("# Expanded Random Split vs Leave-One-Dataset-Out Inflation\n\n")
    report.append("## Purpose\n\n")
    report.append("This analysis quantifies how much performance is inflated when cells are randomly split while preserving dataset mixture, compared with leave-one-dataset-out transfer.\n\n")
    report.append("## Mean MAE by validation mode\n\n")
    report.append(summary.to_string(index=False))
    report.append("\n\n")
    report.append("## LODO-to-random inflation\n\n")
    report.append(wide.to_string(index=False))
    report.append("\n\n")
    report.append("## Interpretation\n\n")
    report.append("- Random splits can share dataset/protocol fingerprints between train and test.\n")
    report.append("- LODO evaluation removes entire dataset domains and is therefore a stronger transfer test.\n")
    report.append("- Large LODO-to-random MAE ratios indicate benchmark leakage or domain-shift inflation.\n")
    report_path.write_text("".join(report), encoding="utf-8")

    # Plot ratio
    plot = wide.copy()
    plot["label"] = plot["feature_set"] + "\n" + plot["model"]
    plot = plot.sort_values("lodo_to_random_mae_ratio")

    plt.figure(figsize=(11, 6))
    y = np.arange(len(plot))
    plt.barh(y, plot["lodo_to_random_mae_ratio"].values)
    plt.axvline(1.0, linewidth=1)
    plt.yticks(y, plot["label"], fontsize=7)
    plt.xlabel("Mean MAE ratio: LODO / random split")
    plt.title("Random-split performance inflation relative to leave-one-dataset-out transfer")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "expanded_random_split_vs_lodo_inflation_ratio.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "expanded_random_split_vs_lodo_inflation_ratio.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close()

    # Plot MAE side-by-side
    mae_plot = summary.copy()
    mae_plot["label"] = mae_plot["feature_set"] + "\n" + mae_plot["model"]
    order = mae_plot[mae_plot["validation_mode"] == "leave_one_dataset_out"].sort_values("mean_mae")["label"].tolist()

    plt.figure(figsize=(12, 6))
    x = np.arange(len(order))
    width = 0.38

    rand = mae_plot[mae_plot["validation_mode"] == "random_stratified_split"].set_index("label").loc[order]
    lodo = mae_plot[mae_plot["validation_mode"] == "leave_one_dataset_out"].set_index("label").loc[order]

    plt.bar(x - width/2, rand["mean_mae"].values, width=width, label="Random split")
    plt.bar(x + width/2, lodo["mean_mae"].values, width=width, label="Leave-one-dataset-out")
    plt.xticks(x, order, rotation=45, ha="right", fontsize=7)
    plt.ylabel("Mean MAE")
    plt.title("Random split vs leave-one-dataset-out performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "expanded_random_split_vs_lodo_mean_mae.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "expanded_random_split_vs_lodo_mean_mae.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close()

    print("")
    print("=" * 80)
    print("Saved:", report_path)
    print("Saved:", summary_csv)
    print("Saved:", inflation_csv)
    print("")
    print("LODO-to-random inflation:")
    print(wide.to_string(index=False))

if __name__ == "__main__":
    main()
