from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "batteryinverse"
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "final_capacity_retention"

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
    "ridge": Ridge(alpha=1.0),
    "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=20000, random_state=42),
    "knn": KNeighborsRegressor(n_neighbors=7, weights="distance"),
    "random_forest": RandomForestRegressor(n_estimators=700, random_state=42, min_samples_leaf=2, n_jobs=-1),
    "extra_trees": ExtraTreesRegressor(n_estimators=700, random_state=42, min_samples_leaf=2, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42, max_iter=300),
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

def evaluate(df, heldout, feature_set, features, model_name, model):
    train = df[df["dataset_id"] != heldout].copy()
    test = df[df["dataset_id"] == heldout].copy()

    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])

    if len(test) < 2 or len(train) < 20:
        return None

    features = [c for c in features if c in df.columns]

    X_train = train[features]
    y_train = train[TARGET].astype(float)
    X_test = test[features]
    y_test = test[TARGET].astype(float).to_numpy()

    pipe = Pipeline([
        ("preprocess", make_preprocessor(features, train)),
        ("model", model),
    ])

    try:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
    except Exception as e:
        return {
            "heldout_dataset": heldout,
            "feature_set": feature_set,
            "model": model_name,
            "status": "error",
            "error": repr(e),
        }

    baseline = np.repeat(float(np.median(y_train)), len(y_test))
    mae = mean_absolute_error(y_test, pred)
    bmae = mean_absolute_error(y_test, baseline)

    return {
        "heldout_dataset": heldout,
        "feature_set": feature_set,
        "model": model_name,
        "status": "ok",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae_model": float(mae),
        "rmse_model": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2_model": float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan,
        "mae_train_median": float(bmae),
        "mae_improvement_baseline_minus_model": float(bmae - mae),
    }

def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
    datasets = sorted(df["dataset_id"].dropna().unique())

    rows = []
    for heldout in datasets:
        print("")
        print("=" * 80)
        print("Heldout:", heldout)
        for fs, features in FEATURE_SETS.items():
            for mn, model in MODELS.items():
                r = evaluate(df, heldout, fs, features, mn, model)
                if r is None:
                    continue
                rows.append(r)
                if r["status"] == "ok":
                    print(fs, mn, "MAE", round(r["mae_model"], 4), "baseline", round(r["mae_train_median"], 4), "impr", round(r["mae_improvement_baseline_minus_model"], 4))
                else:
                    print(fs, mn, "ERROR", r["error"])

    res = pd.DataFrame(rows)
    csv = OUT_DIR / "expanded_model_family_robustness.csv"
    json_path = OUT_DIR / "expanded_model_family_robustness.json"
    report = OUT_DIR / "EXPANDED_MODEL_FAMILY_ROBUSTNESS.md"

    res.to_csv(csv, index=False, float_format="%.12g")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(res.to_json(orient="records")), f, indent=2)

    ok = res[res["status"] == "ok"].copy()

    summary = ok.groupby(["feature_set", "model"]).agg(
        n_evaluations=("heldout_dataset", "nunique"),
        mean_mae=("mae_model", "mean"),
        median_mae=("mae_model", "median"),
        mean_improvement=("mae_improvement_baseline_minus_model", "mean"),
        median_improvement=("mae_improvement_baseline_minus_model", "median"),
        n_positive=("mae_improvement_baseline_minus_model", lambda x: int((x > 0).sum())),
        n_nonpositive=("mae_improvement_baseline_minus_model", lambda x: int((x <= 0).sum())),
    ).reset_index()

    summary_csv = OUT_DIR / "expanded_model_family_robustness_summary.csv"
    summary.to_csv(summary_csv, index=False, float_format="%.12g")

    lines = []
    lines.append("# Expanded Model-Family Robustness\n\n")
    lines.append("## Purpose\n\n")
    lines.append("This analysis tests whether the expanded transfer-gap conclusions depend on one model family. Multiple regressors are evaluated under leave-one-dataset-out transfer.\n\n")
    lines.append("## Summary\n\n")
    lines.append(summary.to_string(index=False))
    lines.append("\n\n")
    lines.append("## Interpretation\n\n")
    lines.append("- If no model family robustly beats the train-median baseline across datasets, the transfer-gap conclusion is model-family robust.\n")
    lines.append("- If metadata-only raw retention remains strong only for some datasets/models, this supports dataset-prior rather than universal degradation learning.\n")
    lines.append("- The strongest manuscript claim should focus on patterns that persist across model families.\n")
    report.write_text("".join(lines), encoding="utf-8")

    # Plot mean improvement by model and feature family
    plot = summary.pivot(index="model", columns="feature_set", values="mean_improvement")
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(plot.index))
    width = 0.18
    cols = list(plot.columns)
    for i, col in enumerate(cols):
        plt.bar(x + (i - (len(cols)-1)/2) * width, plot[col].values, width=width, label=col)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, plot.index, rotation=35, ha="right")
    plt.ylabel("Mean MAE improvement vs train-median")
    plt.title("Model-family robustness under leave-one-dataset-out transfer")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "expanded_model_family_robustness_mean_improvement.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "expanded_model_family_robustness_mean_improvement.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close()

    print("")
    print("=" * 80)
    print("Saved:", csv)
    print("Saved:", summary_csv)
    print("Saved:", report)
    print("")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
