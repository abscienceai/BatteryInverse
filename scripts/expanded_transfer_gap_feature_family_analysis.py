from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "batteryinverse"
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "final_capacity_retention"


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


def safe_corr(x, y, method):
    s = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return np.nan
    if s["x"].nunique() < 2 or s["y"].nunique() < 2:
        return np.nan
    return float(s["x"].corr(s["y"], method=method))


def evaluate_one(df, heldout, feature_set, features):
    train = df[df["dataset_id"] != heldout].copy()
    test = df[df["dataset_id"] == heldout].copy()

    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])

    features = [c for c in features if c in df.columns]

    if len(train) < 20 or len(test) < 2:
        return None, None

    X_train = train[features]
    y_train = train[TARGET].astype(float)
    X_test = test[features]
    y_test = test[TARGET].astype(float).to_numpy()

    pre = make_preprocessor(features, train)
    X_train_z = pre.fit_transform(X_train)
    X_test_z = pre.transform(X_test)

    n_neighbors = min(5, len(train))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_train_z)
    dist, _ = nn.kneighbors(X_test_z)

    nn1 = dist[:, 0]
    nn5 = dist.mean(axis=1)

    model = RandomForestRegressor(
        n_estimators=900,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("preprocess", make_preprocessor(features, train)),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    median_pred = np.repeat(float(np.median(y_train)), len(y_test))
    abs_err = np.abs(y_test - pred)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    baseline_mae = mean_absolute_error(y_test, median_pred)

    metrics = {
        "heldout_dataset": heldout,
        "source_family_heldout": str(test["source_family"].mode().iloc[0]) if "source_family" in test.columns and len(test) else "",
        "feature_set": feature_set,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(features)),
        "mae_model": float(mae),
        "rmse_model": float(rmse),
        "r2_model": float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan,
        "mae_train_median": float(baseline_mae),
        "mae_improvement_baseline_minus_model": float(baseline_mae - mae),
        "heldout_target_mean": float(np.mean(y_test)),
        "train_target_mean": float(np.mean(y_train)),
        "target_mean_shift_abs": float(abs(np.mean(y_test) - np.mean(y_train))),
        "mean_nn1_distance": float(np.mean(nn1)),
        "median_nn1_distance": float(np.median(nn1)),
        "mean_nn5_distance": float(np.mean(nn5)),
        "spearman_nn1_distance_vs_abs_error": safe_corr(nn1, abs_err, "spearman"),
        "pearson_nn1_distance_vs_abs_error": safe_corr(nn1, abs_err, "pearson"),
        "transfer_gap_index_mean_nn1_x_mae": float(np.mean(nn1) * mae),
        "transfer_gap_index_mean_nn5_x_mae": float(np.mean(nn5) * mae),
    }

    pred_df = test[["source_family", "dataset_id", "cell_id", TARGET]].copy()
    pred_df["feature_set"] = feature_set
    pred_df["prediction"] = pred
    pred_df["absolute_error"] = abs_err
    pred_df["train_median_prediction"] = median_pred
    pred_df["train_median_absolute_error"] = np.abs(y_test - median_pred)
    pred_df["nn1_distance"] = nn1
    pred_df["nn5_mean_distance"] = nn5

    return metrics, pred_df


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_sets = {
        "absolute_capacity_only": [
            "cap_cycle_1",
            "cap_cycle_10",
            "cap_cycle_50",
            "cap_cycle_100",
            "cap_mean_first_100",
            "cap_std_first_100",
            "cap_slope_first_100",
            "cap_fade_1_to_100",
        ],
        "normalized_capacity_only": [
            "cap_ratio_10_to_1",
            "cap_ratio_50_to_1",
            "cap_ratio_100_to_1",
            "cap_mean_first_100_norm",
            "cap_std_first_100_norm",
            "cap_slope_first_100_norm",
            "cap_fade_fraction_1_to_100",
        ],
        "absolute_plus_normalized_capacity": [
            "cap_cycle_1",
            "cap_cycle_10",
            "cap_cycle_50",
            "cap_cycle_100",
            "cap_mean_first_100",
            "cap_std_first_100",
            "cap_slope_first_100",
            "cap_fade_1_to_100",
            "cap_ratio_10_to_1",
            "cap_ratio_50_to_1",
            "cap_ratio_100_to_1",
            "cap_mean_first_100_norm",
            "cap_std_first_100_norm",
            "cap_slope_first_100_norm",
            "cap_fade_fraction_1_to_100",
        ],
        "metadata_only": [
            "nominal_capacity_Ah",
            "min_voltage_V",
            "max_voltage_V",
            "chemistry",
            "form_factor",
            "anode_material",
            "cathode_material",
            "ambient_temperature_C",
        ],
    }

    datasets = sorted(df["dataset_id"].dropna().unique().tolist())

    rows = []
    preds = []

    for heldout in datasets:
        print("")
        print("=" * 80)
        print("Heldout:", heldout)
        for fs_name, features in feature_sets.items():
            metrics, pred_df = evaluate_one(df, heldout, fs_name, features)
            if metrics is None:
                print(" skipped:", fs_name)
                continue
            rows.append(metrics)
            preds.append(pred_df)
            print(
                fs_name,
                "n_test=", metrics["n_test"],
                "MAE=", round(metrics["mae_model"], 5),
                "baseline=", round(metrics["mae_train_median"], 5),
                "TGI=", round(metrics["transfer_gap_index_mean_nn1_x_mae"], 5),
            )

    metrics_df = pd.DataFrame(rows)
    pred_all = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()

    metrics_csv = OUT_DIR / "expanded_transfer_gap_feature_family_metrics.csv"
    metrics_json = OUT_DIR / "expanded_transfer_gap_feature_family_metrics.json"
    pred_csv = OUT_DIR / "expanded_transfer_gap_feature_family_predictions.csv"
    matrix_csv = OUT_DIR / "expanded_transfer_gap_feature_family_mae_matrix.csv"
    report_path = OUT_DIR / "EXPANDED_TRANSFER_GAP_FEATURE_FAMILY_ANALYSIS.md"

    metrics_df.to_csv(metrics_csv, index=False, float_format="%.12g")
    pred_all.to_csv(pred_csv, index=False, float_format="%.12g")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(json.loads(metrics_df.to_json(orient="records")), f, indent=2)

    mae_matrix = metrics_df.pivot(index="heldout_dataset", columns="feature_set", values="mae_model")
    mae_matrix.to_csv(matrix_csv, float_format="%.12g")

    report = []
    report.append("# Expanded Transfer-Gap Feature-Family Analysis\n\n")
    report.append("## Purpose\n\n")
    report.append("This analysis extends leave-one-dataset-out transfer diagnostics from the original BatteryML-derived aging table to the expanded 567-cell table including selected BatteryLife processed datasets.\n\n")
    report.append("## Dataset sizes\n\n")
    report.append(df["dataset_id"].value_counts().sort_index().to_string())
    report.append("\n\n")
    report.append("## MAE matrix\n\n")
    report.append(mae_matrix.to_string())
    report.append("\n\n")
    report.append("## Full metrics\n\n")
    report.append(metrics_df.to_string(index=False))
    report.append("\n\n")
    report.append("## Interpretation guide\n\n")
    report.append("- Large gaps between random-looking performance and leave-one-dataset-out performance indicate domain shift.\n")
    report.append("- Metadata-only success can indicate dataset/protocol priors rather than transferable degradation physics.\n")
    report.append("- Normalized capacity features test whether capacity-scale confounding is reduced.\n")
    report.append("- Transfer-Gap Index combines feature-space distance and observed error; high values indicate risky extrapolation.\n")
    report.append("- Very small held-out datasets, especially UL_PUR, should be interpreted cautiously.\n")

    report_path.write_text("".join(report), encoding="utf-8")

    plt.figure(figsize=(10, 5.5))
    plot_matrix = mae_matrix.copy()
    x = np.arange(len(plot_matrix.index))
    width = 0.18
    for i, col in enumerate(plot_matrix.columns):
        plt.bar(x + (i - 1.5) * width, plot_matrix[col].values, width=width, label=col)
    plt.xticks(x, plot_matrix.index, rotation=45, ha="right")
    plt.ylabel("MAE on held-out dataset")
    plt.title("Expanded leave-one-dataset-out MAE by feature family")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "expanded_transfer_gap_feature_family_mae.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "expanded_transfer_gap_feature_family_mae.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close()

    plt.figure(figsize=(7, 5))
    for fs_name, sub in metrics_df.groupby("feature_set"):
        plt.scatter(sub["mean_nn1_distance"], sub["mae_model"], label=fs_name)
    plt.xlabel("Mean nearest-train distance")
    plt.ylabel("Held-out MAE")
    plt.title("Expanded transfer gap: distance vs error")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "expanded_transfer_gap_distance_vs_error.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "expanded_transfer_gap_distance_vs_error.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close()

    print("")
    print("=" * 80)
    print("Saved:", report_path)
    print("Saved:", metrics_csv)
    print("Saved:", matrix_csv)
    print("")
    print("MAE matrix:")
    print(mae_matrix.to_string())


if __name__ == "__main__":
    main()
