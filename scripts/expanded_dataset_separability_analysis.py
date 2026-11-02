from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "batteryinverse"
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


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


def evaluate_feature_set(df, feature_set_name, features):
    features = [c for c in features if c in df.columns]
    work = df.dropna(subset=["dataset_id"]).copy()

    # UL_PUR has only 2 rows; keep it in descriptive counts but exclude from
    # stratified separability because stable train/test split is not meaningful.
    counts = work["dataset_id"].value_counts()
    eligible_datasets = counts[counts >= 5].index.tolist()
    eval_df = work[work["dataset_id"].isin(eligible_datasets)].copy()

    X = eval_df[features]
    y = eval_df["dataset_id"].astype(str)

    splitter = StratifiedShuffleSplit(
        n_splits=30,
        test_size=0.30,
        random_state=42,
    )

    rows = []
    pred_rows = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        train = eval_df.iloc[train_idx].copy()
        test = eval_df.iloc[test_idx].copy()

        model = RandomForestClassifier(
            n_estimators=700,
            random_state=42 + split_id,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

        pipe = Pipeline([
            ("preprocess", make_preprocessor(features, train)),
            ("model", model),
        ])

        pipe.fit(train[features], train["dataset_id"].astype(str))
        pred = pipe.predict(test[features])

        rows.append({
            "feature_set": feature_set_name,
            "split_id": split_id,
            "n_total": int(len(eval_df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_classes": int(y.nunique()),
            "accuracy": float(accuracy_score(test["dataset_id"].astype(str), pred)),
            "balanced_accuracy": float(balanced_accuracy_score(test["dataset_id"].astype(str), pred)),
            "macro_f1": float(f1_score(test["dataset_id"].astype(str), pred, average="macro")),
            "weighted_f1": float(f1_score(test["dataset_id"].astype(str), pred, average="weighted")),
            "majority_class_accuracy": float(y.value_counts(normalize=True).iloc[0]),
        })

        tmp = test[["source_family", "dataset_id", "cell_id"]].copy()
        tmp["feature_set"] = feature_set_name
        tmp["split_id"] = split_id
        tmp["predicted_dataset_id"] = pred
        tmp["correct"] = tmp["dataset_id"].astype(str).to_numpy() == pred
        pred_rows.append(tmp)

    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True)


def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)

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
        "all_non_dataset_features": [
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

    all_metrics = []
    all_preds = []

    print("Dataset counts:")
    print(df["dataset_id"].value_counts().sort_index().to_string())
    print("")

    for name, features in feature_sets.items():
        print("=" * 80)
        print("Feature set:", name)
        m, p = evaluate_feature_set(df, name, features)
        all_metrics.append(m)
        all_preds.append(p)

        print(m[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]].describe().to_string())
        print("")

    metrics = pd.concat(all_metrics, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    summary = metrics.groupby("feature_set").agg(
        n_splits=("split_id", "count"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_weighted_f1=("weighted_f1", "mean"),
        majority_class_accuracy=("majority_class_accuracy", "mean"),
        n_classes=("n_classes", "mean"),
        n_total=("n_total", "mean"),
    ).reset_index()

    metrics.to_csv(OUT_DIR / "expanded_dataset_separability_split_metrics.csv", index=False, float_format="%.12g")
    summary.to_csv(OUT_DIR / "expanded_dataset_separability_summary.csv", index=False, float_format="%.12g")
    preds.to_csv(OUT_DIR / "expanded_dataset_separability_predictions.csv", index=False, float_format="%.12g")

    with open(OUT_DIR / "expanded_dataset_separability_summary.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

    report = []
    report.append("# Expanded Dataset Separability Analysis\n\n")
    report.append("## Purpose\n\n")
    report.append("This analysis tests whether the held-out dataset identity can be predicted from early-cycle and metadata-derived features in the expanded 567-cell table. High separability indicates dataset fingerprints in the feature space.\n\n")
    report.append("## Dataset counts\n\n")
    report.append(df["dataset_id"].value_counts().sort_index().to_string())
    report.append("\n\n")
    report.append("Datasets with fewer than 5 cells are excluded from stratified separability scoring but retained in descriptive counts.\n\n")
    report.append("## Summary metrics\n\n")
    report.append(summary.to_string(index=False))
    report.append("\n\n")
    report.append("## Interpretation guide\n\n")
    report.append("- Accuracy far above majority-class accuracy indicates strong dataset separability.\n")
    report.append("- High separability using normalized capacity features means normalization does not remove dataset/protocol fingerprints.\n")
    report.append("- High separability using metadata is expected but still important because metadata-only raw-retention success can reflect target-prior transfer rather than cell-level degradation physics.\n")
    report.append("- This result should be interpreted together with the expanded deconfounded transfer analysis.\n")

    (OUT_DIR / "EXPANDED_DATASET_SEPARABILITY_ANALYSIS.md").write_text("".join(report), encoding="utf-8")

    plt.figure(figsize=(8, 4.8))
    x = np.arange(len(summary))
    plt.bar(x, summary["mean_balanced_accuracy"])
    plt.axhline(summary["majority_class_accuracy"].iloc[0], linewidth=1)
    plt.xticks(x, summary["feature_set"], rotation=35, ha="right")
    plt.ylabel("Mean balanced accuracy")
    plt.title("Expanded dataset separability")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "expanded_dataset_separability_balanced_accuracy.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "expanded_dataset_separability_balanced_accuracy.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close()

    print("=" * 80)
    print("Expanded dataset separability summary:")
    print(summary.to_string(index=False))
    print("")
    print("Saved:", OUT_DIR / "EXPANDED_DATASET_SEPARABILITY_ANALYSIS.md")


if __name__ == "__main__":
    main()
