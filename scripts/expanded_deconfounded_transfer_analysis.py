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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_TARGET = "final_capacity_retention"


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


def add_deconfounded_targets(df):
    out = df.copy()
    out["raw_retention"] = out[RAW_TARGET].astype(float)

    dataset_mean = out.groupby("dataset_id")["raw_retention"].transform("mean")
    dataset_median = out.groupby("dataset_id")["raw_retention"].transform("median")

    out["dataset_mean_centered_retention"] = out["raw_retention"] - dataset_mean
    out["dataset_median_centered_retention"] = out["raw_retention"] - dataset_median

    # Percentile within dataset. Small datasets are kept but interpreted cautiously.
    out["within_dataset_percentile"] = out.groupby("dataset_id")["raw_retention"].rank(pct=True)

    return out


def evaluate_one(df, heldout, feature_set, features, target):
    train = df[df["dataset_id"] != heldout].copy()
    test = df[df["dataset_id"] == heldout].copy()

    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])

    features = [c for c in features if c in df.columns]

    if len(train) < 20 or len(test) < 2:
        return None

    X_train = train[features]
    y_train = train[target].astype(float)
    X_test = test[features]
    y_test = test[target].astype(float).to_numpy()

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

    mae_model = float(mean_absolute_error(y_test, pred))
    mae_baseline = float(mean_absolute_error(y_test, median_pred))

    return {
        "heldout_dataset": heldout,
        "source_family_heldout": str(test["source_family"].mode().iloc[0]) if "source_family" in test.columns and len(test) else "",
        "target": target,
        "feature_set": feature_set,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(features)),
        "mae_model": mae_model,
        "rmse_model": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2_model": float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan,
        "mae_train_median": mae_baseline,
        "mae_improvement_baseline_minus_model": float(mae_baseline - mae_model),
        "heldout_target_mean": float(np.mean(y_test)),
        "train_target_mean": float(np.mean(y_train)),
        "target_mean_shift_abs": float(abs(np.mean(y_test) - np.mean(y_train))),
    }


def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[RAW_TARGET]).copy()
    df = add_deconfounded_targets(df)

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

    targets = [
        "raw_retention",
        "dataset_mean_centered_retention",
        "dataset_median_centered_retention",
        "within_dataset_percentile",
    ]

    rows = []
    datasets = sorted(df["dataset_id"].dropna().unique().tolist())

    for target in targets:
        print("")
        print("#" * 80)
        print("Target:", target)

        for heldout in datasets:
            print("")
            print("Heldout:", heldout)
            for fs_name, features in feature_sets.items():
                r = evaluate_one(df, heldout, fs_name, features, target)
                if r is None:
                    print(" skipped:", fs_name)
                    continue
                rows.append(r)
                print(
                    fs_name,
                    "MAE=", round(r["mae_model"], 5),
                    "baseline=", round(r["mae_train_median"], 5),
                    "improvement=", round(r["mae_improvement_baseline_minus_model"], 5),
                )

    res = pd.DataFrame(rows)

    csv_path = OUT_DIR / "expanded_deconfounded_transfer_analysis.csv"
    json_path = OUT_DIR / "expanded_deconfounded_transfer_analysis.json"
    matrix_path = OUT_DIR / "expanded_deconfounded_transfer_improvement_matrix.csv"
    report_path = OUT_DIR / "EXPANDED_DECONFOUNDED_TRANSFER_ANALYSIS.md"

    res.to_csv(csv_path, index=False, float_format="%.12g")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(res.to_json(orient="records")), f, indent=2)

    matrix = res.pivot_table(
        index=["target", "heldout_dataset"],
        columns="feature_set",
        values="mae_improvement_baseline_minus_model",
    )
    matrix.to_csv(matrix_path, float_format="%.12g")

    report = []
    report.append("# Expanded Deconfounded Transfer Analysis\n\n")
    report.append("## Purpose\n\n")
    report.append("This analysis tests whether apparent leave-one-dataset-out performance reflects transferable cell-level degradation information or dataset-level target priors. Raw retention is compared with dataset-centered and within-dataset percentile targets.\n\n")
    report.append("## Interpretation\n\n")
    report.append("- Positive improvement means the model beats the train-median baseline.\n")
    report.append("- If a feature family performs well on raw retention but fails on centered or percentile targets, it likely exploits dataset-level priors rather than transferable within-dataset degradation residuals.\n")
    report.append("- Metadata-only performance is especially important: raw success combined with centered-target failure indicates protocol/dataset fingerprinting.\n")
    report.append("- UL_PUR has only 2 held-out cells and should be interpreted cautiously.\n\n")

    report.append("## Improvement matrix\n\n")
    report.append(matrix.to_string())
    report.append("\n\n")

    report.append("## Full metrics\n\n")
    report.append(res.to_string(index=False))
    report.append("\n")

    report_path.write_text("".join(report), encoding="utf-8")

    for target in targets:
        sub = res[res["target"] == target]
        mat = sub.pivot(index="heldout_dataset", columns="feature_set", values="mae_improvement_baseline_minus_model")

        plt.figure(figsize=(10, 5.5))
        x = np.arange(len(mat.index))
        width = 0.18
        for i, col in enumerate(mat.columns):
            plt.bar(x + (i - 1.5) * width, mat[col].values, width=width, label=col)
        plt.axhline(0, linewidth=1)
        plt.xticks(x, mat.index, rotation=45, ha="right")
        plt.ylabel("MAE improvement vs train-median baseline")
        plt.title(f"Expanded deconfounded transfer: {target}")
        plt.legend(fontsize=7)
        plt.tight_layout()

        safe_target = target.replace("/", "_")
        plt.savefig(FIG_DIR / f"expanded_deconfounded_transfer_{safe_target}.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIG_DIR / f"expanded_deconfounded_transfer_{safe_target}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close()

    print("")
    print("=" * 80)
    print("Saved:", report_path)
    print("Saved:", csv_path)
    print("Saved:", matrix_path)
    print("")
    print("Improvement matrix:")
    print(matrix.to_string())


if __name__ == "__main__":
    main()
