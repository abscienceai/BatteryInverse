from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "batteryinverse"
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT_DIR = Path("reports/benchmark_audits")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_TARGET = "final_capacity_retention"
N_RANDOM_SPLITS = 30
RNG = np.random.default_rng(42)

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
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
    "extra_trees": ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
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

def add_shuffle_targets(df):
    out = df.copy()
    y = out[BASE_TARGET].astype(float).to_numpy()

    out["target_original"] = y
    out["target_global_shuffle"] = RNG.permutation(y)

    out["target_within_dataset_shuffle"] = np.nan
    for ds, idx in out.groupby("dataset_id").groups.items():
        idx = list(idx)
        vals = out.loc[idx, BASE_TARGET].astype(float).to_numpy()
        out.loc[idx, "target_within_dataset_shuffle"] = RNG.permutation(vals)

    return out

def fit_eval(train, test, features, target, model_name, model):
    features = [c for c in features if c in train.columns]
    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])

    X_train = train[features]
    y_train = train[target].astype(float)
    X_test = test[features]
    y_test = test[target].astype(float).to_numpy()

    pipe = Pipeline([
        ("preprocess", make_preprocessor(features, train)),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    baseline = np.repeat(float(np.median(y_train)), len(y_test))
    mae = mean_absolute_error(y_test, pred)
    bmae = mean_absolute_error(y_test, baseline)

    return {
        "mae_model": float(mae),
        "mae_train_median": float(bmae),
        "mae_improvement_baseline_minus_model": float(bmae - mae),
        "r2_model": float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

def random_split_eval(df, target, feature_set, features, model_name, model, split_id):
    work = df.dropna(subset=[target, "dataset_id"]).copy()

    train_idx, test_idx = train_test_split(
        np.arange(len(work)),
        test_size=0.30,
        random_state=1000 + split_id,
        stratify=work["dataset_id"].astype(str),
    )

    train = work.iloc[train_idx].copy()
    test = work.iloc[test_idx].copy()

    r = fit_eval(train, test, features, target, model_name, model)
    r.update({
        "validation_mode": "random_stratified_split",
        "split_id": split_id,
        "heldout_dataset": "",
        "target_variant": target,
        "feature_set": feature_set,
        "model": model_name,
    })
    return r

def lodo_eval(df, target, heldout, feature_set, features, model_name, model):
    train = df[df["dataset_id"] != heldout].copy()
    test = df[df["dataset_id"] == heldout].copy()

    if len(test) < 2:
        return None

    r = fit_eval(train, test, features, target, model_name, model)
    r.update({
        "validation_mode": "leave_one_dataset_out",
        "split_id": 0,
        "heldout_dataset": heldout,
        "target_variant": target,
        "feature_set": feature_set,
        "model": model_name,
    })
    return r

def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[BASE_TARGET, "dataset_id"]).copy()
    df = add_shuffle_targets(df)

    target_variants = [
        "target_original",
        "target_global_shuffle",
        "target_within_dataset_shuffle",
    ]

    rows = []

    # Random split
    for split_id in range(1, N_RANDOM_SPLITS + 1):
        print("")
        print("=" * 80)
        print("Random split:", split_id)
        for target in target_variants:
            print("Target:", target)
            for fs, features in FEATURE_SETS.items():
                for mn, model in MODELS.items():
                    r = random_split_eval(df, target, fs, features, mn, model, split_id)
                    rows.append(r)
                    print(fs, mn, "MAE", round(r["mae_model"], 4), "R2", round(r["r2_model"], 4), "impr", round(r["mae_improvement_baseline_minus_model"], 4))

    # LODO
    datasets = sorted(df["dataset_id"].dropna().unique())
    for heldout in datasets:
        print("")
        print("#" * 80)
        print("LODO heldout:", heldout)
        for target in target_variants:
            print("Target:", target)
            for fs, features in FEATURE_SETS.items():
                for mn, model in MODELS.items():
                    r = lodo_eval(df, target, heldout, fs, features, mn, model)
                    if r is None:
                        continue
                    rows.append(r)
                    print(fs, mn, "MAE", round(r["mae_model"], 4), "R2", round(r["r2_model"], 4), "impr", round(r["mae_improvement_baseline_minus_model"], 4))

    res = pd.DataFrame(rows)

    csv = OUT_DIR / "expanded_negative_control_shuffle_analysis.csv"
    summary_csv = OUT_DIR / "expanded_negative_control_shuffle_summary.csv"
    report = OUT_DIR / "EXPANDED_NEGATIVE_CONTROL_SHUFFLE_ANALYSIS.md"

    res.to_csv(csv, index=False, float_format="%.12g")

    summary = res.groupby(["validation_mode", "target_variant", "feature_set", "model"]).agg(
        n_evaluations=("mae_model", "count"),
        mean_mae=("mae_model", "mean"),
        median_mae=("mae_model", "median"),
        mean_r2=("r2_model", "mean"),
        median_r2=("r2_model", "median"),
        mean_baseline_mae=("mae_train_median", "mean"),
        mean_improvement=("mae_improvement_baseline_minus_model", "mean"),
        median_improvement=("mae_improvement_baseline_minus_model", "median"),
    ).reset_index()

    summary.to_csv(summary_csv, index=False, float_format="%.12g")

    with open(OUT_DIR / "expanded_negative_control_shuffle_summary.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

    lines = []
    lines.append("# Expanded Negative-Control Shuffle Analysis\n\n")
    lines.append("## Purpose\n\n")
    lines.append("This analysis tests whether the pipeline produces apparent performance when the target signal is destroyed globally or within each dataset. Within-dataset shuffling preserves dataset-level target distributions but destroys cell-level feature-target alignment.\n\n")
    lines.append("## Summary\n\n")
    lines.append(summary.to_string(index=False))
    lines.append("\n\n")
    lines.append("## Interpretation guide\n\n")
    lines.append("- Global shuffling should remove meaningful signal.\n")
    lines.append("- If random splits retain non-trivial performance under within-dataset shuffling, that indicates dataset-level priors can drive apparent performance.\n")
    lines.append("- LODO performance under shuffled targets should generally fail to beat the baseline.\n")
    report.write_text("".join(lines), encoding="utf-8")

    # Plot random split mean R2
    for mode in ["random_stratified_split", "leave_one_dataset_out"]:
        sub = summary[summary["validation_mode"] == mode].copy()
        sub["label"] = sub["feature_set"] + "\n" + sub["model"]

        for metric in ["mean_r2", "mean_improvement"]:
            piv = sub.pivot_table(index="label", columns="target_variant", values=metric)
            piv = piv.sort_index()

            x = np.arange(len(piv.index))
            width = 0.25
            plt.figure(figsize=(12, 6))
            for i, col in enumerate(piv.columns):
                plt.bar(x + (i - 1) * width, piv[col].values, width=width, label=col)
            plt.axhline(0, linewidth=1)
            plt.xticks(x, piv.index, rotation=45, ha="right", fontsize=7)
            plt.ylabel(metric)
            plt.title(f"Negative-control shuffle: {mode} / {metric}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            safe = metric.replace("/", "_")
            plt.savefig(FIG_DIR / f"expanded_negative_control_shuffle_{mode}_{safe}.png", dpi=300, bbox_inches="tight")
            plt.savefig(FIG_DIR / f"expanded_negative_control_shuffle_{mode}_{safe}.svg", bbox_inches="tight", metadata={"Date": None})
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
