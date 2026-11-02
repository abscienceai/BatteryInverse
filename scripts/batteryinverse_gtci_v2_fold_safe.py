from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT = Path("reports/gtci_v2_fold_safe")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "final_capacity_retention"
XI = "gtci_xi_fold_safe_target_prior"

ABS_FEATURES = [
    "cap_cycle_1", "cap_cycle_10", "cap_cycle_50", "cap_cycle_100",
    "cap_mean_first_100", "cap_std_first_100", "cap_slope_first_100",
    "cap_fade_1_to_100",
]

NORM_FEATURES = [
    "cap_ratio_10_to_1", "cap_ratio_50_to_1", "cap_ratio_100_to_1",
    "cap_mean_first_100_norm", "cap_std_first_100_norm",
    "cap_slope_first_100_norm", "cap_fade_fraction_1_to_100",
]

META_FEATURES = [
    "nominal_capacity_Ah", "min_voltage_V", "max_voltage_V",
    "chemistry", "form_factor", "anode_material", "cathode_material",
    "ambient_temperature_C",
]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipe(train, features, seed=42):
    numeric = [
        c for c in features
        if c in train.columns and pd.api.types.is_numeric_dtype(train[c])
    ]
    categorical = [
        c for c in features
        if c in train.columns and c not in numeric
    ]

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_ohe()),
                ]),
                categorical,
            )
        )

    pre = ColumnTransformer(transformers, sparse_threshold=0.0)

    model = ExtraTreesRegressor(
        n_estimators=700,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )

    return Pipeline([("pre", pre), ("model", model)])


def add_fold_safe_xi(train, test, mode, seed=42, sigma=0.01):
    """
    Fold-safe synthetic target-prior confound.

    Random split:
      test datasets are represented in train, so xi can carry train-estimated
      dataset-prior information.

    LODO:
      held-out dataset is unseen in train, so its dataset median is unavailable.
      Therefore test xi is assigned the train-global median prior, not its own
      hidden target median. This prevents held-out target leakage.
    """
    rng = np.random.default_rng(seed)
    train = train.copy()
    test = test.copy()

    train_dataset_prior = train.groupby("dataset_id")[TARGET].median()
    train_global_prior = float(train[TARGET].median())

    xi_train_raw = train["dataset_id"].map(train_dataset_prior).fillna(train_global_prior).to_numpy(dtype=float)

    if mode == "random_split":
        xi_test_raw = test["dataset_id"].map(train_dataset_prior).fillna(train_global_prior).to_numpy(dtype=float)
    elif mode == "lodo":
        xi_test_raw = np.repeat(train_global_prior, len(test)).astype(float)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    xi_train_raw = xi_train_raw + rng.normal(0.0, sigma, size=len(train))
    xi_test_raw = xi_test_raw + rng.normal(0.0, sigma, size=len(test))

    mu = float(np.mean(xi_train_raw))
    sd = float(np.std(xi_train_raw))
    if sd < 1e-12:
        sd = 1.0

    train[XI] = (xi_train_raw - mu) / sd
    test[XI] = (xi_test_raw - mu) / sd

    return train, test


def ci(vals):
    vals = pd.Series(vals).dropna().astype(float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(vals.mean()), float(vals.quantile(0.025)), float(vals.quantile(0.975))


def eval_random(df, features, n_splits=100):
    rows = []
    work = df.dropna(subset=[TARGET, "dataset_id"]).copy().reset_index(drop=True)

    for split in range(n_splits):
        train_idx, test_idx = train_test_split(
            np.arange(len(work)),
            test_size=0.30,
            random_state=1000 + split,
            stratify=work["dataset_id"].astype(str),
        )

        train = work.iloc[train_idx].copy()
        test = work.iloc[test_idx].copy()
        train, test = add_fold_safe_xi(train, test, mode="random_split", seed=split)

        pipe = make_pipe(train, features, seed=split)
        pipe.fit(train[features], train[TARGET])
        pred = pipe.predict(test[features])

        baseline = np.repeat(train[TARGET].median(), len(test))
        mae = mean_absolute_error(test[TARGET], pred)
        base_mae = mean_absolute_error(test[TARGET], baseline)

        rows.append({
            "split": split,
            "mae": mae,
            "baseline_mae": base_mae,
            "improvement_vs_train_median": base_mae - mae,
        })

    return pd.DataFrame(rows)


def eval_lodo(df, features):
    rows = []
    work = df.dropna(subset=[TARGET, "dataset_id"]).copy().reset_index(drop=True)

    for j, ds in enumerate(sorted(work["dataset_id"].astype(str).unique())):
        train = work[work["dataset_id"].astype(str) != ds].copy()
        test = work[work["dataset_id"].astype(str) == ds].copy()
        if len(test) < 2:
            continue

        train, test = add_fold_safe_xi(train, test, mode="lodo", seed=2000 + j)

        pipe = make_pipe(train, features, seed=42 + j)
        pipe.fit(train[features], train[TARGET])
        pred = pipe.predict(test[features])

        baseline = np.repeat(train[TARGET].median(), len(test))
        mae = mean_absolute_error(test[TARGET], pred)
        base_mae = mean_absolute_error(test[TARGET], baseline)

        rows.append({
            "heldout_dataset": ds,
            "n_test": len(test),
            "mae": mae,
            "baseline_mae": base_mae,
            "improvement_vs_train_median": base_mae - mae,
        })

    return pd.DataFrame(rows)


def summarize(name, validation, table):
    m, lo, hi = ci(table["mae"])
    im, ilo, ihi = ci(table["improvement_vs_train_median"])
    return {
        "feature_set": name,
        "validation": validation,
        "mean_mae": m,
        "ci_low_mae": lo,
        "ci_high_mae": hi,
        "mean_improvement": im,
        "ci_low_improvement": ilo,
        "ci_high_improvement": ihi,
    }


def permutation_xi_random(df, features):
    if XI not in features:
        return None

    work = df.dropna(subset=[TARGET, "dataset_id"]).copy().reset_index(drop=True)

    train_idx, test_idx = train_test_split(
        np.arange(len(work)),
        test_size=0.30,
        random_state=777,
        stratify=work["dataset_id"].astype(str),
    )

    train = work.iloc[train_idx].copy()
    test = work.iloc[test_idx].copy()
    train, test = add_fold_safe_xi(train, test, mode="random_split", seed=777)

    pipe = make_pipe(train, features, seed=777)
    pipe.fit(train[features], train[TARGET])

    pi = permutation_importance(
        pipe,
        test[features],
        test[TARGET],
        n_repeats=50,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    idx = features.index(XI)
    vals = pi.importances[idx]
    return {
        "feature_set": ",".join(features),
        "xi_permutation_importance_mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET, "dataset_id"]).copy().reset_index(drop=True)

    abs_features = [c for c in ABS_FEATURES if c in df.columns]
    norm_features = [c for c in NORM_FEATURES if c in df.columns]
    meta_features = [c for c in META_FEATURES if c in df.columns]

    feature_sets = {
        "xi_only": [XI],
        "absolute_clean": abs_features,
        "absolute_plus_xi": abs_features + [XI],
        "normalized_clean": norm_features,
        "normalized_plus_xi": norm_features + [XI],
        "all_clean": abs_features + norm_features + meta_features,
        "all_plus_xi": abs_features + norm_features + meta_features + [XI],
    }

    summaries = []
    raw_tables = {}
    perm_rows = []

    for name, features in feature_sets.items():
        print("=" * 80)
        print(name)

        rs = eval_random(df, features)
        ld = eval_lodo(df, features)

        rs.to_csv(OUT / f"{name}_random_split.csv", index=False)
        ld.to_csv(OUT / f"{name}_lodo.csv", index=False)

        raw_tables[(name, "random_split")] = rs
        raw_tables[(name, "lodo")] = ld

        summaries.append(summarize(name, "random_split", rs))
        summaries.append(summarize(name, "leave_one_dataset_out", ld))

        pi = permutation_xi_random(df, features)
        if pi:
            pi["feature_set_name"] = name
            perm_rows.append(pi)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "GTCI_V2_FOLD_SAFE_SUMMARY.csv", index=False)

    perm = pd.DataFrame(perm_rows)
    perm.to_csv(OUT / "GTCI_V2_XI_PERMUTATION_IMPORTANCE.csv", index=False)

    delta_rows = []
    pairs = [
        ("absolute", "absolute_clean", "absolute_plus_xi"),
        ("normalized", "normalized_clean", "normalized_plus_xi"),
        ("all", "all_clean", "all_plus_xi"),
    ]

    for family, clean_name, plus_name in pairs:
        for validation, key in [
            ("random_split", "random_split"),
            ("leave_one_dataset_out", "lodo"),
        ]:
            clean = raw_tables[(clean_name, key)]["mae"].reset_index(drop=True)
            plus = raw_tables[(plus_name, key)]["mae"].reset_index(drop=True)
            n = min(len(clean), len(plus))
            delta = plus.iloc[:n] - clean.iloc[:n]
            dm, dlo, dhi = ci(delta)

            delta_rows.append({
                "family": family,
                "validation": validation,
                "delta_mae_plus_xi_minus_clean": dm,
                "ci_low": dlo,
                "ci_high": dhi,
                "interpretation": "negative means xi improves apparent MAE",
            })

    delta = pd.DataFrame(delta_rows)
    delta.to_csv(OUT / "GTCI_V2_DELTA_TABLE.csv", index=False)

    lines = []
    lines.append("# GTCI v2 fold-safe target-prior confound report\n\n")
    lines.append(
        "This version is split-aware. Random-split xi is computed from train-split dataset medians. "
        "LODO test xi does not use the held-out dataset median; it receives only the train-global unknown-domain prior. "
        "Therefore LODO cannot access held-out target-prior information through xi.\n\n"
    )

    lines.append("## Summary\n\n")
    lines.append(summary.to_string(index=False))
    lines.append("\n\n")

    lines.append("## Delta table\n\n")
    lines.append(delta.to_string(index=False))
    lines.append("\n\n")

    lines.append("## Xi permutation importance under random split\n\n")
    lines.append(perm.to_string(index=False))
    lines.append("\n\n")

    lines.append("## Interpretation guide\n\n")
    lines.append("- Strong positive-control behaviour: xi improves random-split MAE but not LODO MAE.\n")
    lines.append("- xi_only should look useful under random split but collapse to global-prior behaviour under LODO.\n")
    lines.append("- If LODO improves strongly after xi, check for leakage.\n")

    (OUT / "GTCI_V2_FOLD_SAFE_REPORT.md").write_text("".join(lines), encoding="utf-8")

    print("Done. Outputs:")
    for p in sorted(OUT.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
