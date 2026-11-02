from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors

DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT = Path("reports/gtci_conformal_active")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "final_capacity_retention"

BASE_FEATURES = [
    "cap_cycle_1", "cap_cycle_10", "cap_cycle_50", "cap_cycle_100",
    "cap_mean_first_100", "cap_std_first_100", "cap_slope_first_100", "cap_fade_1_to_100",
    "cap_ratio_10_to_1", "cap_ratio_50_to_1", "cap_ratio_100_to_1",
    "cap_mean_first_100_norm", "cap_std_first_100_norm",
    "cap_slope_first_100_norm", "cap_fade_fraction_1_to_100",
    "nominal_capacity_Ah", "min_voltage_V", "max_voltage_V",
    "chemistry", "form_factor", "anode_material", "cathode_material", "ambient_temperature_C",
]

K_VALUES = [3, 5, 10]
N_RANDOM = 100
RNG = np.random.default_rng(42)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipe(train, features, random_state=42):
    numeric = [c for c in features if c in train.columns and pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c in train.columns and c not in numeric]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_ohe()),
            ]), categorical),
        ],
        sparse_threshold=0.0,
    )

    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline([("pre", pre), ("model", model)])


def add_known_confound(df, alpha=1.0, sigma=0.05):
    out = df.copy()
    dataset_codes = pd.Categorical(out["dataset_id"].astype(str)).codes.astype(float)
    dataset_codes = (dataset_codes - dataset_codes.mean()) / (dataset_codes.std() + 1e-8)
    noise = RNG.normal(0, sigma, size=len(out))
    out["gtci_xi_dataset_confound"] = alpha * dataset_codes + noise
    return out


def eval_random_split(df, features, target=TARGET, n_splits=50):
    rows = []
    work = df.dropna(subset=[target, "dataset_id"]).copy()

    for split in range(n_splits):
        train_idx, test_idx = train_test_split(
            np.arange(len(work)),
            test_size=0.30,
            random_state=1000 + split,
            stratify=work["dataset_id"].astype(str),
        )
        train = work.iloc[train_idx].copy()
        test = work.iloc[test_idx].copy()

        pipe = make_pipe(train, features, random_state=42 + split)
        pipe.fit(train[features], train[target])
        pred = pipe.predict(test[features])

        baseline = np.repeat(train[target].median(), len(test))
        rows.append({
            "split": split,
            "mae": mean_absolute_error(test[target], pred),
            "baseline_mae": mean_absolute_error(test[target], baseline),
            "improvement": mean_absolute_error(test[target], baseline) - mean_absolute_error(test[target], pred),
        })

    return pd.DataFrame(rows)


def eval_lodo(df, features, target=TARGET):
    rows = []
    work = df.dropna(subset=[target, "dataset_id"]).copy()

    for ds in sorted(work["dataset_id"].astype(str).unique()):
        train = work[work["dataset_id"].astype(str) != ds].copy()
        test = work[work["dataset_id"].astype(str) == ds].copy()
        if len(test) < 2:
            continue

        pipe = make_pipe(train, features, random_state=42)
        pipe.fit(train[features], train[target])
        pred = pipe.predict(test[features])

        baseline = np.repeat(train[target].median(), len(test))
        rows.append({
            "heldout_dataset": ds,
            "n_test": len(test),
            "mae": mean_absolute_error(test[target], pred),
            "baseline_mae": mean_absolute_error(test[target], baseline),
            "improvement": mean_absolute_error(test[target], baseline) - mean_absolute_error(test[target], pred),
        })

    return pd.DataFrame(rows)


def summarize(vals):
    vals = pd.Series(vals).dropna().astype(float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return vals.mean(), vals.quantile(0.025), vals.quantile(0.975)


def run_gtci(df):
    contaminated = add_known_confound(df)

    base_features = [c for c in BASE_FEATURES if c in df.columns]
    contaminated_features = base_features + ["gtci_xi_dataset_confound"]

    results = []

    for label, data, features in [
        ("clean", df, base_features),
        ("gtci_contaminated", contaminated, contaminated_features),
    ]:
        rs = eval_random_split(data, features)
        lodo = eval_lodo(data, features)

        for mode, table in [("random_split", rs), ("leave_one_dataset_out", lodo)]:
            m, lo, hi = summarize(table["mae"])
            im, ilo, ihi = summarize(table["improvement"])
            results.append({
                "condition": label,
                "validation": mode,
                "mean_mae": m,
                "ci_low_mae": lo,
                "ci_high_mae": hi,
                "mean_improvement": im,
                "ci_low_improvement": ilo,
                "ci_high_improvement": ihi,
            })

        # Permutation importance for xi on random split.
        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=0.30,
            random_state=999,
            stratify=data["dataset_id"].astype(str),
        )
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        pipe = make_pipe(train, features)
        pipe.fit(train[features], train[TARGET])

        if "gtci_xi_dataset_confound" in features:
            pi = permutation_importance(
                pipe,
                test[features],
                test[TARGET],
                n_repeats=20,
                random_state=42,
                scoring="neg_mean_absolute_error",
            )
            idx = features.index("gtci_xi_dataset_confound")
            results.append({
                "condition": label,
                "validation": "permutation_importance",
                "mean_mae": np.nan,
                "ci_low_mae": np.nan,
                "ci_high_mae": np.nan,
                "mean_improvement": float(pi.importances_mean[idx]),
                "ci_low_improvement": float(np.quantile(pi.importances[idx], 0.025)),
                "ci_high_improvement": float(np.quantile(pi.importances[idx], 0.975)),
            })

        rs.to_csv(OUT / f"gtci_{label}_random_split.csv", index=False)
        lodo.to_csv(OUT / f"gtci_{label}_lodo.csv", index=False)

    summary = pd.DataFrame(results)
    summary.to_csv(OUT / "GTCI_SUMMARY.csv", index=False)
    return summary


def conformal_quantile(residuals, alpha=0.10):
    residuals = np.sort(np.asarray(residuals, dtype=float))
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(residuals[k])


def run_conformal(df):
    features = [c for c in BASE_FEATURES if c in df.columns]
    lodo_preds = []

    for ds in sorted(df["dataset_id"].astype(str).unique()):
        train = df[df["dataset_id"].astype(str) != ds].copy()
        test = df[df["dataset_id"].astype(str) == ds].copy()
        if len(test) < 2:
            continue

        pipe = make_pipe(train, features)
        pipe.fit(train[features], train[TARGET])
        pred = pipe.predict(test[features])
        residual = np.abs(test[TARGET].to_numpy() - pred)

        for i, (_, row) in enumerate(test.iterrows()):
            lodo_preds.append({
                "heldout_dataset": ds,
                "y": float(row[TARGET]),
                "pred": float(pred[i]),
                "residual": float(residual[i]),
            })

    pred_df = pd.DataFrame(lodo_preds)
    pred_df.to_csv(OUT / "lodo_residuals_for_conformal.csv", index=False)

    q90 = conformal_quantile(pred_df["residual"], alpha=0.10)
    q80 = conformal_quantile(pred_df["residual"], alpha=0.20)

    coverage90 = np.mean(pred_df["residual"] <= q90)
    coverage80 = np.mean(pred_df["residual"] <= q80)

    by_dataset = []
    for ds, g in pred_df.groupby("heldout_dataset"):
        by_dataset.append({
            "heldout_dataset": ds,
            "n": len(g),
            "coverage90_global": float(np.mean(g["residual"] <= q90)),
            "coverage80_global": float(np.mean(g["residual"] <= q80)),
            "median_residual": float(np.median(g["residual"])),
            "mean_residual": float(np.mean(g["residual"])),
        })

    summary = pd.DataFrame([
        {
            "method": "global_lodo_conformal",
            "alpha": 0.10,
            "q": q90,
            "interval_half_width": q90,
            "empirical_coverage": coverage90,
            "rank_safe_if_half_width_le_0.10": bool(q90 <= 0.10),
        },
        {
            "method": "global_lodo_conformal",
            "alpha": 0.20,
            "q": q80,
            "interval_half_width": q80,
            "empirical_coverage": coverage80,
            "rank_safe_if_half_width_le_0.10": bool(q80 <= 0.10),
        },
    ])

    summary.to_csv(OUT / "CONFORMAL_SUMMARY.csv", index=False)
    pd.DataFrame(by_dataset).to_csv(OUT / "CONFORMAL_BY_DATASET.csv", index=False)
    return summary, pd.DataFrame(by_dataset)


def transformed_features(train, pool, features):
    numeric = [c for c in features if c in train.columns and pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c in train.columns and c not in numeric]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_ohe()),
            ]), categorical),
        ],
        sparse_threshold=0.0,
    )
    x_train = pre.fit_transform(train[features])
    x_pool = pre.transform(pool[features])
    return np.asarray(x_train), np.asarray(x_pool)


def greedy_coreset(x_train, x_pool, k):
    if len(x_pool) == 0:
        return []

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(x_train)
    dists = nn.kneighbors(x_pool, return_distance=True)[0].ravel()

    selected = []
    current_dists = dists.copy()

    for _ in range(min(k, len(x_pool))):
        idx = int(np.argmax(current_dists))
        selected.append(idx)
        new_center = x_pool[idx:idx + 1]
        new_d = np.linalg.norm(x_pool - new_center, axis=1)
        current_dists = np.minimum(current_dists, new_d)

    return selected


def run_active_coreset(df):
    features = [c for c in BASE_FEATURES if c in df.columns]
    rows = []

    for ds in sorted(df["dataset_id"].astype(str).unique()):
        train0 = df[df["dataset_id"].astype(str) != ds].copy()
        pool = df[df["dataset_id"].astype(str) == ds].copy()
        if len(pool) < 5:
            continue

        x_train, x_pool = transformed_features(train0, pool, features)

        # Base model before acquisition.
        pipe0 = make_pipe(train0, features)
        pipe0.fit(train0[features], train0[TARGET])
        pred0 = pipe0.predict(pool[features])
        base_mae = mean_absolute_error(pool[TARGET], pred0)

        for k in K_VALUES:
            if k >= len(pool):
                continue

            selected = greedy_coreset(x_train, x_pool, k)
            selected_idx = pool.iloc[selected].index

            train_core = pd.concat([train0, pool.loc[selected_idx]], axis=0)
            test_remaining = pool.drop(index=selected_idx)

            pipe = make_pipe(train_core, features)
            pipe.fit(train_core[features], train_core[TARGET])
            pred = pipe.predict(test_remaining[features])
            core_mae = mean_absolute_error(test_remaining[TARGET], pred)

            random_maes = []
            for r in range(N_RANDOM):
                rand_idx = RNG.choice(pool.index.to_numpy(), size=k, replace=False)
                train_rand = pd.concat([train0, pool.loc[rand_idx]], axis=0)
                test_rand = pool.drop(index=rand_idx)

                pipe_r = make_pipe(train_rand, features, random_state=1000 + r)
                pipe_r.fit(train_rand[features], train_rand[TARGET])
                pred_r = pipe_r.predict(test_rand[features])
                random_maes.append(mean_absolute_error(test_rand[TARGET], pred_r))

            rows.append({
                "heldout_dataset": ds,
                "k": k,
                "n_pool": len(pool),
                "base_mae": base_mae,
                "coreset_mae": core_mae,
                "random_mae_mean": float(np.mean(random_maes)),
                "random_mae_ci_low": float(np.quantile(random_maes, 0.025)),
                "random_mae_ci_high": float(np.quantile(random_maes, 0.975)),
                "coreset_gap_closed_vs_base": float(base_mae - core_mae),
                "random_gap_closed_vs_base": float(base_mae - np.mean(random_maes)),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ACTIVE_CORESET_SUMMARY.csv", index=False)
    return out


def write_report(gtci, conformal, conformal_by_ds, active):
    lines = []
    lines.append("# GTCI + conformal + active acquisition report\n\n")

    lines.append("## Ground-Truth Confound Injection\n\n")
    lines.append(gtci.to_string(index=False))
    lines.append("\n\n")

    lines.append("## Dataset-Mondrian / LODO conformal calibration\n\n")
    lines.append(conformal.to_string(index=False))
    lines.append("\n\n")
    lines.append(conformal_by_ds.to_string(index=False))
    lines.append("\n\n")

    lines.append("## Retrospective core-set acquisition\n\n")
    lines.append(active.to_string(index=False))
    lines.append("\n\n")

    lines.append("## Interpretation guide\n\n")
    lines.append("- If GTCI improves random split but not LODO, the injected feature behaves as a known shortcut.\n")
    lines.append("- If conformal half-width is >0.10 retention points, rank-safe use should be rejected under a ±10-point rule.\n")
    lines.append("- If core-set acquisition closes more gap than random acquisition, this supports the prospective validation sampling plan.\n")

    (OUT / "GTCI_CONFORMAL_ACTIVE_REPORT.md").write_text("".join(lines), encoding="utf-8")


def main():
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET, "dataset_id"]).copy()
    df = df.reset_index(drop=True)

    print("Rows:", len(df))
    print("Running GTCI...")
    gtci = run_gtci(df)

    print("Running conformal calibration...")
    conformal, conformal_by_ds = run_conformal(df)

    print("Running retrospective core-set acquisition...")
    active = run_active_coreset(df)

    write_report(gtci, conformal, conformal_by_ds, active)

    print("Done. Outputs:")
    for p in sorted(OUT.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
