from pathlib import Path
import json
import urllib.request
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")

ROOT = Path(".")
OUT = ROOT / "reports" / "cross_domain_moleculenet_esol_audit"
DATA_DIR = ROOT / "data" / "secondary_domain" / "moleculenet_esol"
FIG_DIR = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
CSV_PATH = DATA_DIR / "delaney-processed.csv"

RNG = 42
rng = np.random.default_rng(RNG)

def download_if_needed():
    if CSV_PATH.exists():
        print(f"Using existing CSV: {CSV_PATH}")
        return
    print(f"Downloading ESOL CSV from {CSV_URL}")
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)
    print(f"Saved: {CSV_PATH}")

def detect_cols(df):
    smiles_col = None
    target_col = None

    for c in df.columns:
        if c.lower() == "smiles":
            smiles_col = c

    for c in df.columns:
        lc = c.lower()
        if "measured" in lc and "solubility" in lc:
            target_col = c

    if smiles_col is None:
        raise RuntimeError(f"Could not find SMILES column. Columns={list(df.columns)}")
    if target_col is None:
        raise RuntimeError(f"Could not find measured solubility target column. Columns={list(df.columns)}")

    return smiles_col, target_col

def mol_from_smiles(smi):
    try:
        return Chem.MolFromSmiles(str(smi))
    except Exception:
        return None

def scaffold_from_mol(mol):
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaf is None or scaf == "":
            return "acyclic_or_no_scaffold"
        return scaf
    except Exception:
        return None

def morgan_fp(mol, n_bits=1024, radius=2):
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)
    except Exception:
        return None

def bootstrap_ci(vals, n_boot=5000):
    vals = np.asarray(pd.Series(vals).dropna().astype(float))
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    if len(vals) == 1:
        return (float(vals[0]), np.nan, np.nan)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

def empirical_interval(vals):
    vals = np.asarray(pd.Series(vals).dropna().astype(float))
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def nearest_distance(X_train, X_test):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(Xtr)
    d, _ = nn.kneighbors(Xte)
    return float(d.mean())

download_if_needed()

df0 = pd.read_csv(CSV_PATH)
smiles_col, target_col = detect_cols(df0)

rows = []
fps = []

for idx, row in df0.iterrows():
    smi = row[smiles_col]
    y = row[target_col]
    mol = mol_from_smiles(smi)
    if mol is None:
        continue
    scaf = scaffold_from_mol(mol)
    fp = morgan_fp(mol)
    if scaf is None or fp is None:
        continue
    rows.append({
        "row_id": idx,
        "smiles": smi,
        "target": float(y),
        "scaffold": scaf,
    })
    fps.append(fp)

df = pd.DataFrame(rows)
X = np.vstack(fps).astype(np.float32)
y = df["target"].values.astype(float)

# Keep scaffold domains with enough samples.
min_domain_n = 15
scaf_counts = df["scaffold"].value_counts()
eligible_scaffolds = scaf_counts[scaf_counts >= min_domain_n].index.tolist()

work_mask = df["scaffold"].isin(eligible_scaffolds).values
dfw = df.loc[work_mask].reset_index(drop=True)
Xw = X[work_mask]
yw = dfw["target"].values.astype(float)
domains = dfw["scaffold"].values

domain_counts = dfw["scaffold"].value_counts().rename_axis("scaffold").reset_index(name="n")
domain_counts.to_csv(OUT / "esol_scaffold_domain_counts.csv", index=False)

print(f"Original molecules: {len(df)}")
print(f"Eligible scaffold-domain molecules: {len(dfw)}")
print(f"Eligible scaffold domains n>={min_domain_n}: {len(eligible_scaffolds)}")

# ---------------------------------------------------------------------
# 1. Scaffold-domain separability
# ---------------------------------------------------------------------
sep_records = []
if len(eligible_scaffolds) >= 3:
    splitter = StratifiedShuffleSplit(n_splits=30, test_size=0.25, random_state=RNG)
    for split_id, (tr, te) in enumerate(splitter.split(Xw, domains)):
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=RNG + split_id,
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_features="sqrt",
        )
        clf.fit(Xw[tr], domains[tr])
        pred = clf.predict(Xw[te])
        ba = balanced_accuracy_score(domains[te], pred)
        sep_records.append({"split": split_id, "balanced_accuracy": ba})

sep_df = pd.DataFrame(sep_records)
sep_df.to_csv(OUT / "esol_scaffold_separability_splits.csv", index=False)

sep_mean, sep_lo, sep_hi = bootstrap_ci(sep_df["balanced_accuracy"]) if len(sep_df) else (np.nan, np.nan, np.nan)

# ---------------------------------------------------------------------
# 2. Random split regression and explicit scaffold-prior baseline
# ---------------------------------------------------------------------
rand_records = []
rs = ShuffleSplit(n_splits=30, test_size=0.25, random_state=RNG)

for split_id, (tr, te) in enumerate(rs.split(Xw)):
    model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=RNG + split_id,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    model.fit(Xw[tr], yw[tr])
    pred = model.predict(Xw[te])

    global_median = float(np.median(yw[tr]))
    global_pred = np.full_like(yw[te], global_median, dtype=float)

    train_df = pd.DataFrame({"domain": domains[tr], "y": yw[tr]})
    med_map = train_df.groupby("domain")["y"].median().to_dict()
    scaffold_pred = np.array([med_map.get(d, global_median) for d in domains[te]], dtype=float)

    rand_records.append({
        "split": split_id,
        "model_mae": mean_absolute_error(yw[te], pred),
        "global_median_mae": mean_absolute_error(yw[te], global_pred),
        "scaffold_median_mae": mean_absolute_error(yw[te], scaffold_pred),
        "model_r2": r2_score(yw[te], pred),
        "n_train": len(tr),
        "n_test": len(te),
    })

rand_df = pd.DataFrame(rand_records)
rand_df["model_improvement_over_global_median"] = rand_df["global_median_mae"] - rand_df["model_mae"]
rand_df["scaffold_prior_improvement_over_global_median"] = rand_df["global_median_mae"] - rand_df["scaffold_median_mae"]
rand_df.to_csv(OUT / "esol_random_split_regression_and_prior.csv", index=False)

rand_model_mae = bootstrap_ci(rand_df["model_mae"])
rand_global_mae = bootstrap_ci(rand_df["global_median_mae"])
rand_scaf_mae = bootstrap_ci(rand_df["scaffold_median_mae"])
rand_prior_gain = bootstrap_ci(rand_df["scaffold_prior_improvement_over_global_median"])

# ---------------------------------------------------------------------
# 3. Leave-one-scaffold-domain-out transfer
# ---------------------------------------------------------------------
lodo_records = []
for heldout in eligible_scaffolds:
    test_mask = domains == heldout
    train_mask = ~test_mask
    if test_mask.sum() < min_domain_n or train_mask.sum() < 50:
        continue

    Xtr, Xte = Xw[train_mask], Xw[test_mask]
    ytr, yte = yw[train_mask], yw[test_mask]

    model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=RNG,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    train_global = float(np.median(ytr))
    baseline_pred = np.full_like(yte, train_global, dtype=float)

    mae_model = mean_absolute_error(yte, pred)
    mae_baseline = mean_absolute_error(yte, baseline_pred)

    dist = nearest_distance(Xtr, Xte)

    lodo_records.append({
        "heldout_scaffold": heldout,
        "n_test": int(test_mask.sum()),
        "model_mae": mae_model,
        "train_global_median_mae": mae_baseline,
        "mae_improvement_baseline_minus_model": mae_baseline - mae_model,
        "model_r2": r2_score(yte, pred) if len(yte) > 2 else np.nan,
        "mean_nearest_train_distance": dist,
        "heldout_target_mean": float(np.mean(yte)),
        "train_target_mean": float(np.mean(ytr)),
        "target_mean_shift_abs": float(abs(np.mean(yte) - np.mean(ytr))),
    })

lodo_df = pd.DataFrame(lodo_records)
if len(lodo_df):
    med_d = np.median(lodo_df["mean_nearest_train_distance"])
    med_e = np.median(lodo_df["model_mae"])
    lodo_df["scaled_tgi"] = (lodo_df["mean_nearest_train_distance"] / med_d) * (lodo_df["model_mae"] / med_e)
else:
    lodo_df["scaled_tgi"] = []

lodo_df.to_csv(OUT / "esol_leave_one_scaffold_domain_out.csv", index=False)

lodo_model_mae = empirical_interval(lodo_df["model_mae"])
lodo_baseline_mae = empirical_interval(lodo_df["train_global_median_mae"])
lodo_improvement = empirical_interval(lodo_df["mae_improvement_baseline_minus_model"])

# Random vs LODO inflation
random_mae_mean = rand_model_mae[0]
lodo_mae_mean = lodo_model_mae[0]
inflation_ratio = float(lodo_mae_mean / random_mae_mean) if random_mae_mean and not np.isnan(random_mae_mean) else np.nan

# ---------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # Separability
    plt.figure(figsize=(5.5, 4))
    plt.hist(sep_df["balanced_accuracy"], bins=10)
    plt.axvline(sep_mean, linestyle="--")
    plt.xlabel("Balanced accuracy")
    plt.ylabel("Count")
    plt.title("ESOL scaffold-domain separability")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "esol_scaffold_domain_separability.png", dpi=300)
    plt.close()

    # Random vs LODO MAE
    plt.figure(figsize=(5.5, 4))
    labels = ["Random split", "Leave-one-scaffold-domain-out"]
    vals = [random_mae_mean, lodo_mae_mean]
    plt.bar(labels, vals)
    plt.ylabel("MAE")
    plt.title("ESOL random vs scaffold-domain transfer")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "esol_random_vs_lodo_mae.png", dpi=300)
    plt.close()

    # TGI scatter
    plt.figure(figsize=(5.5, 4))
    plt.scatter(lodo_df["mean_nearest_train_distance"], lodo_df["model_mae"])
    for _, r in lodo_df.iterrows():
        if r["scaled_tgi"] >= lodo_df["scaled_tgi"].quantile(0.80):
            plt.annotate(str(r["heldout_scaffold"])[:12], (r["mean_nearest_train_distance"], r["model_mae"]), fontsize=7)
    plt.xlabel("Mean nearest-train fingerprint distance")
    plt.ylabel("LODO MAE")
    plt.title("ESOL scaffold-domain TGI diagnostic")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "esol_tgi_distance_vs_error.png", dpi=300)
    plt.close()
except Exception as e:
    print("Figure generation failed:", e)

# ---------------------------------------------------------------------
# 5. Summary report
# ---------------------------------------------------------------------
summary = {
    "dataset": "MoleculeNet ESOL / Delaney",
    "n_original_molecules": int(len(df)),
    "n_eligible_molecules": int(len(dfw)),
    "min_domain_n": int(min_domain_n),
    "n_scaffold_domains": int(len(eligible_scaffolds)),
    "scaffold_domain_separability_balanced_accuracy": {
        "mean": sep_mean,
        "ci95_low": sep_lo,
        "ci95_high": sep_hi,
        "basis": "30 stratified scaffold-domain splits",
    },
    "random_split_model_mae": {
        "mean": rand_model_mae[0],
        "ci95_low": rand_model_mae[1],
        "ci95_high": rand_model_mae[2],
    },
    "random_split_global_median_mae": {
        "mean": rand_global_mae[0],
        "ci95_low": rand_global_mae[1],
        "ci95_high": rand_global_mae[2],
    },
    "random_split_scaffold_median_prior_mae": {
        "mean": rand_scaf_mae[0],
        "ci95_low": rand_scaf_mae[1],
        "ci95_high": rand_scaf_mae[2],
    },
    "random_split_scaffold_prior_gain_over_global_median": {
        "mean": rand_prior_gain[0],
        "ci95_low": rand_prior_gain[1],
        "ci95_high": rand_prior_gain[2],
    },
    "lodo_model_mae_empirical_domain_interval": {
        "mean": lodo_model_mae[0],
        "p2p5": lodo_model_mae[1],
        "p97p5": lodo_model_mae[2],
    },
    "lodo_global_median_baseline_mae_empirical_domain_interval": {
        "mean": lodo_baseline_mae[0],
        "p2p5": lodo_baseline_mae[1],
        "p97p5": lodo_baseline_mae[2],
    },
    "lodo_improvement_baseline_minus_model_empirical_domain_interval": {
        "mean": lodo_improvement[0],
        "p2p5": lodo_improvement[1],
        "p97p5": lodo_improvement[2],
    },
    "random_to_lodo_mae_ratio": inflation_ratio,
    "top_tgi_domains": lodo_df.sort_values("scaled_tgi", ascending=False).head(10).to_dict(orient="records") if len(lodo_df) else [],
}

(OUT / "ESOL_CROSS_DOMAIN_AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

def interval_text(tup, d=3):
    return f"{tup[0]:.{d}f} [{tup[1]:.{d}f} to {tup[2]:.{d}f}]"

report = []
report.append("# Cross-domain BatteryInverse audit: MoleculeNet ESOL\n\n")
report.append("This pilot applies the BatteryInverse audit chain to a second scientific ML domain: molecular property prediction.\n\n")
report.append("## Domain definition\n\n")
report.append("- Dataset: MoleculeNet ESOL / Delaney solubility\n")
report.append("- Features: 1024-bit Morgan fingerprints\n")
report.append("- Target: measured log solubility\n")
report.append("- Domain label: Bemis-Murcko scaffold domain\n")
report.append(f"- Eligible molecules: {len(dfw)}\n")
report.append(f"- Scaffold domains with n >= {min_domain_n}: {len(eligible_scaffolds)}\n\n")

report.append("## Headline diagnostics\n\n")
report.append(f"- Scaffold-domain separability balanced accuracy: {sep_mean:.3f} [{sep_lo:.3f} to {sep_hi:.3f}]\n")
report.append(f"- Random-split model MAE: {interval_text(rand_model_mae)}\n")
report.append(f"- LODO scaffold-domain model MAE: {interval_text(lodo_model_mae)} empirical domain interval\n")
report.append(f"- Random-to-LODO MAE ratio: {inflation_ratio:.3f}\n")
report.append(f"- Random-split global-median prior MAE: {interval_text(rand_global_mae)}\n")
report.append(f"- Random-split scaffold-median prior MAE: {interval_text(rand_scaf_mae)}\n")
report.append(f"- Scaffold-prior gain over global median: {interval_text(rand_prior_gain)}\n")
report.append(f"- LODO improvement over train-global-median baseline: {interval_text(lodo_improvement)} empirical domain interval\n\n")

report.append("## Interpretation\n\n")
report.append("The ESOL pilot is a cross-domain generalization check, not a new molecular-property benchmark. ")
report.append("It asks whether the same audit chain can expose scaffold-domain structure in a second scientific ML setting. ")
report.append("If scaffold identity is recoverable from fingerprints and random-split performance differs from leave-one-scaffold-domain-out behavior, ")
report.append("then the BatteryInverse audit logic is not battery-specific, even though the main case study remains battery degradation.\n\n")

report.append("## Output files\n\n")
report.append("- esol_scaffold_separability_splits.csv\n")
report.append("- esol_random_split_regression_and_prior.csv\n")
report.append("- esol_leave_one_scaffold_domain_out.csv\n")
report.append("- ESOL_CROSS_DOMAIN_AUDIT_SUMMARY.json\n")
report.append("- figures/esol_scaffold_domain_separability.png\n")
report.append("- figures/esol_random_vs_lodo_mae.png\n")
report.append("- figures/esol_tgi_distance_vs_error.png\n")

(OUT / "ESOL_CROSS_DOMAIN_AUDIT_REPORT.md").write_text("".join(report), encoding="utf-8")

print("\n".join(report))
print("\nSaved outputs to:", OUT)
