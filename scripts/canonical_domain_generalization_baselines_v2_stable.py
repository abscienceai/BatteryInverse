from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(".")
DATA = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
OUT = ROOT / "reports/canonical_domain_generalization_baselines_v2_stable"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = 42
DEVICE = "cpu"

def detect_col(cols, candidates, contains=None):
    for c in candidates:
        if c in cols:
            return c
    if contains:
        for c in cols:
            lc = c.lower()
            if all(x in lc for x in contains):
                return c
    return None

df = pd.read_csv(DATA)
cols = list(df.columns)

domain_col = detect_col(cols, ["dataset_id", "dataset", "dataset_name", "source_dataset"], contains=["dataset"])
target_col = detect_col(cols, ["final_capacity_retention", "target_original", "retention"], contains=["retention"])

if domain_col is None:
    raise RuntimeError("Domain/dataset column not detected.")
if target_col is None:
    raise RuntimeError("Target column not detected.")

work = df[df[target_col].notna()].copy().reset_index(drop=True)
counts = work[domain_col].value_counts()
eligible_domains = counts[counts >= 3].index.tolist()
work = work[work[domain_col].isin(eligible_domains)].reset_index(drop=True)

domains = work[domain_col].astype(str).values
y = work[target_col].astype(float).values

exclude_keywords = [
    "target", "retention", "life", "cycle_life", "rul",
    "dataset", "source", "cell_id", "id", "name"
]

feature_cols = []
for c in work.columns:
    lc = c.lower()
    if c in [domain_col, target_col]:
        continue
    if any(k in lc for k in exclude_keywords):
        continue
    feature_cols.append(c)

Xdf = work[feature_cols].copy()
num_cols = [c for c in Xdf.columns if pd.api.types.is_numeric_dtype(Xdf[c])]
cat_cols = [c for c in Xdf.columns if c not in num_cols]

num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xdf[num_cols]), columns=num_cols) if num_cols else pd.DataFrame(index=Xdf.index)
cat = pd.get_dummies(Xdf[cat_cols].astype(str).fillna("missing"), dummy_na=False) if cat_cols else pd.DataFrame(index=Xdf.index)

X = pd.concat([num, cat], axis=1).astype(float).values.astype(np.float32)

print("Detected domain_col:", domain_col)
print("Detected target_col:", target_col)
print("Rows:", len(work))
print("Domains:", len(eligible_domains), sorted(map(str, eligible_domains)))
print("Feature matrix:", X.shape)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class MLPDG(nn.Module):
    def __init__(self, in_dim, n_domains, hidden=64, z_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, z_dim),
            nn.ReLU()
        )
        self.regressor = nn.Linear(z_dim, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains)
        )

    def forward(self, x, grl_lambda=0.0, detach_domain=False):
        z = self.encoder(x)
        pred = self.regressor(z).squeeze(-1)
        dz = grad_reverse(z, grl_lambda) if grl_lambda > 0 else (z.detach() if detach_domain else z)
        dlogits = self.domain_head(dz)
        return pred, dlogits, z

def standardize_X(Xtr_raw, Xte_raw):
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xtr_raw)).astype(np.float32)
    Xte = sc.transform(imp.transform(Xte_raw)).astype(np.float32)
    return Xtr, Xte

def train_model(method, Xtr, ytr_z, env_labels, seed=0, epochs=500, lr=5e-4, irm_lambda=1.0, dann_lambda=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = sorted(pd.Series(env_labels).astype(str).unique())
    env_to_id = {e:i for i,e in enumerate(envs)}
    eid = np.array([env_to_id[str(e)] for e in env_labels], dtype=np.int64)

    model = MLPDG(Xtr.shape[1], len(envs)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr_z, dtype=torch.float32, device=DEVICE)
    et = torch.tensor(eid, dtype=torch.long, device=DEVICE)

    for ep in range(epochs):
        opt.zero_grad()

        if method == "erm":
            pred, _, _ = model(Xt, detach_domain=True)
            loss = F.huber_loss(pred, yt, delta=1.0)

        elif method == "dann":
            pred, dlogits, _ = model(Xt, grl_lambda=dann_lambda)
            pred_loss = F.huber_loss(pred, yt, delta=1.0)
            dom_loss = F.cross_entropy(dlogits, et)
            loss = pred_loss + 0.2 * dom_loss

        elif method == "irm":
            scale = torch.tensor(1.0, device=DEVICE, requires_grad=True)
            env_losses = []
            penalties = []
            for e in torch.unique(et):
                mask = et == e
                if mask.sum() < 2:
                    continue
                pred, _, _ = model(Xt[mask], detach_domain=True)
                env_loss = F.huber_loss(pred * scale, yt[mask], delta=1.0)
                grad = torch.autograd.grad(env_loss, [scale], create_graph=True)[0]
                env_losses.append(env_loss)
                penalties.append(torch.sum(grad ** 2))
            if len(env_losses) == 0:
                pred, _, _ = model(Xt, detach_domain=True)
                loss = F.huber_loss(pred, yt, delta=1.0)
            else:
                lam = irm_lambda if ep > 100 else 0.1
                loss = torch.stack(env_losses).mean() + lam * torch.stack(penalties).mean()
        else:
            raise ValueError(method)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

    return model

def predict_z(model, Xarr):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xarr, dtype=torch.float32, device=DEVICE)
        pred, _, z = model(Xt, detach_domain=True)
    return pred.cpu().numpy(), z.cpu().numpy()

def rep_dataset_ba(Z, labels, seed=0):
    labels = np.asarray(labels).astype(str)
    vc = pd.Series(labels).value_counts()
    keep = vc[vc >= 4].index
    mask = pd.Series(labels).isin(keep).values
    Z = Z[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 3:
        return np.nan

    bas = []
    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=seed)
    for i, (tr, te) in enumerate(sss.split(Z, labels)):
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=seed+i,
            n_jobs=-1,
            max_features="sqrt",
            class_weight="balanced_subsample"
        )
        clf.fit(Z[tr], labels[tr])
        pred = clf.predict(Z[te])
        bas.append(balanced_accuracy_score(labels[te], pred))
    return float(np.mean(bas))

methods = ["erm", "irm", "dann"]
seeds = [0, 1, 2]

records = []

for heldout in sorted(pd.Series(domains).unique()):
    test_mask = domains == heldout
    train_mask = ~test_mask

    if test_mask.sum() < 3:
        continue

    Xtr_raw, Xte_raw = X[train_mask], X[test_mask]
    ytr, yte = y[train_mask], y[test_mask]
    env_train = domains[train_mask]

    Xtr, Xte = standardize_X(Xtr_raw, Xte_raw)

    y_mean = float(np.mean(ytr))
    y_std = float(np.std(ytr) + 1e-8)
    ytr_z = (ytr - y_mean) / y_std

    clip_lo = float(np.quantile(ytr, 0.01))
    clip_hi = float(np.quantile(ytr, 0.99))

    baseline = np.full_like(yte, np.median(ytr), dtype=float)
    baseline_mae = mean_absolute_error(yte, baseline)

    for method in methods:
        preds = []
        rep_bas = []

        for seed in seeds:
            model = train_model(method, Xtr, ytr_z, env_train, seed=seed)

            pz, _ = predict_z(model, Xte)
            pred = pz * y_std + y_mean
            pred_clipped = np.clip(pred, clip_lo, clip_hi)
            preds.append(pred_clipped)

            Xall = np.vstack([Xtr, Xte])
            labels_all = np.concatenate([domains[train_mask], domains[test_mask]])
            _, z_all = predict_z(model, Xall)
            rep_bas.append(rep_dataset_ba(z_all, labels_all, seed=seed))

        pred_mean = np.mean(np.vstack(preds), axis=0)

        mae = mean_absolute_error(yte, pred_mean)
        r2 = r2_score(yte, pred_mean) if len(yte) > 2 else np.nan

        records.append({
            "heldout_dataset": heldout,
            "method": method,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "model_mae": float(mae),
            "train_median_baseline_mae": float(baseline_mae),
            "mae_improvement_baseline_minus_model": float(baseline_mae - mae),
            "model_r2": float(r2),
            "representation_dataset_balanced_accuracy_mean": float(np.nanmean(rep_bas)),
            "clip_low": clip_lo,
            "clip_high": clip_hi
        })

res = pd.DataFrame(records)
res.to_csv(OUT / "canonical_dg_lodo_results_v2_stable.csv", index=False)

summary = res.groupby("method").agg(
    mean_lodo_mae=("model_mae", "mean"),
    median_lodo_mae=("model_mae", "median"),
    mean_baseline_mae=("train_median_baseline_mae", "mean"),
    median_baseline_mae=("train_median_baseline_mae", "median"),
    mean_improvement=("mae_improvement_baseline_minus_model", "mean"),
    median_improvement=("mae_improvement_baseline_minus_model", "median"),
    mean_representation_dataset_ba=("representation_dataset_balanced_accuracy_mean", "mean"),
    median_representation_dataset_ba=("representation_dataset_balanced_accuracy_mean", "median"),
    n_folds=("heldout_dataset", "nunique")
).reset_index()

summary.to_csv(OUT / "canonical_dg_summary_v2_stable.csv", index=False)

intervals = []
for method, g in res.groupby("method"):
    for col in ["model_mae", "mae_improvement_baseline_minus_model", "representation_dataset_balanced_accuracy_mean"]:
        vals = g[col].dropna().astype(float).values
        intervals.append({
            "method": method,
            "quantity": col,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p2p5": float(np.quantile(vals, 0.025)),
            "p97p5": float(np.quantile(vals, 0.975)),
            "n": int(len(vals)),
            "interval_type": "empirical_heldout_dataset_interval"
        })

interval_df = pd.DataFrame(intervals)
interval_df.to_csv(OUT / "canonical_dg_empirical_intervals_v2_stable.csv", index=False)

order = ["erm", "irm", "dann"]

plt.figure(figsize=(6,4))
plt.bar(order, [summary.loc[summary.method==m, "mean_lodo_mae"].iloc[0] for m in order])
plt.ylabel("Mean LODO MAE")
plt.title("Stable canonical DG baselines")
plt.tight_layout()
plt.savefig(FIG / "canonical_dg_mean_lodo_mae_v2_stable.png", dpi=300)
plt.close()

plt.figure(figsize=(6,4))
plt.bar(order, [summary.loc[summary.method==m, "mean_representation_dataset_ba"].iloc[0] for m in order])
plt.ylabel("Dataset BA from learned representation")
plt.title("Representation closure audit")
plt.tight_layout()
plt.savefig(FIG / "canonical_dg_representation_closure_v2_stable.png", dpi=300)
plt.close()

report = []
report.append("# Stable canonical domain-generalization baseline audit\n\n")
report.append("This experiment compares small neural ERM, IRM and DANN-style baselines under leave-one-dataset-out battery validation using target standardization, Huber loss and prediction clipping to the training target range. The goal is to test whether canonical mitigation objectives remove dataset-prior shortcut risk under a fairer stable neural setup.\n\n")
report.append("## Detected setup\n\n")
report.append(f"- Rows: {len(work)}\n")
report.append(f"- Domain column: {domain_col}\n")
report.append(f"- Target column: {target_col}\n")
report.append(f"- Feature matrix shape: {X.shape}\n")
report.append(f"- Held-out datasets: {len(eligible_domains)}\n\n")
report.append("## Summary\n\n")
for _, r in summary.iterrows():
    report.append(
        f"- {r['method']}: mean LODO MAE = {r['mean_lodo_mae']:.3f}; "
        f"median LODO MAE = {r['median_lodo_mae']:.3f}; "
        f"mean improvement over train median = {r['mean_improvement']:.3f}; "
        f"learned-representation dataset BA = {r['mean_representation_dataset_ba']:.3f}\n"
    )
report.append("\n## Interpretation\n\n")
report.append("If ERM, IRM and DANN remain close in LODO error and their learned representations remain highly dataset-identifiable, the result supports a conservative conclusion: canonical domain-generalization objectives do not by themselves certify transferable scientific learning. They require the same closure audits used elsewhere in BatteryInverse.\n")

(OUT / "CANONICAL_DG_BASELINE_REPORT_V2_STABLE.md").write_text("".join(report), encoding="utf-8")

(OUT / "CANONICAL_DG_BASELINE_SUMMARY_V2_STABLE.json").write_text(
    json.dumps({
        "domain_col": domain_col,
        "target_col": target_col,
        "n_rows": int(len(work)),
        "n_domains": int(len(eligible_domains)),
        "n_features": int(X.shape[1]),
        "summary": summary.to_dict(orient="records"),
        "intervals": interval_df.to_dict(orient="records")
    }, indent=2),
    encoding="utf-8"
)

print("".join(report))
print("Saved outputs to:", OUT)
