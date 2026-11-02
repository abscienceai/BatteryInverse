from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv")
OUT_DIR = Path("reports/batteryinverse_net_confoundadam_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "final_capacity_retention"

SEQ_COLS_CANDIDATE = [
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
]

META_NUM_CANDIDATE = [
    "nominal_capacity_Ah",
    "min_voltage_V",
    "max_voltage_V",
    "ambient_temperature_C",
]

META_CAT_CANDIDATE = [
    "chemistry",
    "form_factor",
    "anode_material",
    "cathode_material",
]


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class BatteryTorchDataset(Dataset):
    def __init__(self, seq, meta, y_scaled, y_raw, domain, row_id):
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.meta = torch.tensor(meta, dtype=torch.float32)
        self.y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32)
        self.domain = torch.tensor(domain, dtype=torch.long)
        self.row_id = torch.tensor(row_id, dtype=torch.long)

    def __len__(self):
        return len(self.y_scaled)

    def __getitem__(self, idx):
        return {
            "seq": self.seq[idx],
            "meta": self.meta[idx],
            "y_scaled": self.y_scaled[idx],
            "y_raw": self.y_raw[idx],
            "domain": self.domain[idx],
            "row_id": self.row_id[idx],
        }


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)


class BatteryInverseNet(nn.Module):
    def __init__(self, seq_len, meta_dim, n_domains, hidden_dim=64, z_dim=32):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len),
        )

        self.predictor = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_domains),
        )

    def encode(self, seq, meta):
        _, h = self.gru(seq)
        h = h[-1]
        if meta.shape[1] > 0:
            h = torch.cat([h, meta], dim=1)
        return self.encoder(h)

    def forward(self, seq, meta, grl_alpha=0.0):
        z = self.encode(seq, meta)
        recon = self.decoder(z)
        pred_scaled = self.predictor(z).squeeze(1)
        domain_logits = self.domain_head(grad_reverse(z, grl_alpha))
        return {
            "z": z,
            "recon": recon,
            "pred_scaled": pred_scaled,
            "domain_logits": domain_logits,
        }


class StableConfoundEMA:
    """
    Stable ConfoundAdam-style per-sample weighting.

    Difference from v0:
    - Uses non-negative robust unfamiliarity risk.
    - Avoids negative log-loss denominator instability.
    - Uses a higher minimum weight to prevent silent domain deletion.
    - Stores normalized weights, not only raw clipped weights.
    """

    def __init__(self, n_samples, beta_u=0.90, beta_s=0.99, eps=1e-8, min_weight=0.20):
        self.n_samples = n_samples
        self.beta_u = beta_u
        self.beta_s = beta_s
        self.eps = eps
        self.min_weight = min_weight

        self.u = np.zeros(n_samples, dtype=np.float64)
        self.s = np.zeros(n_samples, dtype=np.float64)
        self.t = np.zeros(n_samples, dtype=np.int64)

        self.last_w_raw = np.ones(n_samples, dtype=np.float64)
        self.last_w_norm = np.ones(n_samples, dtype=np.float64)
        self.last_unfamiliarity = np.zeros(n_samples, dtype=np.float64)
        self.last_risk = np.zeros(n_samples, dtype=np.float64)

    def update_and_weight(self, row_ids, unfamiliarity):
        row_ids = np.asarray(row_ids, dtype=np.int64)
        unfamiliarity = np.asarray(unfamiliarity, dtype=np.float64)

        center = np.median(unfamiliarity)
        mad = np.median(np.abs(unfamiliarity - center)) + self.eps

        # Positive risk. Familiar samples near/below median get moderate risk;
        # high reconstruction-unfamiliarity samples get higher risk.
        risk = np.log1p(np.exp((unfamiliarity - center) / mad))

        raw_weights = []
        for idx, r_now, u_now in zip(row_ids, risk, unfamiliarity):
            self.t[idx] += 1
            t = self.t[idx]

            u_new = self.beta_u * self.u[idx] + (1.0 - self.beta_u) * r_now
            s_new = self.beta_s * self.s[idx] + (1.0 - self.beta_s) * (r_now - u_new) ** 2

            self.u[idx] = u_new
            self.s[idx] = s_new

            u_hat = u_new / (1.0 - self.beta_u ** t)
            s_hat = s_new / (1.0 - self.beta_s ** t)

            denom = 1.0 + u_hat / math.sqrt(s_hat + self.eps)
            w = 1.0 / denom
            if not np.isfinite(w):
                w = 1.0

            w = max(self.min_weight, min(3.0, w))

            self.last_w_raw[idx] = w
            self.last_unfamiliarity[idx] = u_now
            self.last_risk[idx] = r_now
            raw_weights.append(w)

        raw_weights = np.asarray(raw_weights, dtype=np.float64)
        norm_weights = raw_weights / max(raw_weights.mean(), self.eps)

        for idx, w_norm in zip(row_ids, norm_weights):
            self.last_w_norm[idx] = w_norm

        return norm_weights.astype(np.float32)


def prepare_fold_data(df, train_mask, test_mask, seq_cols, meta_num, meta_cat, use_metadata):
    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()

    seq_train_raw = train[seq_cols].replace([np.inf, -np.inf], np.nan)
    med = seq_train_raw.median()
    seq_train = seq_train_raw.fillna(med).to_numpy(dtype=np.float32)
    seq_test = test[seq_cols].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=np.float32)

    seq_mean = seq_train.mean(axis=0, keepdims=True)
    seq_std = seq_train.std(axis=0, keepdims=True)
    seq_std[seq_std < 1e-6] = 1.0

    seq_train = (seq_train - seq_mean) / seq_std
    seq_test = (seq_test - seq_mean) / seq_std

    seq_train = seq_train[:, :, None]
    seq_test = seq_test[:, :, None]

    if use_metadata:
        numeric_cols = [c for c in meta_num if c in df.columns]
        categorical_cols = [c for c in meta_cat if c in df.columns]
    else:
        numeric_cols = []
        categorical_cols = []

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_onehot()),
                ]),
                categorical_cols,
            )
        )

    if transformers:
        pre = ColumnTransformer(transformers, sparse_threshold=0.0)
        meta_train = pre.fit_transform(train)
        meta_test = pre.transform(test)
    else:
        pre = None
        meta_train = np.zeros((len(train), 0), dtype=np.float32)
        meta_test = np.zeros((len(test), 0), dtype=np.float32)

    meta_train = np.asarray(meta_train, dtype=np.float32)
    meta_test = np.asarray(meta_test, dtype=np.float32)

    y_train_raw = train[TARGET].to_numpy(dtype=np.float32)
    y_test_raw = test[TARGET].to_numpy(dtype=np.float32)

    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if y_std < 1e-6:
        y_std = 1.0

    y_train_scaled = (y_train_raw - y_mean) / y_std
    y_test_scaled = (y_test_raw - y_mean) / y_std

    domain_values = sorted(train["dataset_id"].astype(str).unique())
    domain_map = {d: i for i, d in enumerate(domain_values)}

    d_train = train["dataset_id"].astype(str).map(domain_map).fillna(0).to_numpy(dtype=np.int64)
    d_test = np.zeros(len(test), dtype=np.int64)

    return {
        "train": train,
        "test": test,
        "seq_train": seq_train,
        "seq_test": seq_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "y_train_scaled": y_train_scaled,
        "y_test_scaled": y_test_scaled,
        "y_mean": y_mean,
        "y_std": y_std,
        "d_train": d_train,
        "d_test": d_test,
        "row_train": train["row_id"].to_numpy(dtype=np.int64),
        "row_test": test["row_id"].to_numpy(dtype=np.int64),
        "n_domains": len(domain_values),
        "domain_map": domain_map,
    }


def grl_schedule(epoch, epochs, max_alpha):
    p = epoch / max(1, epochs)
    return float(max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


def train_model(fold, n_total, mode, args, device):
    train_ds = BatteryTorchDataset(
        fold["seq_train"],
        fold["meta_train"],
        fold["y_train_scaled"],
        fold["y_train_raw"],
        fold["d_train"],
        fold["row_train"],
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = BatteryInverseNet(
        seq_len=fold["seq_train"].shape[1],
        meta_dim=fold["meta_train"].shape[1],
        n_domains=max(1, fold["n_domains"]),
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = StableConfoundEMA(n_total, min_weight=args.min_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        alpha = grl_schedule(epoch, args.epochs, args.grl_alpha) if "adv" in mode else 0.0

        losses = []
        for batch in loader:
            seq = batch["seq"].to(device)
            meta = batch["meta"].to(device)
            y_scaled = batch["y_scaled"].to(device)
            domain = batch["domain"].to(device)
            row_id = batch["row_id"].cpu().numpy()

            out = model(seq, meta, grl_alpha=alpha)
            recon_target = seq.squeeze(-1)

            recon_per = F.mse_loss(out["recon"], recon_target, reduction="none").mean(dim=1)
            pred_per = torch.abs(out["pred_scaled"] - y_scaled)

            unfamiliarity = torch.log(recon_per.detach() + 1e-8).cpu().numpy()

            if "confound" in mode:
                weights_np = ema.update_and_weight(row_id, unfamiliarity)
                weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
            else:
                weights = torch.ones_like(pred_per)

            loss_pred = (weights * pred_per).mean()
            loss_recon = recon_per.mean()
            loss = loss_pred + args.lambda_recon * loss_recon

            if "adv" in mode and fold["n_domains"] > 1:
                loss_adv = F.cross_entropy(out["domain_logits"], domain)
                loss = loss + args.gamma_adv * loss_adv

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        if epoch in [1, args.epochs] or epoch % 25 == 0:
            print(f"  {mode} epoch {epoch:03d}/{args.epochs}: loss={np.mean(losses):.4f}, alpha={alpha:.3f}")

    return model, ema


@torch.no_grad()
def predict_model(model, fold, split, device):
    if split == "train":
        seq = fold["seq_train"]
        meta = fold["meta_train"]
        y_raw = fold["y_train_raw"]
    else:
        seq = fold["seq_test"]
        meta = fold["meta_test"]
        y_raw = fold["y_test_raw"]

    model.eval()
    seq_t = torch.tensor(seq, dtype=torch.float32, device=device)
    meta_t = torch.tensor(meta, dtype=torch.float32, device=device)

    out = model(seq_t, meta_t, grl_alpha=0.0)
    pred_scaled = out["pred_scaled"].cpu().numpy()
    pred_raw = pred_scaled * fold["y_std"] + fold["y_mean"]

    # Safety clipping for physical retention scale.
    pred_raw = np.clip(pred_raw, -0.2, 1.5)

    z = out["z"].cpu().numpy()
    recon_per = F.mse_loss(out["recon"], seq_t.squeeze(-1), reduction="none").mean(dim=1)
    u = torch.log(recon_per + 1e-8).cpu().numpy()

    mae = mean_absolute_error(y_raw, pred_raw)
    r2 = r2_score(y_raw, pred_raw) if len(y_raw) > 1 else np.nan

    return pred_raw, z, u, mae, r2


def dataset_separability(X, y, n_splits=50):
    X = np.asarray(X)
    y = pd.Series(y).astype(str)

    counts = y.value_counts()
    valid = counts[counts >= 2].index
    mask = y.isin(valid).to_numpy()
    X = X[mask]
    y = y[mask].to_numpy()

    if len(set(y)) < 2:
        return np.nan, np.nan, np.nan

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.30, random_state=42)
    vals = []
    for tr, te in sss.split(X, y):
        clf = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        vals.append(balanced_accuracy_score(y[te], pred))

    vals = np.asarray(vals)
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run_lodo(df, args, use_metadata, device):
    seq_cols = [c for c in SEQ_COLS_CANDIDATE if c in df.columns]
    meta_num = [c for c in META_NUM_CANDIDATE if c in df.columns]
    meta_cat = [c for c in META_CAT_CANDIDATE if c in df.columns]

    rows = []
    n_total = len(df)

    modes = ["vanilla_adam", "adv_only", "confoundadam", "confoundadam_adv"]

    for heldout in sorted(df["dataset_id"].astype(str).unique()):
        print("")
        print("=" * 90)
        print("Held-out:", heldout, "| metadata:", use_metadata)

        train_mask = df["dataset_id"].astype(str) != heldout
        test_mask = df["dataset_id"].astype(str) == heldout
        if test_mask.sum() < 2:
            continue

        fold = prepare_fold_data(df, train_mask, test_mask, seq_cols, meta_num, meta_cat, use_metadata)

        for mode in modes:
            print("Training:", mode)
            model, ema = train_model(fold, n_total, mode, args, device)
            pred, z, u, mae, r2 = predict_model(model, fold, "test", device)

            baseline = np.repeat(np.median(fold["y_train_raw"]), len(fold["y_test_raw"]))
            baseline_mae = mean_absolute_error(fold["y_test_raw"], baseline)

            rows.append({
                "heldout_dataset": heldout,
                "use_metadata": bool(use_metadata),
                "mode": mode,
                "n_train": int(len(fold["train"])),
                "n_test": int(len(fold["test"])),
                "mae": float(mae),
                "r2": float(r2),
                "train_median_mae": float(baseline_mae),
                "improvement_vs_train_median": float(baseline_mae - mae),
                "mean_unfamiliarity_test": float(np.mean(u)),
            })

            print(f"  MAE={mae:.4f}, baseline={baseline_mae:.4f}, improvement={baseline_mae - mae:.4f}, R2={r2:.4f}")

    return pd.DataFrame(rows)


def run_closure(df, args, use_metadata, device):
    seq_cols = [c for c in SEQ_COLS_CANDIDATE if c in df.columns]
    meta_num = [c for c in META_NUM_CANDIDATE if c in df.columns]
    meta_cat = [c for c in META_CAT_CANDIDATE if c in df.columns]

    fold = prepare_fold_data(
        df,
        np.ones(len(df), dtype=bool),
        np.ones(len(df), dtype=bool),
        seq_cols,
        meta_num,
        meta_cat,
        use_metadata,
    )

    model, ema = train_model(fold, len(df), "confoundadam_adv", args, device)
    pred, z, u, mae, r2 = predict_model(model, fold, "test", device)

    out = df[["row_id", "dataset_id", TARGET]].copy()
    out["pred"] = pred
    out["unfamiliarity"] = u
    out["confound_weight_raw"] = ema.last_w_raw[out["row_id"].to_numpy()]
    out["confound_weight_norm"] = ema.last_w_norm[out["row_id"].to_numpy()]
    out["confound_risk"] = ema.last_risk[out["row_id"].to_numpy()]
    out["use_metadata"] = bool(use_metadata)

    z_mean, z_lo, z_hi = dataset_separability(z, df["dataset_id"])
    w_mean, w_lo, w_hi = dataset_separability(out[["confound_weight_norm"]].to_numpy(), df["dataset_id"])
    u_mean, u_lo, u_hi = dataset_separability(out[["unfamiliarity"]].to_numpy(), df["dataset_id"])

    closure = pd.DataFrame([
        {
            "use_metadata": bool(use_metadata),
            "feature_source": "learned_z",
            "dataset_balanced_accuracy_mean": z_mean,
            "ci_low": z_lo,
            "ci_high": z_hi,
        },
        {
            "use_metadata": bool(use_metadata),
            "feature_source": "confound_weight_norm",
            "dataset_balanced_accuracy_mean": w_mean,
            "ci_low": w_lo,
            "ci_high": w_hi,
        },
        {
            "use_metadata": bool(use_metadata),
            "feature_source": "unfamiliarity_U",
            "dataset_balanced_accuracy_mean": u_mean,
            "ci_low": u_lo,
            "ci_high": u_hi,
        },
    ])

    weights = out.groupby("dataset_id").agg(
        n=("row_id", "count"),
        mean_weight_norm=("confound_weight_norm", "mean"),
        median_weight_norm=("confound_weight_norm", "median"),
        mean_weight_raw=("confound_weight_raw", "mean"),
        mean_unfamiliarity=("unfamiliarity", "mean"),
        mean_risk=("confound_risk", "mean"),
        full_model_mae=(TARGET, lambda s: np.nan),
    ).reset_index()

    maes = []
    for ds, g in out.groupby("dataset_id"):
        maes.append({"dataset_id": ds, "full_model_mae": mean_absolute_error(g[TARGET], g["pred"])})
    weights = weights.drop(columns=["full_model_mae"]).merge(pd.DataFrame(maes), on="dataset_id")

    return closure, weights, out


def write_report(lodo, closure, weights, args):
    report = OUT_DIR / "BATTERYINVERSE_NET_CONFOUNDADAM_V1_STABILITY_REPORT.md"

    lines = []
    lines.append("# BatteryInverse-Net + Stable ConfoundAdam v1 report\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This v1 experiment fixes target scaling, prediction clipping, unstable negative-unfamiliarity weighting, "
        "and weight-collapse reporting from v0. It should still not be used as a manuscript claim unless the "
        "closure diagnostics support it.\n\n"
    )

    lines.append("## Settings\n\n")
    for k, v in vars(args).items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n")

    lines.append("## LODO summary\n\n")
    if len(lodo):
        summary = lodo.groupby(["use_metadata", "mode"]).agg(
            n_folds=("heldout_dataset", "count"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_improvement=("improvement_vs_train_median", "mean"),
            median_improvement=("improvement_vs_train_median", "median"),
        ).reset_index()
        lines.append(summary.to_string(index=False))
        lines.append("\n\n")

    lines.append("## Closure separability\n\n")
    lines.append(closure.to_string(index=False))
    lines.append("\n\n")

    lines.append("## Dataset weight summary\n\n")
    lines.append(weights.to_string(index=False))
    lines.append("\n\n")

    lines.append("## Decision rules\n\n")
    lines.append("- If learned_z separability remains high, do not claim adversarial invariance solved dataset fingerprints.\n")
    lines.append("- If weights collapse to a narrow range or encode dataset identity, do not claim ConfoundAdam is successful.\n")
    lines.append("- If LODO improvement is not positive relative to train-median, report the method as an attempted mitigation rather than a successful mitigation.\n")

    report.write_text("".join(lines), encoding="utf-8")
    print("Saved:", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda-recon", type=float, default=0.5)
    parser.add_argument("--gamma-adv", type=float, default=1.0)
    parser.add_argument("--grl-alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--min-weight", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET, "dataset_id"]).copy()
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    all_lodo = []
    all_closure = []
    all_weights = []
    all_preds = []

    for use_metadata in [False, True]:
        print("")
        print("#" * 90)
        print("RUN use_metadata =", use_metadata)
        lodo = run_lodo(df, args, use_metadata, device)
        closure, weights, preds = run_closure(df, args, use_metadata, device)

        all_lodo.append(lodo)
        all_closure.append(closure)
        all_weights.append(weights.assign(use_metadata=bool(use_metadata)))
        all_preds.append(preds)

    lodo = pd.concat(all_lodo, ignore_index=True)
    closure = pd.concat(all_closure, ignore_index=True)
    weights = pd.concat(all_weights, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    lodo.to_csv(OUT_DIR / "batteryinverse_net_v1_lodo_results.csv", index=False)
    closure.to_csv(OUT_DIR / "batteryinverse_net_v1_closure_separability.csv", index=False)
    weights.to_csv(OUT_DIR / "batteryinverse_net_v1_dataset_weight_summary.csv", index=False)
    preds.to_csv(OUT_DIR / "batteryinverse_net_v1_full_predictions_weights.csv", index=False)

    write_report(lodo, closure, weights, args)

    print("")
    print("Done. Outputs:")
    for p in sorted(OUT_DIR.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
