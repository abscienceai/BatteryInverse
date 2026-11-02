from pathlib import Path
import zipfile
import pickle
import json
import re
import numpy as np
import pandas as pd

RAW_DIR = Path("data/00_external_raw/aging/BATTERYLIFE_PROCESSED/original")
OUT_DIR = Path("data/03_processed/BATTERYLIFE_SELECTED")
REPORT_DIR = Path("reports/data_audits")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def to_numeric_array(x):
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, np.ndarray):
        arr = x.astype(float, copy=False) if np.issubdtype(x.dtype, np.number) else np.array([], dtype=float)
    elif isinstance(x, (list, tuple)):
        vals = []
        for v in x:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        arr = np.array(vals, dtype=float)
    else:
        try:
            arr = np.array([float(x)], dtype=float)
        except Exception:
            arr = np.array([], dtype=float)
    return arr[np.isfinite(arr)]


def max_or_nan(x):
    arr = to_numeric_array(x)
    return float(np.nanmax(arr)) if arr.size else np.nan


def median_or_nan(x):
    arr = to_numeric_array(x)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def mean_or_nan(x):
    arr = to_numeric_array(x)
    return float(np.nanmean(arr)) if arr.size else np.nan


def extract_temperature_from_cell_id(cell_id):
    # Examples: 25C, -5C, 45C, 15C, 23C
    m = re.search(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)C(?![A-Za-z])", str(cell_id))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    return np.nan


def extract_protocol_text(protocol):
    try:
        return json.dumps(protocol, sort_keys=True, default=str)
    except Exception:
        return repr(protocol)


def robust_first_last_capacity(cap_series):
    s = pd.Series(cap_series, dtype="float64").dropna()
    if len(s) == 0:
        return np.nan, np.nan
    first = float(s.head(min(3, len(s))).median())
    last = float(s.tail(min(3, len(s))).median())
    return first, last


def nearest_capacity(traj, target_cycle):
    if traj.empty:
        return np.nan
    idx = (traj["cycle_number"] - target_cycle).abs().idxmin()
    return float(traj.loc[idx, "discharge_capacity_Ah"])


def early_slope(traj, max_cycle=100):
    sub = traj[(traj["cycle_number"] <= max_cycle)].dropna(subset=["discharge_capacity_Ah"]).copy()
    if len(sub) < 2:
        return np.nan
    x = sub["cycle_number"].to_numpy(dtype=float)
    y = sub["discharge_capacity_Ah"].to_numpy(dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return np.nan


def summarize_cell(obj, archive_name, pkl_name):
    cell_id = obj.get("cell_id", Path(pkl_name).stem)
    dataset_id = Path(archive_name).stem

    cycle_data = obj.get("cycle_data", [])
    rows = []

    for i, cyc in enumerate(cycle_data):
        if not isinstance(cyc, dict):
            continue

        cycle_number = cyc.get("cycle_number", i + 1)
        try:
            cycle_number = int(cycle_number)
        except Exception:
            cycle_number = i + 1

        discharge_capacity = max_or_nan(cyc.get("discharge_capacity_in_Ah"))
        charge_capacity = max_or_nan(cyc.get("charge_capacity_in_Ah"))
        mean_voltage = mean_or_nan(cyc.get("voltage_in_V"))
        mean_current = mean_or_nan(cyc.get("current_in_A"))
        median_temp = median_or_nan(cyc.get("temperature_in_C"))
        max_time = max_or_nan(cyc.get("time_in_s"))
        ir = max_or_nan(cyc.get("internal_resistance_in_ohm"))

        rows.append({
            "dataset_id": dataset_id,
            "cell_id": cell_id,
            "source_archive": archive_name,
            "source_file": pkl_name,
            "cycle_number": cycle_number,
            "discharge_capacity_Ah": discharge_capacity,
            "charge_capacity_Ah": charge_capacity,
            "mean_voltage_V": mean_voltage,
            "mean_current_A": mean_current,
            "median_temperature_C": median_temp,
            "max_time_s": max_time,
            "internal_resistance_ohm": ir,
        })

    traj = pd.DataFrame(rows)
    if len(traj):
        traj = traj.sort_values("cycle_number").reset_index(drop=True)

    valid = traj.dropna(subset=["discharge_capacity_Ah"]) if len(traj) else pd.DataFrame()

    first_cap, last_cap = robust_first_last_capacity(valid["discharge_capacity_Ah"]) if len(valid) else (np.nan, np.nan)
    retention = float(last_cap / first_cap) if np.isfinite(first_cap) and first_cap != 0 and np.isfinite(last_cap) else np.nan

    cap_1 = nearest_capacity(valid, 1) if len(valid) else np.nan
    cap_10 = nearest_capacity(valid, 10) if len(valid) else np.nan
    cap_50 = nearest_capacity(valid, 50) if len(valid) else np.nan
    cap_100 = nearest_capacity(valid, 100) if len(valid) else np.nan

    slope_100 = early_slope(valid, 100) if len(valid) else np.nan

    nominal = obj.get("nominal_capacity_in_Ah", np.nan)
    try:
        nominal = float(nominal)
    except Exception:
        nominal = np.nan

    temp_from_id = extract_temperature_from_cell_id(cell_id)
    observed_temp = float(valid["median_temperature_C"].median()) if len(valid) and valid["median_temperature_C"].notna().any() else np.nan
    ambient_temp = observed_temp if np.isfinite(observed_temp) else temp_from_id

    quality_flags = []

    if len(valid) < 20:
        quality_flags.append("few_valid_capacity_cycles")
    if not np.isfinite(first_cap) or not np.isfinite(last_cap) or not np.isfinite(retention):
        quality_flags.append("missing_retention")
    if np.isfinite(retention) and retention <= 0:
        quality_flags.append("nonpositive_retention")
    if np.isfinite(retention) and retention > 1.20:
        quality_flags.append("retention_above_1p2")
    if np.isfinite(nominal) and np.isfinite(first_cap):
        ratio = first_cap / nominal if nominal != 0 else np.nan
        if np.isfinite(ratio) and ratio < 0.40:
            quality_flags.append("first_capacity_below_40pct_nominal")
        if np.isfinite(ratio) and ratio > 1.60:
            quality_flags.append("first_capacity_above_160pct_nominal")

    if not quality_flags:
        quality_flags = ["ok"]

    summary = {
        "dataset_id": dataset_id,
        "cell_id": cell_id,
        "source_archive": archive_name,
        "source_file": pkl_name,
        "n_cycles_total": int(len(traj)),
        "n_valid_capacity_cycles": int(len(valid)),
        "first_capacity_robust_Ah": first_cap,
        "last_capacity_robust_Ah": last_cap,
        "capacity_retention_robust": retention,
        "cap_cycle_1": cap_1,
        "cap_cycle_10": cap_10,
        "cap_cycle_50": cap_50,
        "cap_cycle_100": cap_100,
        "cap_mean_first_100": float(valid[valid["cycle_number"] <= 100]["discharge_capacity_Ah"].mean()) if len(valid) else np.nan,
        "cap_std_first_100": float(valid[valid["cycle_number"] <= 100]["discharge_capacity_Ah"].std()) if len(valid) else np.nan,
        "cap_slope_first_100": slope_100,
        "cap_fade_1_to_100": float(cap_1 - cap_100) if np.isfinite(cap_1) and np.isfinite(cap_100) else np.nan,
        "nominal_capacity_Ah": nominal,
        "form_factor": obj.get("form_factor"),
        "anode_material": obj.get("anode_material"),
        "cathode_material": obj.get("cathode_material"),
        "electrolyte_material": obj.get("electrolyte_material"),
        "depth_of_charge": obj.get("depth_of_charge"),
        "depth_of_discharge": obj.get("depth_of_discharge"),
        "already_spent_cycles": obj.get("already_spent_cycles"),
        "max_voltage_V": obj.get("max_voltage_limit_in_V"),
        "min_voltage_V": obj.get("min_voltage_limit_in_V"),
        "max_current_A": obj.get("max_current_limit_in_A"),
        "min_current_A": obj.get("min_current_limit_in_A"),
        "ambient_temperature_C": ambient_temp,
        "temperature_from_cell_id_C": temp_from_id,
        "observed_temperature_median_C": observed_temp,
        "charge_protocol_text": extract_protocol_text(obj.get("charge_protocol")),
        "discharge_protocol_text": extract_protocol_text(obj.get("discharge_protocol")),
        "SOC_interval_text": extract_protocol_text(obj.get("SOC_interval")),
        "quality_flags": ";".join(quality_flags),
    }

    return summary, traj


def main():
    summaries = []
    trajectories = []
    errors = []

    archives = sorted(RAW_DIR.glob("*.zip"))

    print("Archives:", [a.name for a in archives])

    for zpath in archives:
        print("")
        print("=" * 80)
        print("Processing archive:", zpath.name)

        with zipfile.ZipFile(zpath, "r") as z:
            pkl_files = [i for i in z.infolist() if i.filename.endswith(".pkl")]
            print("n_pkl:", len(pkl_files))

            for j, info in enumerate(pkl_files, start=1):
                print(f"[{j}/{len(pkl_files)}] {info.filename}")
                try:
                    with z.open(info.filename, "r") as f:
                        obj = pickle.load(f)

                    summary, traj = summarize_cell(obj, zpath.name, info.filename)
                    summaries.append(summary)
                    trajectories.append(traj)

                except Exception as e:
                    errors.append({
                        "archive": zpath.name,
                        "file": info.filename,
                        "error": repr(e),
                    })
                    print("ERROR:", repr(e))

    summary_df = pd.DataFrame(summaries)
    trajectory_df = pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame()
    errors_df = pd.DataFrame(errors)

    summary_csv = OUT_DIR / "batterylife_selected_cell_summary.csv"
    summary_parquet = OUT_DIR / "batterylife_selected_cell_summary.parquet"
    traj_csv = OUT_DIR / "batterylife_selected_capacity_trajectory.csv"
    traj_parquet = OUT_DIR / "batterylife_selected_capacity_trajectory.parquet"
    errors_csv = OUT_DIR / "batterylife_selected_parse_errors.csv"

    summary_df.to_csv(summary_csv, index=False, float_format="%.12g")
    summary_df.to_parquet(summary_parquet, index=False)
    trajectory_df.to_csv(traj_csv, index=False, float_format="%.12g")
    trajectory_df.to_parquet(traj_parquet, index=False)
    errors_df.to_csv(errors_csv, index=False)

    report = []
    report.append("# BatteryLife Selected Dataset Audit\n\n")
    report.append("## Scope\n\n")
    report.append("This audit summarizes selected BatteryLife processed archives downloaded from Zenodo record 14934405. The selected subset includes CALB, HNEI, MICH, MICH_EXP, NA-ion, SNL, UL_PUR, and XJTU.\n\n")

    report.append("## Summary\n\n")
    report.append(f"- Parsed cells: {len(summary_df)}\n")
    report.append(f"- Capacity trajectory rows: {len(trajectory_df)}\n")
    report.append(f"- Parse errors: {len(errors_df)}\n\n")

    if len(summary_df):
        report.append("## Cells by dataset\n\n")
        report.append(summary_df["dataset_id"].value_counts().sort_index().to_string())
        report.append("\n\n")

        report.append("## Chemistry counts\n\n")
        report.append(summary_df["cathode_material"].fillna("unknown").value_counts().to_string())
        report.append("\n\n")

        report.append("## Form factor counts\n\n")
        report.append(summary_df["form_factor"].fillna("unknown").value_counts().to_string())
        report.append("\n\n")

        report.append("## Quality flags\n\n")
        report.append(summary_df["quality_flags"].fillna("missing").value_counts().to_string())
        report.append("\n\n")

        numeric_cols = [
            "n_cycles_total",
            "n_valid_capacity_cycles",
            "first_capacity_robust_Ah",
            "last_capacity_robust_Ah",
            "capacity_retention_robust",
            "nominal_capacity_Ah",
            "ambient_temperature_C",
        ]
        report.append("## Numeric summary\n\n")
        report.append(summary_df[numeric_cols].describe().to_string())
        report.append("\n\n")

    if len(errors_df):
        report.append("## Parse errors\n\n")
        report.append(errors_df.to_string(index=False))
        report.append("\n\n")

    report_path = REPORT_DIR / "BATTERYLIFE_SELECTED_AUDIT.md"
    report_path.write_text("".join(report), encoding="utf-8")

    print("")
    print("=" * 80)
    print("Parsed cells:", len(summary_df))
    print("Capacity trajectory rows:", len(trajectory_df))
    print("Parse errors:", len(errors_df))
    print("")
    print("Cells by dataset:")
    print(summary_df["dataset_id"].value_counts().sort_index().to_string())
    print("")
    print("Chemistry counts:")
    print(summary_df["cathode_material"].fillna("unknown").value_counts().to_string())
    print("")
    print("Quality flags:")
    print(summary_df["quality_flags"].fillna("missing").value_counts().to_string())
    print("")
    print("Saved:", summary_csv)
    print("Saved:", traj_csv)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
