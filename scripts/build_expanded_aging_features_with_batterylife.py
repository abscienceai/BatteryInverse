from pathlib import Path
import numpy as np
import pandas as pd

MAIN_PATH = Path("data/05_model_ready/aging/early_cycle_features_with_labels.csv")
BL_PATH = Path("data/03_processed/BATTERYLIFE_SELECTED/batterylife_selected_cell_summary.csv")
OUT_DIR = Path("data/05_model_ready/aging")
REPORT_DIR = Path("reports/data_audits")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "expanded_aging_features_with_batterylife.csv"
OUT_PARQUET = OUT_DIR / "expanded_aging_features_with_batterylife.parquet"
REPORT = REPORT_DIR / "EXPANDED_AGING_FEATURES_WITH_BATTERYLIFE.md"


def add_normalized_features(df):
    out = df.copy()
    eps = 1e-12

    base = pd.to_numeric(out.get("cap_cycle_1"), errors="coerce").replace(0, np.nan)

    for col in [
        "cap_cycle_10",
        "cap_cycle_50",
        "cap_cycle_100",
        "cap_mean_first_100",
        "cap_std_first_100",
        "cap_slope_first_100",
        "cap_fade_1_to_100",
    ]:
        if col in out.columns:
            out[col + "_norm"] = pd.to_numeric(out[col], errors="coerce") / (base + eps)

    if "cap_cycle_100" in out.columns:
        out["cap_ratio_100_to_1"] = pd.to_numeric(out["cap_cycle_100"], errors="coerce") / (base + eps)

    if "cap_cycle_50" in out.columns:
        out["cap_ratio_50_to_1"] = pd.to_numeric(out["cap_cycle_50"], errors="coerce") / (base + eps)

    if "cap_cycle_10" in out.columns:
        out["cap_ratio_10_to_1"] = pd.to_numeric(out["cap_cycle_10"], errors="coerce") / (base + eps)

    if "cap_fade_1_to_100" in out.columns:
        out["cap_fade_fraction_1_to_100"] = pd.to_numeric(out["cap_fade_1_to_100"], errors="coerce") / (base + eps)

    return out


def harmonize_main(main):
    out = main.copy()

    if "final_capacity_retention" not in out.columns:
        if "capacity_retention" in out.columns:
            out["final_capacity_retention"] = out["capacity_retention"]
        elif "capacity_retention_robust" in out.columns:
            out["final_capacity_retention"] = out["capacity_retention_robust"]

    if "chemistry" not in out.columns:
        if "cathode_material" in out.columns:
            out["chemistry"] = out["cathode_material"]
        else:
            out["chemistry"] = out["dataset_id"].astype(str)

    if "source_family" not in out.columns:
        out["source_family"] = "BatteryML_main"

    if "source_file" not in out.columns:
        out["source_file"] = ""

    return out


def harmonize_batterylife(bl):
    out = bl.copy()

    out["source_family"] = "BatteryLife_selected"
    out["final_capacity_retention"] = out["capacity_retention_robust"]

    out["chemistry"] = out["cathode_material"].fillna("unknown")
    out["min_voltage_V"] = out["min_voltage_V"]
    out["max_voltage_V"] = out["max_voltage_V"]

    if "cap_cycle_10" not in out.columns:
        out["cap_cycle_10"] = np.nan
    if "cap_cycle_50" not in out.columns:
        out["cap_cycle_50"] = np.nan

    # Compatibility placeholders with earlier feature tables.
    for c in [
        "long_life_label",
        "high_retention_label",
        "long_life_label_within_dataset",
        "high_retention_label_within_dataset",
        "aging_quality_score_proxy",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    return out


def main():
    main_df = pd.read_csv(MAIN_PATH)
    bl_df = pd.read_csv(BL_PATH)

    main_df = harmonize_main(main_df)
    bl_df = harmonize_batterylife(bl_df)

    common_preferred = [
        "source_family",
        "dataset_id",
        "cell_id",
        "source_file",
        "final_capacity_retention",
        "cap_cycle_1",
        "cap_cycle_10",
        "cap_cycle_50",
        "cap_cycle_100",
        "cap_mean_first_100",
        "cap_std_first_100",
        "cap_slope_first_100",
        "cap_fade_1_to_100",
        "nominal_capacity_Ah",
        "min_voltage_V",
        "max_voltage_V",
        "chemistry",
        "form_factor",
        "anode_material",
        "cathode_material",
        "ambient_temperature_C",
        "quality_flags",
        "long_life_label",
        "high_retention_label",
        "long_life_label_within_dataset",
        "high_retention_label_within_dataset",
        "aging_quality_score_proxy",
    ]

    for c in common_preferred:
        if c not in main_df.columns:
            main_df[c] = np.nan
        if c not in bl_df.columns:
            bl_df[c] = np.nan

    combined = pd.concat(
        [main_df[common_preferred], bl_df[common_preferred]],
        ignore_index=True,
    )

    combined = add_normalized_features(combined)

    combined.to_csv(OUT_CSV, index=False, float_format="%.12g")
    combined.to_parquet(OUT_PARQUET, index=False)

    report = []
    report.append("# Expanded Aging Features with BatteryLife Selected\n\n")
    report.append("## Summary\n\n")
    report.append(f"- Main rows: {len(main_df)}\n")
    report.append(f"- BatteryLife selected rows: {len(bl_df)}\n")
    report.append(f"- Combined rows: {len(combined)}\n\n")

    report.append("## Rows by source family\n\n")
    report.append(combined["source_family"].value_counts(dropna=False).to_string())
    report.append("\n\n")

    report.append("## Rows by dataset\n\n")
    report.append(combined["dataset_id"].value_counts(dropna=False).sort_index().to_string())
    report.append("\n\n")

    report.append("## Chemistry counts\n\n")
    report.append(combined["chemistry"].fillna("unknown").value_counts().to_string())
    report.append("\n\n")

    report.append("## Quality flags\n\n")
    report.append(combined["quality_flags"].fillna("missing").value_counts().to_string())
    report.append("\n\n")

    report.append("## Final retention by dataset\n\n")
    retention_summary = combined.groupby("dataset_id")["final_capacity_retention"].describe()
    report.append(retention_summary.to_string())
    report.append("\n\n")

    REPORT.write_text("".join(report), encoding="utf-8")

    print("Expanded aging feature table created.")
    print("Main rows:", len(main_df))
    print("BatteryLife selected rows:", len(bl_df))
    print("Combined rows:", len(combined))
    print("")
    print("Rows by source family:")
    print(combined["source_family"].value_counts(dropna=False).to_string())
    print("")
    print("Rows by dataset:")
    print(combined["dataset_id"].value_counts(dropna=False).sort_index().to_string())
    print("")
    print("Chemistry counts:")
    print(combined["chemistry"].fillna("unknown").value_counts().to_string())
    print("")
    print("Saved:", OUT_CSV)
    print("Saved:", REPORT)


if __name__ == "__main__":
    main()
