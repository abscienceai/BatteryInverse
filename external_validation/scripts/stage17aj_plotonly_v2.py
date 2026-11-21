#!/usr/bin/env python3
"""Stage17AJ-V2: plot-only repair; no changes to original experiments."""
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path.cwd().resolve()
R = ROOT / "external_validation/reports"

CSV = R / "stage17aj_nca_66cell_publication_figure_source.csv"
PREV = R / "stage17aj_nca_endpoint_figure_provenance.json"
PRED = R / "stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv"

PDF = R / "stage17aj_v2_nca_endpoint_window_publication.pdf"
PNG = R / "stage17aj_v2_nca_endpoint_window_publication.png"
REPORT = R / "stage17aj_v2_figure_only_provenance.json"

LOCKED_SHA = (
    "cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757"
)


def need(ok, detail):
    if not ok:
        raise RuntimeError(detail)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1048576), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def numeric(value, description):
    result = float(value)
    need(math.isfinite(result), "Nonfinite " + description)
    return result


def main():
    for path in (CSV, PREV, PRED):
        need(path.is_file(), "Missing source: " + str(path))

    for path in (PDF, PNG, REPORT):
        need(not path.exists(), "Refusing overwrite: " + str(path))

    prior = json.loads(PREV.read_text(encoding="utf-8"))

    need(
        prior.get("all_archive_physical_cells") == 66
        and prior.get("preoutcome_selected_cells") == 30,
        "Previous cohort mismatch"
    )

    need(
        prior.get("figure_created") is False,
        "Original figure was already created"
    )

    need(
        sha(PRED) == LOCKED_SHA
        == prior.get("locked_predictions_sha256"),
        "Locked predictions changed"
    )

    relative = CSV.relative_to(ROOT).as_posix()

    need(
        prior.get("outputs_sha256", {}).get(relative)
        == sha(CSV),
        "Stage17AJ source CSV SHA256 mismatch"
    )

    with CSV.open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        reader = csv.DictReader(stream)

        required = {
            "cell_id",
            "program",
            "selected_before_outcome",
            "last_recorded_cycle",
            "last_Qmax_mAh",
            "last_within25mAh_of_2500",
            "cycle1_Q_mAh_selected_only",
            "last_retention_selected_only",
            "2500_div_Q1_selected_only_POSTOUTCOME",
            "excess_div_Q1_selected_only",
        }

        need(
            required <= set(reader.fieldnames or []),
            "Missing figure source columns"
        )

        rows = list(reader)

    need(
        len(rows) == 66
        and len({r["cell_id"] for r in rows}) == 66,
        "Incorrect 66-cell roster"
    )

    selected = [
        r for r in rows
        if r["selected_before_outcome"].strip().lower()
        == "true"
    ]

    need(len(selected) == 30, "Incorrect selected cohort")

    near66 = sum(
        r["last_within25mAh_of_2500"].strip().lower()
        == "true"
        for r in rows
    )

    near30 = sum(
        r["last_within25mAh_of_2500"].strip().lower()
        == "true"
        for r in selected
    )

    need(
        (near66, near30) == (42, 17),
        "Stage17AJ capacity counts changed"
    )

    for row in rows:
        row["cycle"] = numeric(
            row["last_recorded_cycle"], "cycle"
        )
        row["q"] = numeric(
            row["last_Qmax_mAh"], "last discharge"
        )

        need(
            row["cycle"] >= 1 and row["q"] >= 2500,
            "Unexpected endpoint value"
        )

    for row in selected:
        row["q1"] = numeric(
            row["cycle1_Q_mAh_selected_only"], "Q1"
        )
        row["ratio"] = numeric(
            row["last_retention_selected_only"], "last ratio"
        )
        row["floor"] = numeric(
            row["2500_div_Q1_selected_only_POSTOUTCOME"],
            "2500/Q1"
        )
        row["excess"] = numeric(
            row["excess_div_Q1_selected_only"],
            "excess/Q1"
        )

        need(
            row["q1"] > 0
            and abs(
                row["ratio"] - row["q"] / row["q1"]
            ) < 1e-10
            and abs(
                row["ratio"] - row["floor"] - row["excess"]
            ) < 1e-10,
            "Decomposition mismatch: " + row["cell_id"]
        )

    programs = sorted({r["program"] for r in rows})

    palette = plt.get_cmap("tab10")
    colors = {
        name: palette(i)
        for i, name in enumerate(programs)
    }

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "savefig.facecolor": "white",
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5.5)
    )

    for program in programs:
        for chosen in (False, True):
            group = [
                r for r in rows
                if r["program"] == program
                and (
                    r["selected_before_outcome"]
                    .strip().lower() == "true"
                ) == chosen
            ]

            if not group:
                continue

            ax1.scatter(
                [r["cycle"] for r in group],
                [r["q"] for r in group],
                s=40 if chosen else 35,
                marker="o",
                c=[colors[program]] if chosen else "none",
                edgecolors=[colors[program]],
                alpha=0.88,
                linewidths=1.1,
                zorder=3 if chosen else 2,
            )

        group2 = [
            r for r in selected
            if r["program"] == program
        ]

        if group2:
            ax2.scatter(
                [r["floor"] for r in group2],
                [r["ratio"] for r in group2],
                s=43,
                c=[colors[program]],
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
            )

    ax1.axhline(
        2500,
        color="black",
        linestyle="--",
        linewidth=1.1,
    )

    ax1.set_xlabel("Last recorded instrument-cycle number")
    ax1.set_ylabel(
        "Last recorded maximum discharge capacity (mAh)"
    )
    ax1.set_title(
        "(a) Archive endpoints - 66 physical cells",
        loc="left", weight="bold"
    )

    ax1.set_xlim(
        0, max(r["cycle"] for r in rows) * 1.04
    )
    ax1.set_ylim(
        2450, max(r["q"] for r in rows) + 55
    )

    ax1.text(
        0.03, 0.95,
        "42/66 within 2500-2525 mAh",
        va="top",
        transform=ax1.transAxes,
        fontsize=9,
    )
    ax1.grid(alpha=0.2, linewidth=0.65)

    xmin = min(
        min(r["floor"] for r in selected),
        min(r["ratio"] for r in selected),
    ) - 0.015

    xmax = max(
        max(r["floor"] for r in selected),
        max(r["ratio"] for r in selected),
    ) + 0.015

    ax2.plot(
        [xmin, xmax], [xmin, xmax],
        linestyle="--", color="black", linewidth=1.1
    )

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(xmin, xmax)
    ax2.set_aspect("equal", adjustable="box")

    ax2.set_xlabel(
        "2500 / Q1 (post-outcome diagnostic; not a forecast)"
    )
    ax2.set_ylabel("Measured last Qmax / Q1")
    ax2.set_title(
        "(b) Locked cohort - 30 physical cells",
        loc="left", weight="bold"
    )

    ax2.text(
        0.03, 0.95,
        "17/30 within 2500-2525 mAh",
        transform=ax2.transAxes,
        va="top",
        fontsize=9,
    )
    ax2.grid(alpha=0.2, linewidth=0.65)

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color=colors[p],
            markerfacecolor=colors[p],
            markersize=7,
            label=p,
        )
        for p in programs
    ]

    handles.extend([
        Line2D(
            [0], [0],
            marker="o", linestyle="",
            color="black",
            markerfacecolor="black",
            markersize=6,
            label="Selected (panel a)",
        ),
        Line2D(
            [0], [0],
            marker="o", linestyle="",
            color="black",
            markerfacecolor="none",
            markersize=6,
            label="Excluded (panel a)",
        ),
    ])

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.88,
        bottom=0.24,
        wspace=0.28,
    )

    fig.suptitle(
        "NCA observation-window endpoints: "
        "post-outcome diagnostic",
        fontsize=13,
        weight="bold",
        y=0.97,
    )

    fig.text(
        0.5, 0.14,
        "2500 mAh is a published analysis-window "
        "reference, not a verified per-cell "
        "stopping command or certified EOL.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    with tempfile.TemporaryDirectory(
        prefix="stage17aj_v2_", dir=R
    ) as directory:
        temp = Path(directory)

        temp_pdf = temp / "figure.pdf"
        temp_png = temp / "figure.png"
        temp_report = temp / "report.json"

        fig.savefig(temp_pdf)
        fig.savefig(temp_png, dpi=260)
        plt.close(fig)

        provenance = {
            "stage": "17AJ_V2_FIGURE_ONLY_REPAIR",
            "prior_figure_created": False,
            "archive_cells": 66,
            "selected_cells": 30,
            "within25mAh_all66": near66,
            "within25mAh_selected30": near30,
            "input_sha256": {
                relative: sha(CSV),
                PRED.relative_to(ROOT).as_posix(): sha(PRED),
                PREV.relative_to(ROOT).as_posix(): sha(PREV),
            },
            "figure_sha256": {
                PDF.relative_to(ROOT).as_posix(): sha(temp_pdf),
                PNG.relative_to(ROOT).as_posix(): sha(temp_png),
            },
            "endpoint_caveat": (
                "2500 mAh is a post-outcome "
                "diagnostic reference, not a "
                "predictive baseline or per-cell stop rule."
            ),
            "model_fit": False,
            "new_inference": False,
            "prism_updated": False,
            "matplotlib_version": matplotlib.__version__,
        }

        temp_report.write_text(
            json.dumps(
                provenance,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        need(
            not any(p.exists() for p in (PDF, PNG, REPORT)),
            "Output appeared; refusing overwrite"
        )

        os.replace(temp_pdf, PDF)
        os.replace(temp_png, PNG)
        os.replace(temp_report, REPORT)

    print("=== STAGE 17AJ-V2 FIGURE-ONLY REPAIR ===")
    print("PREDICTION_LOCK_UNCHANGED=YES")
    print("ARCHIVE66=YES SELECTED30=YES")
    print("FIGURE_CREATED=YES")
    print("FIGURE_PDF=", PDF)
    print("FIGURE_PNG=", PNG)
    print("PDF_SHA256=", sha(PDF))
    print("PNG_SHA256=", sha(PNG))
    print("PROVENANCE_JSON=", REPORT)
    print("MODEL_FIT=NO MODEL_INFERENCE=NO PRISM_UPDATED=NO")
    print("=== END STAGE 17AJ-V2 ===")


if __name__ == "__main__":
    main()
