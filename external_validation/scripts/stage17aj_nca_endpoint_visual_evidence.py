#!/usr/bin/env python3
"""Stage17AJ: post-outcome NCA endpoint visual evidence.

No fitting, inference, retrospective selection of model branches,
or claim that last-recorded capacity is certified physical EOL.
"""
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
R = ROOT / "external_validation/reports"

AH = R / "stage17ah_nca_absolute_endpoint_and_censoring_report.json"
AH_C = R / "stage17ah_nca_66cell_absolute_last_capacity_audit.csv"
AI = R / "stage17ai_nca_2500mAh_endpoint_interpretation.json"
AI_C = R / "stage17ai_nca_nominal_floor_mechanistic_decomposition_30.csv"
PRED = R / "stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv"

OUT_C = R / "stage17aj_nca_66cell_publication_figure_source.csv"
OUT_PDF = R / "stage17aj_nca_endpoint_window_diagnostic_figure.pdf"
OUT_PNG = R / "stage17aj_nca_endpoint_window_diagnostic_figure.png"
OUT_J = R / "stage17aj_nca_endpoint_figure_provenance.json"

PRED_SHA = (
    "cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757"
)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1048576), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def getcsv(path, required):
    need(path.is_file(), f"Missing: {path}")

    with path.open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        reader = csv.DictReader(stream)

        need(
            set(required) <= set(reader.fieldnames or []),
            f"Columns missing in {path.name}"
        )
        return list(reader)


def number(x, label):
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"Invalid {label}: {x!r}"
        ) from exc

    need(math.isfinite(v), f"Nonfinite {label}")
    return v


def boolean(x):
    return str(x).strip().lower() == "true"


def verify_map(mapping, description):
    for rel, expected in mapping.items():
        p = Path(rel)

        need(
            not p.is_absolute()
            and ".." not in p.parts,
            f"Unsafe {description} path: {rel}"
        )

        f = (ROOT / p).resolve()

        need(
            f.is_relative_to(ROOT)
            and f.is_file()
            and sha(f) == expected,
            f"Changed {description} source: {rel}"
        )


def main():
    inputs = (AH, AH_C, AI, AI_C, PRED)
    outputs = (OUT_C, OUT_PDF, OUT_PNG, OUT_J)

    for p in inputs:
        need(p.is_file(), f"Missing input: {p}")

    for p in outputs:
        need(
            not p.exists(),
            f"Refusing overwrite: {p}"
        )

    ah = json.loads(
        AH.read_text(encoding="utf-8")
    )
    ai = json.loads(
        AI.read_text(encoding="utf-8")
    )

    need(
        ah.get("original_66_archive_count") == 66
        and ah.get("selected_30_count") == 30
        and ai.get("locked_30_selected_count") == 30,
        "Cohort lock mismatch"
    )

    need(
        ai.get("model_refitted") is False
        and ai.get("frozen_predictions_changed") is False
        and ah.get(
            "preoutcome_locked_predictions_unchanged"
        ) is True,
        "Prior study provenance mismatch"
    )

    need(
        sha(PRED) == PRED_SHA
        == ah.get("prediction_sha256"),
        "NCA predictions are not locked"
    )

    verify_map(
        ah.get("source_sha256", {}),
        "17AH"
    )
    verify_map(
        ai.get("source_sha256", {}),
        "17AI"
    )

    source = getcsv(
        AH_C,
        [
            "cell_id",
            "program",
            "selected_preoutcome_30",
            "last_instrument_cycle",
            "last_segment_discharge_mAh"
        ]
    )

    subset = getcsv(
        AI_C,
        [
            "cell_id",
            "program",
            "first_cycle_Q1_mAh",
            "last_Qmax_mAh",
            "observed_last_ratio_Qlast_over_Q1_NOT_EOL",
            "published_window_lower_edge_over_Q1_DIAGNOSTIC_NOT_PREDICTION",
            "residual_excess_over_Q1"
        ]
    )

    need(
        len(source) == 66 and len(subset) == 30,
        "Unexpected 66/30 row counts"
    )

    need(
        len({r["cell_id"] for r in source}) == 66
        and len({r["cell_id"] for r in subset}) == 30,
        "Duplicate physical cell ID"
    )

    sub = {
        r["cell_id"]: r for r in subset
    }

    selected = {
        r["cell_id"]
        for r in source
        if boolean(r["selected_preoutcome_30"])
    }

    need(
        set(sub) == selected,
        "Selected cohort differs"
    )

    data = []

    for r in sorted(
        source, key=lambda x: x["cell_id"]
    ):
        cid = r["cell_id"]

        cy = number(
            r["last_instrument_cycle"],
            "last cycle"
        )

        q = number(
            r["last_segment_discharge_mAh"],
            "Qlast"
        )

        need(
            cy >= 1
            and int(cy) == cy
            and q >= 2500,
            f"Invalid recorded endpoint: {cid}"
        )

        chosen = cid in selected

        entry = {
            "cell_id": cid,
            "program": r["program"],
            "selected_before_outcome": chosen,
            "last_recorded_cycle": int(cy),
            "last_Qmax_mAh": q,
            "last_within25mAh_of_2500":
                2500 <= q <= 2525,
            "cycle1_Q_mAh_selected_only": "",
            "last_retention_selected_only": "",
            "2500_div_Q1_selected_only_POSTOUTCOME": "",
            "excess_div_Q1_selected_only": "",
        }

        if chosen:
            s = sub[cid]

            need(
                s["program"] == r["program"],
                f"Program differs: {cid}"
            )

            q1 = number(
                s["first_cycle_Q1_mAh"], "Q1"
            )
            qa = number(
                s["last_Qmax_mAh"], "AI Qlast"
            )
            y = number(
                s[
                    "observed_last_ratio_Qlast_over_Q1_NOT_EOL"
                ],
                "last ratio"
            )
            x = number(
                s[
                    "published_window_lower_edge_over_Q1_DIAGNOSTIC_NOT_PREDICTION"
                ],
                "2500/Q1"
            )
            excess = number(
                s["residual_excess_over_Q1"],
                "excess/Q1"
            )

            need(
                q1 > 0
                and abs(q - qa) < 1e-8
                and abs(q / q1 - y) < 1e-10
                and abs(2500 / q1 - x) < 1e-10
                and abs(y - x - excess) < 1e-10,
                f"Decomposition identity broken: {cid}"
            )

            entry.update({
                "cycle1_Q_mAh_selected_only": q1,
                "last_retention_selected_only": y,
                "2500_div_Q1_selected_only_POSTOUTCOME": x,
                "excess_div_Q1_selected_only": excess,
            })

        data.append(entry)

    within66 = sum(
        r["last_within25mAh_of_2500"]
        for r in data
    )

    within30 = sum(
        r["selected_before_outcome"]
        and r["last_within25mAh_of_2500"]
        for r in data
    )

    need(
        within66 == 42 and within30 == 17,
        "Existing Stage17AI counts changed"
    )

    selected_data = [
        r for r in data
        if r["selected_before_outcome"]
    ]

    y = np.array([
        r["last_retention_selected_only"]
        for r in selected_data
    ], dtype=float)

    x = np.array([
        r[
            "2500_div_Q1_selected_only_POSTOUTCOME"
        ]
        for r in selected_data
    ], dtype=float)

    need(
        np.all(y >= x - 1e-12),
        "Endpoint below nominal analysis bound"
    )

    fig_possible = True
    error = None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

    except ImportError as exc:
        fig_possible = False
        error = str(exc)

    with tempfile.TemporaryDirectory(
        prefix="stage17aj_",
        dir=R
    ) as name:
        t = Path(name)
        csvpath = t / "source.csv"

        with csvpath.open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(data[0])
            )
            writer.writeheader()
            writer.writerows(data)

        figures = {}

        if fig_possible:
            labels = sorted({
                r["program"] for r in data
            })

            cmap = plt.get_cmap("tab10")
            colors = {
                g: cmap(i)
                for i, g in enumerate(labels)
            }

            fig, (ax1, ax2) = plt.subplots(
                1, 2,
                figsize=(12, 4.8),
                constrained_layout=True
            )

            for group in labels:
                color = colors[group]

                for is_selected in (True, False):
                    a = [
                        r for r in data
                        if r["program"] == group
                        and r["selected_before_outcome"]
                        == is_selected
                    ]

                    if not a:
                        continue

                    ax1.scatter(
                        [
                            r["last_recorded_cycle"]
                            for r in a
                        ],
                        [
                            r["last_Qmax_mAh"]
                            for r in a
                        ],
                        s=34,
                        marker="o",
                        facecolor=(
                            color if is_selected
                            else "none"
                        ),
                        edgecolor=color,
                        linewidth=1.2,
                        alpha=0.83
                    )

                b = [
                    r for r in selected_data
                    if r["program"] == group
                ]

                if b:
                    ax2.scatter(
                        [
                            r[
                                "2500_div_Q1_selected_only_POSTOUTCOME"
                            ]
                            for r in b
                        ],
                        [
                            r[
                                "last_retention_selected_only"
                            ]
                            for r in b
                        ],
                        s=40,
                        color=color,
                        edgecolor="white",
                        linewidth=0.45
                    )

            ax1.axhline(
                2500,
                linestyle="--",
                linewidth=1.2,
                color="black"
            )

            ax1.set_xlabel(
                "Last recorded instrument-cycle number"
            )
            ax1.set_ylabel(
                "Last recorded Qmax (mAh)"
            )
            ax1.set_title(
                "(a) Archive endpoints: 66 physical cells"
            )

            ax1.text(
                0.01, 0.97,
                "Dashed line: 2500 mAh analysis-window edge",
                transform=ax1.transAxes,
                va="top",
                fontsize=8
            )

            xmin = min(
                float(x.min()), float(y.min())
            ) - 0.012

            xmax = max(
                float(x.max()), float(y.max())
            ) + 0.012

            ax2.plot(
                [xmin, xmax],
                [xmin, xmax],
                "--",
                color="black",
                linewidth=1.2
            )

            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(xmin, xmax)

            ax2.set_xlabel(
                "2500 / Q1 (post-outcome diagnostic, NOT predictor)"
            )
            ax2.set_ylabel(
                "Measured last Qmax / Q1"
            )
            ax2.set_title(
                "(b) Locked cohort: 30 physical cells"
            )

            ax2.text(
                0.03, 0.97,
                "17/30 within 25 mAh of 2500",
                transform=ax2.transAxes,
                va="top",
                fontsize=9
            )

            handles = [
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=colors[g],
                    markeredgecolor=colors[g],
                    markersize=6,
                    label=g
                )
                for g in labels
            ]

            handles.extend([
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    color="black",
                    markerfacecolor="black",
                    markersize=5,
                    label="Selected pre-outcome (panel a)"
                ),
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    color="black",
                    markerfacecolor="none",
                    markersize=5,
                    label="Not selected (panel a)"
                ),
            ])

            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=4,
                bbox_to_anchor=(0.5, -0.05),
                fontsize=8,
                frameon=False
            )

            plt.rcParams["pdf.fonttype"] = 42
            plt.rcParams["ps.fonttype"] = 42

            fig.savefig(
                t / "figure.pdf",
                bbox_inches="tight"
            )
            fig.savefig(
                t / "figure.png",
                dpi=240,
                bbox_inches="tight"
            )

            plt.close(fig)

            figures = {
                OUT_PDF.relative_to(ROOT).as_posix():
                    sha(t / "figure.pdf"),
                OUT_PNG.relative_to(ROOT).as_posix():
                    sha(t / "figure.png")
            }

        report = {
            "stage":
                "17AJ_POSTOUTCOME_NCA_ENDPOINT_VISUAL_EVIDENCE",

            "all_archive_physical_cells": 66,
            "preoutcome_selected_cells": 30,

            "all_archive_last_q_at_or_above_2500":
                True,

            "all_archive_within_25mAh_above_2500":
                within66,

            "selected30_within_25mAh_above_2500":
                within30,

            "selected30_last_ratio_median":
                float(np.median(y)),

            "selected30_2500_over_Q1_median":
                float(np.median(x)),

            "selected30_postoutcome_floor_proxy_absolute_gap_mean":
                float(np.mean(y - x)),

            "program_counts_all66":
                dict(Counter(
                    r["program"] for r in data
                )),

            "program_counts_selected30":
                dict(Counter(
                    r["program"]
                    for r in selected_data
                )),

            "published_2500mAh_window_edge_is_not_per_cell_verified_stop_rule":
                True,

            "2500_over_Q1_is_postoutcome_algebraic_diagnostic_not_forecast":
                True,

            "last_recorded_capacity_not_physical_EOL":
                True,

            "figure_created": fig_possible,
            "figure_warning": error,

            "original_model_retrained": False,
            "new_model_predictions_generated": False,
            "prism_updated": False,

            "locked_predictions_sha256":
                sha(PRED),

            "inputs_sha256": {
                p.relative_to(ROOT).as_posix(): sha(p)
                for p in (AH, AH_C, AI, AI_C, PRED)
            },

            "outputs_sha256": {
                OUT_C.relative_to(ROOT).as_posix():
                    sha(csvpath),
                **figures
            }
        }

        jpath = t / "report.json"

        jpath.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        need(
            not any(
                p.exists()
                for p in outputs
            ),
            "Output appeared during computation"
        )

        os.replace(csvpath, OUT_C)

        if fig_possible:
            os.replace(
                t / "figure.pdf",
                OUT_PDF
            )
            os.replace(
                t / "figure.png",
                OUT_PNG
            )

        os.replace(jpath, OUT_J)

    print(
        "=== STAGE 17AJ NCA ENDPOINT FIGURE EVIDENCE ==="
    )

    print(
        "PREDICTION_LOCK_SHA256_VERIFIED=YES"
    )

    print(
        f"ALL66_WITHIN25mAh_2500={within66} "
        f"SELECTED30_WITHIN25mAh_2500={within30}"
    )

    print(
        "NCA_SELECTED30_LAST_RATIO_MEDIAN=",
        report["selected30_last_ratio_median"]
    )

    print(
        "NCA_SELECTED30_2500_OVER_Q1_MEDIAN=",
        report["selected30_2500_over_Q1_median"]
    )

    print(
        "GROUP_COUNTS_ALL66=",
        json.dumps(report["program_counts_all66"])
    )

    print(
        "GROUP_COUNTS_SELECTED30=",
        json.dumps(
            report["program_counts_selected30"]
        )
    )

    print("FIGURE_CREATED=", fig_possible)

    if fig_possible:
        print("FIGURE_PDF=", OUT_PDF)
        print("FIGURE_PNG=", OUT_PNG)
    else:
        print(
            "FIGURE_SKIPPED_MATPLOTLIB_MISSING=",
            error
        )

    print("SOURCE_CSV=", OUT_C)
    print("PROVENANCE_JSON=", OUT_J)

    print(
        "DIAGNOSTIC_NOT_NEW_PREDICTION=YES "
        "STOP_RULE_CERTIFIED=NO"
    )

    print(
        "MODEL_FIT=NO MODEL_INFERENCE=NO "
        "PRISM_UPDATED=NO"
    )

    print("=== END STAGE 17AJ ===")


if __name__ == "__main__":
    main()
