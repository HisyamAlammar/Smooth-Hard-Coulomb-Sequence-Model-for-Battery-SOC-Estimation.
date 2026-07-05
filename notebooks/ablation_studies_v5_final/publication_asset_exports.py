from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
ASSET_DIR = ROOT / "outputs" / "manuscript_assets"

COLORS = {
    "teal": "#2A9D8F",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#7E57C2",
    "gray": "#8C8C8C",
    "light_gray": "#D8DEE9",
    "ink": "#263238",
}

FIG_SINGLE = (3.35, 2.45)
FIG_FULL = (6.85, 3.1)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
})


def _read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _read_json(rel: str):
    with (ROOT / rel).open(encoding="utf-8") as f:
        return json.load(f)


def _save_figure(fig, stem: str, dpi: int = 600):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    # Empirical remediation policy: SVG export disabled for manuscript assets.
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        paths[ext] = path
    plt.close(fig)
    return paths


def _write_csv(path: Path, rows: list[dict] | pd.DataFrame):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _metric(mean, std=None, nd=2):
    return _fmt(mean, nd) if std in (None, "", np.nan) else f"{_fmt(mean, nd)} ± {_fmt(std, nd)}"


def _basic_axes(ax):
    ax.tick_params(axis="both", length=3, color="#607D8B")
    ax.set_axisbelow(True)


def export_fig01_final_architecture():
    fig, ax = plt.subplots(figsize=FIG_FULL)
    ax.axis("off")
    boxes = [
        ("Input window\nV, I, T, features", 0.04, 0.68, 0.17, 0.14, "#E8EEF7"),
        ("LSTM encoder\nh₁ … h_T", 0.28, 0.68, 0.17, 0.14, "#E3F2EA"),
        ("Anchor head\nh_T", 0.56, 0.76, 0.16, 0.11, "#F1E8F7"),
        ("Delta head\n|Δŝ_t|", 0.56, 0.56, 0.16, 0.11, "#FFF0D9"),
        ("Hard-Coulomb route\nsign(I_t), |I_t|·η·γ", 0.28, 0.35, 0.25, 0.13, "#E8F1F7"),
        ("SOC trajectory\nŝ₀ + ΣΔŝ", 0.71, 0.37, 0.20, 0.13, "#DCEFD8"),
        ("Carried inference\nη*=2.0", 0.36, 0.12, 0.22, 0.12, "#F7F1D8"),
    ]
    for text, x, y, w, h, fill in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor=COLORS["ink"], lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2)
    arrows = [((0.21, 0.75), (0.28, 0.75)), ((0.45, 0.75), (0.56, 0.81)),
              ((0.45, 0.72), (0.56, 0.61)), ((0.64, 0.56), (0.50, 0.48)),
              ((0.64, 0.76), (0.77, 0.50)), ((0.53, 0.42), (0.71, 0.43)),
              ((0.80, 0.37), (0.52, 0.24)), ((0.47, 0.24), (0.42, 0.35))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.0, color="#546E7A"))
    ax.text(0.04, 0.23, "Structural claim: measured-current sign consistency.\nOut of scope: sensor-fault physical correctness.",
            ha="left", va="center", fontsize=7.8, color=COLORS["ink"])
    return _save_figure(fig, "fig01_final_architecture", 600)


def export_fig02_research_evolution_flowchart():
    fig, ax = plt.subplots(figsize=FIG_FULL)
    ax.axis("off")
    labels = ["Seq2Point\ncold start", "Vanilla\nblindness", "Soft penalty\ninsufficient",
              "Post-hoc\nclamp fails", "HC first\nanchor limit", "v5c\ncorrection",
              "anchor_last", "carried\ninference", "η*=2.0", "final\nsystem"]
    fills = ["#F8DCDC", "#F8DCDC", "#FFF0D9", "#F8DCDC", "#E8F1F7",
             "#E8EEF7", "#DCEFD8", "#E3F2EA", "#FFF0D9", "#DCEFD8"]
    xs = np.linspace(0.05, 0.95, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.text(x, 0.56, label, ha="center", va="center", fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.22", fc=fills[i], ec=COLORS["ink"], lw=0.7))
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.035, 0.56), xytext=(x + 0.035, 0.56),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="#546E7A"))
    return _save_figure(fig, "fig02_research_evolution_flowchart", 600)


def export_fig03_dataset_v5_correction():
    df = _read_csv("results/v5/dataset_variant_comparison.csv")
    a = df[df["scenario"].eq("A")].copy()
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.bar(a["variant"], a["label_shift_vs_v4_mean_pct"], color=[COLORS["gray"], COLORS["orange"], COLORS["sky"], COLORS["vermillion"]])
    ax.set_xlabel("Dataset variant")
    ax.set_ylabel("Mean label shift vs v4 (%SOC)")
    for i, v in enumerate(a["label_shift_vs_v4_mean_pct"]):
        ax.text(i, float(v) + 0.05, f"{float(v):.2f}", ha="center", fontsize=7)
    _basic_axes(ax)
    return _save_figure(fig, "fig03_dataset_v5_correction", 600)


def export_fig04_multiseed_model_comparison():
    ms = _read_csv("results/v5/multiseed/multiseed_summary.csv")
    null_a, null_b = 13.0297, 7.5278
    order = ["null", "vanilla_lstm", "hard_coulomb_lstm", "hard_coulomb_tcn", "hc_anchor_pooled", "hc_anchor_last"]
    labels = ["Null", "Vanilla", "HC first", "HC-TCN", "anchor_pooled", "anchor_last"]
    a_vals, b_vals, a_err, b_err = [], [], [], []
    for m in order:
        if m == "null":
            a_vals.append(null_a); b_vals.append(null_b); a_err.append(0); b_err.append(0)
        else:
            ra = ms[(ms.model == m) & (ms.scenario == "scenario_A")].iloc[0]
            rb = ms[(ms.model == m) & (ms.scenario == "scenario_B")].iloc[0]
            a_vals.append(ra.rmse_pct_mean); b_vals.append(rb.rmse_pct_mean)
            a_err.append(ra.rmse_pct_std); b_err.append(rb.rmse_pct_std)
    fig, ax = plt.subplots(figsize=FIG_FULL)
    x = np.arange(len(order))
    ax.bar(x - 0.18, a_vals, 0.36, yerr=a_err, label="Scenario A", color=COLORS["sky"], capsize=2)
    ax.bar(x + 0.18, b_vals, 0.36, yerr=b_err, label="Scenario B", color=COLORS["vermillion"], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("RMSE (%SOC)")
    ax.legend(frameon=False, ncols=2)
    _basic_axes(ax)
    return _save_figure(fig, "fig04_multiseed_model_comparison", 600)


def export_fig05_recursive_policy_comparison():
    df = _read_csv("results/v5/recursive_inference/recursive_policy_comparison.csv")
    keep = ["windowed_independent", "carried_anchor", "load_gated"]
    sub = df.set_index("policy").loc[keep]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    x = np.arange(len(keep))
    ax.bar(x - 0.17, sub["rmse_pct"], 0.34, label="Overall", color=COLORS["sky"])
    ax.bar(x + 0.17, sub["rmse_n20degC"], 0.34, label="−20 °C", color=COLORS["vermillion"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Windowed", "Carried", "Load-gated"], rotation=12, ha="right")
    ax.set_ylabel("RMSE (%SOC)")
    ax.legend(frameon=False)
    _basic_axes(ax)
    return _save_figure(fig, "fig05_recursive_policy_comparison", 600)


def export_fig06_eta_calibration_delta_ratio():
    rows = _read_json("results/v5/delta_calibration/eta_gamma_sweep.json")["rows"]
    nom = [r for r in rows if r["mode"] == "inference_sweep" and r["gamma_mode"] == "nominal"]
    xs = [r["eta"] for r in nom]
    ratio = [r["gated_delta_ratio"] for r in nom]
    rmse = [r["rec_rmse_pct"] for r in nom]
    fig, ax1 = plt.subplots(figsize=FIG_SINGLE)
    ax1.plot(xs, ratio, marker="o", color=COLORS["teal"], label="Delta ratio")
    ax1.axhline(1.0, color="#546E7A", lw=0.9, ls="--")
    ax1.set_xlabel("eta")
    ax1.set_ylabel("Delta ratio")
    ax2 = ax1.twinx()
    ax2.plot(xs, rmse, marker="s", color=COLORS["vermillion"], label="Recursive RMSE")
    ax2.set_ylabel("RMSE (%SOC)")
    ax1.grid(axis="y", alpha=0.18)
    ax1.text(2.02, 1.04, "η*=2.0", fontsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    return _save_figure(fig, "fig06_eta_calibration_delta_ratio", 600)


def export_fig07_ekf_vs_calibrated_hc():
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    labels = ["Calibrated HC", "Best EKF", "EKF low-R"]
    rmse = [4.4318, 6.8485, 39.1161]
    pvr = [0.0, 5.4755, 33.6695]
    x = np.arange(len(labels))
    ax.bar(x - 0.17, rmse, 0.34, label="RMSE", color=[COLORS["vermillion"], COLORS["purple"], COLORS["purple"]])
    ax.bar(x + 0.17, pvr, 0.34, label="PVR", color=[COLORS["teal"], COLORS["orange"], COLORS["orange"]])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("%")
    ax.legend(frameon=False)
    _basic_axes(ax)
    return _save_figure(fig, "fig07_ekf_vs_calibrated_hc", 600)


def export_tables():
    final = _read_csv("results/v5/final_v5_model_comparison.csv")
    ms = _read_csv("results/v5/multiseed/multiseed_summary.csv")
    rec = _read_csv("results/v5/recursive_inference/recursive_policy_comparison.csv")
    eta_csv = _read_csv("results/v5/delta_calibration/eta_gamma_sweep.csv")
    eta_json_rows = _read_json("results/v5/delta_calibration/eta_gamma_sweep.json")["rows"]
    ekf = _read_csv("results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv")
    claims = _read_json("reports/v5_campaign/claims_register_v2.json")["claims"]

    _write_csv(TABLE_DIR / "table01_dataset_variants.csv", _read_csv("results/v5/dataset_variant_comparison.csv"))

    failure_rows = [
        {"failure_mode": "Windowed cold-start artifact", "evidence": "−20 °C windowed RMSE 16.75; carried cuts cold error", "fix": "carried inference", "final_role": "deployment-like continuity"},
        {"failure_mode": "Physics-blind vanilla trajectory", "evidence": "unconstrained trajectory can violate sign consistency", "fix": "Hard-Coulomb routing", "final_role": "measured-current sign consistency"},
        {"failure_mode": "Post-hoc clamp collapse", "evidence": "25.03/30.07% RMSE", "fix": "training-through-constraint", "final_role": "structural learning"},
        {"failure_mode": "Anchor-first bottleneck", "evidence": "original HC Scenario B 10.63±0.60, 5/5 seed failure", "fix": "anchor_last", "final_role": "window-level anchor observability"},
        {"failure_mode": "Delta-path underestimation", "evidence": "delta ratio 0.751", "fix": "eta*=2.0 calibration", "final_role": "rate fidelity 1.002"},
        {"failure_mode": "EKF correction inconsistency", "evidence": "EKF PVR 5–40%", "fix": "calibrated HC comparison", "final_role": "consistency-preserving alternative"},
    ]
    _write_csv(TABLE_DIR / "table02_failure_mode_rationale.csv", failure_rows)

    def final_row(model, scenario):
        return final[(final.model == model) & (final.scenario == scenario)].iloc[0]

    def ms_row(model, scenario):
        return ms[(ms.model == model) & (ms.scenario == scenario)].iloc[0]

    eta2 = next(r for r in eta_json_rows if r["mode"] == "inference_sweep" and r["gamma_mode"] == "nominal" and r["eta"] == 2.0)
    best_ekf = ekf[ekf.model.eq("EKF_1RC_cont[R=0.01]")].iloc[0]
    table03 = [
        {"method": "Null OCV+Coulomb", "scenario_A_RMSE": _fmt(final_row("null[ocv25_qnom]", "scenario_A").rmse_pct), "scenario_B_RMSE": _fmt(final_row("null[ocv25_qnom]", "scenario_B").rmse_pct), "note": "deterministic reference"},
        {"method": "Vanilla LSTM", "scenario_A_RMSE": _metric(ms_row("vanilla_lstm", "scenario_A").rmse_pct_mean, ms_row("vanilla_lstm", "scenario_A").rmse_pct_std), "scenario_B_RMSE": _metric(ms_row("vanilla_lstm", "scenario_B").rmse_pct_mean, ms_row("vanilla_lstm", "scenario_B").rmse_pct_std), "note": "unconstrained"},
        {"method": "Post-hoc clamp", "scenario_A_RMSE": _fmt(final_row("vanilla+posthoc_clamp", "scenario_A").rmse_pct), "scenario_B_RMSE": _fmt(final_row("vanilla+posthoc_clamp", "scenario_B").rmse_pct), "note": "collapse"},
        {"method": "Original HC", "scenario_A_RMSE": _metric(ms_row("hard_coulomb_lstm", "scenario_A").rmse_pct_mean, ms_row("hard_coulomb_lstm", "scenario_A").rmse_pct_std), "scenario_B_RMSE": _metric(ms_row("hard_coulomb_lstm", "scenario_B").rmse_pct_mean, ms_row("hard_coulomb_lstm", "scenario_B").rmse_pct_std), "note": "anchor-first"},
        {"method": "anchor_last", "scenario_A_RMSE": _metric(ms_row("hc_anchor_last", "scenario_A").rmse_pct_mean, ms_row("hc_anchor_last", "scenario_A").rmse_pct_std), "scenario_B_RMSE": _metric(ms_row("hc_anchor_last", "scenario_B").rmse_pct_mean, ms_row("hc_anchor_last", "scenario_B").rmse_pct_std), "note": "selected windowed model"},
        {"method": "Calibrated recursive HC", "scenario_A_RMSE": _fmt(eta2["rec_rmse_pct"]), "scenario_B_RMSE": "n/a", "note": "Scenario A seed 42, eta*=2.0"},
        {"method": "Best EKF 1RC", "scenario_A_RMSE": _fmt(best_ekf.rmse_pct), "scenario_B_RMSE": "n/a", "note": "continuous EKF, literature-like params"},
    ]
    _write_csv(TABLE_DIR / "table03_main_model_comparison.csv", table03)

    table04 = ms[ms["model"].isin(["vanilla_lstm", "hard_coulomb_lstm", "hc_anchor_last", "hc_anchor_pooled", "hard_coulomb_tcn"])][
        ["model", "scenario", "rmse_pct_mean", "rmse_pct_std", "maxe_pct_mean", "maxe_pct_std", "rmse_n20degC_mean", "rmse_40degC_mean"]
    ]
    _write_csv(TABLE_DIR / "table04_multiseed_summary.csv", table04)

    rec_rows = rec[rec["policy"].isin(["windowed_independent", "carried_anchor", "load_gated"])][
        ["policy", "rmse_pct", "maxe_pct", "pvr_disch_eps0", "delta_ratio_disch", "rmse_n20degC", "rmse_40degC", "reanchor_pct"]
    ].copy()
    rec_rows["source"] = "recursive_policy_comparison.csv"
    eta_row = pd.DataFrame([{
        "policy": "eta2_calibrated_carried",
        "rmse_pct": eta2["rec_rmse_pct"],
        "maxe_pct": eta2["rec_maxe_pct"],
        "pvr_disch_eps0": 0.0,
        "delta_ratio_disch": eta2["rec_delta_ratio"],
        "rmse_n20degC": eta2["rec_rmse_n20"],
        "rmse_40degC": eta2["rec_rmse_40"],
        "reanchor_pct": 0.28,
        "source": "eta_gamma_sweep.json",
    }])
    _write_csv(TABLE_DIR / "table05_recursive_eta_ablation.csv", pd.concat([rec_rows, eta_row], ignore_index=True))

    table06 = pd.DataFrame([
        {"method": "Calibrated recursive HC", "rmse_pct": 4.4318, "rmse_n20": 3.5936, "pvr_disch_eps0": 0.0, "limitation": "single checkpoint/scenario"},
        {"method": "Best EKF 1RC R=1e-2", "rmse_pct": best_ekf.rmse_pct, "rmse_n20": best_ekf.rmse_n20, "pvr_disch_eps0": best_ekf.pvr_disch_eps0, "limitation": "literature-like parameters"},
        {"method": "EKF 1RC R=1e-4", "rmse_pct": ekf[ekf.model.eq("EKF_1RC_cont[R=0.0001]")].iloc[0].rmse_pct, "rmse_n20": ekf[ekf.model.eq("EKF_1RC_cont[R=0.0001]")].iloc[0].rmse_n20, "pvr_disch_eps0": ekf[ekf.model.eq("EKF_1RC_cont[R=0.0001]")].iloc[0].pvr_disch_eps0, "limitation": "over-trust voltage"},
    ])
    _write_csv(TABLE_DIR / "table06_ekf_comparison.csv", table06)

    claim_ids = {1, 2, 5, 6, 7, 11, 12, 15, 16, 17}
    table07 = [{"id": c["id"], "status": c["status"], "claim": c["claim"], "evidence": c["evidence"]} for c in claims if c["id"] in claim_ids]
    _write_csv(TABLE_DIR / "table07_claim_summary.csv", table07)


FIGURE_MANIFEST = [
    ("fig01_final_architecture", "15_Anchor_Last_vs_Anchor_First_Observability.ipynb", "experiments/compare_anchor_variants.py; results/v5/delta_calibration/eta_gamma_sweep.json", "Final architecture: anchor_last plus calibrated carried inference.", "Methods: final system overview.", "full-width", "Diagram generated from verified architecture notes, not a measured plot."),
    ("fig02_research_evolution_flowchart", "19_Final_Ablation_Matrix_and_Claims_Register.ipynb", "reports/v5_campaign/phase9_final_v5_report.md; claims_register_v2.json", "Staged failure analysis leading to final system.", "Results: rationale for final system.", "full-width", "Conceptual synthesis figure."),
    ("fig03_dataset_v5_correction", "12_Dataset_v5_Label_Decimation_Correction.ipynb", "results/v5/dataset_variant_comparison.csv", "v5c changes label magnitudes while preserving split comparability.", "Dataset correction subsection.", "one-column", "Shows mean label shift, not full label uncertainty."),
    ("fig04_multiseed_model_comparison", "14_Multiseed_Stability_and_Model_Ranking.ipynb", "results/v5/multiseed/multiseed_summary.csv; results/v5/final_v5_model_comparison.csv", "anchor_last is best mean performer across Scenario A/B with seed variance shown.", "Results: anchor stability.", "full-width", "Null reference has no seed error bar."),
    ("fig05_recursive_policy_comparison", "16_Windowed_vs_Carried_vs_Load_Gated_Recursive_Inference.ipynb", "results/v5/recursive_inference/recursive_policy_comparison.csv", "Recursive policies reduce repeated cold-start behavior with cold/warm tradeoff.", "Results: recursive inference.", "one-column", "Single checkpoint Scenario A seed 42."),
    ("fig06_eta_calibration_delta_ratio", "17_Eta_Calibration_Delta_Path_Rate_Recovery.ipynb", "results/v5/delta_calibration/eta_gamma_sweep.json", "eta*=2.0 recovers delta ratio near 1 and minimizes recursive RMSE.", "Results: eta calibration.", "one-column", "Single checkpoint Scenario A seed 42."),
    ("fig07_ekf_vs_calibrated_hc", "18_Continuous_EKF_1RC_Correction_vs_Consistency_Frontier.ipynb", "results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv; results/v5/delta_calibration/eta_gamma_sweep.json", "Best tested EKF has higher RMSE and nonzero PVR than calibrated HC.", "Results: EKF comparison.", "one-column", "EKF parameters are literature-like, not cell-identified."),
]


TABLE_MANIFEST = [
    ("table01_dataset_variants.csv", "12_Dataset_v5_Label_Decimation_Correction.ipynb", "results/v5/dataset_variant_comparison.csv", "Dataset correction variants and split counts.", "Methods/Data.", "one-column", "Scenario A/B rows retained in CSV."),
    ("table02_failure_mode_rationale.csv", "19_Final_Ablation_Matrix_and_Claims_Register.ipynb", "reports/v5_campaign/phase9_final_v5_report.md; claims_register_v2.json", "Failure mode to evidence to fix rationale.", "Results first subsection.", "full-width", "Synthesis table."),
    ("table03_main_model_comparison.csv", "13_v4_vs_v5_Result_Shift_and_Label_Artifact.ipynb", "results/v5/final_v5_model_comparison.csv; results/v5/multiseed/multiseed_summary.csv", "Main model comparison for manuscript.", "Results main comparison.", "full-width", "Recursive calibrated row is Scenario A seed 42."),
    ("table04_multiseed_summary.csv", "14_Multiseed_Stability_and_Model_Ranking.ipynb", "results/v5/multiseed/multiseed_summary.csv", "Multi-seed stability metrics.", "Results anchor stability.", "full-width", "Null omitted; learned models only."),
    ("table05_recursive_eta_ablation.csv", "16_Windowed_vs_Carried_vs_Load_Gated_Recursive_Inference.ipynb; 17_Eta_Calibration_Delta_Path_Rate_Recovery.ipynb", "results/v5/recursive_inference/recursive_policy_comparison.csv; results/v5/delta_calibration/eta_gamma_sweep.json", "Recursive policy and eta calibration ablation.", "Results recursive/eta.", "full-width", "Single checkpoint Scenario A seed 42."),
    ("table06_ekf_comparison.csv", "18_Continuous_EKF_1RC_Correction_vs_Consistency_Frontier.ipynb", "results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv; results/v5/delta_calibration/eta_gamma_sweep.json", "EKF vs calibrated HC frontier.", "Results EKF comparison.", "one-column", "EKF params literature-like."),
    ("table07_claim_summary.csv", "19_Final_Ablation_Matrix_and_Claims_Register.ipynb", "reports/v5_campaign/claims_register_v2.json", "Supported and limited claims for safe manuscript language.", "Appendix/limitations.", "full-width", "Not full claims register; curated subset."),
]


def write_manifests(validation: dict | None = None):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig_entries = []
    for stem, nb, src, msg, placement, width, caveat in FIGURE_MANIFEST:
        fig_entries.append({
            "asset_filename": f"outputs/figures/{stem}.png",
            "archive_files": [f"outputs/figures/{stem}.pdf"],
            "source_notebook": f"notebooks/ablation_studies_v5_final/{nb}",
            "source_data_file": src,
            "scientific_message": msg,
            "recommended_manuscript_placement": placement,
            "layout": width,
            "caveat": caveat,
        })
    table_entries = []
    for filename, nb, src, msg, placement, width, caveat in TABLE_MANIFEST:
        table_entries.append({
            "asset_filename": f"outputs/tables/{filename}",
            "source_notebook": f"notebooks/ablation_studies_v5_final/{nb}",
            "source_data_file": src,
            "scientific_message": msg,
            "recommended_manuscript_placement": placement,
            "layout": width,
            "caveat": caveat,
        })
    (ASSET_DIR / "figure_manifest.json").write_text(json.dumps(fig_entries, indent=2), encoding="utf-8")
    (ASSET_DIR / "table_manifest.json").write_text(json.dumps(table_entries, indent=2), encoding="utf-8")
    if validation is None:
        validation = validate_assets(write_report=False)
    report = ["# Manuscript Asset Export Report", "", "## Summary"]
    report.append(f"- Required PNG figures present: {validation['png_present']}/{validation['png_required']}")
    report.append(f"- Required CSV tables present: {validation['csv_present']}/{validation['csv_required']}")
    report.append(f"- Blank figure check: {'PASS' if validation['blank_check_pass'] else 'FAIL'}")
    report.append(f"- Notebook export cells present: {validation['notebook_export_cells_present']}/8")
    report.append("",)
    report.append("## Figures")
    for item in fig_entries:
        p = ROOT / item["asset_filename"]
        dims = validation["figures"].get(p.name, {})
        report.append(f"- `{item['asset_filename']}`: {dims.get('width')}x{dims.get('height')} px, {dims.get('size_bytes')} bytes, {item['layout']}. Source: {item['source_data_file']}.")
    report.append("")
    report.append("## Tables")
    for item in table_entries:
        p = ROOT / item["asset_filename"]
        rows = validation["tables"].get(p.name, {})
        report.append(f"- `{item['asset_filename']}`: {rows.get('rows')} rows, {rows.get('columns')} columns. Source: {item['source_data_file']}.")
    report.append("")
    report.append("## Lightweight Notebook Execution")
    report.append("- Export cells call `export_for_notebook('<notebook-id>')`, which loads existing CSV/JSON artifacts only and does not invoke training scripts.")
    report.append(f"- Export cell dry-run status: {validation.get('export_cell_dry_run', 'not_run')}.")
    report.append("- Full `jupyter nbconvert` execution was not run because the local Jupyter CLI is unavailable.")
    report.append("")
    report.append("## Caveats")
    report.append("- Recursive, eta, and EKF assets use Scenario A seed 42 deployment-style evidence as documented in v5 reports.")
    report.append("- Figure captions are intentionally not embedded in images; use manuscript captions.")
    report.append("- PDF archives are exported for every required figure; SVG export is disabled by empirical-only policy.")
    (ASSET_DIR / "asset_export_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def validate_assets(write_report: bool = True) -> dict:
    required_png = [f"{stem}.png" for stem, *_ in FIGURE_MANIFEST]
    required_csv = [filename for filename, *_ in TABLE_MANIFEST]
    result = {
        "png_required": len(required_png),
        "png_present": 0,
        "csv_required": len(required_csv),
        "csv_present": 0,
        "blank_check_pass": True,
        "figures": {},
        "tables": {},
        "notebook_export_cells_present": 0,
        "export_cell_dry_run": "not_run",
    }
    for name in required_png:
        path = FIG_DIR / name
        if path.exists():
            result["png_present"] += 1
            with Image.open(path) as im:
                arr = np.asarray(im.convert("L"))
                blank = bool(arr.std() < 1.0)
                if blank:
                    result["blank_check_pass"] = False
                result["figures"][name] = {
                    "width": im.width,
                    "height": im.height,
                    "size_bytes": path.stat().st_size,
                    "blank": blank,
                }
    for name in required_csv:
        path = TABLE_DIR / name
        if path.exists():
            result["csv_present"] += 1
            df = pd.read_csv(path)
            result["tables"][name] = {"rows": int(len(df)), "columns": int(len(df.columns)), "size_bytes": path.stat().st_size}
    marker = "PUBLICATION_ASSET_EXPORT_V1"
    for nb in (ROOT / "notebooks" / "ablation_studies_v5_final").glob("*.ipynb"):
        if marker in nb.read_text(encoding="utf-8"):
            result["notebook_export_cells_present"] += 1
    if write_report:
        write_manifests(result)
    return result


def export_all():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    export_fig01_final_architecture()
    export_fig02_research_evolution_flowchart()
    export_fig03_dataset_v5_correction()
    export_fig04_multiseed_model_comparison()
    export_fig05_recursive_policy_comparison()
    export_fig06_eta_calibration_delta_ratio()
    export_fig07_ekf_vs_calibrated_hc()
    export_tables()
    validation = validate_assets(write_report=False)
    validation["export_cell_dry_run"] = "helper_execution_passed; full_jupyter_not_available"
    write_manifests(validation)
    return validation


def export_for_notebook(notebook_id: str):
    mapping = {
        "12": [export_fig03_dataset_v5_correction, export_tables],
        "13": [export_tables],
        "14": [export_fig04_multiseed_model_comparison, export_tables],
        "15": [export_fig01_final_architecture],
        "16": [export_fig05_recursive_policy_comparison, export_tables],
        "17": [export_fig06_eta_calibration_delta_ratio, export_tables],
        "18": [export_fig07_ekf_vs_calibrated_hc, export_tables],
        "19": [export_fig02_research_evolution_flowchart, export_tables],
    }
    for fn in mapping.get(str(notebook_id), []):
        fn()
    validation = validate_assets(write_report=False)
    validation["export_cell_dry_run"] = "helper_execution_passed; full_jupyter_not_available"
    write_manifests(validation)
    return validation


if __name__ == "__main__":
    print(json.dumps(export_all(), indent=2))
