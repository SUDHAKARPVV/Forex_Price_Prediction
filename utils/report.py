"""
Human-readable reporting layer.

Turns the raw `evaluation_report.json` structure produced by
`training/evaluate.py` into:
    - PNG charts (overall metrics, directional accuracy vs. random baseline,
      per-horizon error curves, regime-segmented comparison)
    - A markdown metrics table
    - A single self-contained HTML report combining all of the above with
      an auto-generated plain-English narrative (best/worst model per
      metric, a flag when directional accuracy is statistically close to
      a coin flip, and a flag when the proposed Hybrid model underperforms
      a simpler baseline)

This is intentionally decoupled from the training loop: `generate_report()`
takes just the `reports` dict (the same structure written to
evaluation_report.json), so it can be run either automatically at the end
of `main.py`, or standalone against a report file you already have on disk
via `generate_report.py`.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless, no display backend needed
import matplotlib.pyplot as plt
import numpy as np

# Tidepool-inspired categorical palette, fixed order (never cycled)
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
BASELINE_RED = "#e34948"
GRID_GRAY = "#e1e0d9"
TEXT_MUTED = "#898781"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID_GRAY,
        "axes.grid": True,
        "grid.color": GRID_GRAY,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.size": 11,
        "text.color": "#0b0b0b",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
    }
)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# --------------------------------------------------------------------------
# Individual chart builders. Each returns a base64 PNG string and also
# optionally saves a standalone PNG to disk if `save_path` is given.
# --------------------------------------------------------------------------

def plot_price_predictions(dates, actual_price, predictions_by_model: Dict[str, "np.ndarray"], title: str, save_path: str = None) -> str:
    """Line chart of actual price vs. each model's predicted price over
    the test period, at a fixed horizon -- the direct "predict the forex
    price and compare with actual values" plot, matching how Paper 1's
    results are ultimately meant to be read (their tables report
    price-level error; this makes it visual).
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual_price, label="Actual", color="#0b0b0b", linewidth=1.8, zorder=10)
    for i, (model_name, pred) in enumerate(predictions_by_model.items()):
        ax.plot(dates, pred, label=model_name, color=PALETTE[i % len(PALETTE)], linewidth=1.2, alpha=0.85)
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return _fig_to_base64(fig)


def plot_overall_error(reports: Dict, save_path: str = None) -> str:
    models = list(reports.keys())
    mae = [reports[m]["overall"]["MAE"] for m in models]
    rmse = [reports[m]["overall"]["RMSE"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - width / 2, mae, width, label="MAE", color=PALETTE[0])
    ax.bar(x + width / 2, rmse, width, label="RMSE", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Error (log-return space)")
    ax.set_title("Overall forecast error by model — lower is better")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return _fig_to_base64(fig)


def plot_directional_accuracy(reports: Dict, save_path: str = None) -> str:
    models = list(reports.keys())
    acc = [reports[m]["overall"]["DirectionalAccuracy"] for m in models]
    clf_acc = [reports[m]["overall"].get("ClassifierDirectionalAccuracy") for m in models]
    has_clf = any(v is not None for v in clf_acc)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    if has_clf:
        x = np.arange(len(models))
        width = 0.35
        bars1 = ax.bar(x - width / 2, acc, width, label="Regression-derived", color=PALETTE[0])
        clf_plot = [v if v is not None else 0 for v in clf_acc]
        bars2 = ax.bar(x + width / 2, clf_plot, width, label="Classifier-derived", color=PALETTE[2])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        for b, v in zip(bars1, acc):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        for b, v in zip(bars2, clf_acc):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        all_vals = acc + [v for v in clf_acc if v is not None]
    else:
        bars = ax.bar(models, acc, color=PALETTE[0])
        for b, v in zip(bars, acc):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        all_vals = acc

    ax.axhline(0.5, color=BASELINE_RED, linestyle="--", linewidth=1.5, label="Random baseline (0.5)")
    ax.set_ylim(min(0.4, min(all_vals) - 0.03), max(0.55, max(all_vals) + 0.05))
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Directional accuracy vs. a coin-flip baseline")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return _fig_to_base64(fig)


def plot_per_horizon_mae(reports: Dict, save_path: str = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (model, rep) in enumerate(reports.items()):
        mae_curve = rep.get("per_horizon", {}).get("mae")
        if not mae_curve:
            continue
        horizons = np.arange(1, len(mae_curve) + 1)
        ax.plot(horizons, mae_curve, marker="o", markersize=4, linewidth=2, label=model, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Forecast horizon (steps ahead)")
    ax.set_ylabel("MAE")
    ax.set_title("Error growth across the multi-step forecast horizon")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return _fig_to_base64(fig)


def plot_regime_segmented(reports: Dict, metric: str = "directional_accuracy", save_path: str = None) -> str:
    models = [m for m in reports if reports[m].get("regime_segmented")]
    stable_vals, highvol_vals = [], []
    for m in models:
        seg = reports[m]["regime_segmented"]
        stable_vals.append(seg.get("stable", {}).get(metric, np.nan))
        highvol_vals.append(seg.get("high_volatility", {}).get(metric, np.nan))

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - width / 2, stable_vals, width, label="Stable regime", color=PALETTE[0])
    ax.bar(x + width / 2, highvol_vals, width, label="High-volatility regime", color=BASELINE_RED)
    if metric == "directional_accuracy":
        ax.axhline(0.5, color=TEXT_MUTED, linestyle=":", linewidth=1, label="Random baseline (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} — stable vs. high-volatility regime")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return _fig_to_base64(fig)


# --------------------------------------------------------------------------
# Markdown table + auto-narrative
# --------------------------------------------------------------------------

def build_markdown_table(reports: Dict) -> str:
    has_clf = any("ClassifierDirectionalAccuracy" in reports[m]["overall"] for m in reports)
    if has_clf:
        lines = [
            "| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |",
            "|---|---|---|---|---|---|",
        ]
        for model, rep in reports.items():
            o = rep["overall"]
            clf = o.get("ClassifierDirectionalAccuracy")
            clf_str = f"{clf:.4f}" if clf is not None else "n/a"
            lines.append(
                f"| {model} | {o['MAE']:.5f} | {o['RMSE']:.5f} | {o['MAPE']:.1f} | {o['DirectionalAccuracy']:.4f} | {clf_str} |"
            )
    else:
        lines = [
            "| Model | MAE | RMSE | MAPE (%) | Directional accuracy |",
            "|---|---|---|---|---|",
        ]
        for model, rep in reports.items():
            o = rep["overall"]
            lines.append(
                f"| {model} | {o['MAE']:.5f} | {o['RMSE']:.5f} | {o['MAPE']:.1f} | {o['DirectionalAccuracy']:.4f} |"
            )
    return "\n".join(lines)


def build_narrative(reports: Dict) -> list:
    """Auto-generate plain-English observations from the metrics — flags
    the model with best/worst MAE and directional accuracy, and raises a
    caution if the proposed Hybrid model does not beat the simplest
    baseline, or if accuracy is statistically indistinguishable from chance.
    """
    notes = []
    models = list(reports.keys())

    best_mae_model = min(models, key=lambda m: reports[m]["overall"]["MAE"])
    best_acc_model = max(models, key=lambda m: reports[m]["overall"]["DirectionalAccuracy"])
    notes.append(f"Lowest overall MAE: <strong>{best_mae_model}</strong> "
                 f"({reports[best_mae_model]['overall']['MAE']:.5f}).")
    notes.append(f"Highest directional accuracy: <strong>{best_acc_model}</strong> "
                 f"({reports[best_acc_model]['overall']['DirectionalAccuracy']:.4f}).")

    hybrid_name = next((m for m in models if "Hybrid" in m), None)
    if hybrid_name:
        hybrid_acc = reports[hybrid_name]["overall"]["DirectionalAccuracy"]
        hybrid_mae = reports[hybrid_name]["overall"]["MAE"]
        others = [m for m in models if m != hybrid_name]
        beats_all_mae = all(hybrid_mae <= reports[m]["overall"]["MAE"] for m in others)
        beats_all_acc = all(hybrid_acc >= reports[m]["overall"]["DirectionalAccuracy"] for m in others)
        if not beats_all_mae or not beats_all_acc:
            worse_than = [m for m in others if reports[m]["overall"]["DirectionalAccuracy"] > hybrid_acc]
            notes.append(
                "<span style='color:#a32d2d'><strong>Caution:</strong></span> the proposed Hybrid model does "
                f"not outperform {', '.join(worse_than) if worse_than else 'the simpler baselines'} on this run. "
                "On data without a strong, real cross-modal signal, extra model capacity tends to fit noise "
                "rather than add predictive power — see the README for guidance on validating the architecture "
                "against data with a known injected signal, and on real market data once available."
            )

    for model in models:
        acc = reports[model]["overall"]["DirectionalAccuracy"]
        if 0.47 <= acc <= 0.53:
            notes.append(
                f"<strong>{model}</strong>'s directional accuracy ({acc:.4f}) is close to the 0.5 random-guess "
                "baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill."
            )

    return notes


# --------------------------------------------------------------------------
# Full HTML report
# --------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FX Forecasting — Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; color: #0b0b0b; line-height: 1.6; }}
  h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 18px; font-weight: 600; margin-top: 2.5rem; border-top: 1px solid #e1e0d9; padding-top: 1.5rem; }}
  .subtitle {{ color: #52514e; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e1e0d9; }}
  th {{ color: #52514e; font-weight: 600; }}
  img {{ max-width: 100%; border: 1px solid #e1e0d9; border-radius: 8px; margin: 0.5rem 0 1.5rem; }}
  ul.notes li {{ margin-bottom: 0.6rem; }}
  .caveat {{ background: #faece7; border-left: 3px solid #d85a30; padding: 0.75rem 1rem; border-radius: 4px; font-size: 14px; margin: 1rem 0; }}
</style>
</head>
<body>
<h1>FX forecasting — model evaluation report</h1>
<p class="subtitle">Hybrid CNN-LSTM-Transformer vs. Vanilla LSTM, Simplified TFT, and ARIMA baselines</p>

<div class="caveat">This report is generated from the evaluation run's raw metrics. If the underlying
run used the project's synthetic placeholder data, treat all numbers as a pipeline correctness check,
not a real forecasting-skill claim — see the README for details.</div>

<div class="caveat">A note on realistic expectations: published multi-step FX directional-accuracy
results rarely exceed 55-65% on real market data, and even single-step results in the 90s (as some
literature reports) are typically classification tasks on visually-encoded price patterns, not
multi-step return regression. Treat any number materially above that range with the same scrutiny
you'd apply to a headline result in a paper — check the data source and evaluation protocol before
trusting it.</div>

<h2>Summary table</h2>
{table_html}

<h2>Key observations</h2>
<ul class="notes">
{narrative_html}
</ul>

<h2>Overall error (MAE / RMSE)</h2>
<img src="data:image/png;base64,{img_error}" alt="Overall MAE and RMSE by model">

<h2>Directional accuracy vs. random baseline</h2>
<img src="data:image/png;base64,{img_diracc}" alt="Directional accuracy by model against 0.5 baseline">

<h2>Error across the forecast horizon</h2>
<img src="data:image/png;base64,{img_horizon}" alt="Per-horizon MAE curves by model">

<h2>Regime-segmented directional accuracy</h2>
<img src="data:image/png;base64,{img_regime}" alt="Directional accuracy split by stable and high-volatility regime">

{price_chart_section}
</body>
</html>
"""


def _markdown_table_to_html(md_table: str) -> str:
    rows = [r for r in md_table.strip().split("\n") if r.strip()]
    header_cells = [c.strip() for c in rows[0].strip("|").split("|")]
    body_rows = rows[2:]
    html = ["<table>", "<tr>" + "".join(f"<th>{c}</th>" for c in header_cells) + "</tr>"]
    for row in body_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        html.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    html.append("</table>")
    return "\n".join(html)


def generate_report(reports: Dict, output_dir: str = "report", price_predictions: Dict = None) -> str:
    """Build the full human-readable report (charts + table + narrative)
    from a `reports` dict shaped like evaluation_report.json.

    `price_predictions`, if given, should be:
        {"dates": ..., "actual": ..., "by_model": {"ModelName": pred_array, ...}, "horizon_label": "..."}
    and adds a price-level "predicted vs actual" chart to the report --
    see utils/price_reconstruction.py for how to build this.

    Writes:
        {output_dir}/report.html          -- full combined report
        {output_dir}/SUMMARY.md           -- markdown table + narrative, for quick reading
        {output_dir}/charts/*.png         -- standalone PNG charts

    Returns the path to report.html.
    """
    os.makedirs(output_dir, exist_ok=True)
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    img_error = plot_overall_error(reports, save_path=os.path.join(charts_dir, "overall_error.png"))
    img_diracc = plot_directional_accuracy(reports, save_path=os.path.join(charts_dir, "directional_accuracy.png"))
    img_horizon = plot_per_horizon_mae(reports, save_path=os.path.join(charts_dir, "per_horizon_mae.png"))
    img_regime = plot_regime_segmented(reports, save_path=os.path.join(charts_dir, "regime_segmented.png"))

    price_chart_section = ""
    if price_predictions:
        img_price = plot_price_predictions(
            price_predictions["dates"],
            price_predictions["actual"],
            price_predictions["by_model"],
            title=f"Predicted vs. actual price ({price_predictions.get('horizon_label', 't+1')}, test period)",
            save_path=os.path.join(charts_dir, "price_predictions.png"),
        )
        price_chart_section = (
            "<h2>Predicted vs. actual price</h2>\n"
            f'<img src="data:image/png;base64,{img_price}" alt="Predicted vs actual price over the test period">'
        )

    md_table = build_markdown_table(reports)
    table_html = _markdown_table_to_html(md_table)
    narrative = build_narrative(reports)
    narrative_html = "\n".join(f"<li>{n}</li>" for n in narrative)

    html = _HTML_TEMPLATE.format(
        table_html=table_html,
        narrative_html=narrative_html,
        img_error=img_error,
        img_diracc=img_diracc,
        img_horizon=img_horizon,
        img_regime=img_regime,
        price_chart_section=price_chart_section,
    )

    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(html)

    summary_md = "# FX forecasting — evaluation summary\n\n" + md_table + "\n\n## Key observations\n\n"
    summary_md += "\n".join(f"- {_strip_html(n)}" for n in narrative)
    with open(os.path.join(output_dir, "SUMMARY.md"), "w") as f:
        f.write(summary_md)

    return html_path


def _strip_html(text: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", text)
