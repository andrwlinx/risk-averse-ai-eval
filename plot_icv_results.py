#!/usr/bin/env python3
"""
Plot ICV sweep results from a directory of run_icv_eval.py JSON files.

Reads all eval_*.json files in --results_dir, groups by base_model, and
produces one PNG per model with subplots per layer configuration.

Usage:
    python plot_icv_results.py --results_dir results/2026-03-12_icv_ood
    python plot_icv_results.py --results_dir results/my_run --metric linear_rate
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_LABELS = {
    "linear_rate":    "Linear (risk-averse) rate",
    "cooperate_rate": "Cooperate rate",
    "rebel_rate":     "Rebel rate",
    "steal_rate":     "Steal rate",
    "cara_rate":      "CARA rate",
    "parse_rate":     "Parse rate",
}

# Map metric key in JSON → column name after normalisation
METRIC_KEYS = {
    "linear_rate":    "best_linear_rate",
    "cooperate_rate": "cooperate_rate",
    "rebel_rate":     "rebel_rate",
    "steal_rate":     "steal_rate",
    "cara_rate":      "best_cara_rate",
    "parse_rate":     "parse_rate",
}


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("eval_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        cfg = d.get("evaluation_config", {})
        met = d.get("metrics", {})
        rows.append({
            "base_model":    cfg.get("base_model", "unknown"),
            "alpha":         cfg.get("steering_alpha", float("nan")),
            "layer_label":   str(cfg.get("layer_label", "?")),
            "linear_rate":   met.get("best_linear_rate", float("nan")),
            "cooperate_rate":met.get("cooperate_rate",   float("nan")),
            "rebel_rate":    met.get("rebel_rate",       float("nan")),
            "steal_rate":    met.get("steal_rate",       float("nan")),
            "cara_rate":     met.get("best_cara_rate",   float("nan")),
            "parse_rate":    met.get("parse_rate",       float("nan")),
        })
    if not rows:
        raise ValueError(f"No eval_*.json files found in {results_dir}")
    return pd.DataFrame(rows)


def natural_layer_sort_key(lbl: str):
    """Sort layer labels numerically: single layers before multilayer combos."""
    parts = [int(x) for x in lbl.split("+")]
    return (len(parts), parts)


def plot_model(df_model: pd.DataFrame, model_name: str, metric: str,
               output_path: Path):
    layer_labels = sorted(df_model["layer_label"].unique(), key=natural_layer_sort_key)
    num_layers = len(layer_labels)
    ncols = min(4, num_layers)
    nrows = math.ceil(num_layers / ncols)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows + 1),
                             squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    # Baseline = alpha=0 linear_rate (averaged across layers)
    base_rows = df_model[df_model["alpha"] == 0.0]["linear_rate"]
    base_lin = base_rows.mean() if len(base_rows) > 0 else float("nan")

    for i, lbl in enumerate(layer_labels):
        ax = axes_flat[i]
        layer_data = df_model[df_model["layer_label"] == lbl].sort_values("alpha")

        # Primary metric
        ax.plot(layer_data["alpha"], layer_data[metric],
                marker="o", color="steelblue", label=METRIC_LABELS.get(metric, metric))

        # Always overlay linear_rate if plotting something else
        if metric != "linear_rate":
            ax.plot(layer_data["alpha"], layer_data["linear_rate"],
                    marker="s", color="green", linestyle="--", label="Linear rate")

        # Parse rate as faint background indicator
        ax.fill_between(layer_data["alpha"], 0, layer_data["parse_rate"],
                        alpha=0.08, color="gray", label="Parse rate")

        # Baseline alpha=0 line
        if not math.isnan(base_lin):
            ax.axhline(base_lin, color="gray", linestyle=":", alpha=0.7,
                       label=f"Baseline lin={base_lin:.0%}")

        title = f"Layers {lbl}" if "+" in lbl else f"Layer {lbl}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=7)

    # Hide unused subplots
    for j in range(num_layers, len(axes_flat)):
        axes_flat[j].set_visible(False)

    short = model_name.split("/")[-1]
    fig.suptitle(f"{short} — ICV steering sweep (OOD val set)\n"
                 f"Primary metric: {METRIC_LABELS.get(metric, metric)}",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing eval_*.json files")
    parser.add_argument("--metric", type=str, default="linear_rate",
                        choices=list(METRIC_LABELS.keys()),
                        help="Primary metric to plot (default: linear_rate)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save PNGs (default: same as results_dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    print(f"Loaded {len(df)} eval records from {results_dir}")
    print(f"Models: {df['base_model'].unique().tolist()}")

    for model_name, df_model in df.groupby("base_model"):
        short = model_name.split("/")[-1].replace("-", "").replace(".", "")
        out_path = output_dir / f"icv_sweep_{short}.png"
        plot_model(df_model, model_name, args.metric, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
