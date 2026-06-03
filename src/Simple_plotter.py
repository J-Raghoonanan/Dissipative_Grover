#!/usr/bin/env python3
"""
Replot larger-system dissipative Grover figure from saved CSV data.

This script recreates the 2x2 larger-system plot directly from the CSV output
of the Hamiltonian simulation script, without recomputing any dynamics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CSV_NAME = (
    "GroverDissipative_fig_larger_system_common_blue_data_larger_system_"
    "n10_N1024_M1_rtop10_rbottom9_C5_dtpi_periods1p2.csv"
)

OUTPUT_PDF_NAME = CSV_NAME.replace(".csv", "_replot.pdf")


COLORS = {
    "standard": "tab:blue",
    "dissipative": "tab:red",
    "bj": "tab:green",
}


def configure_plot_style() -> None:
    """Use the same manuscript-style settings as the simulation script."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 13,
        "figure.titlesize": 24,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.6,
    })


def panel_sort_key(panel: str) -> int:
    """Sort panels a,b,c,d even if stored as '(a)' or 'a'."""
    panel = str(panel).strip().strip("()").lower()
    order = {"a": 0, "b": 1, "c": 2, "d": 3}
    return order.get(panel, 99)


def plot_curve(ax: plt.Axes, sub: pd.DataFrame, evolution: str, curve: str) -> None:
    """Plot one saved curve."""
    if sub.empty:
        return

    sub = sub.sort_values("t_over_tau")

    if curve == "standard_common_reference":
        label = r"$r=0$"
        kwargs = {"color": COLORS["standard"], "label": label, "linewidth": 1.0, "alpha": 0.4}
    elif curve == "standard":
        label = r"$r=0$"
        kwargs = {"color": COLORS["standard"], "label": label, "linewidth": 1.0, "alpha": 0.4}
    elif curve == "dissipative":
        r_val = int(sub["curve_r"].iloc[0])
        label = fr"$r={r_val}$"
        kwargs = {"color": COLORS["dissipative"], "label": label, "linewidth": 1.0}
    elif curve == "BJ":
        label = "BJ"
        kwargs = {"color": COLORS["bj"], "label": label, "linewidth": 1.0, "linestyle": "--"}
    else:
        return

    x = sub["t_over_tau"].to_numpy()
    y = sub["F"].to_numpy()

    if evolution == "trotterized" and curve in {"standard_common_reference", "standard", "dissipative"}:
        ax.step(x, y, where="post", **kwargs)
    else:
        ax.plot(x, y, **kwargs)


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / CSV_NAME
    pdf_path = here / OUTPUT_PDF_NAME

    df = pd.read_csv(csv_path)

    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, sharey=True)

    for ax, panel in zip(axes.ravel(), sorted(df["panel"].unique(), key=panel_sort_key)):
        panel_df = df[df["panel"] == panel].copy()
        evolution = str(panel_df["evolution"].iloc[0])
        r_panel = int(panel_df["panel_r_reservoir"].iloc[0])

        for curve in ["standard_common_reference", "standard", "dissipative", "BJ"]:
            curve_df = panel_df[panel_df["curve"] == curve]
            plot_curve(ax, curve_df, evolution=evolution, curve=curve)

        panel_label = str(panel).strip().strip("()")
        ax.set_title(fr"({panel_label}) {evolution}, $r={r_panel}$")
        ax.grid(False)
        ax.legend(loc="lower right", frameon=False)

    axes[0, 0].set_ylabel(r"$F$")
    axes[1, 0].set_ylabel(r"$F$")
    axes[1, 0].set_xlabel(r"$t/\tau$")
    axes[1, 1].set_xlabel(r"$t/\tau$")

    x_max = float(df["t_over_tau"].max())
    for ax in axes.ravel():
        # ax.set_xlim(0.0, x_max)
        ax.set_xlim(0.95, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
