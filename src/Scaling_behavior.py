#!/usr/bin/env python3
"""
Unknown-M scaling study for dissipative Grover search.

This standalone script generates two plots intended to address the referee
request for a more quantitative unknown-M discussion:

  1. Continuous-time success-probability curves for M=1 and several search
     sizes N=2^n, using the conservative unknown-M parameter choice

         Delta = 2 pi / sqrt(C N R),

     or, optionally, the known-M optimized choice

         Delta = C sqrt[M(N-M)] / (N R).

     The convergence curves are plotted as F(t) versus t.

  2. The extracted solution time t_sol, defined as the first time at which
     F(t) reaches a chosen target probability, plotted against sqrt(N).

The default target is F = 0.99.

The dynamics are exact within the invariant symmetric subspace spanned by

    |a>   = uniform non-solution state \otimes |+^r>,
    |b_k> = uniform solution state \otimes |k>, k=0,...,R-1.

For an initial |+^n,+^r> state, this reduced (R+1)-dimensional evolution is
identical to the full Hamiltonian evolution in the relevant sector; no dense
2^(n+r) Hamiltonian is built.

Outputs:
  - one PDF figure
  - one long-form CSV containing all plotted curves and scaling data
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------


def configure_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    if sns is not None:
        sns.set_palette("husl")

    plt.rcParams.update({
        "font.size": 17,
        "axes.titlesize": 19,
        "axes.labelsize": 19,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.3,
    })


# ---------------------------------------------------------------------------
# Parameters and scales
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyParams:
    n_data: int
    M: int = 1
    r_reservoir: int = 4
    C: float = 10.0
    delta_mode: str = "unknown"  # "unknown" or "known"

    @property
    def N(self) -> int:
        return 2 ** self.n_data

    @property
    def R(self) -> int:
        return 2 ** self.r_reservoir

    @property
    def delta(self) -> float:
        if self.delta_mode == "unknown":
            # Conservative unknown-M choice from the revised appendix.
            return 2.0 * np.pi / np.sqrt(self.C * self.N * self.R)
        if self.delta_mode == "known":
            # Known-M optimized choice from the main text.
            return self.C * np.sqrt(self.M * (self.N - self.M)) / (self.N * self.R)
        raise ValueError("delta_mode must be either 'unknown' or 'known'")


def bj_gamma(params: StudyParams) -> float:
    return 2.0 * np.pi * params.M * (params.N - params.M) / (
        params.R * params.delta * params.N**2
    )


def bj_tau(params: StudyParams) -> float:
    return 2.0 * np.pi / params.delta


def bj_physical_success_probability(times: np.ndarray, params: StudyParams) -> np.ndarray:
    """
    Physical BJ estimate before the first revival:

        F_BJ(t) = 1 - [(N-M)/N] exp(-gamma t).

    This includes the initial marked-state weight M/N.
    """
    gamma = bj_gamma(params)
    return 1.0 - ((params.N - params.M) / params.N) * np.exp(-gamma * times)


def bj_solution_time(params: StudyParams, target: float) -> float:
    """First time at which the physical BJ estimate reaches target."""
    prefactor = (params.N - params.M) / params.N
    if target <= 1.0 - prefactor:
        return 0.0
    if target >= 1.0:
        raise ValueError("target must be strictly less than 1")
    return -np.log((1.0 - target) / prefactor) / bj_gamma(params)


# ---------------------------------------------------------------------------
# Exact reduced continuous-time dynamics
# ---------------------------------------------------------------------------


def reduced_initial_state(params: StudyParams) -> np.ndarray:
    psi0 = np.zeros(params.R + 1, dtype=complex)
    psi0[0] = np.sqrt((params.N - params.M) / params.N)
    psi0[1:] = np.sqrt(params.M / params.N) / np.sqrt(params.R)
    return psi0


def reservoir_energies(params: StudyParams) -> np.ndarray:
    k = np.arange(params.R, dtype=float)
    return 1.0 + params.delta * (k - (params.R - 1) / 2.0)


def build_reduced_hamiltonian(params: StudyParams) -> np.ndarray:
    """
    Reduced exact dissipative Grover Hamiltonian in basis [|a>, |b_0>,...,|b_{R-1}>].
    """
    H = np.zeros((params.R + 1, params.R + 1), dtype=complex)
    H[0, 0] = (params.N - params.M) / params.N

    coupling = np.sqrt(params.M * (params.N - params.M)) / (params.N * np.sqrt(params.R))
    H[0, 1:] = coupling
    H[1:, 0] = coupling

    H[1:, 1:] += np.diag(reservoir_energies(params))
    H[1:, 1:] += params.M / (params.N * params.R)
    return H


def evolve_from_spectrum(H: np.ndarray, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    coeffs = evecs.conj().T @ psi0
    phases = np.exp(-1j * np.outer(times, evals))
    return (phases * coeffs) @ evecs.T


def exact_success_probability(times: np.ndarray, params: StudyParams) -> np.ndarray:
    H = build_reduced_hamiltonian(params)
    psi0 = reduced_initial_state(params)
    states = evolve_from_spectrum(H, psi0, times)
    return np.sum(np.abs(states[:, 1:]) ** 2, axis=1)


def first_crossing_time(times: np.ndarray, values: np.ndarray, target: float) -> float:
    """Linear interpolation for first target crossing; returns NaN if no crossing."""
    above = np.flatnonzero(values >= target)
    if len(above) == 0:
        return float("nan")
    idx = int(above[0])
    if idx == 0:
        return float(times[0])
    t0, t1 = times[idx - 1], times[idx]
    y0, y1 = values[idx - 1], values[idx]
    if np.isclose(y1, y0):
        return float(t1)
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))


# ---------------------------------------------------------------------------
# CSV and filename helpers
# ---------------------------------------------------------------------------


def float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def n_values_token(n_values: Sequence[int]) -> str:
    if len(n_values) == 0:
        return "none"
    if len(n_values) > 2 and all((n_values[i + 1] - n_values[i]) == (n_values[1] - n_values[0]) for i in range(len(n_values) - 1)):
        return f"n{n_values[0]}to{n_values[-1]}step{n_values[1]-n_values[0]}"
    return "n" + "_".join(str(n) for n in n_values)


def output_paths(output_stem: str, n_values: Sequence[int], r: int, C: float, target: float, delta_mode: str) -> Tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    token = f"{n_values_token(n_values)}_M1_r{r}_C{float_token(C)}_target{float_token(target)}_{delta_mode}Delta"
    return here / f"{output_stem}_{token}.pdf", here / f"{output_stem}_{token}.csv"


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    fieldnames = [
        "record_type", "n_data", "N", "sqrtN", "M", "r", "R", "C", "delta_mode",
        "Delta", "tau", "gamma", "target", "t", "t_over_sqrtN", "t_over_tau",
        "F_exact", "F_BJ", "t_solution_exact", "t_solution_BJ", "fit_slope_through_origin",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------


def run_study(
    n_values: Sequence[int],
    M: int = 1,
    r: int = 4,
    C: float = 10.0,
    target: float = 0.99,
    delta_mode: str = "unknown",
    curve_tau_fraction: float = 0.75,
    points_per_curve: int = 1200,
    output_stem: str = "unknown_M_scaling",
) -> Tuple[Path, Path]:
    if M != 1:
        raise ValueError("This referee-response study is intended for M=1. Use M=1 for the default analysis.")
    if not (0.0 < target < 1.0):
        raise ValueError("target must be between 0 and 1")
    if delta_mode not in {"unknown", "known"}:
        raise ValueError("delta_mode must be 'unknown' or 'known'")

    configure_plot_style()
    rows: List[Dict[str, Any]] = []

    fig, (ax_curves, ax_scaling) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    scaling_sqrtN: List[float] = []
    scaling_exact: List[float] = []
    scaling_bj: List[float] = []

    for n in n_values:
        params = StudyParams(n_data=n, M=M, r_reservoir=r, C=C, delta_mode=delta_mode)
        tau = bj_tau(params)
        t_end = curve_tau_fraction * tau
        times = np.linspace(0.0, t_end, points_per_curve)
        F_exact = exact_success_probability(times, params)
        F_BJ = bj_physical_success_probability(times, params)

        t_sol_exact = first_crossing_time(times, F_exact, target)
        t_sol_bj = bj_solution_time(params, target)

        sqrtN = np.sqrt(params.N)
        scaling_sqrtN.append(sqrtN)
        scaling_exact.append(t_sol_exact)
        scaling_bj.append(t_sol_bj)

        # Change 1: plot the raw physical time t, not t / sqrt(N).
        ax_curves.plot(times, F_exact, label=fr"$n={n}$")

        metadata = {
            "record_type": "curve",
            "n_data": n,
            "N": params.N,
            "sqrtN": sqrtN,
            "M": M,
            "r": r,
            "R": params.R,
            "C": C,
            "delta_mode": delta_mode,
            "Delta": params.delta,
            "tau": tau,
            "gamma": bj_gamma(params),
            "target": target,
            "t_solution_exact": t_sol_exact,
            "t_solution_BJ": t_sol_bj,
            "fit_slope_through_origin": "",
        }
        for t, fe, fb in zip(times, F_exact, F_BJ):
            row = dict(metadata)
            row.update({
                "t": float(t),
                "t_over_sqrtN": float(t / sqrtN),
                "t_over_tau": float(t / tau),
                "F_exact": float(np.real_if_close(fe)),
                "F_BJ": float(np.real_if_close(fb)),
            })
            rows.append(row)

    x = np.asarray(scaling_sqrtN, dtype=float)
    y = np.asarray(scaling_exact, dtype=float)
    y_bj = np.asarray(scaling_bj, dtype=float)
    finite = np.isfinite(y)
    if np.count_nonzero(finite) >= 1:
        slope = float(np.dot(x[finite], y[finite]) / np.dot(x[finite], x[finite]))
    else:
        slope = float("nan")

    # Curves panel.
    ax_curves.axhline(target, color="0.4", linestyle=":", linewidth=1.2, label=fr"target $F={target:g}$")
    ax_curves.set_xlabel(r"$t$", fontsize=35)
    ax_curves.set_ylabel(r"$F(t)$", fontsize=35)
    # ax_curves.set_title(fr"Convergence curves, $M=1$, $r={r}$")
    ax_curves.set_ylim(-0.03, 1.03)
    ax_curves.grid(False)
    ax_curves.legend(frameon=False, ncol=1, fontsize=12)

    # Scaling panel.
    ax_scaling.plot(x, y, marker="o", linestyle="", label="exact first crossing")
    ax_scaling.plot(x, y_bj, marker="s", linestyle="", label="BJ prediction")
    x_line = np.linspace(0.0, 1.05 * x.max(), 200)
    ax_scaling.plot(x_line, slope * x_line, linestyle="--", label=fr"fit $t={slope:.3g}\sqrt{{N}}$")
    ax_scaling.set_xlabel(r"$\sqrt{N}$", fontsize=35)
    ax_scaling.set_ylabel(fr"$t_{{F={target:g}}}$", fontsize=35)
    # ax_scaling.set_title("Solution-time scaling")
    ax_scaling.grid(False)
    ax_scaling.legend(frameon=False, fontsize=12)

    fig.tight_layout()
    pdf_path, csv_path = output_paths(output_stem, n_values, r, C, target, delta_mode)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Add one compact scaling row per n with the final fit slope.
    for n, sqrtN, t_exact, t_bj in zip(n_values, x, y, y_bj):
        params = StudyParams(n_data=n, M=M, r_reservoir=r, C=C, delta_mode=delta_mode)
        rows.append({
            "record_type": "scaling",
            "n_data": n,
            "N": params.N,
            "sqrtN": sqrtN,
            "M": M,
            "r": r,
            "R": params.R,
            "C": C,
            "delta_mode": delta_mode,
            "Delta": params.delta,
            "tau": bj_tau(params),
            "gamma": bj_gamma(params),
            "target": target,
            "t": "",
            "t_over_sqrtN": "",
            "t_over_tau": "",
            "F_exact": "",
            "F_BJ": "",
            "t_solution_exact": t_exact,
            "t_solution_BJ": t_bj,
            "fit_slope_through_origin": slope,
        })

    write_csv(rows, csv_path)

    print(f"Saved {pdf_path}")
    print(f"Saved {csv_path}")
    print(f"delta_mode={delta_mode}, M={M}, r={r}, R={2**r}, C={C}, target={target}")
    print(f"Fit through origin: t_solution ≈ {slope:.8g} sqrt(N)")
    for n, t_exact, t_bj in zip(n_values, y, y_bj):
        print(f"  n={n:2d}, N={2**n:7d}, t_exact={t_exact:.8g}, t_BJ={t_bj:.8g}")

    return pdf_path, csv_path


def parse_n_values(spec: str) -> List[int]:
    """Parse comma list like '6,8,10' or range form '6:14:2'."""
    spec = spec.strip()
    if ":" in spec:
        parts = [int(p) for p in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError("range form must be start:stop or start:stop:step")
        return list(range(start, stop + 1, step))
    return [int(p.strip()) for p in spec.split(",") if p.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unknown-M dissipative Grover scaling study.")
    parser.add_argument("--n-values", type=str, default="6:14:2", help="n values, e.g. '6,8,10' or '6:14:2'.")
    parser.add_argument("--M", type=int, default=1, help="Marked-state degeneracy. Default and intended value: 1.")
    parser.add_argument("--r", type=int, default=4, help="Reservoir qubits. Default: 4.")
    parser.add_argument("--C", type=float, default=10.0, help="Separation/smoothness parameter. Default: 10.")
    # Change 2: default target is now F = 0.99.
    parser.add_argument("--target", type=float, default=0.99, help="Success-probability threshold for t_solution. Default: 0.99.")
    parser.add_argument("--delta-mode", choices=["unknown", "known"], default="unknown", help="Use unknown-M or known-M Delta choice. Default: unknown.")
    parser.add_argument("--curve-tau-fraction", type=float, default=0.75, help="Plot curves up to this fraction of tau. Default: 0.75.")
    parser.add_argument("--points-per-curve", type=int, default=1200, help="Samples per convergence curve. Default: 1200.")
    parser.add_argument("--output-stem", type=str, default="unknown_M_scaling", help="Output filename stem.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_study(
        n_values=parse_n_values(args.n_values),
        M=args.M,
        r=args.r,
        C=args.C,
        target=args.target,
        delta_mode=args.delta_mode,
        curve_tau_fraction=args.curve_tau_fraction,
        points_per_curve=args.points_per_curve,
        output_stem=args.output_stem,
    )
