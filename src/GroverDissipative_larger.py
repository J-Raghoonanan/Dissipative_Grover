#!/usr/bin/env python3
r"""
Larger-system Fig. 2-style simulation for the Grover-dissipation manuscript.

Creates a new 2x2 figure addressing the referee request for a larger total
number of qubits, and saves the plotted data to a parameterized CSV file.

Default figure:
  (a) continuous dissipative Grover, n = 10 data qubits, r = 4 reservoir qubits
  (b) continuous dissipative Grover, n = 10, r = 3
  (c) discrete/trotterized dissipative Grover, n = 10, r = 4
  (d) discrete/trotterized dissipative Grover, n = 10, r = 3

The dissipative simulations correspond to total sizes n+r = 14 and 13
qubits, respectively.

Implementation note
-------------------
The full Hilbert-space dimension is 2^(n+r), but the Hamiltonian and discrete
iterate preserve a small invariant subspace spanned by

  |a>       = uniform superposition over non-solution data states \otimes |+^r>,
  |b_k>     = uniform solution state \otimes |k>,  k = 0,...,R-1,

where R = 2^r.  The exact dynamics from the standard |+^n,+^r> initial state
are therefore obtained by diagonalizing only an (R+1)x(R+1) matrix.  This is
not an approximation to the manuscript Hamiltonian; it is the exact reduced
symmetric-sector evolution.

The script saves a PDF and one long-form CSV in the same directory as this file.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:  # keep the script usable on minimal Python installations
    sns = None


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------


def configure_plot_style() -> None:
    """Use a manuscript-friendly style close to the original Fig. 2 script."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    if sns is not None:
        sns.set_palette("husl")

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


COLORS = {
    "standard": "tab:blue",
    "dissipative": "tab:red",
    "bj": "tab:green",
}


# ---------------------------------------------------------------------------
# Parameters and analytic scales
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LargeFigureParams:
    """Parameters for one panel pair of the larger-system figure."""

    n_data: int = 10
    n_solutions: int = 1
    r_reservoir: int = 4
    C: float = 5.0
    delta_t: float = np.pi

    @property
    def N(self) -> int:
        return 2 ** self.n_data

    @property
    def M(self) -> int:
        return self.n_solutions

    @property
    def R(self) -> int:
        return 2 ** self.r_reservoir

    @property
    def total_qubits(self) -> int:
        return self.n_data + self.r_reservoir

    @property
    def delta(self) -> float:
        """
        Known-M parameter choice from the manuscript:

            Delta = C sqrt[M(N-M)] / (N R).

        This keeps the BJ decay time independent of R at fixed n, M, C, while
        increasing R moves the revival time and improves the reservoir ladder.
        """
        return self.C * np.sqrt(self.M * (self.N - self.M)) / (self.N * self.R)


def bj_gamma(N: int, M: int, R: int, delta: float) -> float:
    """BJ decay rate gamma = 2 pi M(N-M)/(R Delta N^2)."""
    return 2.0 * np.pi * M * (N - M) / (R * delta * N**2)


def bj_tau(delta: float) -> float:
    """BJ revival time tau = 2 pi / Delta."""
    return 2.0 * np.pi / delta


def bj_amplitude_piecewise(t: np.ndarray, gamma: float, tau: float) -> np.ndarray:
    """
    Piecewise BJ source amplitude, retaining the first revival correction:

      a(t) = exp(-gamma t/2)
             - gamma exp[-gamma(t-tau)/2] (t-tau) Theta(t-tau).
    """
    t = np.asarray(t, dtype=float)
    dt = np.maximum(0.0, t - tau)
    return np.exp(-gamma * t / 2.0) - gamma * np.exp(-gamma * dt / 2.0) * dt


def bj_success_probability(t: np.ndarray, params: LargeFigureParams) -> np.ndarray:
    """
    Physical BJ success-probability estimate consistent with the manuscript
    fidelity convention.

    The ideal BJ transfer probability is 1 - |a(t)|^2 for an initial state
    purely in the source state.  The Grover initial state already has marked
    weight M/N, so the physical estimate is

        F_BJ(t) = M/N + (1 - M/N) [1 - |a(t)|^2]
                = 1 - [(N-M)/N] |a(t)|^2.
    """
    gamma = bj_gamma(params.N, params.M, params.R, params.delta)
    tau = bj_tau(params.delta)
    amp = bj_amplitude_piecewise(t, gamma, tau)
    return 1.0 - ((params.N - params.M) / params.N) * np.abs(amp) ** 2


# ---------------------------------------------------------------------------
# Reduced exact dynamics
# ---------------------------------------------------------------------------


def reduced_initial_state(params: LargeFigureParams) -> np.ndarray:
    """
    Initial |+^n,+^r> in the reduced basis [|a>, |b_0>, ..., |b_{R-1}>].

      |+^n,+^r> = sqrt((N-M)/N)|a>
                  + sqrt(M/N) (1/sqrt(R)) sum_k |b_k>.
    """
    N, M, R = params.N, params.M, params.R
    psi0 = np.zeros(R + 1, dtype=complex)
    psi0[0] = np.sqrt((N - M) / N)
    psi0[1:] = np.sqrt(M / N) / np.sqrt(R)
    return psi0


def reservoir_energies(params: LargeFigureParams) -> np.ndarray:
    """E_k = 1 + Delta [k - (R-1)/2], k = 0,...,R-1."""
    k = np.arange(params.R, dtype=float)
    return 1.0 + params.delta * (k - (params.R - 1) / 2.0)


def build_reduced_dissipative_hamiltonian(params: LargeFigureParams) -> np.ndarray:
    """
    Exact symmetric-sector matrix of H_DG in basis [|a>, |b_0>,...,|b_{R-1}>].

    Matrix elements:
      H_aa       = (N-M)/N
      H_ab_k     = sqrt[M(N-M)]/(N sqrt(R))
      H_bk_bl    = E_k delta_kl + M/(N R)
    """
    N, M, R = params.N, params.M, params.R
    E = reservoir_energies(params)

    H = np.zeros((R + 1, R + 1), dtype=complex)
    H[0, 0] = (N - M) / N

    coupling = np.sqrt(M * (N - M)) / (N * np.sqrt(R))
    H[0, 1:] = coupling
    H[1:, 0] = coupling

    H[1:, 1:] += np.diag(E)
    H[1:, 1:] += M / (N * R)
    return H


def build_reduced_standard_hamiltonian(params: LargeFigureParams) -> np.ndarray:
    """
    Exact two-dimensional standard continuous-time Grover Hamiltonian in basis
    [|a>, |s>], where |s> is the uniform marked-state superposition.
    """
    N, M = params.N, params.M
    H = np.zeros((2, 2), dtype=complex)
    H[0, 0] = (N - M) / N
    H[1, 1] = 1.0 + M / N
    H[0, 1] = H[1, 0] = np.sqrt(M * (N - M)) / N
    return H


def evolve_from_spectrum(H: np.ndarray, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return all states exp(-i H t)|psi0> using one diagonalization."""
    evals, evecs = np.linalg.eigh(H)
    coeffs = evecs.conj().T @ psi0
    phases = np.exp(-1j * np.outer(times, evals))
    return (phases * coeffs) @ evecs.T


def continuous_curves(
    params: LargeFigureParams,
    n_periods: float,
    points_per_period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Continuous standard, continuous dissipative, and BJ curves."""
    tau = bj_tau(params.delta)
    times = np.linspace(0.0, n_periods * tau, int(n_periods * points_per_period) + 1)
    t_norm = times / tau

    # Standard continuous Grover in the 2D invariant subspace.
    psi0_std = np.array([
        np.sqrt((params.N - params.M) / params.N),
        np.sqrt(params.M / params.N),
    ], dtype=complex)
    states_std = evolve_from_spectrum(build_reduced_standard_hamiltonian(params), psi0_std, times)
    F_std = np.abs(states_std[:, 1]) ** 2

    # Dissipative continuous Grover in the exact reduced subspace.
    psi0_dg = reduced_initial_state(params)
    states_dg = evolve_from_spectrum(build_reduced_dissipative_hamiltonian(params), psi0_dg, times)
    F_dg = np.sum(np.abs(states_dg[:, 1:]) ** 2, axis=1)

    F_bj = bj_success_probability(times, params)
    return times, t_norm, F_std, F_dg, F_bj


def build_projector_unitary(projector_vector: np.ndarray, phase_time: float) -> np.ndarray:
    """I - (1-exp(-i phase_time)) |v><v| for normalized |v>."""
    v = projector_vector.astype(complex)
    v = v / np.linalg.norm(v)
    dim = len(v)
    return np.eye(dim, dtype=complex) - (1.0 - np.exp(-1j * phase_time)) * np.outer(v, v.conj())


def discrete_curves(
    params: LargeFigureParams,
    n_periods: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Discrete standard, discrete dissipative, and fine-grid BJ curves."""
    tau = bj_tau(params.delta)
    t_end = n_periods * tau
    n_steps = int(np.floor(t_end / params.delta_t))
    step_times = np.arange(n_steps + 1, dtype=float) * params.delta_t
    t_norm = step_times / tau

    # Standard discrete Grover-like product U_+ U_S in two dimensions.
    psi_plus_std = np.array([
        np.sqrt((params.N - params.M) / params.N),
        np.sqrt(params.M / params.N),
    ], dtype=complex)
    U_plus_std = build_projector_unitary(psi_plus_std, params.delta_t)
    U_S_std = np.diag([1.0, np.exp(-1j * params.delta_t)]).astype(complex)
    U_step_std = U_plus_std @ U_S_std

    psi = psi_plus_std.copy()
    F_std = np.empty(n_steps + 1, dtype=float)
    F_std[0] = np.abs(psi[1]) ** 2
    for j in range(1, n_steps + 1):
        psi = U_step_std @ psi
        F_std[j] = np.abs(psi[1]) ** 2

    # Dissipative discrete product U_+ U_S in the exact reduced subspace.
    psi_plus_dg = reduced_initial_state(params)
    U_plus_dg = build_projector_unitary(psi_plus_dg, params.delta_t)
    phases = np.concatenate(([1.0 + 0.0j], np.exp(-1j * reservoir_energies(params) * params.delta_t)))
    U_S_dg = np.diag(phases)
    U_step_dg = U_plus_dg @ U_S_dg

    psi = psi_plus_dg.copy()
    F_dg = np.empty(n_steps + 1, dtype=float)
    F_dg[0] = np.sum(np.abs(psi[1:]) ** 2)
    for j in range(1, n_steps + 1):
        psi = U_step_dg @ psi
        F_dg[j] = np.sum(np.abs(psi[1:]) ** 2)

    # BJ curve on a fine grid for visual comparison.
    fine_times = np.linspace(0.0, t_end, max(1000, 5 * (n_steps + 1)))
    t_norm_bj = fine_times / tau
    F_bj = bj_success_probability(fine_times, params)
    return step_times, t_norm, F_std, F_dg, fine_times, t_norm_bj, F_bj


# ---------------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------------


def delta_t_token(delta_t: float) -> str:
    """Compact delta_t token for filenames."""
    if np.isclose(delta_t, np.pi):
        return "dtpi"
    if np.isclose(delta_t / np.pi, round(delta_t / np.pi, 8)):
        return f"dt{delta_t / np.pi:.8g}pi".replace(".", "p")
    return f"dt{delta_t:.8g}".replace(".", "p")


def float_token(value: float) -> str:
    """Compact float token for filenames."""
    return f"{value:g}".replace(".", "p")


def make_data_stem(
    output_stem: str,
    n_data: int,
    N: int,
    M: int,
    r_top: int,
    r_bottom: int,
    C: float,
    delta_t: float,
    n_periods: float,
) -> str:
    """Build a descriptive CSV filename stem."""
    return (
        f"{output_stem}_data_larger_system_"
        f"n{n_data}_N{N}_M{M}_rtop{r_top}_rbottom{r_bottom}_"
        f"C{float_token(C)}_{delta_t_token(delta_t)}_periods{float_token(n_periods)}"
    )


def base_metadata(params: LargeFigureParams, panel: str, evolution: str) -> Dict[str, Any]:
    """Metadata attached to every saved data row."""
    return {
        "panel": panel,
        "evolution": evolution,
        "n_data": params.n_data,
        "N": params.N,
        "M": params.M,
        "panel_r_reservoir": params.r_reservoir,
        "R_panel": params.R,
        "total_qubits_panel": params.total_qubits,
        "C": params.C,
        "Delta": params.delta,
        "delta_t": params.delta_t,
        "tau": bj_tau(params.delta),
        "gamma": bj_gamma(params.N, params.M, params.R, params.delta),
    }


def add_curve_rows(
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    curve: str,
    curve_r: int,
    times: np.ndarray,
    t_norm: np.ndarray,
    fidelity: np.ndarray,
) -> None:
    """Append one plotted curve to the long-form CSV rows."""
    for t, tn, f in zip(times, t_norm, fidelity):
        row = dict(metadata)
        row.update({
            "curve": curve,
            "curve_r": curve_r,
            "t": float(t),
            "t_over_tau": float(tn),
            "F": float(np.real_if_close(f)),
        })
        rows.append(row)


def write_csv(rows: Iterable[Dict[str, Any]], csv_path: Path) -> None:
    """Write rows to CSV with a stable column order."""
    rows = list(rows)
    fieldnames = [
        "panel",
        "evolution",
        "curve",
        "curve_r",
        "n_data",
        "N",
        "M",
        "panel_r_reservoir",
        "R_panel",
        "total_qubits_panel",
        "C",
        "Delta",
        "delta_t",
        "tau",
        "gamma",
        "t_over_tau",
        "t",
        "F",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def plot_panel_continuous(
    ax: plt.Axes,
    params: LargeFigureParams,
    n_periods: float,
    points_per_period: int,
    label: str,
) -> List[Dict[str, Any]]:
    times, t, F_std, F_dg, F_bj = continuous_curves(params, n_periods, points_per_period)

    ax.plot(t, F_std, color=COLORS["standard"], label=fr"$r=0$")
    ax.plot(t, F_dg, color=COLORS["dissipative"], label=fr"$r={params.r_reservoir}$")
    ax.plot(t, F_bj, color=COLORS["bj"], linestyle="--", label="BJ")
    ax.set_title(fr"{label} continuous, $r={params.r_reservoir}$")

    rows: List[Dict[str, Any]] = []
    metadata = base_metadata(params, panel=label.strip("()"), evolution="continuous")
    add_curve_rows(rows, metadata, "standard", 0, times, t, F_std)
    add_curve_rows(rows, metadata, "dissipative", params.r_reservoir, times, t, F_dg)
    add_curve_rows(rows, metadata, "BJ", params.r_reservoir, times, t, F_bj)
    return rows


def plot_panel_discrete(
    ax: plt.Axes,
    params: LargeFigureParams,
    n_periods: float,
    label: str,
) -> List[Dict[str, Any]]:
    step_times, t, F_std, F_dg, fine_times, t_bj, F_bj = discrete_curves(params, n_periods)

    ax.step(t, F_std, where="post", color=COLORS["standard"], label=fr"$r=0$")
    ax.step(t, F_dg, where="post", color=COLORS["dissipative"], label=fr"$r={params.r_reservoir}$")
    ax.plot(t_bj, F_bj, color=COLORS["bj"], linestyle="--", label="BJ")
    ax.set_title(fr"{label} trotterized, $r={params.r_reservoir}$")

    rows: List[Dict[str, Any]] = []
    metadata = base_metadata(params, panel=label.strip("()"), evolution="trotterized")
    add_curve_rows(rows, metadata, "standard", 0, step_times, t, F_std)
    add_curve_rows(rows, metadata, "dissipative", params.r_reservoir, step_times, t, F_dg)
    add_curve_rows(rows, metadata, "BJ", params.r_reservoir, fine_times, t_bj, F_bj)
    return rows


def make_larger_system_figure(
    n_data: int = 10,
    M: int = 1,
    r_top: int = 4,
    r_bottom: int = 3,
    C: float = 5.0,
    n_periods: float = 2.0,
    points_per_period: int = 600,
    delta_t: float = np.pi,
    output_stem: str = "GroverDissipative_fig_larger_system",
) -> Tuple[Path, Path]:
    """Create and save the larger-system figure and its plotted data CSV."""
    configure_plot_style()

    params_r4 = LargeFigureParams(n_data=n_data, n_solutions=M, r_reservoir=r_top, C=C, delta_t=delta_t)
    params_r3 = LargeFigureParams(n_data=n_data, n_solutions=M, r_reservoir=r_bottom, C=C, delta_t=delta_t)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, sharey=True)

    rows: List[Dict[str, Any]] = []
    rows.extend(plot_panel_continuous(axes[0, 0], params_r4, n_periods, points_per_period, "(a)"))
    rows.extend(plot_panel_continuous(axes[0, 1], params_r3, n_periods, points_per_period, "(b)"))
    rows.extend(plot_panel_discrete(axes[1, 0], params_r4, n_periods, "(c)"))
    rows.extend(plot_panel_discrete(axes[1, 1], params_r3, n_periods, "(d)"))

    axes[0, 0].set_ylabel(r"$F$")
    axes[1, 0].set_ylabel(r"$F$")
    axes[1, 0].set_xlabel(r"$t/\tau$")
    axes[1, 1].set_xlabel(r"$t/\tau$")

    for ax in axes.ravel():
        ax.set_xlim(0.0, n_periods)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(False)
        ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent
    pdf_path = out_dir / f"{output_stem}.pdf"
    csv_stem = make_data_stem(
        output_stem=output_stem,
        n_data=n_data,
        N=params_r4.N,
        M=M,
        r_top=r_top,
        r_bottom=r_bottom,
        C=C,
        delta_t=delta_t,
        n_periods=n_periods,
    )
    csv_path = out_dir / f"{csv_stem}.csv"

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    write_csv(rows, csv_path)

    print(f"Saved {pdf_path}")
    print(f"Saved {csv_path}")
    print("Panel parameters:")
    for p in (params_r4, params_r3):
        print(
            f"  n={p.n_data}, r={p.r_reservoir}, n+r={p.total_qubits}, "
            f"N={p.N}, R={p.R}, M={p.M}, Delta={p.delta:.8g}, "
            f"tau={bj_tau(p.delta):.8g}, gamma={bj_gamma(p.N, p.M, p.R, p.delta):.8g}"
        )

    return pdf_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a larger-system Fig. 2-style plot.")
    parser.add_argument("--n-data", type=int, default=10, help="Number of data/search qubits. Default: 10.")
    parser.add_argument("--M", type=int, default=1, help="Number of marked states. Default: 1.")
    parser.add_argument("--r-top", type=int, default=4, help="Reservoir qubits in the top row. Default: 4.")
    parser.add_argument("--r-bottom", type=int, default=3, help="Reservoir qubits in the bottom row. Default: 3.")
    parser.add_argument("--C", type=float, default=5.0, help="Constant in Delta=C sqrt[M(N-M)]/(NR). Default: 5.")
    parser.add_argument("--periods", type=float, default=2.0, help="Number of revival periods shown on x-axis. Default: 2.")
    parser.add_argument("--points-per-period", type=int, default=600, help="Continuous-time samples per tau. Default: 600.")
    parser.add_argument("--delta-t", type=float, default=np.pi, help="Discrete iterate time step. Default: pi.")
    parser.add_argument("--output-stem", type=str, default="GroverDissipative_fig_larger_system", help="Output filename stem.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_larger_system_figure(
        n_data=args.n_data,
        M=args.M,
        r_top=args.r_top,
        r_bottom=args.r_bottom,
        C=args.C,
        n_periods=args.periods,
        points_per_period=args.points_per_period,
        delta_t=args.delta_t,
        output_stem=args.output_stem,
    )
