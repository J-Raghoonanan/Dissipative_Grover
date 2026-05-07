#!/usr/bin/env python3
"""
Figure 2 reproduction for "Quantum Search by Dissipation" manuscript.

This script compares:
  • Standard continuous-time Grover Hamiltonian (r = 0)
  • Dissipative Grover Hamiltonian with r ancilla "bath" qubits (continuous)
  • Trotterized (discrete) versions of both
  • Bixon–Jortner (BJ) analytic approximation for the dissipative dynamics

The theory follows the manuscript:

  H_G  = |+^n><+^n| + sum_m |S_m><S_m|
  H_DG = |+^n><+^n| ⊗ |+^r><+^r|
         + sum_m |S_m><S_m| ⊗ sum_k E_k |k><k|

with
  N = 2^n       (data Hilbert space dimension)
  R = 2^r       (reservoir Hilbert space dimension)
  E_k = 1 + Δ ( k - (R - 1)/2 ),  k = 0 ... R-1

The BJ mapping uses:
  γ   = 2π M (N - M) / (R Δ N^2)
  τ   = 2π / Δ

and a simple piecewise approximation to the source amplitude a(t)
valid up to the first revival.  In the BJ model we start with
a(0) = 1, b_m(0) = 0 and interpret

  F_BJ(t) = 1 - |a(t)|^2

as the probability transferred out of the effective source state
(and into the solution manifold) in the reduced BJ description.
This starts at F_BJ(0) = 0 and can reach 1.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Sequence, Tuple

# ---------------------------------------------------------------------------
# Global plotting style
# ---------------------------------------------------------------------------

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 30,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 8
})

COLORS = {
    "standard":   "tab:blue",   # standard Grover (r = 0)
    "dissipative": "tab:red",   # dissipative curves (r = 3, 4)
    "bj":         "tab:green",  # BJ analytic curve
}

# How many τ to show on the x-axis
N_PERIODS = 2

# ---------------------------------------------------------------------------
# Basic linear algebra helpers
# ---------------------------------------------------------------------------

def plus_state(dim: int) -> np.ndarray:
    """
    Return the normalized |+> state in dimension `dim`:
        |+> = (1/√dim) ∑_{x=0}^{dim-1} |x>.
    """
    return np.ones(dim, dtype=complex) / np.sqrt(dim)


def projector(state: np.ndarray) -> np.ndarray:
    """
    Rank-1 projector |ψ><ψ| for a normalized state |ψ>.
    """
    return np.outer(state, np.conjugate(state))


def computational_projector(dim: int, index: int) -> np.ndarray:
    """
    Projector onto computational basis state |index> in `dim` dimensions.
    """
    proj = np.zeros((dim, dim), dtype=complex)
    proj[index, index] = 1.0
    return proj


# ---------------------------------------------------------------------------
# Grover Hamiltonians
# ---------------------------------------------------------------------------

def build_standard_grover_hamiltonian(
    n_qubits: int,
    solutions: Sequence[int],
) -> np.ndarray:
    """
    Continuous-time Grover Hamiltonian on n data qubits:

        H_G = |+^n><+^n| + sum_{m ∈ S} |S_m><S_m|,

    where S is the set of solution indices.

    Returns an (N, N) Hermitian matrix with N = 2^n.
    """
    N = 2**n_qubits
    H = np.zeros((N, N), dtype=complex)

    # |+^n><+^n|
    plus_n = plus_state(N)
    H += projector(plus_n)

    # sum_m |S_m><S_m|
    for s in solutions:
        H += computational_projector(N, s)

    return H


def build_dissipative_grover_hamiltonian(
    n_qubits: int,
    r_ancilla: int,
    solutions: Sequence[int],
    delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dissipative Grover Hamiltonian on data⊗reservoir:

        H_DG = |+^n><+^n| ⊗ |+^r><+^r|
               + sum_{m ∈ S} |S_m><S_m| ⊗ sum_k E_k |k><k|,

    with reservoir energies

        E_k = 1 + Δ ( k - (R - 1)/2 ),  k = 0,...,R-1,  R=2^r.

    Returns:
      H_DG    : full Hamiltonian, shape (N*R, N*R)
      energies: reservoir energies E_k, shape (R,)
    """
    N = 2**n_qubits
    R = 2**r_ancilla

    plus_n = plus_state(N)
    plus_r = plus_state(R)

    P_plus_n = projector(plus_n)
    P_plus_r = projector(plus_r)
    H_plus = np.kron(P_plus_n, P_plus_r)

    k_vals = np.arange(R)
    energies = 1.0 + delta * (k_vals - (R - 1) / 2.0)
    Emat = np.diag(energies)

    P_S = np.zeros((N, N), dtype=complex)
    for s in solutions:
        P_S[s, s] = 1.0

    H_solutions = np.kron(P_S, Emat)

    H = H_plus + H_solutions
    return H, energies


# ---------------------------------------------------------------------------
# Time evolution utilities
# ---------------------------------------------------------------------------

def time_evolution_from_spectrum(
    H: np.ndarray,
    psi0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute |ψ(t)> = e^{-i H t} |ψ(0)> for multiple times using a
    single diagonalization of H.

    Returns an array of shape (len(times), dim).
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    coeffs = np.conjugate(eigvecs).T @ psi0  # components in eigenbasis

    states = []
    for t in times:
        phase = np.exp(-1j * eigvals * t)
        psi_t = eigvecs @ (phase * coeffs)
        states.append(psi_t)

    return np.stack(states, axis=0)


def success_probability_from_states_standard(
    states: np.ndarray,
    n_qubits: int,
    solutions: Sequence[int],
) -> np.ndarray:
    """
    Success probability for standard Grover:

        F(t) = ∑_{m ∈ S} |⟨S_m | ψ(t)⟩|^2.
    """
    N = 2**n_qubits
    assert states.shape[1] == N
    sol = np.array(solutions, dtype=int)

    probs = np.abs(states[:, sol])**2
    return probs.sum(axis=1)


def success_probability_from_states_dissipative(
    states: np.ndarray,
    n_qubits: int,
    r_ancilla: int,
    solutions: Sequence[int],
) -> np.ndarray:
    """
    Success probability for dissipative Grover:

        F(t) = ∑_{m ∈ S} ∑_{k=0}^{R-1} |⟨S_m, k | ψ(t)⟩|^2.
    """
    N = 2**n_qubits
    R = 2**r_ancilla
    assert states.shape[1] == N * R

    sol = np.array(solutions, dtype=int)
    T = states.shape[0]
    F = np.zeros(T, dtype=float)

    for ti in range(T):
        psi = states[ti].reshape(N, R)
        probs_data = np.sum(np.abs(psi)**2, axis=1)  # sum over reservoir
        F[ti] = probs_data[sol].sum()

    return F


# ---------------------------------------------------------------------------
# Discrete (Trotterized) versions
# ---------------------------------------------------------------------------

def build_U_plus_full(
    n_qubits: int,
    r_ancilla: int,
    delta_t: float,
) -> np.ndarray:
    """
    U_+ on data⊗reservoir:

        U_+ = I - (1 - e^{-i δt}) |+^n,+^r><+^n,+^r|.
    """
    N = 2**n_qubits
    R = 2**r_ancilla
    dim = N * R

    plus_full = np.kron(plus_state(N), plus_state(R))
    P_plus_full = projector(plus_full)

    U = np.eye(dim, dtype=complex)
    U -= (1.0 - np.exp(-1j * delta_t)) * P_plus_full
    return U


def build_U_S_full(
    n_qubits: int,
    r_ancilla: int,
    solutions: Sequence[int],
    energies: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """
    U_S on data⊗reservoir:

      |x,k> with x ∈ S acquires phase e^{-i E_k δt},
      others are unchanged.
    """
    N = 2**n_qubits
    R = energies.shape[0]
    dim = N * R

    diag = np.ones(dim, dtype=complex)
    for s in solutions:
        for k in range(R):
            idx = s * R + k
            diag[idx] = np.exp(-1j * energies[k] * delta_t)

    return np.diag(diag)


def build_U_plus_standard(
    n_qubits: int,
    delta_t: float,
) -> np.ndarray:
    """
    U_+ on data only:

        U_+ = I - (1 - e^{-i δt}) |+^n><+^n|.
    """
    N = 2**n_qubits
    plus_n = plus_state(N)
    P_plus_n = projector(plus_n)

    U = np.eye(N, dtype=complex)
    U -= (1.0 - np.exp(-1j * delta_t)) * P_plus_n
    return U


def build_U_S_standard(
    n_qubits: int,
    solutions: Sequence[int],
    delta_t: float,
) -> np.ndarray:
    """
    U_S on data only:

        U_S = e^{-i H_S δt},  H_S = ∑_{m∈S} |S_m><S_m|.
    """
    N = 2**n_qubits
    diag = np.ones(N, dtype=complex)
    for s in solutions:
        diag[s] = np.exp(-1j * delta_t)
    return np.diag(diag)


def discrete_trotter_evolution(
    U_step: np.ndarray,
    psi0: np.ndarray,
    n_steps: int,
    step_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a fixed unitary step U_step repeatedly:

        |ψ_L> = (U_step)^L |ψ_0>,  t_L = L * step_time.
    """
    dim = U_step.shape[0]
    states = np.zeros((n_steps + 1, dim), dtype=complex)
    states[0] = psi0

    psi = psi0.copy()
    for L in range(1, n_steps + 1):
        psi = U_step @ psi
        states[L] = psi

    times = np.arange(n_steps + 1, dtype=float) * step_time
    return times, states


# ---------------------------------------------------------------------------
# Bixon–Jortner model: analytic approximation
# ---------------------------------------------------------------------------

def bj_gamma(N: int, M: int, R: int, delta: float) -> float:
    """
    Decay rate γ for BJ mapping:

        γ = 2π M (N - M) / (R Δ N^2).
    """
    return 2.0 * np.pi * M * (N - M) / (R * delta * N**2)


def bj_tau(delta: float) -> float:
    """
    Revival time τ for evenly spaced ladder spacing Δ:

        τ = 2π / Δ.
    """
    return 2.0 * np.pi / delta


def bj_amplitude_piecewise(t: np.ndarray, gamma: float, tau: float) -> np.ndarray:
    """
    Piecewise approximation to the BJ source amplitude a(t) up to the
    first revival:

        a(t) ≈ e^{-γ t / 2}
               - γ e^{-γ (t - τ)/2} (t - τ) Θ(t - τ).
    """
    t = np.asarray(t, dtype=float)
    a0 = np.exp(-gamma * t / 2.0)
    dt = np.maximum(0.0, t - tau)
    a1 = gamma * np.exp(-gamma * dt / 2.0) * dt
    return a0 - a1


def bj_fidelity(
    t: np.ndarray,
    N: int,
    M: int,
    R: int,
    delta: float,
) -> np.ndarray:
    """
    BJ prediction for the effective success probability in the mapped model:

        F_BJ(t) = 1 - |a(t)|^2,

    where a(t) is the BJ source amplitude with initial condition a(0)=1.
    This starts at 0 and measures population transferred out of the
    source state (into the effective solution manifold) in the BJ model.
    """
    gamma = bj_gamma(N, M, R, delta)
    tau = bj_tau(delta)
    a = bj_amplitude_piecewise(t, gamma, tau)
    return 1.0 - np.abs(a)**2


# ---------------------------------------------------------------------------
# High-level simulation and plotting
# ---------------------------------------------------------------------------

@dataclass
class GroverParams:
    n_qubits: int
    solutions: Sequence[int]
    r_ancilla: int
    delta: float


def simulate_continuous_curves(
    params: GroverParams,
    n_periods: int = 5,
    n_time_points_per_period: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous-time curves:

      • Standard Grover (r = 0) under H_G
      • Dissipative Grover (r = params.r_ancilla) under H_DG
      • BJ analytic approximation F_BJ(t) = 1 - |a(t)|^2

    Returns times in units t/τ, with t ∈ [0, n_periods τ].
    """
    n = params.n_qubits
    M = len(params.solutions)
    N = 2**n
    r = params.r_ancilla
    R = 2**r
    delta = params.delta

    tau = bj_tau(delta)
    t_end = n_periods * tau
    nT = n_periods * n_time_points_per_period
    times = np.linspace(0.0, t_end, nT)
    t_norm = times / tau

    # Standard Grover (r=0)
    H_G = build_standard_grover_hamiltonian(n, params.solutions)
    psi0_G = plus_state(N)
    states_G = time_evolution_from_spectrum(H_G, psi0_G, times)
    F_standard = success_probability_from_states_standard(states_G, n, params.solutions)

    # Dissipative Hamiltonian
    H_DG, _ = build_dissipative_grover_hamiltonian(
        n_qubits=n,
        r_ancilla=r,
        solutions=params.solutions,
        delta=delta,
    )
    psi0_DG = np.kron(plus_state(N), plus_state(R))
    states_DG = time_evolution_from_spectrum(H_DG, psi0_DG, times)
    F_diss = success_probability_from_states_dissipative(
        states_DG,
        n_qubits=n,
        r_ancilla=r,
        solutions=params.solutions,
    )

    # BJ analytic (effective model)
    F_bj = bj_fidelity(times, N=N, M=M, R=R, delta=delta)

    return t_norm, F_standard, F_diss, F_bj


def simulate_discrete_curves(
    params: GroverParams,
    delta_t: float = np.pi,
    n_periods: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Discrete (Trotterized) curves:

      • Standard Grover via U_+ U_S on data
      • Dissipative Grover via U_+ U_S on data⊗reservoir
      • BJ analytic curve F_BJ(t) = 1 - |a(t)|^2 on a fine grid.
    """
    n = params.n_qubits
    M = len(params.solutions)
    N = 2**n
    r = params.r_ancilla
    R = 2**r
    delta = params.delta

    tau = bj_tau(delta)
    t_end = n_periods * tau

    # Number of discrete steps to cover ~n_periods τ
    n_steps = int(np.floor(t_end / delta_t))

    # --- Standard Grover (data only) ---
    U_plus_G = build_U_plus_standard(n, delta_t)
    U_S_G = build_U_S_standard(n, params.solutions, delta_t)
    U_step_G = U_plus_G @ U_S_G

    psi0_G = plus_state(N)
    times_G, states_G = discrete_trotter_evolution(U_step_G, psi0_G, n_steps, step_time=delta_t)
    F_standard = success_probability_from_states_standard(states_G, n, params.solutions)
    t_norm = times_G / tau

    # --- Dissipative Grover (data+reservoir) ---
    _, energies = build_dissipative_grover_hamiltonian(
        n_qubits=n,
        r_ancilla=r,
        solutions=params.solutions,
        delta=delta,
    )
    U_plus_DG = build_U_plus_full(n, r, delta_t)
    U_S_DG = build_U_S_full(n, r, params.solutions, energies, delta_t)
    U_step_DG = U_plus_DG @ U_S_DG

    psi0_DG = np.kron(plus_state(N), plus_state(R))
    times_DG, states_DG = discrete_trotter_evolution(U_step_DG, psi0_DG, n_steps, step_time=delta_t)
    F_diss = success_probability_from_states_dissipative(states_DG, n, r, params.solutions)

    # --- BJ analytic curve on fine grid ---
    t_fine = np.linspace(0.0, t_end, 5 * n_steps if n_steps > 0 else 1)
    F_bj = bj_fidelity(t_fine, N=N, M=M, R=R, delta=delta)
    t_norm_bj = t_fine / tau

    return t_norm, F_standard, F_diss, (t_norm_bj, F_bj)


def make_figure_2(n_periods: int = 3) -> None:
    """
    Generate a 2×2 panel figure analogous to Fig. 2:

      (a) Continuous-time, r = 4
      (b) Trotterized,    r = 4
      (c) Continuous-time, r = 3
      (d) Trotterized,    r = 3

    with parameters exactly as in the original code (apart from plotting
    conventions):

      r = 4, Δ = 0.106  (commented out below)
      r = 3, Δ = 0.15

    The x-axis is t/τ and is plotted from 0 to n_periods.
    """
    n_qubits = 3
    solutions = [0]      # M = 1 marked state

    # Parameter choices
    # params_r4 = GroverParams(n_qubits=n_qubits, solutions=solutions, r_ancilla=4, delta=0.106)
    # params_r3 = GroverParams(n_qubits=n_qubits, solutions=solutions, r_ancilla=3, delta=0.15)

    params_r4 = GroverParams(n_qubits=n_qubits, solutions=solutions,
                             r_ancilla=4, delta=0.1)
    params_r3 = GroverParams(n_qubits=n_qubits, solutions=solutions,
                             r_ancilla=3, delta=0.1)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True, sharey=True)
    axes = axes.ravel()

    # --- (a) Continuous, r=4 ---
    t_norm, F_std, F_diss, F_bj = simulate_continuous_curves(params_r4, n_periods=n_periods)
    axes[0].plot(t_norm, F_std, color=COLORS["standard"], label='r = 0')
    axes[0].plot(t_norm, F_diss, color=COLORS["dissipative"], label='r = 4')
    axes[0].plot(t_norm, F_bj, color=COLORS["bj"], linestyle='--', label='BJ')
    axes[0].set_title('(a) continuous, r = 4', fontsize=24)
    axes[0].set_ylabel('F')

    # --- (b) Discrete, r=4 ---
    t_norm_d, F_std_d, F_diss_d, (t_norm_bj_d, F_bj_d) = simulate_discrete_curves(
        params_r4, delta_t=np.pi, n_periods=n_periods
    )
    axes[1].step(t_norm_d, F_std_d, where='post', color=COLORS["standard"], label='r = 0')
    axes[1].step(t_norm_d, F_diss_d, where='post', color=COLORS["dissipative"], label='r = 4')
    axes[1].plot(t_norm_bj_d, F_bj_d, color=COLORS["bj"], linestyle='--', label='BJ')
    axes[1].set_title('(b) trotterized, r = 4', fontsize=24)

    # --- (c) Continuous, r=3 ---
    t_norm_c, F_std_c, F_diss_c, F_bj_c = simulate_continuous_curves(params_r3, n_periods=n_periods)
    axes[2].plot(t_norm_c, F_std_c, color=COLORS["standard"], label='r = 0')
    axes[2].plot(t_norm_c, F_diss_c, color=COLORS["dissipative"], label='r = 3')
    axes[2].plot(t_norm_c, F_bj_c, color=COLORS["bj"], linestyle='--', label='BJ')
    axes[2].set_title('(c) continuous, r = 3', fontsize=24)
    axes[2].set_xlabel(r'$t/\tau$')
    axes[2].set_ylabel('F')

    # --- (d) Discrete, r=3 ---
    t_norm_cd, F_std_cd, F_diss_cd, (t_norm_bj_cd, F_bj_cd) = simulate_discrete_curves(
        params_r3, delta_t=np.pi, n_periods=n_periods
    )
    axes[3].step(t_norm_cd, F_std_cd, where='post', color=COLORS["standard"], label='r = 0')
    axes[3].step(t_norm_cd, F_diss_cd, where='post', color=COLORS["dissipative"], label='r = 3')
    axes[3].plot(t_norm_bj_cd, F_bj_cd, color=COLORS["bj"], linestyle='--', label='BJ')
    axes[3].set_title('(d) trotterized, r = 3', fontsize=24)
    axes[3].set_xlabel(r'$t/\tau$')

    # Shared formatting (no grid, fixed limits)
    for ax in axes:
        ax.set_xlim(0.0, float(n_periods))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(False)

    # Currently legends in all panels; adjust here if you want only one.
    axes[0].legend(fontsize=14, loc='lower right', frameon=False)
    axes[1].legend(fontsize=14, loc='lower right', frameon=False)
    axes[2].legend(fontsize=14, loc='lower right', frameon=False)
    axes[3].legend(fontsize=14, loc='lower right', frameon=False)

    fig.tight_layout()
    fig.savefig("GroverDissipative_fig.pdf", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use N_PERIODS defined at the top
    make_figure_2(n_periods=N_PERIODS)
