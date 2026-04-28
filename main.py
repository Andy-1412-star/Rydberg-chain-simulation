"""
Main example script for a finite one-dimensional Rydberg atom chain simulation.

This script is intended to be read from top to bottom by a student who is
learning both the physics model and the numerical method. The calculation uses
exact diagonalisation, which is a natural first approach for small Hilbert
spaces.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parameter_scan import run_single_simulation, scan_interaction_strength
from plotting import (
    compare_average_excitation,
    plot_average_excitation,
    plot_correlation_matrix,
    plot_final_site_distribution,
    plot_site_resolved_heatmap,
)
from rydberg_model import build_hamiltonian, check_hermitian, maybe_print_hamiltonian_summary


def main() -> None:
    """
    Run a complete example simulation and generate figures.
    """
    N = 6
    Omega = 1.0
    Delta = 0.0
    V = 10.0
    interaction_type = "power_law"
    # Choose the initial product state using the convention:
    # "0" = |g> and "1" = |r>.
    # For example:
    # "000000" means all atoms start in |g>
    # "100000" means the first atom starts in |r>
    # "101010" means an alternating Rydberg excitation pattern
    initial_bitstring = "000000"

    # We work in units with hbar = 1, so time is measured in inverse energy
    # (or inverse frequency) units.
    times = np.linspace(0.0, 10.0, 300)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    maybe_print_hamiltonian_summary(N, Omega, Delta, V, interaction_type)
    print(f"  Initial product state        : {initial_bitstring}")
    H = build_hamiltonian(N, Omega, Delta, V, interaction_type=interaction_type)
    if not check_hermitian(H):
        raise ValueError("Hamiltonian failed the Hermiticity check.")

    # Because the Hamiltonian is time-independent, we diagonalise it once and
    # then evolve each energy eigencomponent with a phase exp(-i E t).
    result = run_single_simulation(
        N=N,
        Omega=Omega,
        Delta=Delta,
        V=V,
        interaction_type=interaction_type,
        times=times,
        initial_bitstring=initial_bitstring,
    )

    avg_excitation = result["average_excitation"]
    excitation_density = result["excitation_density"]
    site_probabilities = result["site_probabilities"]
    correlation_matrix = result["final_correlation_matrix"]
    connected_correlation_matrix = result["final_connected_correlation_matrix"]

    print()
    print("Basic simulation checks")
    print(f"  Maximum average excitation       : {avg_excitation.max():.6f}")
    print(f"  Final average excitation         : {avg_excitation[-1]:.6f}")
    print(f"  Final excitation density         : {excitation_density[-1]:.6f}")

    plot_average_excitation(
        times,
        avg_excitation,
        N,
        title="Average Total Excitation vs Time",
        save_path=figures_dir / "average_excitation.png",
    )

    plot_average_excitation(
        times,
        excitation_density,
        N,
        title="Excitation Density vs Time",
        save_path=figures_dir / "excitation_density.png",
    )

    plot_site_resolved_heatmap(
        times,
        site_probabilities,
        title="Site-Resolved Excitation Probability Heatmap",
        save_path=figures_dir / "site_resolved_heatmap.png",
    )

    plot_final_site_distribution(
        site_probabilities[-1],
        title="Final-Time Site Excitation Probability",
        save_path=figures_dir / "final_site_distribution.png",
    )

    plot_correlation_matrix(
        correlation_matrix,
        title=r"Final-Time Two-Point Correlation Matrix $\langle n_i n_j\rangle$",
        save_path=figures_dir / "final_correlation_matrix.png",
    )

    plot_correlation_matrix(
        connected_correlation_matrix,
        title=r"Final-Time Connected Correlation Matrix $C_{ij}$",
        save_path=figures_dir / "final_connected_correlation_matrix.png",
    )

    V_values = [0.0, 1.0, 5.0, 10.0, 20.0]
    scan_results = scan_interaction_strength(
        N=N,
        Omega=Omega,
        Delta=Delta,
        V_values=V_values,
        interaction_type=interaction_type,
        times=times,
        initial_bitstring=initial_bitstring,
    )

    compare_average_excitation(
        scan_results,
        N,
        title="Interaction-Strength Comparison",
        save_path=figures_dir / "interaction_strength_comparison.png",
    )

    print()
    print("Saved figures:")
    for file_path in sorted(figures_dir.glob("*.png")):
        print(f"  {file_path}")

    # Only try to open plot windows when a GUI-capable backend is available.
    # In headless environments we still save all figures, which is the most
    # important behaviour for batch runs and remote sessions.
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
