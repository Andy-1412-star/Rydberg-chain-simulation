"""
Helper functions for comparing different physical parameter regimes.

The main idea of this module is to keep scan logic separate from the basic
Hamiltonian and observable definitions. That makes the project easier to extend
later when more complicated scans or fitting procedures are needed.
"""

from __future__ import annotations

import numpy as np

from observables import (
    compute_average_excitation,
    compute_connected_correlation_matrix,
    compute_correlation_matrix,
    compute_site_resolved_excitation,
)
from rydberg_model import (
    build_hamiltonian,
    check_hermitian,
    initial_ground_state,
    initial_product_state,
)


def evolve_state_exact_diagonalization(
    H: np.ndarray,
    initial_state: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Evolve a state under a time-independent Hamiltonian using exact diagonalisation.

    Physical meaning
    ----------------
    For a time-independent Hamiltonian, if

    H = U diag(E) U^dagger

    then the exact quantum evolution is

    psi(t) = U diag(exp(-i E t)) U^dagger psi(0)

    This method is especially transparent for small systems because it exposes
    the energy eigenvalues and uses the exact unitary evolution operator.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Expand the initial state in the energy eigenbasis once. After that, each
    # time step only changes the phase of each eigencomponent.
    initial_in_eigenbasis = eigenvectors.conj().T @ initial_state

    states = np.zeros((times.size, initial_state.size), dtype=complex)
    for time_index, time in enumerate(times):
        phase_factors = np.exp(-1j * eigenvalues * time)
        states[time_index] = eigenvectors @ (phase_factors * initial_in_eigenbasis)

    return states


def run_single_simulation(
    N: int,
    Omega: float,
    Delta: float,
    V: float,
    interaction_type: str,
    times: np.ndarray,
    initial_bitstring: str | None = None,
) -> dict:
    """
    Run one complete Rydberg-chain simulation and return a structured result.

    Physical meaning
    ----------------
    This function packages together the standard workflow:

    1. build the Hamiltonian
    2. start from |ggg...g> or another chosen product state
    3. evolve in time
    4. compute key observables

    It is designed so that parameter scans can reuse the same workflow without
    duplicating code.

    Parameters
    ----------
    N : int
        Number of atoms in the chain.
    Omega : float
        Rabi frequency.
    Delta : float
        Detuning.
    V : float
        Interaction strength scale.
    interaction_type : str
        Interaction model, either ``"power_law"`` or ``"nearest_neighbor"``.
    times : numpy.ndarray
        One-dimensional array of time points.
    initial_bitstring : str or None, optional
        Binary string specifying the initial product state. If ``None``, the
        simulation starts from the all-ground state ``|ggg...g>``. If a string
        is provided, its length must equal ``N``.

    Returns
    -------
    dict
        A dictionary containing the parameters, Hamiltonian, chosen initial
        state, time-evolved states, and several observables.
    """
    H = build_hamiltonian(N, Omega, Delta, V, interaction_type=interaction_type)
    if not check_hermitian(H):
        raise ValueError("Constructed Hamiltonian is not Hermitian.")

    if initial_bitstring is None:
        psi0 = initial_ground_state(N)
    else:
        if len(initial_bitstring) != N:
            raise ValueError("initial_bitstring must have length N.")
        psi0 = initial_product_state(initial_bitstring)

    states = evolve_state_exact_diagonalization(H, psi0, times)

    # Normalization should be preserved by unitary time evolution. Checking it
    # numerically is a useful sanity check when building a simulation template.
    norms = np.sum(np.abs(states) ** 2, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-8):
        raise ValueError("Time-evolved states are not normalized within tolerance.")

    average_excitation = compute_average_excitation(states, N)
    site_probabilities = compute_site_resolved_excitation(states, N)
    final_state = states[-1]
    correlation_matrix = compute_correlation_matrix(final_state, N)
    connected_correlation_matrix = compute_connected_correlation_matrix(final_state, N)

    return {
        "parameters": {
            "N": N,
            "Omega": Omega,
            "Delta": Delta,
            "V": V,
            "interaction_type": interaction_type,
            "initial_bitstring": initial_bitstring,
        },
        "times": times,
        "hamiltonian": H,
        "initial_state": psi0,
        "initial_bitstring": initial_bitstring,
        "states": states,
        "average_excitation": average_excitation,
        "excitation_density": average_excitation / N,
        "site_probabilities": site_probabilities,
        "final_correlation_matrix": correlation_matrix,
        "final_connected_correlation_matrix": connected_correlation_matrix,
    }


def scan_interaction_strength(
    N: int,
    Omega: float,
    Delta: float,
    V_values: list[float],
    interaction_type: str,
    times: np.ndarray,
    initial_bitstring: str | None = None,
) -> dict:
    """
    Compare dynamics while scanning the interaction strength V.

    Physical expectation
    --------------------
    When V / Omega is small, interactions only weakly disturb the laser-driven
    dynamics. When V / Omega becomes large, simultaneous nearby excitations are
    energetically suppressed, which is the basic idea behind the Rydberg
    blockade regime.

    Parameters
    ----------
    N : int
        Number of atoms.
    Omega : float
        Rabi frequency held fixed during the scan.
    Delta : float
        Detuning held fixed during the scan.
    V_values : list[float]
        Interaction strengths to compare.
    interaction_type : str
        Interaction model used during the scan.
    times : numpy.ndarray
        Time grid used for every simulation.
    initial_bitstring : str or None, optional
        Initial product state used for every run in the scan. If ``None``, the
        scan starts from the all-ground state.

    Returns
    -------
    dict
        A dictionary mapping readable labels to complete simulation results.
    """
    results = {}
    for V in V_values:
        label = f"V = {V}"
        results[label] = run_single_simulation(
            N=N,
            Omega=Omega,
            Delta=Delta,
            V=V,
            interaction_type=interaction_type,
            times=times,
            initial_bitstring=initial_bitstring,
        )
    return results


def scan_detuning(
    N: int,
    Omega: float,
    Delta_values: list[float],
    V: float,
    interaction_type: str,
    times: np.ndarray,
    initial_bitstring: str | None = None,
) -> dict:
    """
    Compare dynamics while scanning the detuning Delta.

    Physical expectation
    --------------------
    Changing Delta changes the energetic preference for making Rydberg
    excitations. Negative, zero, and positive detuning can therefore produce
    noticeably different oscillation amplitudes and correlation patterns.

    Parameters
    ----------
    N : int
        Number of atoms.
    Omega : float
        Rabi frequency held fixed.
    Delta_values : list[float]
        Detuning values to compare.
    V : float
        Interaction strength held fixed.
    interaction_type : str
        Interaction model used during the scan.
    times : numpy.ndarray
        Time grid used for every simulation.
    initial_bitstring : str or None, optional
        Initial product state used for every run in the scan. If ``None``, the
        scan starts from the all-ground state.

    Returns
    -------
    dict
        A dictionary mapping readable labels to complete simulation results.
    """
    results = {}
    for Delta in Delta_values:
        label = f"Delta = {Delta}"
        results[label] = run_single_simulation(
            N=N,
            Omega=Omega,
            Delta=Delta,
            V=V,
            interaction_type=interaction_type,
            times=times,
            initial_bitstring=initial_bitstring,
        )
    return results


def scan_driving_strength(
    N: int,
    Omega_values: list[float],
    Delta: float,
    V: float,
    interaction_type: str,
    times: np.ndarray,
    initial_bitstring: str | None = None,
) -> dict:
    """
    Compare dynamics while scanning the driving strength Omega.

    Physical expectation
    --------------------
    Increasing Omega strengthens coherent driving and usually speeds up the
    oscillatory dynamics. The competition between Omega and V is particularly
    important, because the ratio V / Omega is a simple guide to whether the
    system is in a weakly interacting or blockade-dominated regime.

    Parameters
    ----------
    N : int
        Number of atoms.
    Omega_values : list[float]
        Rabi frequencies to compare.
    Delta : float
        Detuning held fixed.
    V : float
        Interaction strength held fixed.
    interaction_type : str
        Interaction model used during the scan.
    times : numpy.ndarray
        Time grid used for every simulation.
    initial_bitstring : str or None, optional
        Initial product state used for every run in the scan. If ``None``, the
        scan starts from the all-ground state.

    Returns
    -------
    dict
        A dictionary mapping readable labels to complete simulation results.
    """
    results = {}
    for Omega in Omega_values:
        label = f"Omega = {Omega}"
        results[label] = run_single_simulation(
            N=N,
            Omega=Omega,
            Delta=Delta,
            V=V,
            interaction_type=interaction_type,
            times=times,
            initial_bitstring=initial_bitstring,
        )
    return results
