"""
Observable operators and expectation-value utilities for the Rydberg chain.

This module collects the measurements we want to perform on many-body quantum
states generated during the simulation. The focus is on observables that are
physically meaningful for excitation dynamics and correlations.
"""

from __future__ import annotations

import numpy as np

from rydberg_model import local_operator, number_operator


def total_excitation_operator(N: int) -> np.ndarray:
    """
    Build the total excitation number operator sum_i n_i.

    Physical meaning
    ----------------
    Each local number operator n_i checks whether site i is in the Rydberg
    state. Summing over all sites gives the total number of excited atoms:

    N_r = sum_i n_i

    The expectation value <N_r> is the average number of Rydberg excitations
    present in the chain.

    Parameters
    ----------
    N : int
        Number of atoms in the chain.

    Returns
    -------
    numpy.ndarray
        The many-body operator for the total excitation number.
    """
    n_ops = site_excitation_operators(N)
    return np.sum(n_ops, axis=0)


def site_excitation_operators(N: int) -> list[np.ndarray]:
    """
    Build the list of local excitation number operators n_i for all sites.

    Physical meaning
    ----------------
    The operator n_i = |r_i><r_i| measures whether atom i is excited. Its
    expectation value <n_i> is the excitation probability of site i.

    Parameters
    ----------
    N : int
        Number of atoms in the chain.

    Returns
    -------
    list of numpy.ndarray
        A list containing the many-body operators n_0, n_1, ..., n_{N-1}.
    """
    local_n = number_operator()
    return [local_operator(local_n, site, N) for site in range(N)]


def two_point_operator(N: int, i: int, j: int) -> np.ndarray:
    """
    Build the two-point operator n_i n_j.

    Physical meaning
    ----------------
    The product n_i n_j checks whether sites i and j are both excited. Its
    expectation value <n_i n_j> is therefore the joint excitation probability
    for that pair of sites.

    This is useful when studying correlations and blockade effects, because a
    strong Rydberg interaction can suppress simultaneous excitation of nearby
    atoms.

    Parameters
    ----------
    N : int
        Number of atoms in the chain.
    i, j : int
        Site indices.

    Returns
    -------
    numpy.ndarray
        The many-body operator n_i n_j.
    """
    n_ops = site_excitation_operators(N)
    return n_ops[i] @ n_ops[j]


def expectation_value(state: np.ndarray, operator: np.ndarray) -> float:
    """
    Compute the expectation value <state| operator |state>.

    Physical meaning
    ----------------
    This is the standard quantum-mechanical expectation value. It extracts the
    average result of measuring an observable represented by ``operator`` in
    the state ``state``.

    Parameters
    ----------
    state : numpy.ndarray
        A normalized state vector representing the many-body wavefunction.
    operator : numpy.ndarray
        A square matrix acting on the same Hilbert space as the state vector.

    Returns
    -------
    float
        The expectation value, returned as a real number whenever any tiny
        imaginary part is only numerical floating-point noise.

    Raises
    ------
    ValueError
        If the state is not one-dimensional, if the operator shape does not
        match the state dimension, or if the state is not normalized to within
        a small tolerance.
    """
    if state.ndim != 1:
        raise ValueError("state must be a one-dimensional state vector.")
    if operator.shape != (state.size, state.size):
        raise ValueError("operator shape must match the Hilbert space dimension.")

    norm = np.vdot(state, state)
    if not np.isclose(norm, 1.0, atol=1e-8):
        raise ValueError("state must be normalized before computing expectations.")

    value = np.vdot(state, operator @ state)
    return float(np.real_if_close(value, tol=1000))


def compute_average_excitation(states: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the average total excitation number <N_r(t)> for all times.

    Physical meaning
    ----------------
    This observable answers the question:

    "How many atoms are excited on average as a function of time?"

    It is one of the most direct diagnostics of the competition between laser
    driving, detuning, and interactions.

    Parameters
    ----------
    states : numpy.ndarray
        Array of shape (num_times, 2^N) containing a normalized state vector at
        each time.
    N : int
        Number of atoms.

    Returns
    -------
    numpy.ndarray
        A one-dimensional array containing <N_r(t)> at each time point.
    """
    total_op = total_excitation_operator(N)
    return np.array([expectation_value(state, total_op) for state in states])


def compute_site_resolved_excitation(states: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the site-resolved excitation probabilities <n_i(t)>.

    Physical meaning
    ----------------
    This tells us where excitations are located in the chain. For a finite open
    system, different sites can behave differently, especially near the edges.

    Parameters
    ----------
    states : numpy.ndarray
        Array of shape (num_times, 2^N) containing the state at each time.
    N : int
        Number of atoms.

    Returns
    -------
    numpy.ndarray
        An array of shape (num_times, N), where entry [t, i] is <n_i(t)>.
    """
    n_ops = site_excitation_operators(N)
    site_probs = np.zeros((states.shape[0], N), dtype=float)

    for time_index, state in enumerate(states):
        for site_index, n_i in enumerate(n_ops):
            site_probs[time_index, site_index] = expectation_value(state, n_i)

    return site_probs


def compute_correlation_matrix(state: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the matrix of two-point correlations <n_i n_j> at one time.

    Physical meaning
    ----------------
    The quantity <n_i n_j> measures the probability that sites i and j are
    excited together. The diagonal elements satisfy

    <n_i n_i> = <n_i>

    because n_i is a projector.

    Parameters
    ----------
    state : numpy.ndarray
        A normalized many-body state vector.
    N : int
        Number of atoms.

    Returns
    -------
    numpy.ndarray
        An N x N matrix with entries <n_i n_j>.
    """
    n_ops = site_excitation_operators(N)
    matrix = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            matrix[i, j] = expectation_value(state, n_ops[i] @ n_ops[j])

    return matrix


def compute_connected_correlation_matrix(state: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the connected correlation matrix C_ij.

    Physical meaning
    ----------------
    The raw two-point quantity <n_i n_j> contains both genuine correlations and
    a trivial part coming from the individual excitation probabilities. The
    connected correlation

    C_ij = <n_i n_j> - <n_i><n_j>

    removes that trivial factorised contribution.

    This means:

    - <n_i n_j> tells us how often sites i and j are excited together
    - C_ij tells us how much that joint behaviour differs from what we would
      expect if the two sites were statistically independent

    In blockade physics, connected correlations often make suppression or
    enhancement patterns clearer than the raw correlator alone.

    Parameters
    ----------
    state : numpy.ndarray
        A normalized many-body state vector.
    N : int
        Number of atoms.

    Returns
    -------
    numpy.ndarray
        An N x N connected correlation matrix.
    """
    correlation = compute_correlation_matrix(state, N)
    n_ops = site_excitation_operators(N)
    site_expectations = np.array([expectation_value(state, n_i) for n_i in n_ops])
    disconnected = np.outer(site_expectations, site_expectations)
    return correlation - disconnected
