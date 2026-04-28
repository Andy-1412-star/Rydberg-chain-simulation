"""
Tools for building a finite one-dimensional Rydberg atom chain Hamiltonian.

This module contains the core many-body model construction used throughout the
project. The main goal is clarity: each atom is treated as a two-level system,
and many-body operators are assembled explicitly using tensor products.

Basis convention
----------------
For each site we use

- 0 = |g>  (ground state)
- 1 = |r>  (Rydberg excited state)

If there are N atoms, the many-body Hilbert space has dimension 2^N because
each site contributes a two-dimensional local Hilbert space.
"""

from __future__ import annotations

import numpy as np


def pauli_x() -> np.ndarray:
    """
    Return the single-site Pauli-x operator.

    Physical meaning
    ----------------
    In the two-level basis {|g>, |r>}, the Pauli-x operator changes one local
    state into the other. In other words, it flips the state of a single atom:

    - |g> -> |r>
    - |r> -> |g>

    This is why it appears in the coherent laser driving term of the
    Hamiltonian. A sum of local sigma_x operators models laser-induced
    transitions between the ground and Rydberg states.

    Returns
    -------
    numpy.ndarray
        A 2 x 2 complex-valued array representing the Pauli-x matrix.

    Notes
    -----
    We return a complex-valued array even though the entries are real, because
    the full quantum state evolution uses complex amplitudes.
    """
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def number_operator() -> np.ndarray:
    """
    Return the single-site Rydberg excitation number operator.

    Physical meaning
    ----------------
    The operator

    n = |r><r|

    checks whether a given atom is in the Rydberg excited state. In the basis
    convention

    - 0 = |g>
    - 1 = |r>

    this becomes the matrix

    [[0, 0],
     [0, 1]]

    Its expectation value is the probability that the atom is excited.

    Returns
    -------
    numpy.ndarray
        A 2 x 2 complex-valued array representing the local number operator.
    """
    return np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)


def identity_operator() -> np.ndarray:
    """
    Return the single-site identity operator.

    Physical meaning
    ----------------
    The identity operator acts trivially on one site. It is used when we want
    to place a non-trivial operator on one site of the chain while leaving all
    other sites unchanged.

    Returns
    -------
    numpy.ndarray
        A 2 x 2 complex-valued identity matrix.
    """
    return np.eye(2, dtype=complex)


def local_operator(operator: np.ndarray, site: int, N: int) -> np.ndarray:
    """
    Embed a single-site operator into the full many-body Hilbert space.

    Physical meaning
    ----------------
    A local operator such as sigma_x or n acts on only one atom. However, the
    many-body state of the whole chain lives in a Hilbert space of dimension
    2^N. To apply a single-site operator to site ``site``, we build the tensor
    product

    I ⊗ I ⊗ ... ⊗ operator ⊗ ... ⊗ I

    where the chosen operator appears at the selected site and the identity
    acts on all other sites.

    Parameters
    ----------
    operator : numpy.ndarray
        A 2 x 2 single-site operator.
    site : int
        The site index where the operator should be placed. We use
        zero-based indexing, so valid sites are 0, 1, ..., N-1.
    N : int
        Number of atoms in the chain.

    Returns
    -------
    numpy.ndarray
        A (2^N) x (2^N) matrix representing the operator acting on the full
        many-body Hilbert space.

    Raises
    ------
    ValueError
        If ``N`` is not positive, if ``site`` is out of range, or if the input
        operator is not 2 x 2.

    Notes
    -----
    This explicit tensor-product construction is very readable and suitable for
    small systems. For larger systems, one would usually switch to sparse
    matrices or other more efficient representations.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if site < 0 or site >= N:
        raise ValueError(f"site must be between 0 and {N - 1}.")
    if operator.shape != (2, 2):
        raise ValueError("operator must have shape (2, 2).")

    full_operator = np.array([[1.0]], dtype=complex)
    identity = identity_operator()

    # We grow the full many-body operator one site at a time.
    # At each position we choose either the non-trivial local operator or the
    # identity, depending on which site we want to act on.
    for current_site in range(N):
        factor = operator if current_site == site else identity
        full_operator = np.kron(full_operator, factor)

    return full_operator


def interaction_strength(i: int, j: int, V: float, interaction_type: str) -> float:
    """
    Return the interaction coefficient V_ij for a chosen pair of sites.

    Physical meaning
    ----------------
    The interaction term in the Hamiltonian is

    H_int = sum_{i<j} V_ij n_i n_j

    It contributes only when both sites i and j are simultaneously excited,
    because n_i n_j projects onto states where both atoms are in |r>.

    Two simple interaction models are supported:

    - ``"power_law"``:
      V_ij = V / |i - j|^6
    - ``"nearest_neighbor"``:
      V_ij = V for nearest neighbours, and 0 otherwise

    Parameters
    ----------
    i, j : int
        Site indices with i != j.
    V : float
        Overall interaction scale.
    interaction_type : str
        Name of the interaction model.

    Returns
    -------
    float
        The interaction coefficient for the pair (i, j).

    Raises
    ------
    ValueError
        If an unsupported interaction type is provided.
    """
    distance = abs(i - j)

    if interaction_type == "power_law":
        return V / (distance**6)
    if interaction_type == "nearest_neighbor":
        return V if distance == 1 else 0.0

    raise ValueError(
        "interaction_type must be either 'power_law' or 'nearest_neighbor'."
    )


def build_hamiltonian(
    N: int,
    Omega: float,
    Delta: float,
    V: float,
    interaction_type: str = "power_law",
) -> np.ndarray:
    """
    Construct the full many-body Hamiltonian for a finite Rydberg chain.

    Physical meaning
    ----------------
    The Hamiltonian implemented here is

    H = H_drive + H_detuning + H_int

    with

    H_drive = (Omega / 2) * sum_i sigma_x^(i)
    H_detuning = -Delta * sum_i n_i
    H_int = sum_{i<j} V_ij n_i n_j

    This describes a chain of laser-driven two-level atoms with open boundary
    conditions. The drive flips local states, the detuning shifts the energy of
    excited atoms, and the interaction penalises configurations with multiple
    Rydberg excitations.

    Parameters
    ----------
    N : int
        Number of atoms in the chain.
    Omega : float
        Rabi frequency controlling the coherent drive strength.
    Delta : float
        Laser detuning. Positive or negative detuning changes how favourable
        Rydberg excitation is energetically.
    V : float
        Overall interaction strength.
    interaction_type : str, optional
        Interaction model to use. Supported values are ``"power_law"`` and
        ``"nearest_neighbor"``.

    Returns
    -------
    numpy.ndarray
        A Hermitian (2^N) x (2^N) Hamiltonian matrix.

    Raises
    ------
    ValueError
        If ``N`` is not positive or if ``interaction_type`` is invalid.

    Notes
    -----
    This exact matrix construction is intended for small system sizes. The
    matrix size doubles every time one more atom is added, so this approach
    becomes expensive quickly.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if interaction_type not in {"power_law", "nearest_neighbor"}:
        raise ValueError(
            "interaction_type must be either 'power_law' or 'nearest_neighbor'."
        )

    dimension = 2**N
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    sigma_x = pauli_x()
    n_op = number_operator()

    # Precompute local operators so we do not rebuild the same tensor products
    # repeatedly. This also makes the physics structure easier to see:
    # one list for local spin flips and one list for local excitation counters.
    sigma_x_sites = [local_operator(sigma_x, site, N) for site in range(N)]
    number_sites = [local_operator(n_op, site, N) for site in range(N)]

    # Coherent laser driving: each sigma_x term flips atom i between |g> and |r>.
    for sigma_x_i in sigma_x_sites:
        hamiltonian += 0.5 * Omega * sigma_x_i

    # Detuning: each local excitation contributes an energy shift -Delta.
    for n_i in number_sites:
        hamiltonian += -Delta * n_i

    # Interaction: n_i n_j contributes only if both sites are excited.
    # This is why the blockade effect appears when V is large: double
    # excitations on nearby sites become energetically costly.
    for i in range(N):
        for j in range(i + 1, N):
            Vij = interaction_strength(i, j, V, interaction_type)
            if Vij != 0.0:
                hamiltonian += Vij * (number_sites[i] @ number_sites[j])

    return hamiltonian


def initial_ground_state(N: int) -> np.ndarray:
    """
    Return the many-body state |ggg...g> as a normalized state vector.

    Physical meaning
    ----------------
    This is the product state in which every atom starts in the ground state.
    In the computational basis, it corresponds to the basis vector whose bit
    string is all zeros.

    It is a natural initial state for studying how coherent driving creates
    Rydberg excitations and how interactions then modify those dynamics.

    Parameters
    ----------
    N : int
        Number of atoms in the chain.

    Returns
    -------
    numpy.ndarray
        A complex-valued vector of length 2^N representing |ggg...g>.

    Raises
    ------
    ValueError
        If ``N`` is not positive.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    state = np.zeros(2**N, dtype=complex)
    state[0] = 1.0 + 0.0j
    return state


def initial_product_state(bitstring: str) -> np.ndarray:
    """
    Return a computational-basis product state specified by a binary string.

    Physical meaning
    ----------------
    This function prepares a simple product state in which each atom is placed
    either in the ground state |g> or in the Rydberg excited state |r>.

    The basis convention is

    - ``"0"`` means ``|g>``
    - ``"1"`` means ``|r>``

    so a bitstring such as

    - ``"000000"`` means ``|gggggg>``
    - ``"100000"`` means ``|rggggg>``
    - ``"101010"`` means ``|rgrgrg>``

    The leftmost character corresponds to site 0 and the rightmost character
    corresponds to site N - 1. Because this is a computational-basis product
    state, the final many-body state vector has a single non-zero entry.

    Parameters
    ----------
    bitstring : str
        Binary string specifying the product state of the chain.

    Returns
    -------
    numpy.ndarray
        A normalized complex-valued state vector of length 2^N, where N is the
        length of the bitstring.

    Raises
    ------
    ValueError
        If the bitstring is empty or contains characters other than ``"0"``
        and ``"1"``.

    Notes
    -----
    The binary string maps directly to the computational basis index. For
    example, the bitstring ``"101"`` is the binary representation of the
    integer 5, so the returned state vector has amplitude 1 at index 5 and 0
    everywhere else.
    """
    if not bitstring:
        raise ValueError("bitstring must not be empty.")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("bitstring must contain only '0' and '1'.")

    N = len(bitstring)
    dimension = 2**N
    state = np.zeros(dimension, dtype=complex)

    # The many-body computational basis is ordered by binary integers.
    # Reading the whole bitstring as a binary number therefore tells us which
    # basis vector represents the chosen product state.
    basis_index = int(bitstring, 2)
    state[basis_index] = 1.0 + 0.0j

    return state


def check_hermitian(H: np.ndarray, atol: float = 1e-10) -> bool:
    """
    Check whether a matrix is Hermitian within a numerical tolerance.

    Physical meaning
    ----------------
    A closed-system Hamiltonian must be Hermitian so that its energy spectrum
    is real and time evolution is unitary. This is an important sanity check
    for a numerically constructed quantum model.

    Parameters
    ----------
    H : numpy.ndarray
        Matrix to test.
    atol : float, optional
        Absolute tolerance used in the comparison.

    Returns
    -------
    bool
        ``True`` if H is Hermitian within the given tolerance, otherwise
        ``False``.
    """
    return np.allclose(H, H.conj().T, atol=atol)


def maybe_print_hamiltonian_summary(
    N: int,
    Omega: float,
    Delta: float,
    V: float,
    interaction_type: str,
) -> None:
    """
    Print a short human-readable summary of the chosen Hamiltonian parameters.

    Physical meaning
    ----------------
    This is not required for the simulation itself, but it is helpful when
    reading output and checking that a run corresponds to the intended physical
    regime, such as weak interactions or strong blockade.

    Parameters
    ----------
    N : int
        Number of atoms.
    Omega : float
        Rabi frequency.
    Delta : float
        Detuning.
    V : float
        Interaction strength scale.
    interaction_type : str
        Interaction model name.

    Returns
    -------
    None
        This function prints information and returns nothing.
    """
    dimension = 2**N
    print("Rydberg chain Hamiltonian summary")
    print(f"  Number of atoms N           : {N}")
    print(f"  Hilbert space dimension     : {dimension}")
    print(f"  Rabi frequency Omega        : {Omega}")
    print(f"  Detuning Delta              : {Delta}")
    print(f"  Interaction scale V         : {V}")
    print(f"  Interaction type            : {interaction_type}")
    print("  Boundary conditions         : open")
