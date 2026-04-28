# Numerical Study of Excitation Dynamics and Correlations in a Rydberg Atom Chain

## 1. Project Overview

This project is a clear first numerical framework for studying a finite one-dimensional chain of laser-driven Rydberg atoms. It is designed as a starting template for a summer research project, so the main goals are:

- readable code
- modular structure
- beginner-friendly explanations
- easy future modification

Each atom is treated as a two-level system:

- \(|g\rangle\) = ground state
- \(|r\rangle\) = Rydberg excited state

The basis convention used throughout the code is:

- `0 = |g>`
- `1 = |r>`

For a chain of \(N\) atoms, the many-body Hilbert space has dimension

```math
2^N
```

because each atom has two possible states and the full basis contains all binary strings of length \(N\).

This first version is intended for small systems such as \(N = 4, 5, 6\), and can often still be used a bit beyond that. However, exact simulation becomes expensive quickly as \(N\) increases.

## 2. Physical Model

The system is modelled as an interacting spin-1/2 chain with coherent laser driving. The Hamiltonian is

```math
H = \frac{\Omega}{2}\sum_i \sigma_x^{(i)}
    - \Delta \sum_i n_i
    + \sum_{i<j} V_{ij} n_i n_j
```

The physical meaning of the parameters is:

- \(\Omega\): Rabi frequency, which sets the coherent drive strength
- \(\Delta\): laser detuning
- \(V\): overall interaction strength scale

The local number operator is

```math
n_i = |r_i\rangle\langle r_i|
```

so \(n_i\) measures whether atom \(i\) is in the Rydberg state.

## 3. Hamiltonian Terms

### Coherent Driving

```math
H_\mathrm{drive} = \frac{\Omega}{2}\sum_i \sigma_x^{(i)}
```

This term flips atoms between \(|g\rangle\) and \(|r\rangle\).

### Detuning

```math
H_\mathrm{detuning} = -\Delta \sum_i n_i
```

This changes the energy cost of creating Rydberg excitations.

### Interaction

```math
H_\mathrm{int} = \sum_{i<j} V_{ij} n_i n_j
```

This term contributes when both sites \(i\) and \(j\) are excited. Physically, it captures the fact that two nearby Rydberg excitations can strongly interact.

## 4. Interaction Options

Two interaction models are included.

### Power-law interaction

```math
V_{ij} = \frac{V}{|i-j|^6}
```

Use:

```python
interaction_type = "power_law"
```

This is a simple van der Waals-type model.

### Nearest-neighbour-only interaction

```math
V_{ij} =
\begin{cases}
V, & j = i + 1 \\
0, & \text{otherwise}
\end{cases}
```

Use:

```python
interaction_type = "nearest_neighbor"
```

This is a simpler truncated model which is easier to compare with basic lattice models.

## 5. Initial State and Time Evolution

The default initial state is

```math
|ggg\cdots g\rangle
```

which means all atoms start in the ground state.

This is a natural starting point because it is simple and lets us watch how the laser drive creates excitations over time.

The current code uses **exact diagonalisation** for a **time-independent Hamiltonian**:

```math
\psi(t) = U e^{-iEt} U^\dagger \psi(0)
```

where \(E\) contains the energy eigenvalues and \(U\) contains the eigenvectors of the Hamiltonian.

We work in units where

```math
\hbar = 1
```

so time is measured in inverse energy or inverse frequency units.

## 6. Observables

The code calculates the following quantities:

### Average total excitation number

```math
\langle N_r(t)\rangle = \left\langle \psi(t)\middle|\sum_i n_i\middle|\psi(t)\right\rangle
```

### Excitation density

```math
\frac{\langle N_r(t)\rangle}{N}
```

### Site-resolved excitation probability

```math
\langle n_i(t)\rangle
```

### Two-point correlation function

```math
\langle n_i n_j\rangle
```

### Connected correlation function

```math
C_{ij} = \langle n_i n_j\rangle - \langle n_i\rangle\langle n_j\rangle
```

The connected correlation removes the part that would already be present if the two sites behaved independently.

## 7. What the Example Script Produces

Running `main.py` generates plots such as:

- average excitation versus time
- excitation density versus time
- site-resolved excitation heatmap
- final-time site excitation distribution
- final-time two-point correlation matrix
- final-time connected correlation matrix
- comparison of excitation dynamics for several interaction strengths

These plots help show:

- coherent oscillations
- the effect of detuning
- how strong interactions suppress multiple excitations
- blockade-related spatial correlation patterns

## 8. How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

The script will save figures into a folder called `figures/`.

## 9. Project Structure

```text
rydberg_chain_project/
    README.md
    main.py
    rydberg_model.py
    observables.py
    plotting.py
    parameter_scan.py
    requirements.txt
```

## 10. Numerical Limitations

This first version is based on exact Hamiltonian construction and exact diagonalisation. That is useful because it is transparent and easy to explain, but it also means the method becomes expensive for larger systems.

The main reason is the Hilbert space growth:

```math
\dim(\mathcal{H}) = 2^N
```

So every extra atom doubles the Hilbert space dimension.

Current limitations of this first version:

- closed-system dynamics only
- time-independent Hamiltonian only
- open boundary conditions only
- no decoherence or dissipation
- no shaped pulses
- no direct comparison with experimental data yet
- dense matrices rather than sparse methods

## 11. Possible Future Extensions

This project was written to make later changes easier. Natural next steps include:

- time-dependent \(\Omega(t)\) or \(\Delta(t)\)
- pulse shaping
- spontaneous decay or dephasing
- Lindblad open-system evolution
- site-dependent parameters
- disorder
- periodic boundary conditions
- sparse-matrix methods
- comparison with lab data

## 12. Which File to Modify

If your supervisor changes the Hamiltonian, the first file to edit is:

- `rydberg_model.py`

If you want to add or change observables:

- `observables.py`

If you want to change the plotting style or add figures:

- `plotting.py`

If you want to add parameter comparisons:

- `parameter_scan.py`
