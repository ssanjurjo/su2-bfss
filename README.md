# Numerics for "Bound States of Two D0-Branes and the Three-Graviton Coupling"

Code and small data behind the numbers, the tables and the figure of *Bound States of Two D0-Branes and the Three-Graviton Coupling* by Sergio Sanjurjo and Themistocles Zikopoulos.
The paper solves numerically for the ground state of the SU(2) BFSS matrix model, the zero-energy bound state of two
D0-branes, and extracts the coefficient of its power-law tail, which is the three-graviton coupling. This repository
contains the numerics of the ground state: the reduction to the singlet sector, the finite-element solver, and the
post-processing that gives the tail coefficient, the moments, the virial terms and the figure.

## Units and conventions

`g = 2` and lightlike radius `R_- = 1`, with the Hamiltonian in the coordinates `q_IA` of Fröhlich, Graf, Hasler,
Hoppe and Yau; the eleven-dimensional Planck length is `l_p = (16 pi^2)^(-1/6) = 0.430`. `R` is the separation of the
two D0-branes along the valley, where the density of the ground state is `P(R) = (8/9) Omega_8 |C|^2 R^-10`.

## Requirements and running

Python 3.12 with the packages of `requirements.txt` (`python -m pip install -r requirements.txt`). The figure needs a
LaTeX installation. The large finite-element solves use Julia 1.11 for the sparse Cholesky factorization
(`julia --project=ground_state/jl -e 'using Pkg; Pkg.instantiate()'`; set `JULIA` if `julia` is not on the path);
without Julia, `EIGSOLVER=superlu` uses the sparse LU of SciPy, which is enough for small grids.
The solver starts its worker processes by fork, so it needs Linux; set `OMP_NUM_THREADS=1` so that the processes
do not compete for threads.

Every script runs from any folder, for example `python ground_state/moments.py`, and its docstring gives the
arguments. A script that checks its numbers prints one `PASS` or `FAIL` line per number and exits with status 1 on
a mismatch.

## Scripts

Runtimes are for one core unless stated. "Saved state" means a finite-element state, which is not shipped (see Data).

| part of the paper | script | what it reproduces | runtime |
|---|---|---|---|
| Appendix: Reduction for the Ground State of SU(2) BFSS | `reduction/fspace.py` (with `fermions.py`, `common/gamma.py`) | the basis of the 2066 states and the fermionic operators of the reduced Hamiltonian, written to `fspace_ops.pkl` | 10 minutes, 40 GB with 8 processes |
| The ground state: the numerical solution. Appendix: Details on the Numerical Solution (solving for the ground state) | `ground_state/solve_ch.py` (with `fem3d.py`, `jl/eig_chol.jl`) | the variational state: channels, enrichment, eigenvalue problem | 1 to 12 hours on 80 cores, up to 200 GB |
| The numerical solution: results | `ground_state/box_levels.py` | `E_h = 0.0003`, `E_1 = 0.1742` and the overlap estimate `1 - E_h/E_1`, about `0.998` | seconds; saved state |
| the same | `ground_state/virial.py` | the virial terms `<T> = 27.166`, `<V_B> = 27.164`, `<V_F> = -54.329` (result in `data/virial/`) | ten minutes to two hours, up to 20 GB; saved state |
| The numerical solution: tail coefficient. Appendix: Details on the Numerical Solution (asymptotic tail of the wavefunction) | `ground_state/tail_fit.py --save` | the density of the largest singular value of a state (`data/profiles/`) | 30 minutes to tens of core-hours; saved state |
| the same | `ground_state/tail_c2/c2_table.py --json table.json`, then `ground_state/tail_c2/final.py table.json` | `\|C\|^2 = 286.4 +- 0.2`: the 24 fits of the best state, the values with coarser elements along the valley, the changes with the 12 x 12 mesh and with the pruning tolerance, and the uncertainty | seconds |
| The numerical solution: moments. Appendix: Details on the Numerical Solution (moments) | `ground_state/observables.py` | the moments of a state, completed with the tail (`data/observables_lowtol/`) | a minute to half an hour, up to 20 GB; saved state |
| the same | `ground_state/moments.py` | both tables of moments with their errors, in the two normalizations; the root-mean-square hyperradius `7.4 l_p` | seconds |
| The figure (the far field of the bound state) | `figures/fig_density.py` | the figure; the local coefficient relative to its asymptote at `R = 7, 10, 15` and the coefficient `D_3` of the `R^-3` correction | seconds |

## Data

`data/` holds what the post-processing scripts produced from the finite-element states: the radial profiles, the moments
and the virial terms, and the expected output of the fit of the tail. `data/README.md` describes each file and the
states. The states themselves (0.3 to 1.5 GB each) are not shipped; the same file gives the
commands that produce them, with the size and checksum of the best state. The operators file behind them is shipped as
`data/fspace_ops.pkl` (19 MB): pass it with `--ops`. `data/runs.json` records, for every state, the settings
of its run and how the eigensolver ended.

## Notes on the shipped results

- The moments are completed beyond the matching radius with the predicted tail coefficient `893025/(32 pi^4) = 286.49`,
  the default of `observables.py`. With the fitted `286.4` the largest change is `-0.25` in
  `<t1^4> = 16140(10)`. The fit of the tail coefficient itself does not use the prediction.
- For the best state and for `Q4L22G16V35` and `Q4N12L22G16` the eigensolver stopped at its limit of 80 iterations,
  with relative residuals `2.6e-9`, `1.6e-9` and `1.1e-9` for the lowest state, above its tolerance of `1e-10`
  (the three higher levels of the best state are below `1e-11`). The residual is measured relative to
  `|E| |M x| + |H x|`, which for the lowest state is of the order of `E_h` itself.
- A state written by this version records its Gauss orders, its final number of channels, the residuals of the eigensolver
  and the checksum of the operators file. The states behind the shipped data were written before that and record none of
  them: they must be read with the Gauss orders given in `data/README.md`.

## Tests

    python -m pytest -q tests

runs the scripts of the table that need no saved state and take seconds, recomputes the tail coefficient from the shipped
profiles and compares it with `data/tail_c2/final.txt`, checks the numbers that follow directly from the shipped data
(the peak of the density, the probability beyond `R = 4`, the virial terms), and draws the figure if LaTeX is available;
all of it takes under a minute.

## License

MIT, see `LICENSE`.
