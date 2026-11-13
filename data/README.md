# Data

Units as in the top-level README. Every file here was written by a script of this repository from a saved finite-element
state; the states themselves are too large to ship.

## Files

| path | content | written by |
|---|---|---|
| `profiles/solve_ch_<tag>_PR.npz` | the density `P` of the largest singular value on the grid `R = 0.05, 0.10, ...` and the energy `E` of the state (`E_h`), for the 18 states below | `ground_state/tail_fit.py <state> --save` |
| `tail_c2/table.json` | the tail coefficient of every profile: the direct far-field fit of the best state (`C_ref`) and, per state, `E`, the position of the wall `Rw`, `C2`, the window amplitude `dN` with its spread, the two-term fit on `[10, 13]` (`null` for the long boxes), `C2` with three other windows, and the integral of the profile | `ground_state/tail_c2/c2_table.py --json table.json` |
| `tail_c2/final.txt` | the tail coefficient with the entries of its uncertainty | `ground_state/tail_c2/final.py table.json` |
| `fspace_ops.pkl` | the operators of the reduced Hamiltonian behind every shipped state (a Python pickle of sparse matrices; checksum below) | `reduction/fspace.py` |
| `runs.json` | for every state: mesh, channels per node, element and Gauss orders, levels computed, seeds, and how the eigensolver ended | the logs of the solver runs |
| `observables_lowtol/obs_<tag>.pkl` | moments of a state inside the box and completed with the tail at the matching radii 8, 10, 12, and histograms of `r1` and of the hyperradius, for the eight states that enter the tables of moments (a Python pickle of a dictionary of numbers, lists and NumPy arrays) | `ground_state/observables.py <state>` |
| `virial/virial_Q4L22G16V25_r0.json` | the kinetic (radial and angular), bosonic and Yukawa terms of the best state, their sum and the stored `E_h` | `ground_state/virial.py <state>` |
| `../figures/data/profile_Q4L22G16V25_r0.npz` | a copy of the profile of the best state, read by the figure | |

The profiles of the larger states were computed with the function of `tail_fit.py` distributed over the values of `R`.

## States

All states live in the box `a, b <= 3`, `c <= L_c`, with quadratic elements along the valley that start at 0.15 and grow by
factors of 1.4 up to the largest size `h`. `_r<k>` is the state after `k` enrichment rounds of its run. Channels: saved
states interpolated to the grid ("seeds") plus the lowest local eigenvectors.

| tag | mesh in `(a, b)` | elements in `a, b` | `h` | pruning tolerance | `L_c` | Gauss points in `a, b` | channels | `E_h` |
|---|---|---|---|---|---|---|---|---|
| `LE10B15` | 10 x 10 | cubic | 0.5 | `1e-12` | 15 | 6 | 8 local, 10 rounds | `0.008705` |
| `P16B15_r0` | 10 x 10 | cubic | 0.5 | `1e-16` | 15 | 6 | seeds LE10B15 and its rounds 6 to 9 + 8 local | `0.003381` |
| `P16B15_r1` | 10 x 10 | cubic | 0.5 | `1e-16` | 15 | 6 | the same, one more round | `0.003314` |
| `T24B15_r0` | 10 x 10 | cubic | 0.5 | `1e-24` | 15 | 6 | seeds P16B15_r0, LE10B15 and its round 9 + 8 local | `0.003372` |
| `N12P16B15_r0` | 12 x 12 | cubic | 0.5 | `1e-16` | 15 | 6 | seeds: two 12 x 12 states at `1e-12`, P16B15_r0, T24B15_r0 + 8 local | `0.001936` |
| `N12P16B15_r1` | 12 x 12 | cubic | 0.5 | `1e-16` | 15 | 6 | the same, one more round | `0.001902` |
| `Q4P16B15_r0` | 10 x 10 | quartic | 0.5 | `1e-16` | 15 | 6 | seeds P16B15_r1, P16B15_r0 + 8 local | `0.000909` |
| `Q4N12P16B15_r0` | 12 x 12 | quartic | 0.5 | `1e-16` | 15 | 6 | seeds N12P16B15_r1, N12P16B15_r0 + 8 local | `0.000846` |
| `L22P16B22_r0` | 10 x 10 | cubic | 0.5 | `1e-16` | 22 | 6 | seeds P16B15_r0, T24B15_r0, LE10B22 and its round 9 + 8 local | `0.003364` |
| `L22P16B22_r1` | 10 x 10 | cubic | 0.5 | `1e-16` | 22 | 6 | the same, one more round | `0.003315` |
| `Q4L22G16_r0` | 10 x 10 | quartic | 0.5 | `1e-16` | 22 | 16 | seeds Q4L22P16B22_r0, L22P16B22_r1 + 8 local | `0.000910` |
| `Q4L22G12_r0` | 10 x 10 | quartic | 0.5 | `1e-16` | 22 | 12 | as Q4L22G16_r0 | `0.000910` |
| `Q4L22G16E_r0` | 10 x 10 | quartic | 0.5 | `1e-16` | 22 | 16 | seed Q4L22P16B22_r0 + 4 local | `0.000910` |
| `Q4L22G16E_r1` | 10 x 10 | quartic | 0.5 | `1e-16` | 22 | 16 | the same, one more round | `0.000897` |
| `Q4L22G16T24_r0` | 10 x 10 | quartic | 0.5 | `1e-24` | 22 | 16 | seeds Q4L22P16B22_r0, L22P16B22_r1 + 4 local | `0.000889` |
| `Q4N12L22G16_r0` | 12 x 12 | quartic | 0.5 | `1e-16` | 22 | 16 | seed Q4L22G16_r0 + 4 local | `0.000901` |
| `Q4L22G16V35_r0` | 10 x 10 | quartic | 0.35 | `1e-16` | 22 | 16 | seed Q4L22G16_r0 + 4 local | `0.000513` |
| `Q4L22G16V25_r0` | 10 x 10 | quartic | 0.25 | `1e-16` | 22 | 16 | seed Q4L22G16_r0 + 4 local | `0.000308` |

`Q4L22G16V25_r0` is the best state. The others measure the effect of one parameter at a time, or supply
its channels.

## Producing the states

The operators of the reduced Hamiltonian come first. They are shipped as `data/fspace_ops.pkl` (18.9 MB), and
`python reduction/fspace.py --out fspace_ops.pkl`, run from the root of the repository, rebuilds them (10 minutes, 40 GB
with 8 processes). The shipped file has SHA-256
`0e53fa6f749bb6ea555742f5ed757a9102989bd5d3c999886961c23ff1f56b32`; with one thread per process (`OMP_NUM_THREADS=1`)
and the packages of `requirements.txt` the script reproduces it byte for byte, while multithreaded BLAS can rotate the
basis within degenerate blocks, which changes the file and not the physics.
The states are then written by `ground_state/solve_ch.py`. The commands below are run from the root of the repository, after
`export BFSS_FSPACE_OPS=$PWD/data/fspace_ops.pkl` (or with `--ops data/fspace_ops.pkl` in each of them), and write the states
to the folder `states`. The chain that leads to the best state is

    python ground_state/solve_ch.py 10 10 0 3.0 3.0 15.0 8 --h0 0.25 --hmaxc 0.5 --enrich 10 --outdir states --out _LE10B15
    python ground_state/solve_ch.py 10 10 0 3.0 3.0 22.0 8 --h0 0.25 --hmaxc 0.5 --enrich 10 --outdir states --out _LE10B22
    python ground_state/solve_ch.py 10 10 0 3.0 3.0 15.0 8 --h0 0.25 --hmaxc 0.5 --ptol 1e-16 --enrich 1 --seed states/solve_ch_LE10B15.pkl \
        --seed states/solve_ch_LE10B15_r9.pkl --seed states/solve_ch_LE10B15_r8.pkl --seed states/solve_ch_LE10B15_r7.pkl \
        --seed states/solve_ch_LE10B15_r6.pkl --outdir states --out _P16B15
    python ground_state/solve_ch.py 10 10 0 3.0 3.0 15.0 8 --h0 0.25 --hmaxc 0.5 --ptol 1e-24 --enrich 0 --seed states/solve_ch_P16B15_r0.pkl \
        --seed states/solve_ch_LE10B15.pkl --seed states/solve_ch_LE10B15_r9.pkl --outdir states --out _T24B15
    FEM_ORDER_AB=4 python ground_state/solve_ch.py 10 10 0 3.0 3.0 15.0 8 --h0 0.25 --hmaxc 0.5 --ptol 1e-16 --enrich 0 \
        --seed states/solve_ch_P16B15_r1.pkl --seed states/solve_ch_P16B15_r0.pkl --outdir states --out _Q4P16B15
    python ground_state/solve_ch.py 10 10 0 3.0 3.0 22.0 8 --h0 0.25 --hmaxc 0.5 --ptol 1e-16 --enrich 1 --seed states/solve_ch_P16B15_r0.pkl \
        --seed states/solve_ch_T24B15_r0.pkl --seed states/solve_ch_LE10B22.pkl --seed states/solve_ch_LE10B22_r9.pkl --outdir states --out _L22P16B22
    FEM_ORDER_AB=4 python ground_state/solve_ch.py 10 10 0 3.0 3.0 22.0 8 --h0 0.25 --hmaxc 0.5 --ptol 1e-16 --enrich 0 \
        --seed states/solve_ch_L22P16B22_r1.pkl --seed states/solve_ch_L22P16B22_r0.pkl --seed states/solve_ch_Q4P16B15_r0.pkl --outdir states --out _Q4L22P16B22
    FEM_ORDER_AB=4 FEM_NGAUSS_AB=16 python ground_state/solve_ch.py 10 10 0 3.0 3.0 22.0 8 --h0 0.25 --hmaxc 0.5 --ptol 1e-16 --enrich 0 \
        --nev 4 --seed states/solve_ch_Q4L22P16B22_r0.pkl --seed states/solve_ch_L22P16B22_r1.pkl --outdir states --out _Q4L22G16
    FEM_ORDER_AB=4 FEM_NGAUSS_AB=16 python ground_state/solve_ch.py 10 10 0 3.0 3.0 22.0 4 --h0 0.25 --hmaxc 0.25 --ptol 1e-16 --enrich 0 \
        --nev 4 --seed states/solve_ch_Q4L22G16_r0.pkl --outdir states --out _Q4L22G16V25

The first two runs take ten to twelve hours each on 80 cores, with up to 180 GB, the others between one and four hours;
the last one takes four hours and about 200 GB. `Q4L22P16B22` (six Gauss points) serves only as a seed. The other
states of the table differ from one of these commands as the table says: `Q4L22G16V35` has `--hmaxc 0.35`, `Q4N12L22G16`
has `12 12` and `--hmaxc 0.5`, `Q4L22G12` has `FEM_NGAUSS_AB=12`, `Q4L22G16E` has four local channels, one seed and
`--enrich 1`, `Q4L22G16T24` has four local channels and `--ptol 1e-24`, and the 12 x 12 states of the box 15 have `12 12`
and the seeds of the table. The two 12 x 12 seeds of `N12P16B15` are the rounds 0 and 1 of a run on that mesh at
tolerance `1e-12`, itself seeded by `LE10B15`, its rounds 8 and 9, and the rounds 4 and 5 of the same mesh solved without
seeds. `runs.json` lists, for every shipped state, the mesh, the channels per node, the element and Gauss orders, the
number of levels computed (`--nev`), the seeds, and how the eigensolver ended (iterations and residuals); the states
solved to the tolerance `1e-8` used `JL_TOL1=1e-8` and `JL_MAXIT=30`.

| file | size | SHA-256 |
|---|---|---|
| `solve_ch_Q4L22G16V25_r0.pkl`, the best state | 1.47 GB | `9782356f46262db0cf129722465e8d2636e1fc2c180a066fc5fe4dbf0a04d5f7` |
| `solve_ch_Q4L22G16_r0.pkl`, its seed | 785 MB | `9b719d1ebc359f10a64c615f5f2e911ab7369da65fd2cb4e13095e6cef9527fd` |

A state is a pickle with the lowest eigenvalues `ev`, the node fermion vectors `Phi` of the lowest state (single
precision), the element grid `grid`, the element orders `qs` and the kept node indices `keep`. A state computed with
`FEM_NGAUSS_AB` different from six must be post-processed with the same value. From a state,

    FEM_NGAUSS_AB=16 python ground_state/tail_fit.py solve_ch_Q4L22G16V25_r0.pkl --save --outdir data/profiles
    FEM_NGAUSS_AB=16 OBS_NQ=12 python ground_state/observables.py solve_ch_Q4L22G16V25_r0.pkl --outdir data/observables_lowtol
    FEM_NGAUSS_AB=16 python ground_state/virial.py solve_ch_Q4L22G16V25_r0.pkl --ops fspace_ops.pkl --outdir data/virial

write the files of this folder (`FEM_NGAUSS_AB=16` and `OBS_NQ=12` for the tags with `G16`, `FEM_NGAUSS_AB=12` for `G12`,
the defaults for the others).
