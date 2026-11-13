"""Lowest levels of the box and the overlap estimate 1 - E_h/E_1.

Expanding the variational state in the eigenstates of the box, whose two lowest levels are E_0 and E_1, its squared overlap
with the ground state of the box is at least (E_1 - E_h)/(E_1 - E_0) >= 1 - E_h/E_1.  The script reads the eigenvalues stored
in a saved state and prints E_h, E_1 and 1 - E_h/E_1; for the best state, Q4L22G16V25_r0, they are compared with
the values quoted there, 0.0003, 0.1742 and about 0.998.  The computed E_1 is itself a variational value, which lies above the
exact level, so that the number is an estimate and not a bound.
Usage: python box_levels.py solve_ch_<tag>.pkl [...]   (seconds; the saved states are not shipped, see data/README.md)"""
import os, pickle, sys
import numpy as np

bad = 0
if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip()); sys.exit(0 if len(sys.argv) > 1 else 2)
for f in sys.argv[1:]:
    d = pickle.load(open(f, "rb")); ev = np.sort(np.asarray(d["ev"]).real); cmax = float(d["grid"][2][-1])
    print(f"{f}: box c <= {cmax:g}, E_h = {ev[0]:.5f}, E_1 = {ev[1]:.5f}, overlap estimate 1 - E_h/E_1 = {1 - ev[0] / ev[1]:.4f}")
    if os.path.basename(f) == "solve_ch_Q4L22G16V25_r0.pkl":
        for name, got, ref in (("E_h", f"{ev[0]:.4f}", "0.0003"), ("E_1", f"{ev[1]:.4f}", "0.1742"),
                               ("overlap estimate 1 - E_h/E_1", f"{1 - ev[0] / ev[1]:.3f}", "0.998")):
            ok = got == ref; bad += not ok
            print(f"{'PASS' if ok else 'FAIL'}  best state, {name}: computed {got}, expected {ref}")
sys.exit(1 if bad else 0)
