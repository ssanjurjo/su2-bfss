"""Quick checks of the package.  Each script compares its numbers with the expected values and exits with status 1 on a mismatch;
the tail coefficient is recomputed from the shipped profiles and compared with data/tail_c2/final.txt, and the figure is
drawn if LaTeX is installed.  Usage (from the root of the package): python -m pytest -q tests   (under a minute)"""
import json, os, re, shutil, subprocess, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = [
    ("ground_state", ["moments.py"]),
]


def run(folder, args, env=None):
    p = subprocess.run([sys.executable, *args], cwd=os.path.join(ROOT, folder), capture_output=True, text=True, timeout=1800,
                       env=dict(os.environ, **(env or {})))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout


@pytest.mark.parametrize("folder,args", SCRIPTS, ids=[f"{f}/{a[0]}" for f, a in SCRIPTS])
def test_script(folder, args):
    out = run(folder, args)
    assert "FAIL" not in out, out[-3000:]


def numbers(text):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?", text)]


def test_tail_coefficient(tmp_path):
    table = str(tmp_path / "table.json")
    run("ground_state/tail_c2", ["c2_table.py", "--json", table])
    out = run("ground_state/tail_c2", ["final.py", table])
    expected = open(os.path.join(ROOT, "data", "tail_c2", "final.txt")).read()
    got, ref = numbers(out), numbers(expected)
    assert len(got) == len(ref) and np.allclose(got, ref, rtol=0, atol=2e-3), out
    new, old = json.load(open(table)), json.load(open(os.path.join(ROOT, "data", "tail_c2", "table.json")))
    assert [r["tag"] for r in new["rows"]] == [r["tag"] for r in old["rows"]]
    assert np.allclose([r["C2"] for r in new["rows"]], [r["C2"] for r in old["rows"]], rtol=0, atol=1e-6)


def test_size_of_the_state():
    """the numbers of the paragraph on the size of the state that follow from the density of the largest singular value"""
    z = np.load(os.path.join(ROOT, "data", "profiles", "solve_ch_Q4L22G16V25_r0_PR.npz")); R, P = z["R"], z["P"]
    lp = (16 * np.pi ** 2) ** (-1 / 6)
    i = int(np.argmax(P)); a = np.polyfit(R[i - 2:i + 3], P[i - 2:i + 3], 2); peak = -a[1] / (2 * a[0])
    assert abs(np.trapezoid(P, R) - 1) < 1e-4
    assert f"{peak / lp:.1f}" == "5.5"                                   # the density peaks at 5.5 l_p
    assert f"{100 * np.trapezoid(P[R >= 4], R[R >= 4]):.1f}" == "0.4"    # 0.4% of the probability beyond R = 4
    f = np.load(os.path.join(ROOT, "figures", "data", "profile_Q4L22G16V25_r0.npz"))
    assert np.array_equal(f["R"], R) and np.array_equal(f["P"], P)


def test_virial_terms():
    v = json.load(open(os.path.join(ROOT, "data", "virial", "virial_Q4L22G16V25_r0.json")))
    assert (f"{v['T']:.3f}", f"{v['V_B']:.3f}", f"{v['V_F']:.3f}") == ("27.166", "27.164", "-54.329")
    assert f"{100 * (v['T'] / v['V_B'] - 1):.2f}" == "0.01" and abs(v["sum"] - v["E_h"]) < 1e-7    # they agree to 0.01%


@pytest.mark.skipif(shutil.which("latex") is None, reason="the figure needs LaTeX")
def test_figure(tmp_path):
    out = run("figures", ["fig_density.py"], env={"BFSS_FIGURE_OUT": str(tmp_path)})
    assert os.path.getsize(tmp_path / "fig_density.pdf") > 10000, out
