#!/usr/bin/env python3
"""Polynomial fits of erf / erfc for the native inference engine (phase 1c).

Our own fits (mpmath Chebyshev interpolation at 60 digits), no code taken from any
library: InferErf / InferErfc in src/infer/mormot.ai.infer.backend.cpu.pas evaluate
  [0, 1]        erf(x)  = x * P(x^2)
  [k, k + 1/2]  erfc(x) = exp(-x^2) * Q_k(x - k - 1/4), k = 1, 1.5, ... 5.5
and fall back to the continued fraction beyond 6 (erfc(6) < 3e-17). The script
prints the Pascal constants and the worst relative error of each piece, measured
in double precision (the arithmetic of the engine) on a dense grid.

  ~/.venvs/lai-infer-ref/bin/python tools/infer-ref/fit_erf.py > /tmp/erf.inc
"""
import struct

import mpmath as mp
import numpy as np

mp.mp.dps = 60
DEG_ERF = 14      # P degree in x^2 on [0, 1]
DEG_ERFC = 17     # Q_k degree on each half unit


def cheb_coeffs(f, a, b, deg):
    """monomial coefficients of the Chebyshev interpolant of f on [a, b], in the
    variable u = x - (a + b) / 2"""
    m = (a + b) / 2
    h = (b - a) / 2
    n = deg + 1
    nodes = [m + h * mp.cos(mp.pi * (2 * i + 1) / (2 * n)) for i in range(n)]
    # solve the Vandermonde system in u at the Chebyshev nodes (60 digits)
    A = mp.matrix(n, n)
    y = mp.matrix(n, 1)
    for i, x in enumerate(nodes):
        u = x - m
        for j in range(n):
            A[i, j] = u ** j
        y[i] = f(x)
    c = mp.lu_solve(A, y)
    return [c[j] for j in range(n)], m


def bits(v):
    """a double as two cardinals (low, high) with its value as comment: every
    compiler reads them exactly - Delphi 7 does not read every float literal so"""
    q = struct.unpack("<Q", struct.pack("<d", v))[0]
    return f"${q & 0xFFFFFFFF:08X}, ${q >> 32:08X} {{{v!r}}}"


def horner(c, u):
    r = 0.0
    for v in reversed(c):
        r = r * u + v
    return r


def main():
    out = []
    # erf(x) / x as a polynomial in t = x^2 on [0, 1]
    f = lambda t: mp.erf(mp.sqrt(t)) / mp.sqrt(t) if t > 0 else 2 / mp.sqrt(mp.pi)
    c, m = cheb_coeffs(f, mp.mpf(0), mp.mpf(1), DEG_ERF)
    cf = [float(v) for v in c]
    worst = 0
    for x in np.linspace(1e-9, 1.0, 20001):
        t = x * x
        mine = x * horner(cf, t - float(m))
        ref = mp.erf(mp.mpf(x))
        worst = max(worst, abs(mine - ref) / abs(ref))
    out.append(f"  // erf(x) = x * P(x^2 - {float(m)!r}) on [0, 1], worst relative error {float(worst):.2e}")
    out.append(f"  ERF_P: array[0..{2 * len(cf) - 1}] of cardinal = (")
    out.append(",\n".join(f"    {bits(v)}" for v in cf) + ");")
    # erfc(x) * exp(x^2) on half units from 1 to 6
    starts = [1 + 0.5 * k for k in range(10)]
    out.append(f"  // erfc(x) = exp(-x^2) * Q_k(x - mid_k) on [1 + k/2, 1.5 + k/2], k = 0..9")
    out.append(f"  ERFC_Q: array[0..9, 0..{2 * DEG_ERFC + 1}] of cardinal = (")
    rows = []
    worst_all = 0
    for s in starts:
        a, b = mp.mpf(s), mp.mpf(s) + mp.mpf("0.5")
        g = lambda x: mp.erfc(x) * mp.exp(x * x)
        c, m = cheb_coeffs(g, a, b, DEG_ERFC)
        cf = [float(v) for v in c]
        worst = 0
        for x in np.linspace(float(a), float(b), 4001):
            mine = np.exp(-x * x) * horner(cf, x - float(m))
            ref = mp.erfc(mp.mpf(x))
            worst = max(worst, abs(mine - ref) / ref)
        worst_all = max(worst_all, worst)
        rows.append(f"    // [{s}, {s + 0.5}], worst relative error {float(worst):.2e}\n"
                    "    (" + ",\n     ".join(f"{bits(v)}" for v in cf) + ")")
    out.append(",\n".join(rows) + ");")
    print("\n".join(out))
    print(f"  // worst relative error of all erfc pieces: {float(worst_all):.2e}")


main()
