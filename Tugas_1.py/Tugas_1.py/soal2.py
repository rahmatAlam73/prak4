# SOAL 2: Interpolasi linier (x=5) dan Lagrange (x=11)
import numpy as np

X = [3, 7, 10, 12, 15, 17]
Y = [31, 151, 304, 436, 679, 871]

# (A) Interpolasi linier pada x = 5 (titik apit: 3 dan 7)
def interpolasi_linier(x, x0, y0, x1, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

xa = 5
hasil_A = interpolasi_linier(xa, 3, 31, 7, 151)
print(f"(A) Interpolasi linier: y({xa}) = {hasil_A}")

# (B) Interpolasi Lagrange pada x = 11
def lagrange(x, xs, ys):
    total = 0.0
    for i in range(len(xs)):
        L = 1.0
        for j in range(len(xs)):
            if i != j:
                L *= (x - xs[j]) / (xs[i] - xs[j])
        total += ys[i] * L
    return total

xb = 11
print(f"(B) Lagrange, 6 titik            : y({xb}) = {lagrange(xb, X, Y):.4f}")
# Variasi: hanya 3 titik terdekat (10, 12, 15) dan 4 titik (7,10,12,15)
print(f"    Lagrange, 3 titik (10,12,15) : y({xb}) = {lagrange(xb, X[2:5], Y[2:5]):.4f}")
print(f"    Lagrange, 4 titik (7,10,12,15): y({xb}) = {lagrange(xb, X[1:5], Y[1:5]):.4f}")