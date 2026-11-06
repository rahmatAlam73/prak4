# SOAL 3: (A) Regresi linier manual + pengecekan Python, prediksi x=10 dan x=14
#         (B) Spline linier, estimasi y pada x=8
# Wajib: numpy.  Opsional: scipy, scikit-learn (dilewati bila tidak terpasang)
import numpy as np

X = np.array([0, 4, 6, 9, 10, 12], dtype=float)
Y = np.array([-5, 15, 35, 65, 75, 95], dtype=float)
n = len(X)

# ---- (A) Cara manual (kuadrat terkecil) ----
sx, sy = X.sum(), Y.sum()
sxy, sxx = (X * Y).sum(), (X ** 2).sum()
b = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
a = Y.mean() - b * X.mean()
print("=== Regresi Linier MANUAL ===")
print("n=%d, Sx=%g, Sy=%g, Sxy=%g, Sxx=%g" % (n, sx, sy, sxy, sxx))
print("b (slope)     = %.6f" % b)
print("a (intercept) = %.6f" % a)
print("Persamaan: y = %.4f + %.4fx" % (a, b))
for x in (10, 14):
    print("  y(%d) = %.4f" % (x, a + b * x))

# ---- Pengecekan: numpy.polyfit ----
print("\n=== Pengecekan Python: numpy.polyfit ===")
slope, intercept = np.polyfit(X, Y, 1)
print("slope = %.6f, intercept = %.6f" % (slope, intercept))
for x in (10, 14):
    print("  y(%d) = %.4f" % (x, slope * x + intercept))

# R^2
yhat = a + b * X
r2 = 1 - ((Y - yhat) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
print("R^2 = %.6f" % r2)

# ---- Pengecekan: scipy (opsional) ----
try:
    from scipy import stats
    r = stats.linregress(X, Y)
    print("\n=== Pengecekan Python: scipy.stats.linregress ===")
    print("slope=%.6f, intercept=%.6f" % (r.slope, r.intercept))
except ImportError:
    print("\n(scipy tidak terpasang, dilewati)")

# ---- Pengecekan: scikit-learn (opsional) ----
try:
    from sklearn.linear_model import LinearRegression
    m = LinearRegression().fit(X.reshape(-1, 1), Y)
    print("\n=== Pengecekan Python: scikit-learn ===")
    print("slope=%.6f, intercept=%.6f" % (m.coef_[0], m.intercept_))
    print("  y(10), y(14) =", np.round(m.predict(np.array([[10.0], [14.0]])), 4))
except ImportError:
    print("(scikit-learn tidak terpasang, dilewati)")

# Padanan Excel: =SLOPE(Y,X)  =INTERCEPT(Y,X)  =FORECAST(x,Y,X)

# ---- (B) Spline linier x = 8 (di antara (6,35) dan (9,65)) ----
x_est = 8
print("\n(B) Spline linier: y(%d) = %.4f" % (x_est, np.interp(x_est, X, Y)))
print("    Manual: 35 + (8-6)*(65-35)/(9-6) = %.4f" % (35 + (8 - 6) * (65 - 35) / (9 - 6)))