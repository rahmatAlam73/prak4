# SOAL 4: Forecasting penjualan 2024 & 2025
#   (1) Linear Trend Projection   (2) Exponential Smoothing
# Library: numpy  (pip install numpy)
import numpy as np

jual = np.array([1024, 1075, 1128, 1184, 1243, 1305, 1370, 1438, 1509, 1584], dtype=float)
n = len(jual)
t = np.arange(1, n + 1)  # 2014 = 1, ..., 2023 = 10

# ---- (1) Linear Trend Projection: y = a + b t ----
b = (n * (t * jual).sum() - t.sum() * jual.sum()) / (n * (t ** 2).sum() - t.sum() ** 2)
a = jual.mean() - b * t.mean()
print("=== (1) Linear Trend Projection ===")
print("y = %.4f + %.4f t" % (a, b))
for th, tk in [(2024, 11), (2025, 12)]:
    print("  Forecast %d (t=%d) = %.2f" % (th, tk, a + b * tk))


# ---- (2) Simple Exponential Smoothing: F(t+1) = alpha*A(t) + (1-alpha)*F(t) ----
def ses(data, alpha):
    F = [data[0]]                                   # F1 = A1
    for i in range(1, len(data)):
        F.append(alpha * data[i - 1] + (1 - alpha) * F[i - 1])
    nxt = alpha * data[-1] + (1 - alpha) * F[-1]    # forecast periode berikutnya (2024)
    return np.array(F), nxt


print("\n=== (2) Exponential Smoothing ===")
print("%-6s %-12s %-14s" % ("alpha", "MSE", "Forecast 2024"))
hasil = []
for k in range(1, 10):
    alpha = k / 10.0
    F, nxt = ses(jual, alpha)
    mse = float(np.mean((jual[1:] - F[1:]) ** 2))
    hasil.append((alpha, mse, nxt))
    print("%-6.1f %-12.2f %-14.2f" % (alpha, mse, nxt))

alpha, mse, f2024 = min(hasil, key=lambda r: r[1])
f2025 = f2024    # tanpa data aktual 2024, SES memberi nilai sama
print("\nalpha terbaik (MSE terkecil) = %.1f" % alpha)
print("  Forecast 2024 = %.2f" % f2024)
print("  Forecast 2025 = %.2f" % f2025)
print("Catatan: SES menghasilkan nilai datar, sehingga tertinggal dari data yang trennya naik.")