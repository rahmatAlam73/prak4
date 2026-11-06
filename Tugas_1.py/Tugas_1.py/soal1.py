import os
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))

data = [10, 20, 22, 22, 22, 22, 22, 24, 26, 26, 28, 30, 30, 30, 30, 30,
        32, 34, 36, 38, 40, 42, 44, 46, 46, 48, 48, 50, 50, 60]
s = pd.Series(data, dtype=float)

# a. banyaknya data
n = len(s)
print("a. n =", n)

# b. deskripsi data (setara fungsi Excel)
print("\nb. Deskripsi data")
print("Maksimum             :", s.max())
print("Minimum              :", s.min())
print("Rata-rata            :", round(s.mean(), 4))
print("Varians (sampel)     :", round(s.var(ddof=1), 4), " [VAR.S]")
print("Varians (populasi)   :", round(s.var(ddof=0), 4), " [VAR.P]")
print("Std deviasi (sampel) :", round(s.std(ddof=1), 4), " [STDEV.S]")
print("Std deviasi (pop)    :", round(s.std(ddof=0), 4), " [STDEV.P]")
print("Median               :", s.median())
print("Modus                :", [float(m) for m in s.mode()], " (Excel MODE.SNGL hanya menampilkan yang pertama)")
print("Skewness             :", round(s.skew(), 4), " [SKEW]")
print("Kurtosis             :", round(s.kurt(), 4), " [KURT]")
for q in [0, 0.25, 0.5, 0.75, 1]:
    print("Kuantil %3d%%         : %s  [QUARTILE.INC]" % (int(q * 100), s.quantile(q)))

# c. grafik
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(range(1, n + 1), data, color="tab:blue")
ax[0].set_title("(A) Scatter Plot")
ax[0].set_xlabel("Urutan data")
ax[0].set_ylabel("Nilai")
ax[0].grid(True)

ax[1].boxplot(data)
ax[1].set_title("(B) Boxplot")
ax[1].set_ylabel("Nilai")
plt.tight_layout()

out = os.path.join(FOLDER, "soal1_grafik.png")
plt.savefig(out, dpi=150)
print("\nGrafik disimpan di:", out)
plt.show()