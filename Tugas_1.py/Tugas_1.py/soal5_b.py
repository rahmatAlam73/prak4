# SOAL 5(b): ARIMA - Monthly Milk Production (168 bulan), forecast bulan 169-173
# Padanan Python dari RapidMiner Time Series extension.
# Library: numpy, pandas, matplotlib, statsmodels
#   pip install numpy pandas matplotlib statsmodels
#
# Data: simpan sebagai "monthly_milk.csv" di folder yang sama dengan file ini
# (kolom bulan boleh ada; kolom angka terakhir dipakai sebagai produksi).
# Pemisah koma/titik koma dikenali otomatis. Tanpa CSV, dipakai data tiruan.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    raise SystemExit("statsmodels belum terpasang. Jalankan: pip install statsmodels")

warnings.filterwarnings("ignore")

FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(FOLDER, "monthly_milk.csv")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    y = None
    for col in reversed(list(df.columns)):             # ambil kolom angka paling kanan
        angka = pd.to_numeric(df[col], errors="coerce")
        if angka.notna().sum() >= 0.9 * len(df):
            y = angka.dropna().astype(float).values
            print("Kolom produksi:", col)
            break
    if y is None:
        raise SystemExit("Tidak ada kolom angka di monthly_milk.csv")
else:
    print("(monthly_milk.csv tidak ditemukan -> memakai data TIRUAN, bukan data asli)")
    rng = np.random.default_rng(0)
    t = np.arange(168)
    y = 600 + 1.2 * t + 40 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 8, 168)

print("Jumlah data:", len(y), "(seharusnya 168)")

# Pilih orde ARIMA(p,d,q) dengan AIC terkecil
best = None
for p in range(0, 4):
    for d in range(0, 3):
        for q in range(0, 4):
            try:
                res = ARIMA(y, order=(p, d, q)).fit()
                if np.isfinite(res.aic) and (best is None or res.aic < best[0]):
                    best = (res.aic, (p, d, q), res)
            except Exception:
                continue

if best is None:
    raise SystemExit("Tidak ada model ARIMA yang berhasil dihitung.")

aic, order, res = best
print("Model terbaik ARIMA%s, AIC = %.2f" % (str(order), aic))

fc = res.get_forecast(steps=5)
mean = np.asarray(fc.predicted_mean)
ci = np.asarray(fc.conf_int())
mulai = len(y) + 1
hasil = pd.DataFrame({
    "bulan_ke": range(mulai, mulai + 5),
    "prediksi": mean,
    "batas_bawah": ci[:, 0],
    "batas_atas": ci[:, 1],
})
print(hasil.round(2).to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(y) + 1), y, label="Data aktual")
plt.plot(range(mulai, mulai + 5), mean, "r-o", label="Forecast")
plt.legend()
plt.xlabel("Bulan ke-")
plt.ylabel("Produksi susu")
plt.tight_layout()
out = os.path.join(FOLDER, "soal5b_forecast.png")
plt.savefig(out, dpi=150)
print("Grafik disimpan di:", out)
plt.show()