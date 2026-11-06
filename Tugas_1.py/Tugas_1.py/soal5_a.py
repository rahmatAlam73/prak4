# SOAL 5(a): Multiple Linear Regression pada data polynomial
# Padanan Python dari RapidMiner: Retrieve Polynomial -> Split Data -> Linear Regression
#                                  -> Apply Model -> Performance
# Library: numpy, pandas  (pip install numpy pandas)
#
# Data: simpan data Polynomial dari RapidMiner sebagai "polynomial.csv" di FOLDER yang sama
# dengan file ini. Kolom terakhir dianggap target (label), kolom lain fitur.
# Pemisah koma atau titik koma dikenali otomatis. Tanpa CSV, dipakai data tiruan.
import os
import numpy as np
import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(FOLDER, "polynomial.csv")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")      # buang kolom/baris non-angka
    df = df.dropna(axis=1, how="all").dropna()
    print("Data dibaca dari:", CSV_PATH)
else:
    print("(polynomial.csv tidak ditemukan -> memakai data TIRUAN, bukan data RapidMiner)")
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, size=(200, 3))
    y = 2 + 1.5 * x[:, 0] - 0.8 * x[:, 1] ** 2 + 0.3 * x[:, 2] ** 3 + rng.normal(0, 1, 200)
    df = pd.DataFrame(x, columns=["a1", "a2", "a3"])
    df["label"] = y

target = "label" if "label" in df.columns else df.columns[-1]
fitur = [c for c in df.columns if c != target]
print("Fitur :", fitur)
print("Target:", target, " | jumlah baris:", len(df))

# Split data 70% train, 30% test (acak, seed tetap)
rng = np.random.default_rng(1)
idx = rng.permutation(len(df))
ntrain = int(0.7 * len(df))
tr, te = idx[:ntrain], idx[ntrain:]
Xtr = df[fitur].values[tr]
ytr = df[target].values[tr]
Xte = df[fitur].values[te]
yte = df[target].values[te]

# Linear Regression (kuadrat terkecil): y = b0 + b1*x1 + ... + bk*xk
A = np.column_stack([np.ones(len(Xtr)), Xtr])
koef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
print("\nIntercept :", round(float(koef[0]), 4))
for f, c in zip(fitur, koef[1:]):
    print("Koefisien %-6s: %.4f" % (f, c))

# Apply Model + Performance
pred = np.column_stack([np.ones(len(Xte)), Xte]) @ koef
rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
print("\nRMSE (data uji): %.4f" % rmse)
print("R^2  (data uji): %.4f" % r2)
print("\n10 prediksi pertama:")
print(pd.DataFrame({"aktual": yte[:10], "prediksi": pred[:10]}).round(3).to_string())