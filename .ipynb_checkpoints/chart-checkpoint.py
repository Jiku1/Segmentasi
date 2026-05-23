import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Pastikan folder outputs tersedia
os.makedirs("./outputs", exist_ok=True)

# Muat data agregat
df = pd.read_csv("./result_kfold.csv")

# SETTING TEXTURE GRAFIK ACADEMIC LOOK (SERIF)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Urutan degradasi dari bersih (0) hingga gangguan untuk menghitung kemiringan (slope)
# Kita petakan sumbu X menjadi nilai numerik progresif (0 sampai 6) untuk regresi
deg_mapping = {"none": 0, "compression": 1, "gamma": 2, "shadow": 3, "blur": 4, "glare": 5, "noise": 6}

slope_results = []

for model_name in df["model"].unique():
    sub = df[df["model"] == model_name].copy()
    sub["x_val"] = sub["degradation"].map(deg_mapping)
    sub = sub.sort_values("x_val")
    
    # Hitung slope untuk Dice
    slope_dice, _, _, _, _ = linregress(sub["x_val"], sub["Dice_Mean"])
    # Hitung slope untuk Boundary F1
    slope_bound, _, _, _, _ = linregress(sub["x_val"], sub["Boundary_Mean"])
    
    slope_results.append({
        "model": model_name.upper(),
        "Slope_Dice": abs(slope_dice),       # Menggunakan nilai absolut agar grafik batang bernilai positif (laju penurunan)
        "Slope_Boundary": abs(slope_bound)
    })

df_slope = pd.DataFrame(slope_results)
print("\n--- NILAI LAJU PENURUNAN (SLOPE COEFFICIENT) ---")
print(df_slope)

# =====================================================
# PLOT 5: GRAFIK BATANG LAJU PENURUNAN (DEGRADATION SLOPE)
# =====================================================
x = np.arange(len(df_slope["model"]))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
# Laju penurunan Dice
rects1 = ax.bar(x - width/2, df_slope["Slope_Dice"], width, label='Laju Penurunan Dice (Wilayah)', color='#1f77b4')
# Laju penurunan Boundary F1
rects2 = ax.bar(x + width/2, df_slope["Slope_Boundary"], width, label='Laju Penurunan Boundary F1 (Tepi)', color='#ff7f0e')

ax.set_title('Perbandingan Laju Penurunan Performa (Degradation Slope)\nSemakin Tinggi Batang, Semakin Rentan Model Terhadap Gangguan', fontsize=11, fontweight='bold', pad=10)
ax.set_ylabel('Koefisien Kemiringan Linier (Absolut)', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(df_slope["model"])
ax.legend(frameon=True, facecolor='white', edgecolor='none')
ax.grid(True, linestyle='--', alpha=0.4, axis='y')

# Tambahkan label angka di atas batang grafis otomatis
ax.bar_label(rects1, padding=3, fmt='%.4f', fontsize=9)
ax.bar_label(rects2, padding=3, fmt='%.4f', fontsize=9)

plt.tight_layout()
plt.savefig("./outputs/degradation_slope_comparison.png", dpi=300)
plt.close()

print("\n[SUKSES] Grafik 'degradation_slope_comparison.png' berhasil disimpan!")
print("Nilai di atas bisa langsung Anda salin untuk mengisi Tabel 4.4 di naskah Bab IV Anda.")
