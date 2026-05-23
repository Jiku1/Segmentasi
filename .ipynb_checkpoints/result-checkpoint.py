import os
import pandas as pd
import matplotlib.pyplot as plt

# Pastikan folder outputs tersedia
os.makedirs("./outputs", exist_ok=True)

# =====================================================
# LOAD RESULT CSV (Menggunakan file agregat hasil RunPod)
# =====================================================
# Menyesuaikan nama file dengan output skrip RunPod sebelumnya
df = pd.read_csv("./result_kfold.csv")

print("\n--- DATA AGREGAT EKSPERIMEN ---")
print(df)

# SETTING ESTETIKA GRAFIK STANDAR JURNAL AKADEMIK (SERIF/TIMES NEW ROMAN)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# =====================================================
# PLOT 1: DICE COEFFICIENT COMPARISON (Akurasi Wilayah)
# =====================================================
plt.figure(figsize=(8, 4.5))
for model_name in df["model"].unique():
    sub = df[df["model"] == model_name]
    plt.plot(sub["degradation"], sub["Dice_Mean"], marker='o', linewidth=2, label=model_name.upper())

plt.title("Perbandingan Dice Coefficient terhadap Berbagai Skenario Degradasi Visual", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Tipe Degradasi Visual", fontsize=10)
plt.ylabel("Rata-Rata Dice Coefficient", fontsize=10)
plt.ylim(0.5, 0.85)  # Rentang sumbu Y disesuaikan agar fluktuasi terlihat dramatis
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("./outputs/dice_comparison.png", dpi=300)
plt.close()

# =====================================================
# PLOT 2: DICE STANDARD DEVIATION (Stabilitas Wilayah)
# =====================================================
plt.figure(figsize=(8, 4.5))
for model_name in df["model"].unique():
    sub = df[df["model"] == model_name]
    plt.plot(sub["degradation"], sub["Dice_Std"], marker='s', linestyle='--', linewidth=1.5, label=model_name.upper())

plt.title("Standar Deviasi Dice Coefficient pada Berbagai Skenario Degradasi Visual", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Tipe Degradasi Visual", fontsize=10)
plt.ylabel("Standar Deviasi (Std Dev)", fontsize=10)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("./outputs/dice_std.png", dpi=300)
plt.close()


# =====================================================
# PLOT 3: BOUNDARY F1-SCORE COMPARISON (Ketepatan Batas Tepi)
# =====================================================
plt.figure(figsize=(8, 4.5))
for model_name in df["model"].unique():
    sub = df[df["model"] == model_name]
    plt.plot(sub["degradation"], sub["Boundary_Mean"], marker='^', linewidth=2, label=model_name.upper())

plt.title("Perbandingan Boundary F1-Score terhadap Berbagai Skenario Degradasi Visual", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Tipe Degradasi Visual", fontsize=10)
plt.ylabel("Rata-Rata Boundary F1-Score", fontsize=10)
plt.ylim(0.08, 0.15)  # Rentang sumbu Y disesuaikan dengan nilai 0.09 - 0.13 milik data 
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("./outputs/boundary_comparison.png", dpi=300)
plt.close()

# =====================================================
# PLOT 4: BOUNDARY STANDARD DEVIATION (Stabilitas Batas Tepi)
# =====================================================
plt.figure(figsize=(8, 4.5))
for model_name in df["model"].unique():
    sub = df[df["model"] == model_name]
    plt.plot(sub["degradation"], sub["Boundary_Std"], marker='d', linestyle='--', linewidth=1.5, label=model_name.upper())

plt.title("Standar Deviasi Boundary F1-Score pada Berbagai Skenario Degradasi Visual", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Tipe Degradasi Visual", fontsize=10)
plt.ylabel("Standar Deviasi (Std Dev)", fontsize=10)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("./outputs/boundary_std.png", dpi=300)
plt.close()

print("\n=================== VISUALISASI BERHASIL ===================")
print("[1] outputs/dice_comparison.png       -> Rata-rata Luasan (Dice)")
print("[2] outputs/dice_std.png              -> Stabilitas Luasan")
print("[3] outputs/boundary_comparison.png   -> Rata-rata Kontur Tepi (Boundary)")
print("[4] outputs/boundary_std.png          -> Stabilitas Kontur Tepi")
print("============================================================")
