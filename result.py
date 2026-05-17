import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD RESULT CSV
# =========================
df = pd.read_csv("./outputs/thesis_results.csv")

print(df)

# =========================
# PLOT 1: IoU COMPARISON
# =========================
plt.figure()
for model in df["model"].unique():
    sub = df[df["model"] == model]
    plt.plot(sub["degradation"], sub["IoU_mean"], marker='o', label=model)

plt.title("IoU Comparison Across Degradation Types")
plt.xlabel("Degradation Type")
plt.ylabel("IoU Mean")
plt.legend()
plt.grid()
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("./outputs/iou_comparison.png", dpi=300)

# =========================
# PLOT 2: STD ERROR
# =========================
plt.figure()
for model in df["model"].unique():
    sub = df[df["model"] == model]
    plt.plot(sub["degradation"], sub["IoU_std"], marker='o', label=model)

plt.title("IoU Standard Deviation")
plt.xlabel("Degradation Type")
plt.ylabel("Std Dev")
plt.legend()
plt.grid()
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("./outputs/iou_std.png", dpi=300)

print(" Grafik berhasil disimpan di folder outputs/")