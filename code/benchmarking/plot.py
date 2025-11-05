import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

for Q in df["Q"].unique():
    for B in df["B"].unique():
        subset = df[(df["Q"] == Q) & (df["B"] == B)]

        if subset.empty: 
            continue

        plt.figure(figsize=(6,4))
        plt.title(f"Runtime vs N (Q={Q}, B={B})")
        plt.plot(subset["N"], subset["Cub (µs)"], label="CUB", marker="o")
        plt.plot(subset["N"], subset["CUDA (µs)"], label="CUDA", marker="o")
        plt.plot(subset["N"], subset["Futhark (µs)"], label="Futhark", marker="o")
        
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("N (log scale)")
        plt.ylabel("Runtime (µs, log scale)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
