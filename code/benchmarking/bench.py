import subprocess
import csv
import time

Q_values = [4, 8, 16, 32]
B_values = [16, 32, 64]
inputs = [1000, 10000, 100000]

results = []

for Q in Q_values:
    for B in B_values:
        print(f"=== Building Q={Q}, B={B} ===")
        subprocess.run(["make", "clean"])
        subprocess.run(["make", f"Q={Q}", f"B={B}"])

        for n in inputs:
            # run CUDA version
            t0 = time.time()
            output = subprocess.check_output(["./radix", str(n), "1"]).decode()
            cuda_time = time.time() - t0

            # run CUB version
            cub_out = subprocess.check_output(["./cub_binary", str(n)]).decode()

            # run Futhark version
            futhark_out = subprocess.check_output(["./futhark_binary", str(n)]).decode()

            results.append([Q, B, n, cuda_time, cub_out.strip(), futhark_out.strip()])

            print(f"Q={Q}, B={B}, N={n}, CUDA={cuda_time:.5f}s")

# Save CSV
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Q", "B", "N", "CUDA(s)", "CUB", "Futhark"])
    writer.writerows(results)

print("✅ Benchmark complete. Results in results.csv")
