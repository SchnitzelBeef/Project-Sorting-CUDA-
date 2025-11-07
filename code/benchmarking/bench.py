import re
import random
import subprocess
import csv
import time, sys

Q_values = [4]
B_values = [64, 128, 256]
inputs = [1000, 10000, 1000000, 10000000]
BITS_values = [2, 4, 8]

# Q_values = [23]
# B_values = [256]
# inputs = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216]
# BITS_values = [4]

CUDA_RADIX_DIR = "../../code/cuda/radix/"
FUTHARK_DIR = "../futhark"
TXT_OUTFILE = "../cuda/radix/input.txt"
IN_OUTFILE = "../futhark/input.in"

### Create a .txt and .in based on N ####################
def generate_input_file(N):
    
    data = [random.getrandbits(32) for _ in range(N)]

    with open(TXT_OUTFILE, "w") as f:
        f.write("[")
        f.write(", ".join(f"{x}u32" for x in data))
        f.write("]\n")

    with open(IN_OUTFILE, "w") as f:
        f.write("[")
        f.write(", ".join(f"{x}u32" for x in data))
        f.write("]\n")

    print(f"Generated {N} u32 values in {TXT_OUTFILE} and {IN_OUTFILE}")

## Run radix + cub with given N, Q, B ##########
def run_radix(N, Q, B, NUM_BITS):
    cmd = ["make","clean"]
    subprocess.run(cmd,cwd=CUDA_RADIX_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cmd = ["make", f"N={N}", f"RADIX_Q={Q}", f"RADIX_B={B}", f"BITS={NUM_BITS}"]
    output = subprocess.run(cmd,
                            cwd=CUDA_RADIX_DIR,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True   
    )
    cub_t, cuda_t = parse_CUDA_output(output.stdout)
    return cub_t, cuda_t
    
### Extract times from CUDA output ##########n 
def parse_CUDA_output(output):
    cub_time = None
    cuda_time = None
    validated = False

    for line in output.splitlines():
        if "VALID RESULT!" in line:
            validated = True
        elif "CUB Radix Sort Time" in line:
            cub_time = int(line.split(":")[1].strip().split()[0])
        elif "CUDA Radix Sort Time" in line:
            cuda_time = int(line.split(":")[1].strip().split()[0])
        

    if not validated:
        return ("ERR", "ERR")

    return cub_time, cuda_time


### Run futhark benchmark ######################
def run_futhark():
    cmd = ["make", "clean"]
    subprocess.run(cmd,cwd=FUTHARK_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cmd = ["make", "bench"]
    output = subprocess.run(cmd,
                            cwd=FUTHARK_DIR,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True   
    )
    futhark_t = parse_futhark_time(output.stdout)
    return futhark_t


### Extract futhark time from output ##########
def parse_futhark_time(output) :
    # Find a number followed by μs
    match = re.search("([\d\.]+)\s*μs", output)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("Runtime not found in output")
    

total = len(inputs) * len(Q_values) * len(B_values) * len(BITS_values)
done = 0
start_time = time.time()

def print_status():
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0

    sys.stdout.write(
        f"\rProgress: {done}/{total} "
        f"({done/total*100:5.1f}%) "
        f"| Elapsed: {elapsed/60:6.1f}m "
        f"| ETA: {remaining/60:6.1f}m"
    )
    sys.stdout.flush()


results = []

for N in inputs:
    generate_input_file(N)  # only once per N
    futhark_t = run_futhark()
    for Q in Q_values:
        for B in B_values:
            for NUM_BITS in BITS_values:
                print(f"\n=== N={N}, Q={Q}, B={B}, NUM_BITS={NUM_BITS} ===")

                if (B < (1 << NUM_BITS)):
                    print(f"➡️  SKIPPING because {B} < {1<<NUM_BITS}")
                    done += 1
                else:
                    cub_t, cuda_t = run_radix(N, Q, B, NUM_BITS)  

                    # invalid runs
                    if cub_t == "ERR" or cuda_t == "ERR":
                        print("❌ CUDA SORT DID NOT VALIDATE")
                        continue

                    print(f"✅ CUB={cub_t}µs  CUDA={cuda_t}µs  Futhark={futhark_t}µs")

                    results.append([N, Q, B, cub_t, cuda_t, futhark_t])
                    done += 1
                    print_status()


# Write results
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "Q", "B", "Cub (µs)", "CUDA (µs)", "Futhark (µs)"])
    writer.writerows(results)

print("\n🛺 Benchmark complete — results saved to results.csv")