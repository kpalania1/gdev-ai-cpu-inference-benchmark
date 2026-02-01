import csv, os, platform, subprocess, time
from datetime import datetime
import numpy as np
import torch
import torchvision
from torchvision import models

# file name
CSV_FILE = "benchmark_results_final.csv"

# ---- Parameters ----
BATCH_SIZES  = [1, 2, 4, 8, 16]
WARMUP_ITERS = 20
TIMED_ITERS  = 100
SWEEPS_S     = 10    
REPEATS_R    = 3     

# single-threaded inter-op for deterministic results
torch.set_num_interop_threads(1) 
torch.set_grad_enabled(False)
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"

def sh(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception: return ""

def parse_lscpu() -> dict:
    info = {}
    txt = sh("lscpu")
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info

def parse_meminfo() -> dict:
    txt = sh("free -m")
    total_mb = ""
    for line in txt.splitlines():
        if line.lower().startswith("mem:"):
            parts = line.split()
            if len(parts) >= 2: total_mb = parts[1]
    return {"mem_total_mb": total_mb}

lscpu = parse_lscpu()
mem = parse_meminfo()
sysinfo = {
    "hostname": sh("hostname"),
    "os": sh("cat /etc/os-release | egrep '^(NAME|VERSION)=' | tr '\n' ' '"),
    "kernel": platform.release(),
    "arch": platform.machine(),
    "cpu_model": lscpu.get("Model name", ""),
    "cpu_sockets": lscpu.get("Socket(s)", ""),
    "cpu_cores_per_socket": lscpu.get("Core(s) per socket", ""),
    "cpu_threads_per_core": lscpu.get("Thread(s) per core", ""),
    "cpu_logical_cpus": lscpu.get("CPU(s)", ""),
    "mem_total_mb": mem.get("mem_total_mb", ""),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
	"torchvision_version": torchvision.__version__,
}

header = [
    "timestamp_utc", "device", "sweep_id", "repeat_id", "model", "batch_size",
    "threads", "iters", "warmup_iters", "median_ms", "mean_ms", "p99_latency_ms",
    "throughput_ips", "hostname", "os", "kernel", "arch", "cpu_model",
    "cpu_sockets", "cpu_cores_per_socket", "cpu_threads_per_core",
    "cpu_logical_cpus", "mem_total_mb", "python_version", "torch_version", "cuda_available", "torchvision_version"
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(header)

# --- Execution ---
device_name = "cpu"
device = torch.device(device_name)
model_builders = [
	("resnet50", lambda: models.resnet50(pretrained=True)),
	("resnet18", lambda: models.resnet18(pretrained=True)),
]
# Legacy server setting
#TMAX = int(sysinfo["cpu_logical_cpus"]) if sysinfo["cpu_logical_cpus"] else 4
#THREADS_LIST = list(range(1, TMAX + 1))

# Modern server setting
logical_cpus = int(sysinfo["cpu_logical_cpus"]) if sysinfo["cpu_logical_cpus"] else 1
THREADS_LIST = [t for t in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48] if t <= logical_cpus]

print(f"Logging to: {CSV_FILE} (Updated in real-time)")
print("-" * 125)

for model_name, builder in model_builders:
    model = builder().to(device)
    model.eval()

    with torch.inference_mode():
        for s in range(1, SWEEPS_S + 1):
            for t in THREADS_LIST:
                torch.set_num_threads(t)
                for b in BATCH_SIZES:
                    x = torch.randn(b, 3, 224, 224).to(device)
                    
                    for r in range(1, REPEATS_R + 1):
                        # Warmup Phase
                        for _ in range(WARMUP_ITERS): _ = model(x)
                        
                        # Measurement Phase
                        latencies = []
                        for _ in range(TIMED_ITERS):
                            start = time.perf_counter() 
                            _ = model(x)
                            latencies.append((time.perf_counter() - start) * 1000.0)

                        median_ms = float(np.median(latencies))
                        mean_ms   = float(np.mean(latencies))
                        p99_ms    = float(np.percentile(latencies, 99))
                        ips = b / (median_ms / 1000.0)

                        # WRITE TO CSV
                        with open(CSV_FILE, "a", newline="") as f:
                            csv.writer(f).writerow([
                                datetime.utcnow().isoformat(), device_name, s, r, model_name, b, t,
                                TIMED_ITERS, WARMUP_ITERS, 
                                round(median_ms, 3), round(mean_ms, 3), round(p99_ms, 3),
                                round(ips, 3),
                                sysinfo["hostname"], sysinfo["os"], sysinfo["kernel"], sysinfo["arch"],
                                sysinfo["cpu_model"], sysinfo["cpu_sockets"], sysinfo["cpu_cores_per_socket"],
                                sysinfo["cpu_threads_per_core"], sysinfo["cpu_logical_cpus"], sysinfo["mem_total_mb"],
                                sysinfo["python_version"], sysinfo["torch_version"], sysinfo["cuda_available"], sysinfo["torchvision_version"]
                            ])

                        print(f"{model_name:<10} | S:{s:02d} | R:{r} | T:{t:02d} | B:{b:<2} | Med: {median_ms:10.2f} ms | P99: {p99_ms:10.2f} ms | {ips:8.2f} ips")

print("-" * 140)
print("Benchmark Complete.")