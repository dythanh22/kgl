import os, subprocess, shutil, time, platform
info = {"python": platform.python_version(), "gpus": []}
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if out.returncode == 0:
        for line in out.stdout.strip().splitlines():
            n, u, t, g = [x.strip() for x in line.split(",")]
            info["gpus"].append({"name": n, "mem_used_mb": int(u), "mem_total_mb": int(t), "util_pct": int(g)})
    else:
        info["gpu_error"] = out.stderr.strip()[:200]
except FileNotFoundError:
    info["gpu_error"] = "nvidia-smi not found"
except Exception as e:
    info["gpu_error"] = repr(e)[:200]
try:
    mem = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":", 1); mem[k] = int(v.split()[0])
    info["ram_total_mb"] = mem["MemTotal"] // 1024
    info["ram_avail_mb"] = mem.get("MemAvailable", 0) // 1024
except Exception:
    pass
try:
    wd = __WORKDIR__
    du = shutil.disk_usage(wd if os.path.isdir(wd) else "/")
    info["disk_free_gb"] = round(du.free / 1e9, 1)
except Exception:
    pass
uptime = container_uptime()
if uptime is not None:
    info["container_uptime_s"] = uptime
_emit(info)
