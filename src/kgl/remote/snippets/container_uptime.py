import os
import time
from pathlib import Path


def container_uptime():
    try:
        clk = os.sysconf("SC_CLK_TCK")
        start_ticks = int(Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()[19])
        btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                     if line.startswith("btime "))
        return max(0, int(time.time() - (btime + start_ticks / clk)))
    except (OSError, ValueError, IndexError, StopIteration):
        return None
