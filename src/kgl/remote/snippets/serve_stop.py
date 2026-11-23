running = inspect_process()
if running:
    pid, _, saved = running
    if __INSTANCE__ and saved.get("instance") != __INSTANCE__:
        raise RuntimeError("refusing to stop a different server instance")
    os.killpg(pid, signal.SIGTERM)
    for attempt in range(60):
        members = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(fields[2]) == pid and fields[0] != "Z":
                members.append(proc.name)
        if not members:
            break
        if attempt == 50:
            os.killpg(pid, signal.SIGKILL)
        time.sleep(0.2)
    else:
        raise RuntimeError("server did not stop")
pidf.unlink(missing_ok=True)
metadata.unlink(missing_ok=True)
_emit({"stopped": True, "pid": running[0] if running else None})
