running = inspect_process()
args = __ARGS__
if running:
    pid, argv, saved = running
    if saved.get("protocol") != 2:
        raise RuntimeError("legacy server detected; stop it before upgrading")
    if args and argv != args + ["--instance", saved["instance"]]:
        raise RuntimeError("running server configuration conflicts with requested configuration")
    _emit(dict(saved, already_running=True))
else:
    instance = uuid.uuid4().hex
    cmd = [sys.executable, str(d / "serve.pyz")] + args + ["--instance", instance]
    with (d / "serve.log").open("ab") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=d)
    pending = d / f".serve-{instance}.json"
    try:
        fields = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()
        saved = dict(pid=p.pid, start_ticks=fields[19], instance=instance, args=args, protocol=2)
        pending.write_text(json.dumps(saved))
        pending.replace(metadata)
    except BaseException:
        try:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise
    finally:
        pending.unlink(missing_ok=True)
    _emit(dict(saved, started=True))
