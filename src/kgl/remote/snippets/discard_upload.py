from pathlib import Path
Path(__PATH__).unlink(missing_ok=True)
_emit({"ok": True})
