"""Linux ttk launcher for the Qwen checkout example (Graphite theme, VI/EN)."""
from queue import Empty, Queue
from threading import Event, Thread
import time
import tkinter as tk
import tkinter.font as tkf
from tkinter import messagebox, simpledialog, ttk
import webbrowser

from kgl.desktop.launcher import LaunchError, parse_notebook
from launch import desktop_entry_path, install_desktop_entry, make_launcher
from proxy import URL

PALETTE = {
    "bg": "#1d1d1f",
    "card": "#242427",
    "raise": "#2e2e33",
    "hover": "#38383f",
    "border": "#36363c",
    "text": "#ececee",
    "muted": "#a3a3aa",
    "faint": "#6f6f77",
    "accent": "#b9c7ff",
    "accent_hover": "#ced9ff",
    "ink": "#14172a",
    "ok": "#8fd6a2",
    "warn": "#f2c27a",
    "danger": "#ff8f80",
    "danger_bg": "#3a2322",
    "danger_hover": "#4d2b29",
}

XFT_FONTS = ("noto sans", "dejavu sans", "liberation sans", "inter", "ubuntu", "cantarell", "segoe ui")
STEP_KEYS = ("create", "connect", "input", "build", "worker", "proxy")
STEP_INDEX = {key: idx for idx, key in enumerate(STEP_KEYS)}
STEP_INDEX["health"] = STEP_INDEX["worker"]

I18N = {
    "vi": {
        "lang_btn": "EN | English",
        "subtitle_xft": "Qwen3.8-35B Q4_K_M  ·  2 × Tesla T4  ·  32K context",
        "subtitle_x11": "Qwen3.8-35B Q4_K_M  -  2 x Tesla T4  -  32K context",
        "notebook_label": "Notebook",
        "endpoint_info": f"Chat: {URL}/   |   OpenAI API: {URL}/v1   (qwen-local)",
        "copy_api": "Sao chép API",
        "copied_api_xft": "Đã chép ✓",
        "copied_api_x11": "Đã chép [OK]",
        "pin_menu": "Ghim vào Menu Linux",
        "pinned_menu_xft": "Đã ghim Menu ✓",
        "pinned_menu_x11": "Đã ghim Menu [OK]",
        "pin_error": "Không ghi được biểu tượng ứng dụng: {err}",
        "footer": "Notebook tự động lưu sau lần chạy đầu. Phiên GPU đang hoạt động sẽ tính hạn mức tuần trên Kaggle.",
        "quota": "GPU tuần: còn {h}h {m:02d}p / {total}h",
        "checking": "Đang kiểm tra...",
        "elapsed_fmt": "{label} - tổng {s}s",
        "cancelling": "Đang hủy, chờ thao tác hiện tại dừng an toàn...",
        "invalid_slug": "Nhập link notebook Kaggle hoặc owner/slug (ví dụ: owner/test-kgl).",
        "generic_error": "Kiểm tra thư mục phiên và kết nối.",
        "manual_step": "1. Gắn URL thủ công",
        "steps": {
            "create": "1. Cấp máy GPU",
            "connect": "2. Kết nối Jupyter",
            "input": "3. Kiểm 2xT4 & GGUF",
            "build": "4. Build llama.cpp",
            "worker": "5. Nạp Qwen lên GPU",
            "proxy": "6. Mở Proxy 8001",
        },
        "buttons": {
            "start": "Chạy Qwen",
            "restart": "Chạy lại Qwen",
            "chat": "Mở Chat",
            "attach": "Gắn URL thủ công...",
            "refresh": "Làm mới",
            "stop-worker": "Dừng worker",
            "end-session": "Kết thúc GPU",
            "detach": "Ngắt kết nối",
            "cancel": "Hủy khởi chạy",
        },
        "dialogs": {
            "attach_title": "Gắn phiên",
            "attach_prompt": "URL Jupyter từ Kaggle",
            "end_title": "Kết thúc phiên GPU",
            "end_busy": "Worker đang xử lý yêu cầu. ",
            "end_prompt": "{warn}Kết thúc phiên GPU của {nb} trên Kaggle và dừng proxy local?",
            "detach_title": "Ngắt kết nối",
            "detach_busy": "Worker đang xử lý yêu cầu. Vẫn ngắt kết nối local và dừng proxy của launcher?",
            "detach_owned": ("Worker Qwen do launcher tạo vẫn chạy trên VM (chọn Dừng worker nếu muốn tắt trước). "
                             "Tiếp tục ngắt kết nối local?"),
            "detach_plain": "Ngắt kết nối local? GPU và worker còn chạy trên Kaggle.",
            "wait_title": "Đang xử lý",
            "wait_body": "Chờ thao tác hiện tại hoàn tất trước khi đóng cửa sổ.",
            "cancel_title": "Đang chạy",
            "cancel_body": "Hủy khởi chạy và chờ kết quả?",
        },
        "labels": {
            "idle": "Chưa có phiên chạy",
            "create": "Đang cấp máy GPU trên Kaggle",
            "connect": "Kết nối Jupyter",
            "attached": "Đã gắn phiên",
            "input": "Kiểm GGUF và hai T4",
            "build": "Xác minh hoặc khôi phục llama.cpp",
            "worker": "Kiểm checksum và nạp Qwen",
            "health": "Kiểm tra worker",
            "proxy": "Mở proxy local",
            "ready": "Sẵn sàng phục vụ",
            "busy": "Worker đang xử lý yêu cầu",
            "worker-stopped": "Worker đã dừng (GPU còn chạy)",
            "unknown": "Cần đối soát trạng thái",
            "detach": "Ngắt kết nối local",
            "stop-worker": "Đang dừng worker",
            "end-session": "Đang kết thúc phiên GPU",
        },
        "errors": {
            "missing_input": "Thiếu GGUF trong notebook",
            "wrong_gpu": "Cần bật GPU T4 x 2 trên notebook",
            "gpu_unavailable": "Không đọc được GPU trên notebook",
            "input_probe": "Không kiểm tra được input",
            "build_verify": "Bản build llama.cpp không khớp hoặc lỗi CUDA",
            "build_access": "Không khôi phục được dataset build",
            "checksum": "Checksum GGUF khác preset",
            "cancelled": "Đã hủy khởi chạy",
            "connection_lost": "Mất kết nối tới notebook",
            "session_ended": "Phiên GPU trên Kaggle đã kết thúc",
            "proxy_down": "Proxy local chưa chạy",
            "proxy_unreachable": "Không gọi được proxy local",
            "port_conflict": "Cổng 8001 đang thuộc tiến trình khác",
            "proxy_cleanup_failed": "Đã tắt GPU nhưng chưa dừng được proxy 8001",
            "worker_mismatch": "Worker không khớp bản ghi",
            "worker_unready": "Worker chưa sẵn sàng",
            "failed_connect": "Không kết nối được URL Jupyter",
            "auth_failed": "Lỗi xác thực Kaggle (~/.kaggle/kaggle.json)",
            "notebook_not_found": "Không tìm thấy notebook hoặc thiếu quyền",
            "quota_exceeded": "Hết hạn mức GPU hoặc chạm giới hạn phiên Kaggle",
            "sdk_missing": "Chưa cài kagglesdk==0.1.38 cho chế độ tự động",
            "sdk_version": "Phiên bản kagglesdk không khớp bản ghim 0.1.38",
            "metadata_failed": "Không đọc được cấu hình notebook từ Kaggle",
            "create_failed": "Kaggle không khởi tạo được phiên GPU",
            "create_rejected": "Kaggle từ chối yêu cầu tạo phiên",
            "create_uncertain": "Mất phản hồi khi xin GPU (kiểm tra trên Kaggle)",
            "create_timeout": "Quá thời gian chờ Kaggle cấp máy GPU",
            "poll_failed": "Mất liên lạc khi chờ Kaggle cấp máy",
            "missing_url": "Kaggle không trả về URL kết nối Jupyter",
            "cancel_unconfirmed": "Chưa xác nhận được GPU đã tắt (bấm Kết thúc GPU lại)",
            "interrupted_create": "Gián đoạn lúc đang xin GPU (kiểm tra trên Kaggle)",
        },
    },
    "en": {
        "lang_btn": "VI | Tiếng Việt",
        "subtitle_xft": "Qwen3.8-35B Q4_K_M  ·  2 × Tesla T4  ·  32K context",
        "subtitle_x11": "Qwen3.8-35B Q4_K_M  -  2 x Tesla T4  -  32K context",
        "notebook_label": "Notebook",
        "endpoint_info": f"Chat: {URL}/   |   OpenAI API: {URL}/v1   (qwen-local)",
        "copy_api": "Copy API",
        "copied_api_xft": "Copied ✓",
        "copied_api_x11": "Copied [OK]",
        "pin_menu": "Pin to Linux Menu",
        "pinned_menu_xft": "Pinned to Menu ✓",
        "pinned_menu_x11": "Pinned to Menu [OK]",
        "pin_error": "Could not write desktop entry: {err}",
        "footer": "Notebook is saved automatically after first launch. Active GPU sessions consume weekly Kaggle quota.",
        "quota": "Weekly GPU: {h}h {m:02d}m / {total}h left",
        "checking": "Checking...",
        "elapsed_fmt": "{label} - total {s}s",
        "cancelling": "Cancelling, waiting for the active step to stop safely...",
        "invalid_slug": "Enter a Kaggle notebook URL or owner/slug (e.g. owner/test-kgl).",
        "generic_error": "Check the session directory and connection.",
        "manual_step": "1. Manual URL attach",
        "steps": {
            "create": "1. Allocate GPU",
            "connect": "2. Connect Jupyter",
            "input": "3. Check 2xT4 & GGUF",
            "build": "4. Verify llama.cpp",
            "worker": "5. Load Qwen on GPU",
            "proxy": "6. Start Proxy 8001",
        },
        "buttons": {
            "start": "Start Qwen",
            "restart": "Restart Qwen",
            "chat": "Open Chat",
            "attach": "Attach URL...",
            "refresh": "Refresh",
            "stop-worker": "Stop Worker",
            "end-session": "End GPU Session",
            "detach": "Detach",
            "cancel": "Cancel Launch",
        },
        "dialogs": {
            "attach_title": "Attach Session",
            "attach_prompt": "Jupyter Server URL from Kaggle",
            "end_title": "End GPU Session",
            "end_busy": "Worker is currently serving a request. ",
            "end_prompt": "{warn}End Kaggle GPU session for {nb} and stop the local proxy?",
            "detach_title": "Detach Session",
            "detach_busy": "Worker is currently busy. Detach locally and stop the launcher proxy anyway?",
            "detach_owned": ("The launcher-started Qwen worker is still running on the VM (click Stop Worker first if needed). "
                             "Continue local detach?"),
            "detach_plain": "Detach local connection? The GPU session and worker keep running on Kaggle.",
            "wait_title": "Operation in Progress",
            "wait_body": "Please wait for the current operation to finish before closing.",
            "cancel_title": "Launch Running",
            "cancel_body": "Cancel launch and wait for safe cleanup?",
        },
        "labels": {
            "idle": "No active session",
            "create": "Allocating GPU VM on Kaggle",
            "connect": "Connecting to Jupyter",
            "attached": "Session attached",
            "input": "Checking GGUF and 2xT4",
            "build": "Verifying or restoring llama.cpp",
            "worker": "Verifying checksum and loading Qwen",
            "health": "Checking worker readiness",
            "proxy": "Starting local proxy",
            "ready": "Ready to serve",
            "busy": "Worker busy serving request",
            "worker-stopped": "Worker stopped (GPU still running)",
            "unknown": "Status reconciliation needed",
            "detach": "Detaching local connection",
            "stop-worker": "Stopping worker",
            "end-session": "Ending Kaggle GPU session",
        },
        "errors": {
            "missing_input": "Missing GGUF input in notebook",
            "wrong_gpu": "Enable GPU T4 x 2 on the notebook",
            "gpu_unavailable": "GPU unavailable inside notebook",
            "input_probe": "Could not verify notebook inputs",
            "build_verify": "Saved llama.cpp build or CUDA verification failed",
            "build_access": "Could not restore llama.cpp build dataset",
            "checksum": "GGUF checksum differs from preset",
            "cancelled": "Launch cancelled",
            "connection_lost": "Lost connection to notebook",
            "session_ended": "Kaggle GPU session has ended",
            "proxy_down": "Local proxy is not running",
            "proxy_unreachable": "Cannot reach local proxy",
            "port_conflict": "Port 8001 is occupied by another process",
            "proxy_cleanup_failed": "GPU stopped, but port 8001 proxy could not be closed",
            "worker_mismatch": "Worker instance does not match record",
            "worker_unready": "Worker is not ready",
            "failed_connect": "Could not connect with that Jupyter URL",
            "auth_failed": "Kaggle authentication failed (~/.kaggle/kaggle.json)",
            "notebook_not_found": "Notebook not found or inaccessible",
            "quota_exceeded": "Weekly GPU quota or active session limit reached",
            "sdk_missing": "kagglesdk==0.1.38 is required for automatic mode",
            "sdk_version": "Installed kagglesdk does not match pinned 0.1.38",
            "metadata_failed": "Could not read notebook metadata from Kaggle",
            "create_failed": "Kaggle failed to start the GPU session",
            "create_rejected": "Kaggle rejected the session creation request",
            "create_uncertain": "Lost response while requesting GPU (check Kaggle)",
            "create_timeout": "Timed out waiting for Kaggle GPU allocation",
            "poll_failed": "Lost contact while polling Kaggle VM",
            "missing_url": "Kaggle did not return a Jupyter URL",
            "cancel_unconfirmed": "GPU shutdown not yet confirmed (click End GPU Session again)",
            "interrupted_create": "Interrupted during GPU request (check Kaggle)",
        },
    },
}


def resolve_ui_font(root: tk.Misc, lang: str = "vi") -> tuple[str, bool]:
    """Choose a Vietnamese-capable font and detect whether Xft TrueType fonts are available."""
    available = {name.lower(): name for name in tkf.families(root)}
    for candidate in XFT_FONTS:
        if candidate in available:
            return available[candidate], True
    if lang == "vi" and "clearlyu" in available:
        return available["clearlyu"], False
    for candidate in ("helvetica", "nimbus sans l", "clearlyu"):
        if candidate in available:
            return available[candidate], False
    return tkf.nametofont("TkDefaultFont").actual("family"), False


def apply_graphite_theme(root: tk.Misc, lang: str = "vi") -> tuple[str, bool]:
    family, has_xft = resolve_ui_font(root, lang)
    base_size = 10
    small_size = 9 if has_xft else 10
    title_size = 12
    font_base = (family, base_size)
    font_bold = (family, base_size, "bold")
    font_title = (family, title_size, "bold")
    font_small = (family, small_size)

    root.configure(bg=PALETTE["bg"])
    root.option_add("*TCombobox*Listbox.background", PALETTE["card"])
    root.option_add("*TCombobox*Listbox.foreground", PALETTE["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", PALETTE["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", PALETTE["ink"])
    root.option_add("*TCombobox*Listbox.font", font_base)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    style.configure(".", background=PALETTE["bg"], foreground=PALETTE["text"], font=font_base)
    style.configure("App.TFrame", background=PALETTE["bg"])
    style.configure("Card.TFrame", background=PALETTE["card"], borderwidth=1, relief="solid",
                    bordercolor=PALETTE["border"])
    style.configure("Title.TLabel", background=PALETTE["bg"], foreground=PALETTE["text"], font=font_title)
    style.configure("Sub.TLabel", background=PALETTE["bg"], foreground=PALETTE["muted"], font=font_small)
    style.configure("Card.TLabel", background=PALETTE["card"], foreground=PALETTE["text"], font=font_base)
    style.configure("CardBold.TLabel", background=PALETTE["card"], foreground=PALETTE["text"], font=font_bold)
    style.configure("CardMuted.TLabel", background=PALETTE["card"], foreground=PALETTE["muted"], font=font_small)
    style.configure("Footer.TLabel", background=PALETTE["bg"], foreground=PALETTE["faint"], font=font_small)

    style.configure("Graphite.TCombobox", fieldbackground=PALETTE["raise"], background=PALETTE["raise"],
                    foreground=PALETTE["text"], arrowcolor=PALETTE["muted"], bordercolor=PALETTE["border"],
                    lightcolor=PALETTE["border"], darkcolor=PALETTE["border"], insertcolor=PALETTE["text"],
                    padding=6, font=font_base)
    style.map("Graphite.TCombobox",
              fieldbackground=[("disabled", PALETTE["card"]), ("readonly", PALETTE["raise"])],
              foreground=[("disabled", PALETTE["muted"])])

    for name, bg, fg, hover_bg in [
        ("Secondary.TButton", PALETTE["raise"], PALETTE["text"], PALETTE["hover"]),
        ("Accent.TButton", PALETTE["accent"], PALETTE["ink"], PALETTE["accent_hover"]),
        ("Danger.TButton", PALETTE["danger_bg"], PALETTE["danger"], PALETTE["danger_hover"]),
    ]:
        style.configure(name, background=bg, foreground=fg, bordercolor=PALETTE["border"],
                        focusthickness=1, focuscolor=PALETTE["accent"], padding=(12, 7),
                        font=font_bold if name != "Secondary.TButton" else font_base)
        style.map(name,
                  background=[("disabled", PALETTE["card"]), ("active", hover_bg)],
                  foreground=[("disabled", PALETTE["faint"])])

    style.configure("Small.TButton", background=PALETTE["raise"], foreground=PALETTE["muted"],
                    bordercolor=PALETTE["border"], padding=(8, 3), font=font_small)
    style.map("Small.TButton", background=[("active", PALETTE["hover"])], foreground=[("active", PALETTE["text"])])

    style.configure("Graphite.Horizontal.TProgressbar", troughcolor=PALETTE["raise"],
                    background=PALETTE["accent"], bordercolor=PALETTE["border"],
                    lightcolor=PALETTE["accent"], darkcolor=PALETTE["accent"], thickness=6)
    return family, has_xft


def _format_quota(quota: dict, lang: str = "vi") -> tuple[str, float, str]:
    rem = int(quota.get("remaining_s", 0))
    total = max(1.0, float(quota.get("total_s", 1)))
    ratio = max(0.0, min(1.0, rem / total))
    hours, mins = rem // 3600, (rem % 3600) // 60
    total_h = int(round(total / 3600))
    color = PALETTE["danger"] if ratio < 0.1 else (PALETTE["warn"] if ratio < 0.25 else PALETTE["ok"])
    text = I18N[lang]["quota"].format(h=hours, m=mins, total=total_h)
    return text, ratio, color


class Window:
    def __init__(self, root, launcher):
        self.root, self.launcher = root, launcher
        self.events = Queue()
        self.cancel = Event()
        self.running = False
        self.cancellable = False
        self.checked_ready = False
        self.launch_mode = ""
        self.step_started = 0.0
        self.done_steps = set()
        self.failed_step = ""
        self.last_quota = None
        self.custom_notice = ""
        self.lang = launcher.saved_language() if hasattr(launcher, "saved_language") else "vi"
        self.record = launcher.status()

        root.title("KGL Desktop")
        root.minsize(580, 260)
        root.resizable(True, False)
        root.columnconfigure(0, weight=1)
        self.font_family, self.has_xft = apply_graphite_theme(root, self.lang)

        outer = ttk.Frame(root, style="App.TFrame", padding=(18, 16))
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        # 1. Header + Language Switcher + Weekly GPU Quota Badge
        header = ttk.Frame(outer, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        titles = ttk.Frame(header, style="App.TFrame")
        titles.grid(row=0, column=0, sticky="w")
        ttk.Label(titles, text="KGL Desktop", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.subtitle_var = tk.StringVar()
        ttk.Label(titles, textvariable=self.subtitle_var, style="Sub.TLabel").grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )

        right_head = ttk.Frame(header, style="App.TFrame")
        right_head.grid(row=0, column=1, sticky="ne")
        self.lang_var = tk.StringVar()
        self.lang_btn = ttk.Button(right_head, textvariable=self.lang_var, style="Small.TButton",
                                   command=self.toggle_language)
        self.lang_btn.grid(row=0, column=1, sticky="ne", padx=(8, 0))

        self.quota_box = ttk.Frame(right_head, style="Card.TFrame", padding=(10, 6))
        self.quota_var = tk.StringVar(value="")
        ttk.Label(self.quota_box, textvariable=self.quota_var, style="CardMuted.TLabel").grid(row=0, column=0, sticky="e")
        self.quota_bar = tk.Canvas(self.quota_box, width=148, height=4, bg=PALETTE["raise"], highlightthickness=0)
        self.quota_bar.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        # 2. Session & Status Card
        card = ttk.Frame(outer, style="Card.TFrame", padding=(14, 12))
        card.grid(row=1, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)

        status_row = ttk.Frame(card, style="Card.TFrame")
        status_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        status_row.columnconfigure(1, weight=1)
        self.lamp = tk.Canvas(status_row, width=12, height=12, bg=PALETTE["card"], highlightthickness=0)
        self.lamp.grid(row=0, column=0, padx=(0, 8))
        self.lamp_dot = self.lamp.create_oval(2, 2, 10, 10, fill=PALETTE["faint"], outline="")
        self.message = tk.StringVar()
        ttk.Label(status_row, textvariable=self.message, style="CardBold.TLabel",
                  wraplength=510).grid(row=0, column=1, sticky="w")

        self.nb_label_var = tk.StringVar()
        ttk.Label(card, textvariable=self.nb_label_var, style="CardMuted.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 12)
        )
        self.slug = tk.StringVar(value=self.record.notebook)
        self.entry = ttk.Combobox(card, textvariable=self.slug, values=self.launcher.recent_notebooks(),
                                  style="Graphite.TCombobox", width=40)
        self.entry.grid(row=1, column=1, sticky="ew")

        self.endpoint_row = ttk.Frame(card, style="Card.TFrame")
        self.endpoint_row.columnconfigure(0, weight=1)
        self.endpoint_var = tk.StringVar()
        ttk.Label(self.endpoint_row, textvariable=self.endpoint_var, style="CardMuted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.copy_text = tk.StringVar()
        self.copy_btn = ttk.Button(self.endpoint_row, textvariable=self.copy_text,
                                   style="Small.TButton", command=self.copy_api)
        self.copy_btn.grid(row=0, column=1, sticky="e", padx=(8, 0))

        # 3. 6-Step Progress Card (Contextual)
        self.stepper = ttk.Frame(outer, style="Card.TFrame", padding=(14, 10))
        self.stepper.columnconfigure(0, weight=1)
        self.stepper.columnconfigure(1, weight=1)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(self.stepper, variable=self.progress_var, maximum=len(STEP_KEYS),
                                        mode="determinate", style="Graphite.Horizontal.TProgressbar")
        self.progress.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.step_labels = {}
        step_font_size = 9 if self.has_xft else 10
        for idx, key in enumerate(STEP_KEYS):
            lbl = tk.Label(self.stepper, text="", bg=PALETTE["card"], fg=PALETTE["faint"],
                           font=(self.font_family, step_font_size), anchor="w")
            lbl.grid(row=1 + idx // 2, column=idx % 2, sticky="ew", padx=(0, 12), pady=2)
            self.step_labels[key] = lbl

        # 4. Contextual Action Bar
        self.action_bar = ttk.Frame(outer, style="App.TFrame")
        self.action_bar.grid(row=3, column=0, sticky="ew", pady=(14, 6))
        self.action_bar.columnconfigure(4, weight=1)
        self.buttons = {}
        for action, btn_style in [
            ("start", "Accent.TButton"),
            ("chat", "Accent.TButton"),
            ("attach", "Secondary.TButton"),
            ("refresh", "Secondary.TButton"),
            ("stop-worker", "Secondary.TButton"),
            ("end-session", "Danger.TButton"),
            ("detach", "Secondary.TButton"),
            ("cancel", "Danger.TButton"),
        ]:
            self.buttons[action] = ttk.Button(self.action_bar, style=btn_style,
                                              command=lambda a=action: self.click(a))

        # 5. Footer
        footer = ttk.Frame(outer, style="App.TFrame")
        footer.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        footer.columnconfigure(0, weight=1)
        self.footer_var = tk.StringVar()
        ttk.Label(footer, textvariable=self.footer_var, style="Footer.TLabel", wraplength=420).grid(
            row=0, column=0, sticky="w"
        )
        self.shortcut_text = tk.StringVar()
        self.shortcut_btn = ttk.Button(footer, textvariable=self.shortcut_text,
                                       style="Small.TButton", command=self.install_shortcut)
        self.shortcut_btn.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self._apply_static_texts()
        self.show_record()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(80, self.drain)
        self.refresh_quota()
        if self.record.phase in ("ready", "busy"):
            self.click("refresh")

    def _tr(self) -> dict:
        return I18N[self.lang]

    def _apply_static_texts(self):
        tr = self._tr()
        self.lang_var.set(tr["lang_btn"])
        self.subtitle_var.set(tr["subtitle_xft"] if self.has_xft else tr["subtitle_x11"])
        self.nb_label_var.set(tr["notebook_label"])
        self.endpoint_var.set(tr["endpoint_info"])
        self.copy_text.set(tr["copy_api"])
        self.footer_var.set(tr["footer"])
        pinned_key = "pinned_menu_xft" if self.has_xft else "pinned_menu_x11"
        self.shortcut_text.set(tr[pinned_key] if desktop_entry_path().is_file() else tr["pin_menu"])
        step_font_size = 9 if self.has_xft else 10
        for lbl in self.step_labels.values():
            lbl.configure(font=(self.font_family, step_font_size))
        if self.last_quota:
            self._render_quota(self.last_quota)

    def toggle_language(self):
        self.lang = "en" if self.lang == "vi" else "vi"
        if hasattr(self.launcher, "remember_language"):
            self.launcher.remember_language(self.lang)
        self.font_family, self.has_xft = apply_graphite_theme(self.root, self.lang)
        self.custom_notice = ""
        self._apply_static_texts()
        self.show_record()

    def copy_api(self):
        tr = self._tr()
        self.root.clipboard_clear()
        self.root.clipboard_append(URL + "/v1")
        self.copy_text.set(tr["copied_api_xft"] if self.has_xft else tr["copied_api_x11"])
        self.root.after(1500, lambda: self.copy_text.set(self._tr()["copy_api"]))

    def install_shortcut(self):
        tr = self._tr()
        try:
            install_desktop_entry()
        except OSError as exc:
            self.message.set(tr["pin_error"].format(err=exc))
        else:
            self.shortcut_text.set(tr["pinned_menu_xft"] if self.has_xft else tr["pinned_menu_x11"])

    def refresh_quota(self):
        if not self.launcher.can_create:
            return

        def fetch():
            info = self.launcher.quota()
            if info:
                self.events.put(("quota", info))

        Thread(target=fetch, daemon=True).start()

    def _render_quota(self, info: dict):
        self.last_quota = info
        text, ratio, color = _format_quota(info, self.lang)
        self.quota_var.set(text)
        self.quota_box.grid(row=0, column=0, sticky="e")
        self.quota_bar.delete("all")
        self.quota_bar.create_rectangle(0, 0, max(2, int(148 * ratio)), 4, fill=color, outline="")

    def _update_lamp(self):
        if self.running:
            color = PALETTE["warn"]
        elif self.record.phase == "ready" and self.checked_ready:
            color = PALETTE["ok"]
        elif self.record.phase == "busy":
            color = PALETTE["warn"]
        elif self.record.phase == "unknown" or self.record.error_code:
            color = PALETTE["danger"]
        else:
            color = PALETTE["faint"]
        self.lamp.itemconfigure(self.lamp_dot, fill=color)

    def _render_stepper(self):
        show = (self.running and self.cancellable) or bool(self.failed_step)
        if not show:
            self.stepper.grid_remove()
            return
        tr = self._tr()
        marks = ("✓", "●", "○", "✕", "—") if self.has_xft else ("[OK]", "[*]", "[ ]", "[!]", "[-]")
        m_done, m_active, m_wait, m_fail, m_skip = marks
        self.stepper.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        active_key = "worker" if self.current == "health" else self.current
        step_elapsed = int(time.monotonic() - self.step_started) if self.running and self.step_started else 0
        completed_count = len(self.done_steps)
        if self.launch_mode == "attach":
            completed_count = max(completed_count, 1)
        self.progress_var.set(min(len(STEP_KEYS), completed_count + (0.5 if self.running else 0.0)))
        for key, lbl in self.step_labels.items():
            title = tr["steps"][key]
            if key == "create" and self.launch_mode == "attach":
                lbl.configure(text=f"{m_skip}  {tr['manual_step']}", fg=PALETTE["muted"])
            elif key == self.failed_step:
                lbl.configure(text=f"{m_fail}  {title}", fg=PALETTE["danger"])
            elif key in self.done_steps:
                lbl.configure(text=f"{m_done}  {title}", fg=PALETTE["ok"])
            elif self.running and key == active_key:
                lbl.configure(text=f"{m_active}  {title} ({step_elapsed}s)", fg=PALETTE["warn"])
            else:
                lbl.configure(text=f"{m_wait}  {title}", fg=PALETTE["faint"])

    def show_record(self):
        tr = self._tr()
        sep = " · " if self.has_xft else " - "
        last = time.strftime("%H:%M:%S", time.localtime(self.record.checked)) if self.record.checked else ""
        reason = tr["errors"].get(self.record.error_code, self.record.error_code)
        if self.custom_notice:
            self.message.set(self.custom_notice)
        else:
            self.message.set(sep.join(filter(None, [tr["labels"][self.record.phase], last, reason])))
        self.entry.configure(values=self.launcher.recent_notebooks())
        if self.record.phase in ("ready", "busy") and self.checked_ready:
            self.endpoint_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        else:
            self.endpoint_row.grid_remove()
        self._update_lamp()
        self._render_stepper()
        self.controls()

    def controls(self):
        tr = self._tr()
        btns = tr["buttons"]
        play = "▶ " if self.has_xft else "> "
        ext = " ↗" if self.has_xft else ""
        attached = self.record.phase != "idle"
        owns_gpu = self.record.source == "created" and self.record.session_id > 0
        can_start = (not attached and self.launcher.can_create) or (attached and bool(self.record.kernel))
        enabled = {
            "start": can_start,
            "attach": not attached,
            "refresh": attached,
            "chat": self.checked_ready,
            "stop-worker": bool(self.record.owned_worker),
            "end-session": owns_gpu,
            "detach": attached and not owns_gpu,
        }
        start_label = btns["restart"] if self.record.phase == "worker-stopped" else btns["start"]
        self.buttons["start"].configure(text=play + start_label)
        self.buttons["chat"].configure(text=btns["chat"] + ext)
        for action in ("attach", "refresh", "stop-worker", "end-session", "detach", "cancel"):
            self.buttons[action].configure(text=btns[action])

        for action, button in self.buttons.items():
            active = self.running and self.cancellable if action == "cancel" else enabled[action] and not self.running
            button.configure(state="normal" if active else "disabled")
            button.grid_remove()

        left, right = [], []
        if self.running:
            if self.cancellable:
                right.append("cancel")
        elif not attached:
            if self.launcher.can_create:
                left.append("start")
            left.append("attach")
        elif self.checked_ready and self.record.phase == "ready":
            left.extend(["chat", "refresh"])
            if enabled["stop-worker"]:
                left.append("stop-worker")
            right.append("end-session" if owns_gpu else "detach")
        else:
            if enabled["start"]:
                left.append("start")
            left.append("refresh")
            if enabled["stop-worker"]:
                left.append("stop-worker")
            right.append("end-session" if owns_gpu else "detach")

        for col, action in enumerate(left):
            self.buttons[action].grid(row=0, column=col, padx=(0, 8), sticky="w")
        for idx, action in enumerate(right):
            self.buttons[action].grid(row=0, column=5 + idx, padx=(8, 0), sticky="e")

        self.entry.configure(state="disabled" if self.running or attached else "normal")

    def click(self, action):
        tr = self._tr()
        dlg = tr["dialogs"]
        if action == "chat":
            webbrowser.open(URL + "/")
            return
        if action == "cancel":
            self.cancel.set()
            self.message.set(tr["cancelling"])
            return
        if self.running:
            return
        slug = self.slug.get()
        if action == "attach" or (action == "start" and self.record.phase == "idle"):
            normalized = parse_notebook(slug)
            if not normalized:
                self.custom_notice = tr["invalid_slug"]
                self.message.set(self.custom_notice)
                self._update_lamp()
                return
            slug = normalized
            self.slug.set(normalized)
        url = ""
        if action == "attach":
            url = simpledialog.askstring(dlg["attach_title"], dlg["attach_prompt"], show="*", parent=self.root)
            if not url:
                return
        if action == "end-session":
            warn = dlg["end_busy"] if self.record.phase == "busy" else ""
            prompt = dlg["end_prompt"].format(warn=warn, nb=self.record.notebook)
            if not messagebox.askyesno(dlg["end_title"], prompt, parent=self.root):
                return
        if action == "detach":
            if self.record.phase == "busy":
                prompt = dlg["detach_busy"]
            elif self.record.owned_worker:
                prompt = dlg["detach_owned"]
            else:
                prompt = dlg["detach_plain"]
            if not messagebox.askyesno(dlg["detach_title"], prompt, parent=self.root):
                return
        self.custom_notice = ""
        self.running = True
        self.cancellable = action in ("start", "attach")
        self.checked_ready = False
        self.cancel.clear()
        if self.cancellable:
            self.launch_mode = "attach" if action == "attach" or self.record.phase != "idle" else "create"
            self.done_steps = {"create", "connect"} if (action == "start" and self.record.phase != "idle") else set()
            self.failed_step = ""
        self.started = self.step_started = time.monotonic()
        self.current = action
        self.message.set(tr["checking"])
        self._update_lamp()
        self._render_stepper()
        self.controls()

        def work():
            try:
                self.launcher.execute(action, slug if self.cancellable else "", url,
                                      self.events.put, self.cancel, expected_run=self.record.run)
                self.events.put(("finished", ""))
            except LaunchError as exc:
                self.events.put(("finished", str(exc)))
            except Exception:
                self.events.put(("finished", self._tr()["generic_error"]))

        Thread(target=work, daemon=False).start()

    def drain(self):
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            if isinstance(event, dict):
                phase = event["phase"]
                step_key = "worker" if phase == "health" else phase
                outcome = event.get("outcome")
                if outcome == "started":
                    self.current = phase
                    self.step_started = time.monotonic()
                elif outcome == "done" and step_key in STEP_INDEX:
                    self.done_steps.add(step_key)
                elif outcome == "failed" and step_key in STEP_INDEX:
                    self.failed_step = step_key
                self._render_stepper()
            elif event[0] == "quota":
                self._render_quota(event[1])
            elif event[0] == "finished":
                self.running = False
                self.record = self.launcher.status()
                if self.record.notebook:
                    self.slug.set(self.record.notebook)
                self.checked_ready = not event[1] and self.record.phase == "ready"
                if self.checked_ready:
                    self.failed_step = ""
                # Prefer localized error_code translation over raw English LaunchError when error_code is known.
                self.custom_notice = event[1] if (event[1] and not self.record.error_code) else ""
                self.show_record()
                if self.current in ("create", "proxy", "end-session") or not event[1]:
                    self.refresh_quota()
        if self.running and not self.cancel.is_set():
            tr = self._tr()
            elapsed = int(time.monotonic() - self.started)
            label = tr["labels"].get(self.current, tr["checking"])
            self.message.set(tr["elapsed_fmt"].format(label=label, s=elapsed))
            if self.cancellable:
                self._render_stepper()
        self.root.after(80, self.drain)

    def close(self):
        dlg = self._tr()["dialogs"]
        if not self.running:
            self.root.destroy()
        elif self.cancel.is_set() or not self.cancellable:
            messagebox.showinfo(dlg["wait_title"], dlg["wait_body"], parent=self.root)
        elif messagebox.askyesno(dlg["cancel_title"], dlg["cancel_body"], parent=self.root):
            self.click("cancel")


if __name__ == "__main__":
    root = tk.Tk(className="KglDesktop")
    try:
        Window(root, make_launcher())
    except LaunchError as exc:
        messagebox.showerror("KGL Desktop", str(exc))
        root.destroy()
    else:
        root.mainloop()
