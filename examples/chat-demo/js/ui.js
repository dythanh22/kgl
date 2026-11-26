import { buildChatRequest } from "./budget.js";
import { createPixelMascot, drawContextRing, drawStillMascot } from "./mascot.js";
import { displayAnswer } from "./markdown.js";
import { collectAllTags, filterConversations } from "./organize.js";
import {
  MASCOT_KEY,
  PARTIAL_LABELS,
  RAIL_KEY,
  THEME_KEY,
  activeSampling,
  activeSources,
  chatTitle,
  state,
} from "./state.js";
import { storageStats } from "./storage.js";
import { renderActionItemsCard } from "./tasks/action_items_view.js";
import { DEFAULT_MAX_REQUEST_BYTES, formatBytes, sliceSourceContent } from "./tasks/files.js";
import { buildSamplingOverrides } from "./tasks/sampling.js";
import { renderRewriteReview, renderSummaryMeta } from "./tasks/work_templates_view.js";

export const byId = id => document.getElementById(id);

const SAMPLING_LABELS = {
  preset: "Theo preset",
  precise: "Chính xác",
  creative: "Sáng tạo",
  custom: "Tùy chỉnh",
};

let mascotCtrl = null;
let mascotSettleTimer = null;
let toastTimer = null;

export function svgIcon(name, extraClass = "") {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", extraClass ? `icon ${extraClass}` : "icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(ns, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

export function showToast(message, ms = 2200) {
  const el = byId("toast");
  if (!el) return;
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

export function fmtK(n) {
  if (!Number.isFinite(n)) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(Math.round(n));
}

export function updateGreeting() {
  const el = byId("greeting");
  if (!el) return;
  const hour = new Date().getHours();
  el.textContent = hour < 5 ? "Làm việc muộn?" : hour < 12 ? "Chào buổi sáng" : hour < 18 ? "Chào buổi chiều" : "Chào buổi tối";
}

export function initMascot() {
  drawStillMascot(byId("mascot-mini"));
  drawStillMascot(byId("mascot-about"));
  mascotCtrl = createPixelMascot(byId("mascot-canvas"));
  try {
    const saved = localStorage.getItem(MASCOT_KEY);
    state.mascotEnabled = saved === null ? true : saved === "true";
  } catch {
    state.mascotEnabled = true;
  }
  applyMascotVisibility();
  return mascotCtrl;
}

export function applyMascotVisibility() {
  document.body.classList.toggle("no-mascot", !state.mascotEnabled);
  const sw = byId("sw-mascot");
  if (sw) sw.setAttribute("aria-checked", String(state.mascotEnabled));
}

export function setMascotMood(mood, label = "", { error = false, settleMs = 0 } = {}) {
  clearTimeout(mascotSettleTimer);
  mascotCtrl?.set(mood);
  const statusEl = byId("mascot-status");
  if (statusEl) {
    statusEl.textContent = label;
    statusEl.dataset.error = String(Boolean(error));
  }
  if (settleMs > 0) {
    mascotSettleTimer = setTimeout(() => {
      if (!state.running) {
        const hasText = Boolean(byId("prompt")?.value.trim());
        setMascotMood(hasText ? "listening" : "idle", "");
      }
    }, settleMs);
  }
}

export function pulseMascotToken() {
  mascotCtrl?.token();
}

export function getMascotMood() {
  return mascotCtrl?.getState?.() || "idle";
}

export function applyTheme(mode) {
  const valid = ["dark", "light", "system"].includes(mode) ? mode : "dark";
  state.themeMode = valid;
  const resolved = valid === "system"
    ? (window.matchMedia?.("(prefers-color-scheme: light)")?.matches ? "light" : "dark")
    : valid;
  document.documentElement.dataset.theme = resolved;
  const seg = byId("theme-seg");
  if (seg) {
    for (const btn of seg.querySelectorAll("[data-theme-val]")) {
      btn.setAttribute("aria-pressed", String(btn.dataset.themeVal === valid));
    }
  }
  const toggle = byId("theme-toggle");
  if (toggle) {
    toggle.title = resolved === "dark" ? "Chuyển sang giao diện sáng (Fog)" : "Chuyển sang giao diện tối (Graphite)";
  }
  drawStillMascot(byId("mascot-mini"));
  drawStillMascot(byId("mascot-about"));
  renderBudget();
}

export function initTheme() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    applyTheme(saved === "light" || saved === "dark" || saved === "system" ? saved : "dark");
  } catch {
    applyTheme("dark");
  }
  window.matchMedia?.("(prefers-color-scheme: dark)")?.addEventListener?.("change", () => {
    if (state.themeMode === "system") applyTheme("system");
  });
  byId("theme-toggle")?.addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme === "light" ? "light" : "dark";
    const next = cur === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch {}
  });
}

export function applyRailState(open, save = false) {
  state.railOpen = Boolean(open);
  const rail = byId("rail");
  if (rail) rail.hidden = state.vision || !state.railOpen;
  const btn = byId("rail-toggle");
  if (btn) btn.setAttribute("aria-pressed", String(state.railOpen && !state.vision));
  if (save && window.innerWidth > 1160) {
    try { localStorage.setItem(RAIL_KEY, String(state.railOpen)); } catch {}
  }
  updateScrim();
}

export function initRail() {
  let open = window.innerWidth > 1160;
  try {
    const saved = localStorage.getItem(RAIL_KEY);
    if (saved !== null && window.innerWidth > 1160) open = saved === "true";
  } catch {}
  applyRailState(open, false);
}

export function applySidebarState(open) {
  state.sidebarOpen = Boolean(open);
  const aside = byId("conversations");
  if (aside) aside.hidden = state.vision || !state.sidebarOpen;
  const openBtn = byId("sidebar-open");
  if (openBtn) openBtn.hidden = state.vision || state.sidebarOpen;
  updateScrim();
}

export function updateScrim() {
  const scrim = byId("scrim");
  if (!scrim) return;
  const mobileSide = window.innerWidth <= 820 && state.sidebarOpen && !state.vision;
  const floatingRail = window.innerWidth <= 1160 && state.railOpen && !state.vision;
  scrim.hidden = !(mobileSide || floatingRail);
}

export function setStatus(text, stateName = "checking", { reconnect = false, waiting = false } = {}) {
  const notice = byId("connection-notice");
  const retry = byId("reconnect");
  const restoreFocus = document.activeElement === retry;
  notice.hidden = !waiting && !reconnect;
  notice.dataset.state = stateName;
  byId("connection-message").textContent = notice.hidden ? "" : text;
  retry.hidden = !reconnect;
  if (restoreFocus && retry.hidden) {
    byId(state.vision ? "image" : "prompt").focus();
  }
  const status = byId("status");
  if (status) {
    status.textContent = text;
    status.dataset.state = stateName;
    status.title = text;
  }
  const dot = byId("status-dot");
  if (dot) dot.dataset.state = stateName;
  if (stateName === "error") {
    setMascotMood("sad", reconnect ? "" : text, { error: true });
  } else if (stateName === "busy" && !state.running) {
    setMascotMood("thinking");
  }
}

export function showError(error, options) {
  const message = error instanceof TypeError
    ? "Không kết nối được proxy cục bộ. Kiểm tra terminal đang chạy proxy."
    : error.message;
  const hint = /cuda.*out of memory|outofmemoryerror/i.test(message)
    ? " Hãy thử model nhỏ hơn hoặc khởi động lại worker với thêm bộ nhớ GPU."
    : "";
  setStatus(message + hint, "error", options);
}

let sessionWarned15 = false;

export function renderSessionTime() {
  const label = byId("session-time");
  if (label) label.hidden = state.sessionDeadline === undefined;
  if (state.sessionDeadline === undefined) {
    renderEnvironmentCard();
    return;
  }
  const minutes = Math.max(0, Math.ceil((state.sessionDeadline - performance.now()) / 60000));
  const hours = Math.floor(minutes / 60);
  const rem = minutes % 60;
  const formatted = hours ? `${hours}h ${rem}m` : `${minutes}m`;
  if (label) {
    label.textContent = `Kaggle · ${formatted}`;
    label.title = `Thời gian phiên Kaggle ước tính còn lại · ${minutes} phút`;
    label.toggleAttribute("data-low", minutes <= 30);
  }
  renderEnvironmentCard();

  const warnBanner = byId("session-warn");
  if (!warnBanner) return;
  if (minutes > 15) {
    sessionWarned15 = false;
    warnBanner.hidden = true;
  } else if (!sessionWarned15) {
    sessionWarned15 = true;
    byId("session-warn-text").textContent = `Phiên Kaggle ước tính còn ${minutes} phút. Hãy sao lưu lịch sử trước khi phiên kết thúc.`;
    warnBanner.hidden = false;
  }
}

export function updateSessionTime(seconds) {
  state.sessionDeadline = Number.isFinite(seconds) && seconds >= 0
    ? performance.now() + seconds * 1000 : undefined;
  renderSessionTime();
}

export function renderEnvironmentCard() {
  const box = byId("rail-env");
  if (!box) return;
  box.replaceChildren();
  const health = state.health;
  const addKv = (iconName, label, right, title = "") => {
    const row = document.createElement("div");
    row.className = "kv";
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = label;
    if (title) k.title = title;
    const v = document.createElement("span");
    v.className = "v";
    v.textContent = right;
    row.append(svgIcon(iconName), k, v);
    box.append(row);
  };

  if (!health) {
    addKv("bolt", "Worker", "Connecting…");
    return;
  }

  // GPU memory rows from LlamaBackend.health() (e.g. 2x Tesla T4)
  if (Array.isArray(health.gpu_memory) && health.gpu_memory.length) {
    for (const sample of health.gpu_memory) {
      const gb = (sample.memory_mib / 1024).toFixed(1);
      addKv("gpu", `GPU ${sample.gpu}`, `${gb} GB VRAM`);
    }
  } else {
    const dev = health.devices?.join(", ") || health.device || "cpu";
    addKv("gpu", "Device", dev);
  }

  const shortModel = health.alias || health.model?.split("/").pop() || "model";
  const rightTag = [health.quant, health.devices?.length ? "GPU" : ""].filter(Boolean).join(" · ");
  addKv("bolt", shortModel, rightTag || (health.backend || ""), health.model || "");

  if (health.offloaded_layers?.total) {
    addKv("chip", "Layers offloaded", `${health.offloaded_layers.offloaded} / ${health.offloaded_layers.total}`);
  }
  const ctx = health.n_ctx || health.requested_context;
  if (ctx) {
    addKv("sliders", "Context window", `${fmtK(ctx)}${health.n_ctx ? "" : " req"}`);
  }
  if (state.sessionDeadline !== undefined) {
    const mins = Math.max(0, Math.ceil((state.sessionDeadline - performance.now()) / 60000));
    const hrs = Math.floor(mins / 60);
    const rem = mins % 60;
    addKv("clock", "Phiên Kaggle", hrs ? `${hrs}h ${rem}m` : `${mins}m`);
  }
  renderAboutInfo();
}

export function renderAboutInfo() {
  const box = byId("about-info");
  if (!box) return;
  box.replaceChildren();
  const h = state.health || {};
  const rows = [
    ["Backend", h.backend || "—"],
    ["Model", h.alias ? `${h.alias} (${h.model?.split("/").pop() || ""})` : (h.model || "—")],
    ["Quant / Context", [h.quant, h.n_ctx ? `${h.n_ctx.toLocaleString()} ctx` : ""].filter(Boolean).join(" · ") || "—"],
    ["Devices", (h.devices?.join(", ") || h.device || "—")],
    ["Capabilities", Object.entries(state.capabilities || {}).filter(([, v]) => v).map(([k]) => k).join(", ") || "standard"],
  ];
  for (const [k, v] of rows) {
    const row = document.createElement("div");
    row.className = "srow";
    const w = document.createElement("div");
    w.className = "w";
    const b = document.createElement("b");
    b.textContent = k;
    w.append(b);
    const val = document.createElement("span");
    val.className = "muted small-note";
    val.textContent = v;
    row.append(w, val);
    box.append(row);
  }
}

export function syncSendEnabled() {
  const promptVal = byId("prompt")?.value.trim() || "";
  const hasSources = activeSources().some(s => s.selected !== false);
  const runBtn = byId("run");
  if (runBtn) {
    runBtn.disabled = state.running || !state.canSend || (!promptVal && !hasSources);
  }
}

export function resizePrompt() {
  const promptField = byId("prompt");
  const box = byId("composer-box");
  if (!promptField || !box) return;
  promptField.style.height = "auto";
  const nextHeight = Math.min(promptField.scrollHeight, 220);
  promptField.style.height = nextHeight + "px";
  const hasFiles = activeSources().length > 0;
  const grown = hasFiles || promptField.value.includes("\n") || nextHeight > 44 || promptField.value.length > 68;
  box.dataset.grown = String(grown);
  syncSendEnabled();
}

export function renderComposerFiles() {
  const strip = byId("composer-files");
  const underBtn = byId("under-sources-label");
  const railEmpty = byId("rail-sources-empty");
  const railCount = byId("rail-sources-count");
  const list = activeSources();
  const activeCount = list.filter(s => s.selected !== false).length;

  if (railEmpty) railEmpty.hidden = list.length > 0;
  if (railCount) railCount.textContent = list.length ? `${activeCount}/${list.length}` : "";
  if (underBtn) {
    if (state.taskMode === "action_items") underBtn.textContent = "Rút đầu việc";
    else if (state.taskMode === "summary") underBtn.textContent = "Tóm tắt";
    else if (state.taskMode === "rewrite") underBtn.textContent = "Biên tập";
    else if (list.length) underBtn.textContent = `${activeCount} nguồn`;
    else underBtn.textContent = "Nguồn";
  }

  if (!strip) return;
  strip.replaceChildren();
  strip.hidden = state.vision || list.length === 0;
  for (const src of list) {
    const chip = document.createElement("span");
    chip.className = "fchip";
    if (src.selected === false) chip.style.opacity = "0.5";
    const { from, to, total } = sliceSourceContent(src);
    const name = document.createElement("span");
    name.className = "fn";
    name.textContent = src.name;
    name.title = `${src.name} (${from}–${to}/${total} dòng)`;
    const meta = document.createElement("span");
    meta.className = "fm";
    meta.textContent = `${to - from + 1}L`;
    const rm = document.createElement("button");
    rm.type = "button";
    rm.setAttribute("aria-label", `Bỏ ${src.name}`);
    rm.append(svgIcon("x"));
    rm.addEventListener("click", () => {
      if (state.running) return;
      const arr = activeSources();
      const idx = arr.indexOf(src);
      if (idx >= 0) arr.splice(idx, 1);
      strip.dispatchEvent(new CustomEvent("kgl:sources-changed", { bubbles: true }));
    });
    chip.append(svgIcon("file"), name, meta, rm);
    strip.append(chip);
  }
  resizePrompt();
}

export function syncSystemUI() {
  const val = state.active ? (state.active.system ?? "") : state.newSystem;
  byId("system-prompt").value = val;
  byId("toggle-system").dataset.active = String(Boolean(val.trim()));
}

export function syncTaskModeUI() {
  const mode = state.vision ? null : state.taskMode;
  const active = Boolean(mode);
  const banner = byId("task-banner");
  const title = byId("task-mode-title");
  const badge = byId("task-mode-badge");
  const toggle = byId("toggle-action-items");
  const promptField = byId("prompt");
  const summaryOpts = byId("summary-options");
  const rewriteOpts = byId("rewrite-options");

  banner.hidden = !active;
  summaryOpts.hidden = mode !== "summary";
  rewriteOpts.hidden = mode !== "rewrite";
  toggle.setAttribute("aria-pressed", String(mode === "action_items"));

  if (mode === "action_items") {
    title.textContent = "Rút đầu việc có bằng chứng";
    badge.textContent = state.supportsJsonSchema
      ? "JSON Schema · Bảng + CSV/TSV"
      : "Chế độ văn bản · Worker thiếu JSON Schema";
    badge.className = state.supportsJsonSchema ? "badge-accent" : "badge-warn";
    promptField.placeholder = "Dán biên bản hoặc ghi chú tiếng Việt / tiếng Anh để rút đầu việc có câu nguồn…";
  } else if (mode === "summary") {
    title.textContent = "Tóm tắt có cấu trúc";
    badge.textContent = "Quyết định · Đề xuất · Câu hỏi mở";
    badge.className = "badge-accent";
    promptField.placeholder = "Dán văn bản cần tóm tắt (có thể đặt giới hạn số từ ở trên)…";
  } else if (mode === "rewrite") {
    title.textContent = "Soạn / sửa văn bản";
    badge.textContent = "Đối chiếu bản gốc & kiểm tên/số/hạn";
    badge.className = "badge-accent";
    promptField.placeholder = "Dán bản nháp cần biên tập…";
  } else {
    promptField.placeholder = state.active?.messages?.length
      ? "Reply to kgl… (gõ / để mở mẫu prompt)"
      : "Bạn muốn làm việc gì hôm nay? (gõ / để mở mẫu prompt hoặc kéo thả tệp)";
  }
  byId("history-scope").disabled = state.running || active;
  renderComposerFiles();
}

export function syncThinkingUI() {
  const select = byId("thinking");
  const btn = byId("think-toggle");
  const lbl = byId("think-toggle-label");
  if (!select || !btn) return;
  const on = select.value === "on";
  btn.hidden = state.vision || !state.supportsThinking;
  btn.setAttribute("aria-pressed", String(on));
  if (lbl) lbl.textContent = on ? "Có suy luận" : "Suy luận";
}

export function syncSamplingUI() {
  const cfg = activeSampling();
  byId("sampling-mode").value = cfg.mode;
  byId("samp-temp").value = cfg.temperature;
  byId("samp-top-p").value = cfg.top_p;
  byId("samp-presence").value = cfg.presence_penalty;
  byId("samp-repeat").value = cfg.repeat_penalty;

  const label = SAMPLING_LABELS[cfg.mode] || "Theo preset";
  const underLbl = byId("under-sampling-label");
  if (underLbl) underLbl.textContent = label;
  const railFolded = byId("rail-gen-folded");
  if (railFolded) railFolded.textContent = label;

  const pop = byId("sampling-pop");
  if (pop) {
    for (const item of pop.querySelectorAll("[data-samp-mode]")) {
      const checked = item.dataset.sampMode === cfg.mode;
      item.setAttribute("aria-checked", String(checked));
      const ck = item.querySelector(".ck");
      if (ck) ck.textContent = checked ? "✓" : "";
    }
  }

  const canPresence = state.capabilities?.presence_penalty === true;
  const canRepeat = state.capabilities?.repeat_penalty === true;
  byId("samp-presence-wrap").hidden = !canPresence;
  byId("samp-repeat-wrap").hidden = !canRepeat;

  const customEditable = cfg.mode === "custom";
  for (const id of ["samp-temp", "samp-top-p", "samp-presence", "samp-repeat"]) {
    byId(id).disabled = state.running || !customEditable;
  }

  const eff = byId("sampling-effective");
  try {
    const { summary } = buildSamplingOverrides(cfg, state.capabilities);
    eff.textContent = summary;
    eff.style.color = "";
  } catch (error) {
    eff.textContent = error.message;
    eff.style.color = "var(--danger)";
  }
}

export function historyNotice(message) {
  const notice = byId("history-status");
  notice.textContent = message;
  notice.hidden = !message;
}

export function formatMessageStats(message) {
  const parts = [];
  const last = message?.usage;
  const metrics = message?.metrics;
  if (last?.completion_tokens > 0) {
    parts.push(`${last.completion_tokens.toLocaleString()} tokens`);
  }
  if (Number.isFinite(metrics?.totalMs)) {
    parts.push(`${(metrics.totalMs / 1000).toFixed(1)} s`);
  }
  if (!message?.partial && Number.isFinite(metrics?.workerTokPerSec) && metrics.workerTokPerSec > 0) {
    parts.push(`${metrics.workerTokPerSec.toFixed(1)} tok/s`);
  }
  if (Number.isFinite(metrics?.firstContentMs)) {
    parts.push(`TTFT ${(metrics.firstContentMs / 1000).toFixed(2)} s`);
  }
  return parts.join(" · ");
}

export function showUsage() {
  const usage = byId("usage");
  if (!usage) return;
  usage.hidden = true;
  usage.textContent = "";
}

export function scrollToReply() {
  const transcript = byId("transcript");
  transcript.scrollTop = transcript.scrollHeight;
  byId("scroll-bottom").hidden = true;
}

export function isNearBottom() {
  const transcript = byId("transcript");
  return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 90;
}

function makeBranchNav(activeIndex, total, onChange) {
  const nav = document.createElement("span");
  nav.className = "branch-nav";
  const prev = document.createElement("button");
  prev.type = "button";
  prev.textContent = "‹";
  prev.title = "Nhánh trước";
  prev.disabled = state.running || activeIndex <= 0;
  prev.addEventListener("click", () => onChange(activeIndex - 1));
  const label = document.createElement("span");
  label.textContent = `${activeIndex + 1}/${total}`;
  const next = document.createElement("button");
  next.type = "button";
  next.textContent = "›";
  next.title = "Nhánh sau";
  next.disabled = state.running || activeIndex >= total - 1;
  next.addEventListener("click", () => onChange(activeIndex + 1));
  nav.append(prev, label, next);
  return nav;
}

function makeIconCopyButton(value, title) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "ibtn";
  btn.title = title;
  btn.setAttribute("aria-label", title);
  btn.append(svgIcon("copy"));
  let timer;
  btn.addEventListener("click", async () => {
    clearTimeout(timer);
    try {
      await navigator.clipboard.writeText(value);
      btn.replaceChildren(svgIcon("check"));
      showToast("Đã sao chép");
      timer = setTimeout(() => btn.replaceChildren(svgIcon("copy")), 1500);
    } catch {
      showToast("Không thể sao chép tự động");
    }
  });
  return btn;
}

function renderSegmentedAnswer(segments) {
  const frag = document.createDocumentFragment();
  segments.forEach((seg, i) => {
    if (i > 0) {
      const div = document.createElement("div");
      div.className = "continue-divider";
      div.textContent = `Tiếp nối ${i}`;
      frag.append(div);
    }
    frag.append(displayAnswer(seg));
  });
  return frag;
}

export function addMessage(role, content = "", reasoning = "", partial = false, live = false, reason, meta = {}) {
  const transcript = byId("transcript");
  const article = document.createElement("article");
  article.className = role;
  if (partial) article.dataset.partial = "true";
  if (Number.isInteger(meta.index)) article.dataset.msgIndex = String(meta.index);

  const isActionItems = meta.task === "action_items";

  if (role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "user-bubble";
    const p = document.createElement("p");
    p.textContent = content;
    bubble.append(p);
    article.append(bubble);

    if (!live) {
      const actions = document.createElement("div");
      actions.className = "user-meta";
      const hasBranches = Array.isArray(meta.siblings) && meta.siblings.length > 1 && meta.onSwitchSibling;
      if (hasBranches) {
        actions.dataset.always = "true";
        const activeIdx = Number.isInteger(meta.activeSibling) ? meta.activeSibling : 0;
        actions.append(makeBranchNav(activeIdx, meta.siblings.length, next => meta.onSwitchSibling(meta.index, next)));
      }
      actions.append(makeIconCopyButton(content, "Sao chép tin nhắn"));
      if (meta.onEdit) {
        const editBtn = document.createElement("button");
        editBtn.type = "button";
        editBtn.className = "ibtn";
        editBtn.title = "Sửa & gửi lại (tạo nhánh mới)";
        editBtn.setAttribute("aria-label", "Sửa tin nhắn");
        editBtn.disabled = state.running;
        editBtn.append(svgIcon("edit"));
        editBtn.addEventListener("click", () => openInlineEdit(bubble, content, meta.index, meta.onEdit));
        actions.append(editBtn);
      }
      article.append(actions);
    }
    transcript.append(article);
    return { text: bubble, thoughts: null, thoughtSummary: null };
  }

  // Assistant turn (Graphite flat layout matching demo.gif)
  const details = document.createElement("details");
  details.className = "reasoning";
  const summary = document.createElement("summary");
  const thoughts = document.createElement("p");
  summary.textContent = live && !content ? "Đang suy luận…" : "Suy luận";
  thoughts.textContent = reasoning;
  details.hidden = !reasoning;
  details.append(summary, thoughts);
  article.append(details);

  if (partial) {
    const pill = document.createElement("span");
    pill.className = "partial-pill";
    const partialLabel = Object.hasOwn(PARTIAL_LABELS, reason) ? PARTIAL_LABELS[reason] : "Chưa có câu trả lời";
    pill.textContent = `Chưa hoàn tất · ${partialLabel}`;
    article.append(pill);
  }

  const text = document.createElement("div");
  text.className = "answer";
  if (isActionItems) {
    text.append(renderActionItemsCard({
      raw: content,
      sourceText: meta.sourceText || "",
      partial,
      live,
      structured: Boolean(meta.structured),
    }));
  } else if (content) {
    if (Array.isArray(meta.segments) && meta.segments.length > 1) {
      text.append(renderSegmentedAnswer(meta.segments));
    } else {
      text.append(displayAnswer(content));
    }
    if (!live && !partial && meta.task === "summary") {
      text.append(renderSummaryMeta(content, meta.targetWords));
    } else if (!live && !partial && meta.task === "rewrite" && meta.sourceText) {
      text.append(renderRewriteReview(meta.sourceText, content));
    }
  } else if (live && !reasoning) {
    const indicator = document.createElement("span");
    indicator.className = "thinking-indicator";
    indicator.textContent = "Đang đọc yêu cầu…";
    text.append(indicator);
  }
  article.append(text);

  if (!live) {
    const foot = document.createElement("div");
    foot.className = "msg-foot";
    if (content) {
      foot.append(makeIconCopyButton(content, isActionItems && meta.structured ? "Sao chép JSON gốc" : "Sao chép câu trả lời"));
    }
    if (meta.isLast && !partial && meta.onRegenerate) {
      const regenBtn = document.createElement("button");
      regenBtn.type = "button";
      regenBtn.className = "ibtn";
      regenBtn.title = "Sinh lại câu trả lời";
      regenBtn.setAttribute("aria-label", "Tạo lại câu trả lời");
      regenBtn.disabled = state.running || !state.canSend;
      regenBtn.append(svgIcon("redo"));
      regenBtn.addEventListener("click", () => meta.onRegenerate());
      foot.append(regenBtn);
    }
    if (Array.isArray(meta.variants) && meta.variants.length > 1 && meta.onSwitchVariant) {
      const activeIdx = Number.isInteger(meta.activeVariant) ? meta.activeVariant : 0;
      foot.append(makeBranchNav(activeIdx, meta.variants.length, next => meta.onSwitchVariant(meta.index, next)));
    }
    const stats = formatMessageStats(meta.message);
    if (stats) {
      const span = document.createElement("span");
      span.className = "msg-stats";
      span.textContent = stats;
      foot.append(span);
    }
    if (foot.childElementCount) article.append(foot);
  }

  transcript.append(article);
  return { text, thoughts, thoughtSummary: summary };
}

function openInlineEdit(container, originalText, index, onEdit) {
  if (state.running) return;
  container.replaceChildren();
  const box = document.createElement("div");
  box.className = "inline-edit-box";
  const area = document.createElement("textarea");
  area.value = originalText;
  const bar = document.createElement("div");
  bar.className = "inline-edit-actions";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "btn-ghost sm";
  cancelBtn.textContent = "Hủy";
  cancelBtn.addEventListener("click", () => {
    const p = document.createElement("p");
    p.textContent = originalText;
    container.replaceChildren(p);
  });
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "btn sm pri";
  saveBtn.textContent = "Gửi bản sửa";
  saveBtn.disabled = !state.canSend;
  saveBtn.addEventListener("click", () => {
    const next = area.value.trim();
    if (!next) return;
    onEdit(index, next);
  });
  area.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      event.preventDefault();
      cancelBtn.click();
    } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      saveBtn.click();
    }
  });
  bar.append(cancelBtn, saveBtn);
  box.append(area, bar);
  container.append(box);
  area.focus();
  area.setSelectionRange(area.value.length, area.value.length);
}

export function openEditForLastUser(onEdit) {
  const msgs = state.active?.messages || [];
  let lastUserIdx = -1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === "user") {
      lastUserIdx = i;
      break;
    }
  }
  if (lastUserIdx < 0) return false;
  const article = byId("transcript").querySelector(`article[data-msg-index="${lastUserIdx}"]`);
  const container = article?.querySelector(".user-bubble");
  if (!container) return false;
  openInlineEdit(container, msgs[lastUserIdx].content, lastUserIdx, onEdit);
  return true;
}

export function renderStorageMetrics() {
  const el = byId("storage-metrics");
  const folded = byId("rail-storage-folded");
  const bytes = storageStats.lastBytes;
  if (folded) {
    folded.textContent = state.conversations.length ? `${state.conversations.length} chat` : "";
  }
  if (!el) return;
  if (!bytes && !state.conversations.length) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const warn = bytes >= 3.5 * 1024 * 1024 || storageStats.lastWriteMs >= 30;
  el.dataset.warn = String(warn);
  const writeInfo = storageStats.writeCount ? ` · ghi ${storageStats.lastWriteMs.toFixed(1)} ms` : "";
  el.textContent = `Lưu trữ: ${formatBytes(bytes)} · ${state.conversations.length} chat${writeInfo}`
    + (warn ? " · Nên sao lưu v2." : "");
}

function dateGroupOf(chat, nowStart) {
  if (chat.pinned) return "Đã ghim";
  const ts = Date.parse(chat.updatedAt || chat.createdAt || "");
  if (!Number.isFinite(ts)) return "Gần đây";
  const day = 86400000;
  if (ts >= nowStart) return "Hôm nay";
  if (ts >= nowStart - day) return "Hôm qua";
  if (ts >= nowStart - 7 * day) return "7 ngày qua";
  return "Cũ hơn";
}

export function renderHistory(onSelect, onDelete) {
  const historyList = byId("history");
  const tagBar = byId("tag-filter-bar");
  const query = byId("history-search").value.trim();

  const allTags = collectAllTags(state.conversations);
  tagBar.replaceChildren();
  tagBar.hidden = allTags.length === 0;
  if (!tagBar.hidden) {
    const allChip = document.createElement("button");
    allChip.type = "button";
    allChip.className = "chip";
    allChip.textContent = "Tất cả";
    allChip.setAttribute("aria-pressed", String(!state.activeTagFilter));
    allChip.addEventListener("click", () => {
      state.activeTagFilter = "";
      renderHistory(onSelect, onDelete);
    });
    tagBar.append(allChip);
    for (const item of allTags) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = `#${item.label} (${item.count})`;
      const selected = state.activeTagFilter.toLocaleLowerCase() === item.label.toLocaleLowerCase();
      chip.setAttribute("aria-pressed", String(selected));
      chip.addEventListener("click", () => {
        state.activeTagFilter = selected ? "" : item.label;
        renderHistory(onSelect, onDelete);
      });
      tagBar.append(chip);
    }
  }

  historyList.hidden = !state.recentsOpen;
  const matches = filterConversations(state.conversations, {
    query,
    tag: state.activeTagFilter,
    dateRange: state.dateFilter,
  });

  historyList.replaceChildren();
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);
  const nowStart = todayStart.getTime();
  let lastGroup = "";
  const showGroups = matches.length > 3 || matches.some(m => m.chat.pinned);

  for (const { chat, snippet } of matches) {
    if (showGroups) {
      const group = dateGroupOf(chat, nowStart);
      if (group !== lastGroup) {
        lastGroup = group;
        const heading = document.createElement("div");
        heading.className = "history-group";
        heading.textContent = group;
        historyList.append(heading);
      }
    }

    const item = document.createElement("div");
    item.className = "history-item";
    item.setAttribute("role", "button");
    item.tabIndex = 0;
    item.setAttribute("aria-current", String(chat === state.active));
    item.title = chatTitle(chat);

    const row = document.createElement("div");
    row.className = "history-row";
    if (chat.pinned) {
      const pin = document.createElement("span");
      pin.className = "history-pin";
      pin.textContent = "★";
      pin.title = "Đã ghim";
      row.append(pin);
    }
    const label = document.createElement("span");
    label.className = "history-title";
    label.textContent = chatTitle(chat);
    row.append(label);

    if (onDelete) {
      const del = document.createElement("button");
      del.type = "button";
      del.className = "history-del";
      del.title = "Xoá hội thoại";
      del.setAttribute("aria-label", `Xoá ${chatTitle(chat)}`);
      del.append(svgIcon("trash"));
      del.addEventListener("click", event => {
        event.stopPropagation();
        onDelete(chat);
      });
      row.append(del);
    }
    item.append(row);

    if (chat.tags?.length) {
      const tagsWrap = document.createElement("div");
      tagsWrap.className = "history-tags";
      for (const t of chat.tags.slice(0, 4)) {
        const pill = document.createElement("span");
        pill.className = "history-tag";
        pill.textContent = `#${t}`;
        tagsWrap.append(pill);
      }
      item.append(tagsWrap);
    }

    if (snippet) {
      const snip = document.createElement("div");
      snip.className = "history-snippet";
      const mark = document.createElement("mark");
      mark.textContent = snippet.match;
      snip.append(document.createTextNode(snippet.before), mark, document.createTextNode(snippet.after));
      item.append(snip);
    }

    item.addEventListener("click", () => onSelect(chat));
    item.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onSelect(chat);
      }
    });
    historyList.append(item);
  }

  if (!matches.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = (query || state.activeTagFilter || state.dateFilter !== "all")
      ? "Không có hội thoại phù hợp."
      : "Chưa có hội thoại nào.";
    historyList.append(empty);
  }
  renderStorageMetrics();
}

let transcriptHandlers = {};

export function setTranscriptHandlers(handlers) {
  transcriptHandlers = handlers || {};
}

export function renderTranscript(liveMessage) {
  const transcript = byId("transcript");
  const emptyState = byId("empty-state");
  byId("thread-title").textContent = chatTitle(state.active);
  document.title = state.active ? `${chatTitle(state.active)} · kgl` : "kgl workspace chat";

  const hasActive = Boolean(state.active);
  for (const id of ["rename-chat", "delete-chat", "export-md", "pin-chat", "edit-tags", "chat-menu-active-sep", "chat-menu-danger-sep"]) {
    const node = byId(id);
    if (node) node.hidden = !hasActive;
  }
  if (hasActive) {
    const isPinned = Boolean(state.active.pinned);
    const pinBtn = byId("pin-chat");
    pinBtn.setAttribute("aria-pressed", String(isPinned));
    pinBtn.querySelector(".w").textContent = isPinned ? "Bỏ ghim chat" : "Ghim chat";
    const tagCount = state.active.tags?.length || 0;
    byId("edit-tags").querySelector(".w").textContent = tagCount ? `Tags (${tagCount})` : "Gắn Tags";
  }
  byId("handover-chat").hidden = !(state.active?.messages?.length >= 2);
  showUsage();

  transcript.replaceChildren();
  const msgs = state.active?.messages || [];
  emptyState.hidden = state.vision || msgs.length > 0;
  transcript.hidden = state.vision || msgs.length === 0;
  if (!msgs.length && !state.vision) {
    updateGreeting();
  }

  let reply;
  msgs.forEach((message, index) => {
    const isLive = message === liveMessage;
    const nodes = addMessage(
      message.role,
      message.content,
      message.reasoning,
      message.partial && !isLive,
      isLive,
      message.partialReason,
      {
        index,
        isLast: index === msgs.length - 1,
        message,
        task: message.task,
        structured: message.structured,
        sourceText: message.sourceText,
        targetWords: message.targetWords,
        segments: message.segments,
        siblings: message.siblings,
        activeSibling: message.activeSibling,
        variants: message.variants,
        activeVariant: message.activeVariant,
        ...transcriptHandlers,
      },
    );
    if (isLive) reply = nodes;
  });

  const last = msgs.at(-1);
  byId("retry").hidden = !last?.partial;
  byId("continue").hidden = !(last?.partial && last?.partialReason === "length" && last?.content?.trim() && !last?.structured);
  syncTaskModeUI();
  scrollToReply();
  return reply;
}

export function setBusy(busy) {
  state.running = busy;
  byId("run").hidden = busy && !state.vision;
  byId("stop").hidden = !busy || state.vision;
  byId("stop").disabled = !busy;
  byId("run-vision").disabled = busy || !state.canSend;
  byId("refresh").disabled = byId("new-chat").disabled = busy;
  byId("retry").disabled = byId("continue").disabled = busy || !state.canSend;
  byId("history").inert = busy;
  for (const id of [
    "plus-btn", "rename-chat", "delete-chat", "pin-chat", "edit-tags", "export-md", "handover-chat",
    "history-search", "history-date-filter", "import-chat", "toggle-action-items", "exit-task-mode",
    "open-prompts", "attach-file-btn", "pin-note-btn", "sampling-mode", "toggle-sampling-adv", "samp-reset",
    "think-toggle", "under-sampling-btn",
  ]) {
    const el = byId(id);
    if (el) el.disabled = busy;
  }
  byId("prompt").readOnly = busy;
  byId("system-prompt").readOnly = busy;
  byId("max-tokens").readOnly = busy;
  byId("history-scope").disabled = busy || Boolean(state.taskMode);
  byId("thinking").disabled = busy;
  syncSamplingUI();
  syncSendEnabled();
}

export function currentChatRequest(mode = "send", overridePrevious) {
  const limit = byId("max-tokens");
  let samplingOverrides = {};
  try {
    samplingOverrides = buildSamplingOverrides(activeSampling(), state.capabilities).overrides;
  } catch {
    samplingOverrides = {};
  }
  const req = buildChatRequest({
    previous: overridePrevious || state.active?.messages || [],
    prompt: byId("prompt").value,
    systemText: state.active ? (state.active.system ?? "") : state.newSystem,
    scope: byId("history-scope").value,
    limitValue: limit.value,
    limitNumber: limit.valueAsNumber,
    contextLimit: state.contextLimit,
    supportsThinking: state.supportsThinking,
    supportsJsonSchema: state.supportsJsonSchema,
    thinkingValue: byId("thinking").value,
    taskMode: state.taskMode,
    taskOptions: state.taskOptions,
    sources: activeSources(),
    samplingOverrides,
    mode,
  });
  limit.placeholder = req.defaultTokens ?? "";
  return req;
}

export function renderBudget() {
  renderComposerFiles();
  const wrap = byId("budget-wrap");
  const budget = byId("context-budget");
  const fill = byId("budget-fill");
  const handoverHint = byId("handover-hint");
  const ringBtn = byId("context-ring-btn");

  const hasCtx = !state.vision && Boolean(state.contextLimit);
  if (wrap) wrap.hidden = !hasCtx;
  if (ringBtn) ringBtn.hidden = !hasCtx;
  if (!hasCtx) return;

  const retry = !byId("prompt").value.trim() && state.active?.messages.at(-1)?.partial;
  const { sent, estimate, replyTokens, thinking, payloadBytes, breakdown } = currentChatRequest(retry ? "retry" : "send");
  const total = estimate + (replyTokens || 0);
  const limit = state.contextLimit;
  const overCtx = total > limit;
  const overBytes = payloadBytes > DEFAULT_MAX_REQUEST_BYTES;
  const over = overCtx || overBytes;
  const frac = total / limit;
  const pct = Math.min(100, Math.round(frac * 100));

  if (fill) {
    fill.style.width = pct + "%";
    fill.toggleAttribute("data-over", over || pct >= 85);
  }

  // Update Rail Context Card (identical breakdown to demo.gif)
  const pctEl = byId("ctx-pct");
  const ratioEl = byId("ctx-ratio");
  const foldedEl = byId("rail-ctx-folded");
  if (pctEl) pctEl.textContent = `${pct}% used`;
  if (ratioEl) ratioEl.textContent = `${fmtK(total)} / ${fmtK(limit)}`;
  if (foldedEl) foldedEl.textContent = `${pct}%`;

  const wOf = val => `${Math.min(100, Math.max(0, (val / limit) * 100)).toFixed(2)}%`;
  const setW = (id, val) => { const el = byId(id); if (el) el.style.width = wOf(val); };
  setW("seg-msg", breakdown.messages);
  setW("seg-src", breakdown.sources);
  setW("seg-sys", breakdown.system);
  setW("seg-rep", breakdown.reply);

  const legend = byId("ctx-legend");
  if (legend) {
    legend.replaceChildren();
    const rows = [
      ["Messages", breakdown.messages, "var(--part-blue)"],
      ["Sources & files", breakdown.sources, "var(--part-orange)"],
      ["System prompt", breakdown.system, "var(--part-gray)"],
      [thinking ? "Reply + think" : "Reply limit", breakdown.reply, "var(--part-green)"],
      ["Free space", breakdown.free, "var(--raise2)"],
    ];
    for (const [name, tokens, color] of rows) {
      const row = document.createElement("div");
      row.className = "kv";
      const sw = document.createElement("span");
      sw.className = "sw";
      sw.style.background = color;
      const k = document.createElement("span");
      k.className = "k muted";
      k.textContent = name;
      const v = document.createElement("span");
      v.className = "v";
      v.textContent = fmtK(tokens);
      const p = document.createElement("span");
      p.className = "pct";
      p.textContent = `${((tokens / limit) * 100).toFixed(1)}%`;
      row.append(sw, k, v, p);
      legend.append(row);
    }
  }

  const payloadEl = byId("ctx-payload");
  if (payloadEl) {
    payloadEl.textContent = `Payload ${formatBytes(payloadBytes)} / ${formatBytes(DEFAULT_MAX_REQUEST_BYTES)}`;
    payloadEl.classList.toggle("over", overBytes);
  }
  if (handoverHint) {
    handoverHint.disabled = state.running || !(state.active?.messages?.length >= 2);
  }

  // Update 16x16 Context Donut Ring in composer under-bar
  drawContextRing(byId("context-ring"), frac, { over, warn: pct >= 75 });
  const summary = `Sending ${sent.length} msg · ~${estimate.toLocaleString()} in + ${(replyTokens || 0).toLocaleString()} max out / ${limit.toLocaleString()} ctx · Payload ${formatBytes(payloadBytes)} / ${formatBytes(DEFAULT_MAX_REQUEST_BYTES)}`;
  if (ringBtn) ringBtn.title = `${pct}% context · ${summary}`;
  if (budget) {
    budget.textContent = summary;
    budget.toggleAttribute("data-over", over);
  }
}
