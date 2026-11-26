import { buildChatRequest } from "./budget.js";
import { copyButton, displayAnswer } from "./markdown.js";
import { collectAllTags, filterConversations } from "./organize.js";
import { PARTIAL_LABELS, THEME_KEY, activeSampling, activeSources, chatTitle, state } from "./state.js";
import { storageStats } from "./storage.js";
import { renderActionItemsCard } from "./tasks/action_items_view.js";
import { DEFAULT_MAX_REQUEST_BYTES, formatBytes } from "./tasks/files.js";
import { buildSamplingOverrides } from "./tasks/sampling.js";
import { renderRewriteReview, renderSummaryMeta } from "./tasks/work_templates_view.js";

export const byId = id => document.getElementById(id);

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const toggle = byId("theme-toggle");
  toggle.innerHTML = theme === "dark" ? "&#9728;" : "&#9790;";
  toggle.title = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
}

export function initTheme() {
  try {
    const savedTheme = localStorage.getItem(THEME_KEY);
    applyTheme(savedTheme === "light" ? "light" : "dark");
  } catch {
    applyTheme("dark");
  }
  byId("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch {}
  });
}

export function setStatus(text, stateName = "checking") {
  const status = byId("status");
  status.textContent = text;
  status.dataset.state = stateName;
}

export function showError(error) {
  const message = error instanceof TypeError
    ? "Local proxy unavailable. Check the proxy terminal."
    : error.message;
  const hint = /cuda.*out of memory|outofmemoryerror/i.test(message)
    ? " Try a smaller model or restart the worker with more GPU memory available."
    : "";
  setStatus(message + hint, "error");
}

let sessionWarned15 = false;

export function renderSessionTime() {
  const label = byId("session-time");
  label.hidden = state.sessionDeadline === undefined;
  if (label.hidden) return;
  const minutes = Math.max(0, Math.ceil((state.sessionDeadline - performance.now()) / 60000));
  const hours = Math.floor(minutes / 60);
  const rem = minutes % 60;
  const formatted = hours ? `${hours}h ${rem}m` : `${minutes} min`;
  label.textContent = `Session left · ${formatted}`;
  label.title = `Estimated session time left · ${minutes} min`;
  label.toggleAttribute("data-low", minutes <= 30);

  const warnBanner = byId("session-warn");
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

export function resizePrompt() {
  const promptField = byId("prompt");
  promptField.style.height = "auto";
  promptField.style.height = Math.min(promptField.scrollHeight, 224) + "px";
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
    promptField.placeholder = "Ask anything or paste text to summarize, edit, or extract… (type / for prompts)";
  }
  byId("history-scope").disabled = state.running || active;
}

export function syncSamplingUI() {
  const cfg = activeSampling();
  byId("sampling-mode").value = cfg.mode;
  byId("samp-temp").value = cfg.temperature;
  byId("samp-top-p").value = cfg.top_p;
  byId("samp-presence").value = cfg.presence_penalty;
  byId("samp-repeat").value = cfg.repeat_penalty;

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

export function showUsage() {
  const lastMsg = state.active?.messages.at(-1);
  const last = lastMsg?.usage;
  const metrics = lastMsg?.metrics;
  const usage = byId("usage");
  usage.hidden = state.vision;
  if (!lastMsg || lastMsg.role !== "assistant") {
    usage.textContent = "";
    return;
  }
  const parts = [];
  if (last) {
    parts.push(`${last.prompt_tokens.toLocaleString()} in · ${last.completion_tokens.toLocaleString()} out`);
  } else {
    parts.push("thiếu dữ liệu usage");
  }
  if (Number.isFinite(metrics?.firstContentMs)) {
    parts.push(`TTFT ${(metrics.firstContentMs / 1000).toFixed(2)}s`);
  }
  if (Number.isFinite(metrics?.totalMs)) {
    parts.push(`Tổng ${(metrics.totalMs / 1000).toFixed(2)}s`);
  }
  if (!lastMsg.partial) {
    if (Number.isFinite(metrics?.workerTokPerSec) && metrics.workerTokPerSec > 0) {
      parts.push(`${metrics.workerTokPerSec.toFixed(1)} tok/s`);
    } else if (last?.completion_tokens > 0 && Number.isFinite(metrics?.totalMs) && metrics.totalMs > 0) {
      const endToEnd = last.completion_tokens / (metrics.totalMs / 1000);
      parts.push(`${endToEnd.toFixed(1)} token/s đầu cuối`);
    }
  }
  usage.textContent = parts.length ? `Last request · ${parts.join(" · ")}` : "";
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
  const head = document.createElement("div");
  head.className = "msg-head";
  const title = document.createElement("h2");
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const text = document.createElement("div");
  article.className = role;
  if (partial) article.dataset.partial = "true";
  if (Number.isInteger(meta.index)) article.dataset.msgIndex = String(meta.index);
  const partialLabel = Object.hasOwn(PARTIAL_LABELS, reason) ? PARTIAL_LABELS[reason] : "Incomplete";
  const isActionItems = meta.task === "action_items";
  title.textContent = role === "user"
    ? (isActionItems ? "You · Action items" : "You")
    : partial
      ? `Assistant · ${partialLabel}`
      : (isActionItems ? "Assistant · Action items" : "Assistant");
  head.append(title);

  if (role === "user" && !live) {
    if (Array.isArray(meta.siblings) && meta.siblings.length > 1 && meta.onSwitchSibling) {
      const activeIdx = Number.isInteger(meta.activeSibling) ? meta.activeSibling : 0;
      actions.append(makeBranchNav(activeIdx, meta.siblings.length, next => meta.onSwitchSibling(meta.index, next)));
    }
    if (meta.onEdit) {
      const editBtn = document.createElement("button");
      editBtn.type = "button";
      editBtn.className = "copy";
      editBtn.textContent = "Edit";
      editBtn.disabled = state.running;
      editBtn.addEventListener("click", () => openInlineEdit(text, content, meta.index, meta.onEdit));
      actions.append(editBtn);
    }
  }

  if (role === "assistant" && !live) {
    if (Array.isArray(meta.variants) && meta.variants.length > 1 && meta.onSwitchVariant) {
      const activeIdx = Number.isInteger(meta.activeVariant) ? meta.activeVariant : 0;
      actions.append(makeBranchNav(activeIdx, meta.variants.length, next => meta.onSwitchVariant(meta.index, next)));
    }
    if (meta.isLast && !partial && meta.onRegenerate) {
      const regenBtn = document.createElement("button");
      regenBtn.type = "button";
      regenBtn.className = "copy";
      regenBtn.textContent = "Regenerate";
      regenBtn.disabled = state.running || !state.canSend;
      regenBtn.addEventListener("click", () => meta.onRegenerate());
      actions.append(regenBtn);
    }
    if (content) {
      actions.append(copyButton(content, isActionItems && meta.structured ? "Copy raw JSON" : "Copy reply"));
    }
  }

  if (actions.childElementCount) head.append(actions);
  article.append(head);

  let thoughts;
  if (role === "assistant") {
    const details = document.createElement("details");
    details.className = "reasoning";
    const summary = document.createElement("summary");
    thoughts = document.createElement("p");
    summary.textContent = "Reasoning";
    thoughts.textContent = reasoning;
    details.hidden = !reasoning;
    details.append(summary, thoughts);
    article.append(details);

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
    } else if (live) {
      const indicator = document.createElement("span");
      indicator.className = "thinking-indicator";
      indicator.textContent = "Generating reply…";
      text.append(indicator);
    }
  } else {
    const paragraph = document.createElement("p");
    paragraph.textContent = content;
    text.append(paragraph);
  }
  article.append(text);
  transcript.append(article);
  return { text, thoughts };
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
  cancelBtn.className = "btn-ghost";
  cancelBtn.textContent = "Hủy";
  cancelBtn.addEventListener("click", () => {
    const p = document.createElement("p");
    p.textContent = originalText;
    container.replaceChildren(p);
  });
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
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
  const container = article?.querySelector(":scope > div:last-child");
  if (!container) return false;
  openInlineEdit(container, msgs[lastUserIdx].content, lastUserIdx, onEdit);
  return true;
}

export function renderStorageMetrics() {
  const el = byId("storage-metrics");
  if (!el) return;
  const bytes = storageStats.lastBytes;
  if (!bytes && !state.conversations.length) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const warn = bytes >= 3.5 * 1024 * 1024 || storageStats.lastWriteMs >= 30;
  el.dataset.warn = String(warn);
  const writeInfo = storageStats.writeCount ? ` · ghi ${storageStats.lastWriteMs.toFixed(1)} ms` : "";
  el.textContent = `Lưu trữ: ${formatBytes(bytes)} · ${state.conversations.length} chat${writeInfo}`
    + (warn ? " · Nên sao lưu v2 hoặc cân nhắc IndexedDB." : "");
}

export function renderHistory(onSelect, onTagFilter) {
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
      if (onTagFilter) onTagFilter();
      else renderHistory(onSelect, onTagFilter);
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
        if (onTagFilter) onTagFilter();
        else renderHistory(onSelect, onTagFilter);
      });
      tagBar.append(chip);
    }
  }

  const matches = filterConversations(state.conversations, {
    query,
    tag: state.activeTagFilter,
    dateRange: state.dateFilter,
  });

  historyList.replaceChildren();
  for (const { chat, snippet } of matches) {
    const button = document.createElement("button");
    button.type = "button";

    const row = document.createElement("span");
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
    const count = document.createElement("span");
    count.className = "history-count";
    count.textContent = String(Math.ceil(chat.messages.length / 2));
    row.append(label, count);
    button.append(row);

    if (chat.tags?.length) {
      const tagsWrap = document.createElement("span");
      tagsWrap.className = "history-tags";
      for (const t of chat.tags.slice(0, 4)) {
        const pill = document.createElement("span");
        pill.className = "history-tag";
        pill.textContent = `#${t}`;
        tagsWrap.append(pill);
      }
      button.append(tagsWrap);
    }

    if (snippet) {
      const snip = document.createElement("span");
      snip.className = "history-snippet";
      const mark = document.createElement("mark");
      mark.textContent = snippet.match;
      snip.append(document.createTextNode(snippet.before), mark, document.createTextNode(snippet.after));
      button.append(snip);
    }

    button.title = label.textContent;
    button.setAttribute("aria-current", String(chat === state.active));
    button.addEventListener("click", () => onSelect(chat));
    historyList.append(button);
  }
  if (!matches.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = (query || state.activeTagFilter || state.dateFilter !== "all")
      ? "No matching conversations."
      : "No saved conversations yet.";
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
  const hasActive = Boolean(state.active);
  byId("rename-chat").hidden = byId("delete-chat").hidden = byId("export-md").hidden
    = byId("pin-chat").hidden = byId("edit-tags").hidden = !hasActive;
  if (hasActive) {
    const isPinned = Boolean(state.active.pinned);
    byId("pin-chat").setAttribute("aria-pressed", String(isPinned));
    byId("pin-chat").textContent = isPinned ? "Bỏ ghim" : "Ghim chat";
    const tagCount = state.active.tags?.length || 0;
    byId("edit-tags").textContent = tagCount ? `Tags (${tagCount})` : "Tags";
  }
  byId("handover-chat").hidden = !(state.active?.messages?.length >= 2);
  showUsage();
  transcript.replaceChildren();
  const msgs = state.active?.messages || [];
  emptyState.hidden = state.vision || msgs.length > 0;
  transcript.hidden = state.vision || msgs.length === 0;
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
  scrollToReply();
  return reply;
}

export function setBusy(busy) {
  state.running = busy;
  byId("run").disabled = byId("run-vision").disabled = busy || !state.canSend;
  byId("refresh").disabled = byId("new-chat").disabled = busy;
  byId("retry").disabled = byId("continue").disabled = busy || !state.canSend;
  byId("stop").hidden = !busy || state.vision;
  byId("stop").disabled = !busy;
  byId("history").inert = busy;
  for (const id of ["rename-chat", "delete-chat", "pin-chat", "edit-tags", "export-md", "handover-chat", "history-search", "history-date-filter", "import-chat", "toggle-action-items", "exit-task-mode", "open-prompts", "attach-file-btn", "pin-note-btn", "sampling-mode", "toggle-sampling-adv", "samp-reset"]) {
    byId(id).disabled = busy;
  }
  byId("prompt").readOnly = busy;
  byId("system-prompt").readOnly = busy;
  byId("max-tokens").readOnly = busy;
  byId("history-scope").disabled = busy || Boolean(state.taskMode);
  byId("thinking").disabled = busy;
  syncSamplingUI();
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
  const wrap = byId("budget-wrap");
  const budget = byId("context-budget");
  const fill = byId("budget-fill");
  const handoverHint = byId("handover-hint");
  wrap.hidden = state.vision || !state.contextLimit;
  if (wrap.hidden) return;
  const retry = !byId("prompt").value.trim() && state.active?.messages.at(-1)?.partial;
  const { sent, estimate, replyTokens, thinking, payloadBytes } = currentChatRequest(retry ? "retry" : "send");
  const total = estimate + replyTokens;
  const maxLabel = thinking ? "max out incl. thinking" : "max out";
  const overCtx = total > state.contextLimit;
  const overBytes = payloadBytes > DEFAULT_MAX_REQUEST_BYTES;
  const over = overCtx || overBytes;
  const pct = Math.min(100, Math.round((total / state.contextLimit) * 100));
  fill.style.width = pct + "%";
  fill.toggleAttribute("data-over", over || pct >= 85);
  const warnings = [];
  if (overCtx) warnings.push("Over context: reduce history or max tokens.");
  if (overBytes) warnings.push(`Over ${formatBytes(DEFAULT_MAX_REQUEST_BYTES)} proxy limit: uncheck sources or shorten range.`);
  budget.textContent = `Sending ${sent.length} msg · ~${estimate.toLocaleString()} in + ${replyTokens.toLocaleString()} ${maxLabel} / ${state.contextLimit.toLocaleString()} ctx · Payload ${formatBytes(payloadBytes)} / ${formatBytes(DEFAULT_MAX_REQUEST_BYTES)}`
    + (warnings.length ? " · " + warnings.join(" ") : "");
  budget.toggleAttribute("data-over", over);
  handoverHint.hidden = !(pct >= 75 && (state.active?.messages?.length || 0) >= 4);
}
