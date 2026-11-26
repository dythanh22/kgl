import { fetchReady, parseRetryAfter, postInference, readChatStream, readImage } from "./api.js";
import {
  attachVariant,
  downloadMarkdown,
  exportThreadMarkdown,
  forkAtUserMessage,
  prepareRegenerate,
  switchAssistantVariant,
  switchUserSibling,
} from "./branches.js";
import { estimateMessageTokens } from "./budget.js";
import { displayAnswer } from "./markdown.js";
import { mergeTabConversations, normalizeTags } from "./organize.js";
import { createReadinessPoller } from "./poller.js";
import { MASCOT_KEY, KEY_STORAGE, STORAGE_KEY, THEME_KEY, activeSampling, activeSources, chatTitle, state } from "./state.js";
import { createDraftSaver, exportHistory, loadHistory, mergeBackup, saveHistory, snapshot, snapshotV1 } from "./storage.js";
import { supportsActionItemsSchema } from "./tasks/action_items.js";
import { renderActionItemsCard } from "./tasks/action_items_view.js";
import {
  DEFAULT_MAX_REQUEST_BYTES,
  buildHandoverMessages,
  formatBytes,
  measurePayloadBytes,
} from "./tasks/files.js";
import { initSourcesUI } from "./tasks/files_ui.js";
import { initPromptLibrary } from "./tasks/prompts_ui.js";
import { buildSamplingOverrides, defaultSampling } from "./tasks/sampling.js";
import {
  applyMascotVisibility,
  applyRailState,
  applySidebarState,
  applyTheme,
  byId,
  currentChatRequest,
  getMascotMood,
  historyNotice,
  initMascot,
  initRail,
  initTheme,
  isNearBottom,
  openEditForLastUser,
  pulseMascotToken,
  renderBudget,
  renderEnvironmentCard,
  renderHistory,
  renderSessionTime,
  renderStorageMetrics,
  renderTranscript,
  resizePrompt,
  scrollToReply,
  setBusy,
  setMascotMood,
  setStatus,
  setTranscriptHandlers,
  showError,
  showToast,
  syncSamplingUI,
  syncSendEnabled,
  syncSystemUI,
  syncTaskModeUI,
  syncThinkingUI,
  updateGreeting,
  updateScrim,
  updateSessionTime,
} from "./ui.js";

const run = byId("run");
const runVision = byId("run-vision");
const refresh = byId("refresh");
const output = byId("output");
const transcript = byId("transcript");
const emptyState = byId("empty-state");
const promptField = byId("prompt");
const systemField = byId("system-prompt");
const newChat = byId("new-chat");
const thinkingField = byId("thinking");
const keyField = byId("key");
const exportModal = byId("export-md-modal");
const settingsModal = byId("settings-modal");
const exportJsonModal = byId("export-json-modal");

initMascot();
initTheme();
initRail();
updateGreeting();
const narrowSidebar = window.matchMedia("(max-width: 820px)");
const floatingRail = window.matchMedia("(max-width: 1160px)");
applySidebarState(!narrowSidebar.matches);
narrowSidebar.addEventListener("change", event => {
  if (event.matches && state.sidebarOpen) {
    const hadFocus = byId("conversations").contains(document.activeElement);
    applySidebarState(false);
    if (hadFocus) byId("sidebar-open").focus();
  }
});
floatingRail.addEventListener("change", event => {
  if (event.matches && state.railOpen) {
    const hadFocus = byId("rail").contains(document.activeElement);
    applyRailState(false, false);
    if (hadFocus) byId("rail-toggle").focus();
  }
});

// Popover positioning helper (anchors above/below trigger button)
const POPOVERS = [
  { menu: byId("chat-menu"), trigger: byId("thread-title-btn"), placement: "below-left" },
  { menu: byId("plus-menu"), trigger: byId("plus-btn"), placement: "above-left" },
  { menu: byId("sampling-pop"), trigger: byId("under-sampling-btn"), placement: "above-right" },
];

function closeAllPopovers() {
  for (const { menu, trigger } of POPOVERS) {
    if (menu && !menu.hidden) {
      if (menu.contains(document.activeElement)) trigger?.focus();
      menu.hidden = true;
    }
    if (trigger) trigger.setAttribute("aria-expanded", "false");
  }
}

function togglePopover(menuEl, triggerEl, placement) {
  const wasOpen = !menuEl.hidden;
  closeAllPopovers();
  if (wasOpen) return;
  menuEl.hidden = false;
  triggerEl.setAttribute("aria-expanded", "true");
  const rect = triggerEl.getBoundingClientRect();
  const mw = menuEl.offsetWidth;
  const mh = menuEl.offsetHeight;
  let left = rect.left;
  let top = rect.bottom + 6;
  if (placement === "above-left") {
    left = rect.left;
    top = rect.top - mh - 8;
  } else if (placement === "above-right") {
    left = rect.right - mw;
    top = rect.top - mh - 8;
  }
  left = Math.max(10, Math.min(left, window.innerWidth - mw - 10));
  top = Math.max(10, Math.min(top, window.innerHeight - mh - 10));
  menuEl.style.left = `${left}px`;
  menuEl.style.top = `${top}px`;
}

for (const { menu, trigger, placement } of POPOVERS) {
  trigger?.addEventListener("click", event => {
    event.stopPropagation();
    if (state.running && trigger !== byId("thread-title-btn")) return;
    togglePopover(menu, trigger, placement);
  });
}

document.addEventListener("pointerdown", event => {
  const insidePopover = POPOVERS.some(({ menu, trigger }) =>
    menu?.contains(event.target) || trigger?.contains(event.target),
  );
  if (!insidePopover) closeAllPopovers();
});

// Idle -> sleepy mascot after 2 minutes of inactivity
let lastActivity = performance.now();
setInterval(() => {
  if (!state.running && getMascotMood() === "idle" && performance.now() - lastActivity > 120000) {
    setMascotMood("sleepy", "");
  }
}, 5000);
for (const ev of ["pointermove", "keydown", "pointerdown", "wheel"]) {
  window.addEventListener(ev, () => {
    lastActivity = performance.now();
    if (!state.running && getMascotMood() === "sleepy") {
      setMascotMood(promptField.value.trim() ? "listening" : "idle", "");
    }
  }, { passive: true });
}

try {
  keyField.value = sessionStorage.getItem(KEY_STORAGE) || "";
} catch {
  // Tab-scoped key works even when sessionStorage is restricted.
}
keyField.addEventListener("input", () => {
  try {
    if (keyField.value) sessionStorage.setItem(KEY_STORAGE, keyField.value);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {}
});
byId("auth-check")?.addEventListener("click", () => check());

function requestKey() {
  keyField.value = "";
  keyField.dispatchEvent(new Event("input"));
  byId("auth").hidden = false;
  state.canSend = false;
  syncSendEnabled();
  runVision.disabled = byId("retry").disabled = byId("continue").disabled = true;
  keyField.focus();
}

setInterval(renderSessionTime, 15000);

const restored = loadHistory();
if (restored) {
  state.conversations = restored.conversations;
  state.active = restored.active;
  state.newDraft = restored.newDraft;
  state.newSystem = restored.newSystem;
  state.newSampling = restored.newSampling || defaultSampling();
  state.newSources = restored.newSources || [];
  state.customPrompts = restored.customPrompts || [];
}
let chatBase = new Map(state.conversations.map(chat => [chat.id, JSON.stringify(chat)]));

promptField.value = state.active?.draft ?? (state.active ? "" : state.newDraft);
syncSystemUI();
syncSamplingUI();
syncThinkingUI();

function readSharedChats() {
  const stored = JSON.parse(localStorage.getItem(STORAGE_KEY));
  return Array.isArray(stored?.conversations) ? stored.conversations : [];
}

function applySharedChats(conversations, activeId) {
  const previous = state.active;
  const changed = conversations.length !== state.conversations.length
    || conversations.some((chat, index) => chat !== state.conversations[index]);
  state.conversations = conversations;
  state.active = conversations.find(chat => chat.id === activeId) || null;
  if (previous !== state.active) {
    syncSelectedChat();
    renderTranscript();
  }
  if (changed || previous !== state.active) renderHistory(selectChat, deleteChatItem);
}

function reportChatConflicts(conflicts) {
  if (conflicts.length) {
    historyNotice("Có xung đột với tab khác. Bản đang mở ở tab này vẫn có thể xuất JSON. Tải lại để dùng bản đã lưu.");
  }
}

function persist(data = snapshot(state)) {
  const start = performance.now();
  let remote;
  try {
    remote = readSharedChats();
  } catch {
    historyNotice("History could not be read. Export a backup before leaving this page.");
    return false;
  }
  const result = mergeTabConversations(data.conversations, remote, chatBase, state.running ? state.active?.id : undefined);
  const active = result.saved.some(chat => chat.id === data.active) ? data.active : undefined;
  if (!saveHistory({ ...data, conversations: result.saved, active }, historyNotice, start)) return false;
  chatBase = result.nextBase;
  const conflicts = new Set(result.conflicts);
  for (const chat of result.saved) {
    if (!conflicts.has(chat.id)) chatBase.set(chat.id, JSON.stringify(chat));
  }
  applySharedChats(result.merged, data.active);
  reportChatConflicts(result.conflicts);
  renderStorageMetrics();
  return true;
}

const draftSaver = createDraftSaver(() => persist(), 350);

function flushDraftAndPersist(data) {
  draftSaver.cancel();
  return persist(data);
}

function deleteChatItem(chat) {
  if (!chat || state.running) return;
  if (!window.confirm(`Xoá hội thoại "${chatTitle(chat)}" và bản nháp của nó?`)) return;
  draftSaver.cancel();
  const kept = state.conversations.filter(c => c !== chat);
  const nextActiveId = state.active === chat ? null : state.active?.id;
  if (!persist({ ...snapshot(state), conversations: kept, active: nextActiveId })) return;
  if (state.active === chat) selectChat(null, false);
  else renderHistory(selectChat, deleteChatItem);
  showToast("Đã xoá hội thoại");
}

function selectChat(chat, shouldPersist = true) {
  closeAllPopovers();
  draftSaver.flush();
  state.active = chat;
  renderHistory(selectChat, deleteChatItem);
  renderTranscript();
  if (shouldPersist) persist();
  syncSelectedChat();
  if (window.innerWidth <= 820) applySidebarState(false);
  if (!chat) setMascotMood("wave", "", { settleMs: 2200 });
  else setMascotMood(promptField.value.trim() ? "listening" : "idle", "");
  promptField.focus();
}

function syncSelectedChat() {
  promptField.value = state.active?.draft ?? (state.active ? "" : state.newDraft);
  syncSystemUI();
  syncSamplingUI();
  syncThinkingUI();
  sourcesUI?.renderSources();
  resizePrompt();
  renderBudget();
}

function importHistory(raw) {
  draftSaver.flush();
  const { data, mergedPrompts, addedCount, skippedCount } = mergeBackup(state, raw);
  if (!persist(data)) return;
  state.customPrompts = mergedPrompts;
  state.newDraft = data.newDraft;
  state.newSystem = data.newSystem;
  state.newSampling = data.newSampling || defaultSampling();
  state.newSources = data.newSources || [];
  selectChat(state.active, false);
  showToast(`Đã nhập ${addedCount} hội thoại (bỏ qua ${skippedCount} trùng ID)`);
}

function applyReadyHealth(health) {
  byId("auth").hidden = true;
  state.health = health;
  state.vision = health.backend === "python";
  state.capabilities = health.capabilities || {};
  state.supportsThinking = !state.vision && health.backend === "llama" && health.capabilities?.enable_thinking === true;
  state.supportsJsonSchema = !state.vision && supportsActionItemsSchema(health);
  byId("thinking-field").hidden = !state.supportsThinking;
  syncThinkingUI();
  syncTaskModeUI();
  syncSamplingUI();

  byId("title").textContent = state.vision ? "Image classifier" : "Kaggle Worker";
  const modelEl = byId("model");
  const shortModel = health.alias || health.model?.split("/").pop() || "model";
  const context = health.n_ctx || health.requested_context;
  state.contextLimit = Number.isSafeInteger(context) && context > 1 ? context : undefined;

  const limit = byId("max-tokens");
  if (state.contextLimit) limit.max = state.contextLimit - 1;
  else {
    limit.removeAttribute("max");
    limit.placeholder = "";
  }

  modelEl.textContent = [shortModel, health.quant].filter(Boolean).join(" · ");
  modelEl.title = health.model || "";
  const underModel = byId("under-model-name");
  if (underModel) underModel.textContent = shortModel;

  byId("chat").hidden = state.vision;
  byId("vision").hidden = !state.vision;
  applySidebarState(!state.vision && state.sidebarOpen);
  applyRailState(!state.vision && state.railOpen, false);
  if (state.vision) {
    byId("system-bar").hidden = true;
    emptyState.hidden = true;
    transcript.hidden = true;
    byId("thread-title").textContent = "Image classification";
  }

  state.canSend = true;
  renderEnvironmentCard();
  if (!state.vision) {
    renderHistory(selectChat, deleteChatItem);
    renderTranscript();
  }
  renderBudget();
  setStatus("Sẵn sàng", "ready");
  if (!state.running) {
    setMascotMood(state.active?.messages?.length ? "idle" : "wave", "", { settleMs: 2400 });
  }
  runVision.disabled = byId("retry").disabled = byId("continue").disabled = false;
  syncSendEnabled();
}

const readinessPoller = createReadinessPoller({
  async probe() {
    const response = await fetchReady(keyField.value);
    const data = await response.json();
    const health = data.health || data.last_health;
    if (health) state.health = health;
    updateSessionTime(data.approx_session_remaining_s);
    if (response.status === 401) {
      requestKey();
      return { kind: "terminal", message: "Nhập khóa API của proxy rồi bấm Kết nối." };
    }
    if (data.state === "connection_lost") {
      return { kind: "terminal", message: "Phiên Kaggle đã mất kết nối. Kết nối lại bằng kgl rồi thử lại." };
    }
    if (data.state === "wrong_instance") {
      return { kind: "terminal", message: "Worker đã thay đổi. Kiểm tra kgl status rồi kết nối lại." };
    }
    if (health?.status === "error") {
      return { kind: "terminal", message: health.error + " Xem kgl serve logs." };
    }
    if (response.ok && data.state === "ready" && health) {
      return { kind: "ready", health };
    }
    return { kind: "busy", retryAfterSec: parseRetryAfter(response) };
  },
  onReady({ health }) {
    applyReadyHealth(health);
  },
  onTerminal({ message, error }) {
    state.canSend = false;
    syncSendEnabled();
    runVision.disabled = byId("retry").disabled = byId("continue").disabled = true;
    if (error) showError(error, { reconnect: byId("auth").hidden });
    else setStatus(message || "Worker chưa sẵn sàng.", "error", { reconnect: byId("auth").hidden });
  },
  onTick(remainingSec) {
    setStatus(`Đang chờ worker · kiểm tra lại (${remainingSec}s)…`, "busy", { waiting: true });
  },
  onTimeout() {
    setStatus("Worker vẫn đang bận sau 30 giây.", "error", { reconnect: true });
  },
});

async function check() {
  if (refresh.disabled || state.running) return;
  readinessPoller.stop();
  state.canSend = false;
  syncSendEnabled();
  runVision.disabled = byId("retry").disabled = byId("continue").disabled = refresh.disabled = true;
  setStatus("Đang kiểm tra kết nối…", "checking");
  setMascotMood("thinking", "Đang kết nối…");
  try {
    const response = await fetchReady(keyField.value);
    const data = await response.json();
    const health = data.health || data.last_health;
    if (health) {
      state.health = health;
      renderEnvironmentCard();
    }
    updateSessionTime(data.approx_session_remaining_s);
    if (response.status === 401) {
      requestKey();
      throw new Error("Nhập khóa API của proxy rồi bấm Kết nối.");
    }
    if (data.state === "busy") {
      setStatus("Đang chờ worker rảnh…", "busy", { waiting: true });
      readinessPoller.start(parseRetryAfter(response) * 1000);
      return;
    }
    if (health?.status === "error") throw new Error(health.error + " Xem kgl serve logs.");
    if (data.state === "connection_lost") throw new Error("Phiên Kaggle đã mất kết nối. Kết nối lại bằng kgl rồi thử lại.");
    if (data.state === "loading" || data.state === "starting") throw new Error("Model đang nạp. Hãy thử lại sau ít phút.");
    if (data.state === "down") throw new Error("Worker chưa sẵn sàng. Hãy kiểm tra kgl serve status và kgl serve logs.");
    if (!response.ok) {
      const detail = data.error?.message || "Worker hoặc thiết bị khác với cấu hình dự kiến.";
      throw new Error(detail + " Kiểm tra kgl status và kết nối lại nếu phiên Kaggle đã hết.");
    }
    applyReadyHealth(health);
  } catch (error) {
    if (error instanceof TypeError) updateSessionTime();
    showError(error, { reconnect: byId("auth").hidden });
  } finally {
    refresh.disabled = false;
  }
}

function createStreamPainter(renderFn, minIntervalMs = 60) {
  let dirty = false;
  let rafId = null;
  let timerId = null;
  let lastPaint = 0;

  function paintNow() {
    rafId = null;
    timerId = null;
    if (!dirty) return;
    dirty = false;
    lastPaint = performance.now();
    renderFn();
  }

  function schedule() {
    dirty = true;
    if (rafId !== null || timerId !== null) return;
    const elapsed = performance.now() - lastPaint;
    if (elapsed >= minIntervalMs) {
      rafId = requestAnimationFrame(paintNow);
    } else {
      timerId = setTimeout(() => {
        timerId = null;
        if (dirty && rafId === null) rafId = requestAnimationFrame(paintNow);
      }, minIntervalMs - elapsed);
    }
  }

  function flush() {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    if (timerId !== null) { clearTimeout(timerId); timerId = null; }
    if (dirty) { dirty = false; renderFn(); }
  }

  return { schedule, flush };
}

async function send({ mode = "send", overridePrevious = null, priorVariants = null } = {}) {
  if (!state.canSend || state.running || refresh.disabled) return;
  closeAllPopovers();
  draftSaver.flush();
  readinessPoller.stop();

  const prompt = promptField.value.trim();
  const hasSelectedSources = activeSources().some(s => s.selected !== false);
  if (!state.vision && mode === "send" && !prompt && !hasSelectedSources) {
    showError(new Error("Enter a message or attach a source file."));
    promptField.focus();
    return;
  }
  const previous = overridePrevious || state.active?.messages || [];
  if (mode === "retry" && !previous.at(-1)?.partial) return;
  if (mode === "continue" && !(previous.at(-1)?.partial && previous.at(-1)?.partialReason === "length" && previous.at(-1)?.content?.trim())) return;

  const {
    messages, payload: chatPayload, payloadBytes, replyTokens, estimate, thinking, task, structured, sourceText, targetWords,
  } = currentChatRequest(mode, previous);

  if (!state.vision && replyTokens !== undefined
      && (!Number.isInteger(replyTokens) || replyTokens < 1 || (state.contextLimit && replyTokens >= state.contextLimit))) {
    showError(new Error("Choose a reply limit smaller than the context window."));
    return;
  }
  if (!state.vision) {
    try {
      buildSamplingOverrides(activeSampling(), state.capabilities);
    } catch (error) {
      showError(error);
      return;
    }
  }
  if (!state.vision && payloadBytes > DEFAULT_MAX_REQUEST_BYTES
      && !window.confirm(`Kích thước JSON thực gửi (${formatBytes(payloadBytes)}) vượt mốc mặc định ${formatBytes(DEFAULT_MAX_REQUEST_BYTES)} của proxy. Hãy bỏ bớt nguồn hoặc thu hẹp khoảng dòng, hoặc bấm OK để vẫn thử gửi.`)) {
    return;
  }
  if (!state.vision && state.contextLimit && estimate + replyTokens > state.contextLimit
      && !window.confirm(`Estimated input and reply limit exceed ${state.contextLimit.toLocaleString()} tokens. Reduce the history or reply limit, or send anyway.`)) {
    return;
  }

  setBusy(true);
  sourcesUI.renderSources();
  state.controller = new AbortController();
  byId("result").hidden = true;
  let message;
  let painter;
  let shouldAutoPoll = false;
  let pollDelayMs = 0;

  try {
    const payload = state.vision
      ? { image: await readImage(byId("image").files[0]) }
      : chatPayload;

    const startTime = performance.now();
    setStatus("Đang xử lý yêu cầu…", "busy");
    setMascotMood(structured ? "tool" : "thinking", "Đang đọc yêu cầu…");
    const response = await postInference(state.vision, payload, keyField.value, state.controller.signal);
    if (!response.ok) {
      const { error } = await response.json();
      if (response.status === 401) requestKey();
      if (response.status === 413) {
        throw new Error("Request vượt giới hạn byte của proxy (mặc định 512 KiB). Hãy bỏ chọn bớt tệp/nguồn ghim, thu hẹp khoảng dòng hoặc giảm phạm vi lịch sử.");
      }
      if (response.status === 503 && error?.type === "worker_busy") {
        shouldAutoPoll = true;
        pollDelayMs = parseRetryAfter(response) * 1000;
        state.canSend = false;
      }
      if (response.status === 503) state.canSend = false;
      throw new Error(error?.message || "Không thể nhận câu trả lời.");
    }
    if (state.vision) {
      const { predictions } = await response.json();
      output.textContent = predictions.map(item => item.label + " · " + (item.score * 100).toFixed(1) + "%").join("\n")
        || "No predictions returned.";
      byId("result").hidden = false;
      setStatus("Sẵn sàng", "ready");
      return;
    }

    const nowIso = new Date().toISOString();
    if (!state.active) {
      state.active = {
        id: crypto.randomUUID(),
        createdAt: nowIso,
        updatedAt: nowIso,
        system: state.newSystem || undefined,
        sampling: state.newSampling.mode !== "preset" ? { ...state.newSampling } : undefined,
        sources: state.newSources.filter(s => s.pinned),
      };
      state.conversations.unshift(state.active);
      state.newDraft = "";
      state.newSources = [];
    } else {
      state.active.updatedAt = nowIso;
      if (mode === "send" && Array.isArray(state.active.sources)) {
        state.active.sources = state.active.sources.filter(s => s.pinned);
      }
    }
    sourcesUI?.renderSources();
    if (mode === "send") {
      state.active.draft = "";
      promptField.value = "";
      resizePrompt();
    }

    let segmentIndex = 0;
    if (mode === "continue") {
      message = messages.at(-1);
      if (!Array.isArray(message.segments) || !message.segments.length) {
        message.segments = [message.content];
      }
      message.segments.push("");
      segmentIndex = message.segments.length - 1;
      message.partial = true;
      message.partialReason = "interrupted";
      state.active.messages = messages;
    } else {
      message = { role: "assistant", content: "", reasoning: "", partial: true, partialReason: "interrupted" };
      if (task === "action_items") {
        message.task = "action_items";
        message.structured = Boolean(structured);
        message.sourceText = sourceText || "";
      } else if (task === "summary") {
        message.task = "summary";
        if (targetWords) message.targetWords = targetWords;
      } else if (task === "rewrite") {
        message.task = "rewrite";
        message.sourceText = sourceText || "";
      }
      if (priorVariants) attachVariant(message, priorVariants);
      state.active.messages = [...messages, message];
    }

    renderHistory(selectChat, deleteChatItem);
    const reply = renderTranscript(message);
    persist();

    let reasoningDirty = false;
    let contentDirty = false;
    let firstContentAt = 0;

    painter = createStreamPainter(() => {
      const nearBottom = isNearBottom();
      if (reasoningDirty) {
        reasoningDirty = false;
        reply.thoughts.parentElement.hidden = false;
        reply.thoughts.textContent = message.reasoning;
        if (reply.thoughtSummary) {
          reply.thoughtSummary.textContent = firstContentAt ? "Suy luận" : "Đang suy luận…";
        }
      }
      if (contentDirty) {
        contentDirty = false;
        if (reply.thoughtSummary && message.reasoning) {
          reply.thoughtSummary.textContent = "Suy luận";
        }
        if (message.task === "action_items") {
          reply.text.replaceChildren(renderActionItemsCard({
            raw: message.content,
            sourceText: message.sourceText,
            partial: false,
            live: true,
            structured: Boolean(message.structured),
          }));
        } else {
          reply.text.replaceChildren(displayAnswer(message.content));
        }
      }
      if (nearBottom) scrollToReply();
    }, 60);

    const { finishReason, timings } = await readChatStream(response, {
      startTime,
      onUsage(usage) {
        message.usage = usage;
      },
      onReasoning(chunk) {
        message.reasoning += chunk;
        reasoningDirty = true;
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(0);
        setMascotMood("thinking", `Đang suy luận · ${elapsed}s`);
        painter.schedule();
      },
      onContent(chunk) {
        const now = performance.now();
        if (!firstContentAt) firstContentAt = now;
        pulseMascotToken();
        setMascotMood(structured ? "tool" : "talking", "Đang trả lời…");
        if (mode === "continue") {
          message.segments[segmentIndex] += chunk;
          message.content = message.segments.join("\n\n");
        } else {
          message.content += chunk;
        }
        contentDirty = true;
        painter.schedule();
      },
      onSaveCheckpoint() {
        persist();
      },
    });
    painter.flush();
    message.metrics = timings;

    if (finishReason === "length") {
      message.partialReason = "length";
      const detail = message.content.trim() ? "Câu trả lời đã chạm giới hạn token." : "Đã chạm giới hạn token trước khi có câu trả lời.";
      const hint = thinking ? " Tăng giới hạn hoặc tắt suy luận rồi thử lại."
        : " Chọn Viết tiếp hoặc chỉnh giới hạn rồi thử lại.";
      throw new Error(detail + hint);
    }
    if (finishReason !== "stop") throw new Error("Câu trả lời chưa hoàn tất. Kiểm tra kết nối trước khi thử lại.");
    if (!message.content.trim()) {
      message.partialReason = "empty";
      throw new Error("Worker kết thúc lượt mà chưa trả lời. Hãy thử lại.");
    }
    message.partial = false;
    delete message.partialReason;
    setStatus("Sẵn sàng", "ready");
    setMascotMood("happy", "", { settleMs: 2200 });
  } catch (error) {
    painter?.flush();
    if (state.controller?.signal.aborted) {
      if (message) message.partialReason = "stopped";
      state.canSend = false;
      shouldAutoPoll = true;
      setStatus("Đã dừng cục bộ · đang tự kiểm tra khi worker rảnh…", "busy", { waiting: true });
      setMascotMood("sorry", "Đã dừng · đang kiểm tra worker…");
    } else if (shouldAutoPoll) {
      setStatus("Đang chờ worker rảnh…", "busy", { waiting: true });
    } else {
      if (error instanceof TypeError) state.canSend = false;
      showError(error, { reconnect: !state.canSend && byId("auth").hidden });
    }
  } finally {
    state.controller = undefined;
    const keepScroll = !isNearBottom();
    const scrollTop = transcript.scrollTop;
    setBusy(false);
    sourcesUI.renderSources();
    if (message) {
      flushDraftAndPersist();
      renderTranscript();
      if (keepScroll) transcript.scrollTop = scrollTop;
    }
    renderBudget();
    if (shouldAutoPoll) {
      readinessPoller.start(pollDelayMs);
    }
  }
}

setTranscriptHandlers({
  onEdit(index, newText) {
    if (!state.active || state.running || !state.canSend) return;
    draftSaver.flush();
    const forked = forkAtUserMessage(state.active.messages, index, newText);
    state.active.messages = forked;
    persist();
    send({ mode: "edit", overridePrevious: forked });
  },
  onSwitchSibling(index, targetIdx) {
    if (!state.active || state.running) return;
    draftSaver.flush();
    state.active.messages = switchUserSibling(state.active.messages, index, targetIdx);
    persist();
    renderHistory(selectChat, deleteChatItem);
    renderTranscript();
    renderBudget();
  },
  onRegenerate() {
    if (!state.active || state.running || !state.canSend) return;
    draftSaver.flush();
    const { prefix, variants } = prepareRegenerate(state.active.messages);
    send({ mode: "regenerate", overridePrevious: prefix, priorVariants: variants });
  },
  onSwitchVariant(index, targetIdx) {
    if (!state.active || state.running) return;
    draftSaver.flush();
    state.active.messages = switchAssistantVariant(state.active.messages, index, targetIdx);
    persist();
    renderTranscript();
    renderBudget();
  },
});

// ======================== EVENTS ========================
byId("form").addEventListener("submit", event => {
  event.preventDefault();
  send({ mode: "send" });
});
promptField.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send({ mode: "send" });
  } else if (event.key === "ArrowUp" && !promptField.value && !state.running && !event.isComposing) {
    if (openEditForLastUser((idx, text) => {
      if (!state.active || state.running || !state.canSend) return;
      const forked = forkAtUserMessage(state.active.messages, idx, text);
      state.active.messages = forked;
      persist();
      send({ mode: "edit", overridePrevious: forked });
    })) {
      event.preventDefault();
    }
  }
});

thinkingField.addEventListener("change", () => {
  syncThinkingUI();
  renderBudget();
});
byId("think-toggle")?.addEventListener("click", () => {
  if (state.running || !state.supportsThinking) return;
  thinkingField.value = thinkingField.value === "on" ? "off" : "on";
  syncThinkingUI();
  renderBudget();
});

byId("retry").addEventListener("click", () => send({ mode: "retry" }));
byId("continue").addEventListener("click", () => send({ mode: "continue" }));
byId("stop").addEventListener("click", () => {
  byId("stop").disabled = true;
  state.controller?.abort();
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") draftSaver.flush();
});
window.addEventListener("pagehide", () => {
  draftSaver.flush();
  if (state.running && !state.vision) persist();
});

byId("session-warn-export").addEventListener("click", () => {
  draftSaver.flush();
  exportHistory(snapshot(state));
});
byId("session-warn-dismiss").addEventListener("click", () => {
  byId("session-warn").hidden = true;
});

// Sidebar & Rail toggles
byId("sidebar-toggle").addEventListener("click", () => applySidebarState(!state.sidebarOpen));
byId("sidebar-open")?.addEventListener("click", () => applySidebarState(true));
byId("rail-toggle")?.addEventListener("click", () => applyRailState(!state.railOpen, true));
byId("context-ring-btn")?.addEventListener("click", () => applyRailState(!state.railOpen, true));
byId("under-model-btn")?.addEventListener("click", () => applyRailState(true, true));
byId("under-sources-btn")?.addEventListener("click", () => {
  if (activeSources().length) applyRailState(true, true);
  else byId("attach-file-btn").click();
});
byId("scrim")?.addEventListener("click", () => {
  if (window.innerWidth <= 820) applySidebarState(false);
  if (window.innerWidth <= 1160) applyRailState(false, false);
});

// Sidebar Search drawer & Recents fold
byId("search-toggle")?.addEventListener("click", () => {
  state.searchOpen = !state.searchOpen;
  byId("search-drawer").hidden = !state.searchOpen;
  byId("search-toggle").setAttribute("aria-expanded", String(state.searchOpen));
  if (state.searchOpen) byId("history-search").focus();
});
byId("recents-toggle")?.addEventListener("click", () => {
  state.recentsOpen = !state.recentsOpen;
  byId("recents-toggle").setAttribute("aria-expanded", String(state.recentsOpen));
  renderHistory(selectChat, deleteChatItem);
});

// System bar
byId("toggle-system").addEventListener("click", () => {
  closeAllPopovers();
  const bar = byId("system-bar");
  bar.hidden = !bar.hidden;
  byId("toggle-system").setAttribute("aria-expanded", String(!bar.hidden));
  if (!bar.hidden) systemField.focus();
});
byId("close-system-bar")?.addEventListener("click", () => {
  byId("system-bar").hidden = true;
  byId("toggle-system").setAttribute("aria-expanded", "false");
});
for (const chip of document.querySelectorAll(".chip[data-sys]")) {
  chip.addEventListener("click", () => {
    if (state.running) return;
    systemField.value = chip.dataset.sys;
    systemField.dispatchEvent(new Event("input"));
    systemField.focus();
  });
}
systemField.addEventListener("input", () => {
  if (state.active) state.active.system = systemField.value;
  else state.newSystem = systemField.value;
  syncSystemUI();
  draftSaver.schedule();
  renderBudget();
});

function setTaskMode(mode) {
  state.taskMode = mode;
  syncTaskModeUI();
  renderBudget();
  promptField.focus();
}

function updateActiveSampling(patch) {
  const next = { ...activeSampling(), ...patch };
  if (state.active) state.active.sampling = next;
  else state.newSampling = next;
  syncSamplingUI();
  persist();
}

byId("sampling-mode").addEventListener("change", event => {
  const mode = event.target.value;
  updateActiveSampling({ mode });
  if (mode === "custom") {
    byId("sampling-advanced").hidden = false;
    byId("toggle-sampling-adv").setAttribute("aria-expanded", "true");
  }
});
byId("sampling-pop")?.addEventListener("click", event => {
  const btn = event.target.closest("[data-samp-mode]");
  if (!btn || state.running) return;
  const mode = btn.dataset.sampMode;
  closeAllPopovers();
  updateActiveSampling({ mode });
  if (mode === "custom") {
    applyRailState(true, true);
    const genCard = byId("rail-gen-card");
    if (genCard) genCard.open = true;
    byId("sampling-advanced").hidden = false;
    byId("toggle-sampling-adv").setAttribute("aria-expanded", "true");
  }
});
byId("plus-menu")?.addEventListener("click", event => {
  const taskBtn = event.target.closest("[data-pick-task]");
  if (taskBtn) {
    closeAllPopovers();
    if (!state.running) setTaskMode(taskBtn.dataset.pickTask);
    return;
  }
  if (event.target.closest("button")) closeAllPopovers();
});

byId("toggle-sampling-adv").addEventListener("click", () => {
  const panel = byId("sampling-advanced");
  panel.hidden = !panel.hidden;
  byId("toggle-sampling-adv").setAttribute("aria-expanded", String(!panel.hidden));
});
for (const [id, key] of [
  ["samp-temp", "temperature"],
  ["samp-top-p", "top_p"],
  ["samp-presence", "presence_penalty"],
  ["samp-repeat", "repeat_penalty"],
]) {
  byId(id).addEventListener("input", event => {
    updateActiveSampling({ mode: "custom", [key]: event.target.value });
  });
}
byId("samp-reset").addEventListener("click", () => {
  const reset = defaultSampling();
  if (state.active) delete state.active.sampling;
  else state.newSampling = reset;
  syncSamplingUI();
  persist();
});

byId("summary-words").addEventListener("input", event => {
  state.taskOptions.targetWords = event.target.value;
  renderBudget();
});
for (const [id, key] of [
  ["rewrite-audience", "audience"],
  ["rewrite-goal", "goal"],
  ["rewrite-tone", "tone"],
]) {
  byId(id).addEventListener("change", event => {
    state.taskOptions[key] = event.target.value;
    renderBudget();
  });
}

const promptLib = initPromptLibrary({
  promptField,
  state,
  onPersist: () => flushDraftAndPersist(),
  onSetTaskMode: setTaskMode,
  onOpen: closeAllPopovers,
  onDraftChanged: () => {
    if (state.active) state.active.draft = promptField.value;
    else state.newDraft = promptField.value;
    resizePrompt();
    draftSaver.schedule();
    renderBudget();
  },
});

const sourcesUI = initSourcesUI({
  state,
  onPersist: () => flushDraftAndPersist(),
  onBudgetChange: renderBudget,
  onError: showError,
});
sourcesUI.renderSources();

// Quick links inside Right Rail Sources card + composer chip removals
byId("rail-add-file")?.addEventListener("click", () => byId("attach-file-btn").click());
byId("rail-pin-note")?.addEventListener("click", () => byId("pin-note-btn").click());
byId("composer-files")?.addEventListener("kgl:sources-changed", () => {
  flushDraftAndPersist();
  sourcesUI.renderSources();
  renderBudget();
});

// ======================== SETTINGS 2-COLUMN MODAL ========================
function openSettings(pane = "appearance") {
  closeAllPopovers();
  const urlEl = byId("api-base-url");
  if (urlEl) urlEl.textContent = `${location.origin}/v1`;
  selectSettingsPane(pane);
  settingsModal.showModal();
}
function selectSettingsPane(pane) {
  for (const btn of byId("settings-nav").querySelectorAll("[data-spage]")) {
    btn.classList.toggle("on", btn.dataset.spage === pane);
  }
  for (const sec of settingsModal.querySelectorAll(".spane")) {
    sec.hidden = sec.dataset.pane !== pane;
  }
}
byId("open-settings")?.addEventListener("click", () => openSettings("appearance"));
byId("settings-close")?.addEventListener("click", () => { settingsModal.close(); });
settingsModal?.addEventListener("pointerdown", event => {
  if (event.target === settingsModal) settingsModal.close();
});
byId("settings-nav")?.addEventListener("click", event => {
  const btn = event.target.closest("[data-spage]");
  if (btn) selectSettingsPane(btn.dataset.spage);
});
byId("theme-seg")?.addEventListener("click", event => {
  const btn = event.target.closest("[data-theme-val]");
  if (!btn) return;
  applyTheme(btn.dataset.themeVal);
  try { localStorage.setItem(THEME_KEY, btn.dataset.themeVal); } catch {}
});
byId("sw-mascot")?.addEventListener("click", () => {
  state.mascotEnabled = !state.mascotEnabled;
  applyMascotVisibility();
  try { localStorage.setItem(MASCOT_KEY, String(state.mascotEnabled)); } catch {}
});
byId("settings-open-prompts")?.addEventListener("click", () => {
  settingsModal.close();
  byId("open-prompts").click();
});
byId("settings-open-system")?.addEventListener("click", () => {
  settingsModal.close();
  byId("system-bar").hidden = false;
  byId("toggle-system").setAttribute("aria-expanded", "true");
  systemField.focus();
});
byId("copy-api-url")?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(byId("api-base-url").textContent);
    showToast("Đã sao chép Base URL");
  } catch {
    showToast("Không thể sao chép tự động");
  }
});
byId("settings-export-btn")?.addEventListener("click", () => {
  settingsModal.close();
  byId("export-chat").click();
});
byId("settings-import-btn")?.addEventListener("click", () => {
  settingsModal.close();
  byId("import-chat").click();
});
byId("settings-show-auth")?.addEventListener("click", () => {
  settingsModal.close();
  byId("auth").hidden = false;
  keyField.focus();
});

// ======================== HANDOVER MODAL ========================
const handoverModal = byId("handover-modal");
const handoverTurns = byId("handover-turns");
const handoverBudget = byId("handover-budget");
const handoverStatus = byId("handover-status");
const handoverDraft = byId("handover-draft");

function currentHandoverRequest() {
  const msgs = buildHandoverMessages(state.active.messages || [], activeSources(), Number(handoverTurns.value));
  const est = 64 + estimateMessageTokens(msgs[0].content);
  const payload = { messages: msgs, stream: false, max_tokens: 1024 };
  if (state.supportsThinking) payload.chat_template_kwargs = { enable_thinking: false };
  return { payload, est, bytes: measurePayloadBytes(payload) };
}

function updateHandoverBudget() {
  if (!state.active) return;
  const { est, bytes } = currentHandoverRequest();
  const limit = state.contextLimit;
  const over = (limit && est + 1024 > limit) || bytes > DEFAULT_MAX_REQUEST_BYTES;
  handoverBudget.textContent = limit
    ? `~${est.toLocaleString()} in + 1,024 out / ${limit.toLocaleString()} ctx`
    : `~${est.toLocaleString()} in`;
  handoverBudget.textContent += ` · ${formatBytes(bytes)}` + (over ? " · Vượt giới hạn, hãy chọn ít lượt hơn." : "");
  byId("handover-generate").disabled = Boolean(over) || state.running || !state.canSend;
}

function openHandoverModal() {
  closeAllPopovers();
  if (!state.active || state.running) return;
  handoverStatus.hidden = true;
  handoverDraft.value = "";
  handoverModal.showModal();
  updateHandoverBudget();
}

byId("handover-chat").addEventListener("click", openHandoverModal);
byId("handover-hint").addEventListener("click", openHandoverModal);
handoverTurns.addEventListener("change", updateHandoverBudget);
byId("handover-cancel").addEventListener("click", () => { handoverModal.close(); });

byId("handover-generate").addEventListener("click", async () => {
  if (!state.active || state.running || !state.canSend) return;
  const { payload, est, bytes } = currentHandoverRequest();
  if ((state.contextLimit && est + payload.max_tokens > state.contextLimit) || bytes > DEFAULT_MAX_REQUEST_BYTES) {
    updateHandoverBudget();
    return;
  }
  byId("handover-generate").disabled = true;
  handoverStatus.textContent = "Đang tạo bản bàn giao từ Qwen…";
  handoverStatus.hidden = false;
  setMascotMood("tool", "Đang tạo bản bàn giao…");
  try {
    const response = await postInference(false, payload, keyField.value);
    if (!response.ok) {
      const { error } = await response.json();
      throw new Error(error?.message || `HTTP ${response.status}`);
    }
    const data = await response.json();
    const choice = data.choices?.[0];
    const content = choice?.message?.content?.trim() || "";
    if (!content) throw new Error("Worker không trả về nội dung bàn giao.");
    handoverDraft.value = content;
    handoverStatus.textContent = choice?.finish_reason === "length"
      ? "Cảnh báo: Bản bàn giao chạm giới hạn token. Hãy rà soát/bổ sung trước khi mở chat mới."
      : "Đã tạo xong. Hãy đọc, chỉnh sửa nếu cần rồi bấm 'Xác nhận mở chat mới'.";
    setMascotMood("happy", "", { settleMs: 1800 });
  } catch (error) {
    handoverStatus.textContent = `Không thể tạo bản bàn giao (${error.message}). Hội thoại gốc vẫn được giữ nguyên.`;
    setMascotMood("sorry", "", { settleMs: 2500 });
  } finally {
    updateHandoverBudget();
  }
});

byId("handover-confirm").addEventListener("click", () => {
  const text = handoverDraft.value.trim();
  if (!text) {
    handoverStatus.textContent = "Bản bàn giao đang trống. Bấm 'Tạo bản bàn giao' hoặc nhập nội dung trước khi xác nhận.";
    handoverStatus.hidden = false;
    return;
  }
  const orig = state.active;
  const pinnedCopies = (orig?.sources || [])
    .filter(s => s.pinned && s.selected !== false)
    .map(s => ({ ...s, id: crypto.randomUUID() }));
  const nextChat = {
    id: crypto.randomUUID(),
    title: `Bàn giao · ${chatTitle(orig)}`,
    parentId: orig?.id,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    system: orig?.system,
    sampling: orig?.sampling ? { ...orig.sampling } : undefined,
    sources: pinnedCopies,
    messages: [
      { role: "user", content: text },
      { role: "assistant", content: "Đã nhận bản ghi chú bàn giao. Bạn muốn tiếp tục công việc từ mục nào?" },
    ],
  };
  state.conversations.unshift(nextChat);
  handoverModal.close();
  selectChat(nextChat, true);
  showToast("Đã mở hội thoại mới từ bản bàn giao");
});

byId("toggle-action-items").addEventListener("click", () => {
  closeAllPopovers();
  if (state.running) return;
  setTaskMode(state.taskMode === "action_items" ? null : "action_items");
});
byId("exit-task-mode").addEventListener("click", () => {
  if (state.running) return;
  setTaskMode(null);
});
for (const starter of document.querySelectorAll(".starter[data-task]")) {
  starter.addEventListener("click", () => {
    if (state.running) return;
    setTaskMode(starter.dataset.task);
  });
}
for (const starter of document.querySelectorAll(".starter[data-prompt]")) {
  starter.addEventListener("click", () => {
    setTaskMode(null);
    promptField.value = starter.dataset.prompt;
    promptField.dispatchEvent(new Event("input"));
    promptField.focus();
    promptField.setSelectionRange(promptField.value.length, promptField.value.length);
  });
}

// Export single thread as Markdown (.md)
byId("export-md").addEventListener("click", () => {
  closeAllPopovers();
  if (!state.active) return;
  const msgs = state.active.messages || [];
  const partialCount = msgs.filter(m => m.partial).length;
  byId("export-md-scope").textContent = `Hội thoại "${chatTitle(state.active)}" · ${msgs.length} tin nhắn trong nhánh hiện tại`
    + (partialCount ? ` (${partialCount} lượt dở).` : ".");
  byId("export-md-reasoning").checked = false;
  exportModal.showModal();
  byId("export-md-confirm").focus();
});
byId("export-md-cancel").addEventListener("click", () => { exportModal.close(); });
byId("export-md-confirm").addEventListener("click", () => {
  if (!state.active) return;
  const includeReasoning = byId("export-md-reasoning").checked;
  const md = exportThreadMarkdown(state.active, { includeReasoning });
  downloadMarkdown(chatTitle(state.active), md);
  exportModal.close();
  showToast("Đã tải tệp .md");
});

// Close prompt library backdrop on outside click
byId("prompt-library")?.addEventListener("pointerdown", event => {
  if (event.target === byId("prompt-library")) promptLib.closeLibrary();
});

// Global keyboard shortcuts
document.addEventListener("keydown", event => {
  if (event.isComposing || document.querySelector(".modal-backdrop[open]")) return;
  const mod = event.ctrlKey || event.metaKey;
  if (mod && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (!state.vision) {
      applySidebarState(true);
      state.searchOpen = true;
      byId("search-drawer").hidden = false;
      byId("search-toggle").setAttribute("aria-expanded", "true");
      byId("history-search").focus();
      byId("history-search").select();
    }
    return;
  }
  if (mod && event.shiftKey && event.key.toLowerCase() === "o") {
    if (!state.running && !state.vision) {
      event.preventDefault();
      selectChat(null);
    }
    return;
  }
  if (event.key === "Escape") {
    if (POPOVERS.some(({ menu }) => menu && !menu.hidden)) {
      closeAllPopovers();
      return;
    }
    if (!byId("slash-menu").hidden) {
      promptLib.hideSlashMenu();
      return;
    }
    if (narrowSidebar.matches && state.sidebarOpen) {
      applySidebarState(false);
      byId("sidebar-open").focus();
      event.preventDefault();
      return;
    }
    if (floatingRail.matches && state.railOpen) {
      applyRailState(false, false);
      byId("rail-toggle").focus();
      event.preventDefault();
      return;
    }
    if (state.running && !state.vision) {
      event.preventDefault();
      byId("stop").click();
    }
  }
});

transcript.addEventListener("scroll", () => {
  const dist = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
  byId("scroll-bottom").hidden = dist < 120;
});
byId("scroll-bottom").addEventListener("click", scrollToReply);

byId("image").addEventListener("change", () => {
  const preview = byId("vision-preview");
  const file = byId("image").files[0];
  if (!file) {
    preview.hidden = true;
    preview.removeAttribute("src");
    return;
  }
  const url = URL.createObjectURL(file);
  preview.onload = () => URL.revokeObjectURL(url);
  preview.src = url;
  preview.hidden = false;
});

newChat.addEventListener("click", () => selectChat(null));
byId("history-search").addEventListener("input", () => renderHistory(selectChat, deleteChatItem));
byId("history-date-filter").addEventListener("change", event => {
  state.dateFilter = event.target.value;
  renderHistory(selectChat, deleteChatItem);
});
byId("pin-chat").addEventListener("click", () => {
  closeAllPopovers();
  if (!state.active || state.running) return;
  state.active.pinned = !state.active.pinned;
  state.active.updatedAt = new Date().toISOString();
  renderHistory(selectChat, deleteChatItem);
  renderTranscript();
  flushDraftAndPersist();
});
byId("edit-tags").addEventListener("click", () => {
  closeAllPopovers();
  if (!state.active || state.running) return;
  const current = (state.active.tags || []).join(", ");
  const entered = window.prompt("Nhập các tag, phân cách bằng dấu phẩy (vd: họp, báo-cáo, qwen):", current);
  if (entered === null) return;
  const tags = normalizeTags(entered.split(","));
  if (tags.length) state.active.tags = tags;
  else delete state.active.tags;
  state.active.updatedAt = new Date().toISOString();
  renderHistory(selectChat, deleteChatItem);
  renderTranscript();
  flushDraftAndPersist();
});
byId("rename-chat").addEventListener("click", () => {
  closeAllPopovers();
  const title = window.prompt("Tên hội thoại", chatTitle(state.active));
  if (title === null || !title.trim()) return;
  state.active.title = title.trim();
  state.active.updatedAt = new Date().toISOString();
  renderHistory(selectChat, deleteChatItem);
  renderTranscript();
  flushDraftAndPersist();
});

byId("export-chat").addEventListener("click", () => {
  draftSaver.flush();
  byId("export-json-scope").textContent = `Hiện có ${state.conversations.length} hội thoại và ${state.customPrompts.length} mẫu prompt cá nhân.`;
  exportJsonModal.showModal();
});
byId("export-v2-btn").addEventListener("click", () => {
  exportHistory(snapshot(state), "kgl-conversations-v2.json");
  exportJsonModal.close();
  showToast("Đã xuất backup v2");
});
byId("export-v1-btn").addEventListener("click", () => {
  exportHistory(snapshotV1(state), "kgl-conversations-v1.json");
  exportJsonModal.close();
  showToast("Đã xuất backup v1");
});
byId("export-json-cancel").addEventListener("click", () => { exportJsonModal.close(); });

// Two-tab synchronization & conflict detection
window.addEventListener("storage", event => {
  if (event.key !== STORAGE_KEY) return;
  try {
    const { merged, conflicts, nextBase } = mergeTabConversations(
      state.conversations, readSharedChats(), chatBase, state.running ? state.active?.id : undefined,
    );
    chatBase = nextBase;
    applySharedChats(merged, state.active?.id);
    reportChatConflicts(conflicts);
  } catch {}
});

byId("import-chat").addEventListener("click", () => byId("import-file").click());
byId("import-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  try {
    const raw = JSON.parse(await file.text());
    if (state.running) throw new Error("Wait for the current reply before importing.");
    importHistory(raw);
  } catch (error) {
    historyNotice(error.message);
  }
});
byId("delete-chat").addEventListener("click", () => {
  closeAllPopovers();
  deleteChatItem(state.active);
});

byId("history-scope").addEventListener("change", renderBudget);
byId("max-tokens").addEventListener("input", renderBudget);
promptField.addEventListener("input", () => {
  if (state.active) state.active.draft = promptField.value;
  else state.newDraft = promptField.value;
  resizePrompt();
  draftSaver.schedule();
  renderBudget();
  if (!state.running) {
    setMascotMood(promptField.value.trim() ? "listening" : "idle", "");
  }
});
refresh.addEventListener("click", check);
byId("reconnect").addEventListener("click", check);
window.addEventListener("resize", () => updateScrim());

// Initial paint
renderHistory(selectChat, deleteChatItem);
renderTranscript();
renderEnvironmentCard();
resizePrompt();
check();
