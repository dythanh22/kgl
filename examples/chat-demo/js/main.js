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
import { displayAnswer } from "./markdown.js";
import { mergeTabConversations, normalizeTags } from "./organize.js";
import { createReadinessPoller } from "./poller.js";
import { KEY_STORAGE, STORAGE_KEY, activeSampling, activeSources, chatTitle, state } from "./state.js";
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
import { estimateMessageTokens } from "./budget.js";
import {
  byId,
  currentChatRequest,
  historyNotice,
  initTheme,
  isNearBottom,
  openEditForLastUser,
  renderBudget,
  renderHistory,
  renderSessionTime,
  renderStorageMetrics,
  renderTranscript,
  resizePrompt,
  scrollToReply,
  setBusy,
  setStatus,
  setTranscriptHandlers,
  showError,
  syncSamplingUI,
  syncSystemUI,
  syncTaskModeUI,
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

initTheme();

try {
  keyField.value = sessionStorage.getItem(KEY_STORAGE) || "";
} catch {
  // A key entered in this tab still works when browser storage is disabled.
}
keyField.addEventListener("input", () => {
  try {
    if (keyField.value) sessionStorage.setItem(KEY_STORAGE, keyField.value);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    // Keep using the input value for this page.
  }
});

function requestKey() {
  keyField.value = "";
  keyField.dispatchEvent(new Event("input"));
  byId("auth").hidden = false;
  state.canSend = false;
  run.disabled = runVision.disabled = byId("retry").disabled = byId("continue").disabled = true;
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
  if (changed || previous !== state.active) renderHistory(selectChat);
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

function selectChat(chat, shouldPersist = true) {
  draftSaver.flush();
  state.active = chat;
  renderHistory(selectChat);
  renderTranscript();
  if (shouldPersist) persist();
  syncSelectedChat();
  promptField.focus();
}

function syncSelectedChat() {
  promptField.value = state.active?.draft ?? (state.active ? "" : state.newDraft);
  syncSystemUI();
  syncSamplingUI();
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
  historyNotice(`Imported ${addedCount} · skipped ${skippedCount} existing.`);
}

function applyReadyHealth(health) {
  byId("auth").hidden = true;
  state.vision = health.backend === "python";
  state.capabilities = health.capabilities || {};
  state.supportsThinking = !state.vision && health.backend === "llama" && health.capabilities?.enable_thinking === true;
  state.supportsJsonSchema = !state.vision && supportsActionItemsSchema(health);
  byId("thinking-field").hidden = !state.supportsThinking;
  syncTaskModeUI();
  syncSamplingUI();
  byId("title").textContent = state.vision ? "Image classification" : "kgl workspace chat";
  const model = byId("model");
  const devices = health.devices?.join(", ") || health.device;
  const context = health.n_ctx || health.requested_context;
  const contextLabel = context
    ? context.toLocaleString() + " ctx" + (health.n_ctx ? "" : " req")
    : "";
  (state.vision ? document.querySelector(".brand-meta") : byId("environment-info")).prepend(model);
  model.title = health.model || "";
  state.contextLimit = Number.isSafeInteger(context) && context > 1 ? context : undefined;
  const limit = byId("max-tokens");
  if (state.contextLimit) limit.max = state.contextLimit - 1;
  else {
    limit.removeAttribute("max");
    limit.placeholder = "";
  }
  model.textContent = [health.model?.split("/").pop(), devices, contextLabel]
    .filter(Boolean).join(" · ");
  model.hidden = false;
  byId("worker-memory").textContent = (health.gpu_memory || []).map(
    sample => `GPU ${sample.gpu} · ${sample.memory_mib} MiB at load`
  ).join(" · ");
  byId("chat").hidden = state.vision;
  byId("vision").hidden = !state.vision;
  byId("sidebar-toggle").hidden = state.vision;
  byId("conversations").hidden = state.vision || compactWorkspace.matches;
  byId("generation").hidden = state.vision;
  byId("thread-head").hidden = state.vision;
  if (state.vision) {
    byId("system-bar").hidden = true;
    emptyState.hidden = true;
    transcript.hidden = true;
  }
  byId("usage").hidden = state.vision;
  byId("panel").hidden = false;
  state.canSend = true;
  if (!state.vision) {
    renderHistory(selectChat);
    renderTranscript();
  }
  renderBudget();
  setStatus("Ready", "ready");
  run.disabled = runVision.disabled = byId("retry").disabled = byId("continue").disabled = false;
}

const readinessPoller = createReadinessPoller({
  async probe() {
    const response = await fetchReady(keyField.value);
    const data = await response.json();
    const health = data.health || data.last_health;
    updateSessionTime(data.approx_session_remaining_s);
    if (response.status === 401) {
      requestKey();
      return { kind: "terminal", message: "Enter your proxy API key, then check the connection." };
    }
    if (data.state === "connection_lost") {
      return { kind: "terminal", message: "Kaggle session unavailable. Reconnect kgl and check again." };
    }
    if (data.state === "wrong_instance") {
      return { kind: "terminal", message: "Worker instance changed. Check kgl status and reconnect." };
    }
    if (health?.status === "error") {
      return { kind: "terminal", message: health.error + " See kgl serve logs." };
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
    run.disabled = runVision.disabled = byId("retry").disabled = byId("continue").disabled = true;
    if (error) showError(error);
    else setStatus(message || "Worker unavailable.", "error");
  },
  onTick(remainingSec) {
    setStatus(`Worker busy · tự kiểm tra lại (${remainingSec}s)…`, "busy");
  },
  onTimeout() {
    setStatus("Worker vẫn đang bận sau 30s. Bấm Check connection để kiểm tra lại.", "error");
  },
});

async function check() {
  readinessPoller.stop();
  state.canSend = false;
  run.disabled = runVision.disabled = byId("retry").disabled = byId("continue").disabled = refresh.disabled = true;
  setStatus("Checking model…", "checking");
  try {
    const response = await fetchReady(keyField.value);
    const data = await response.json();
    const health = data.health || data.last_health;
    updateSessionTime(data.approx_session_remaining_s);
    if (response.status === 401) {
      requestKey();
      throw new Error("Enter your proxy API key, then check the connection.");
    }
    if (data.state === "busy") {
      setStatus("Worker busy · đang tự kiểm tra lại…", "busy");
      readinessPoller.start(parseRetryAfter(response) * 1000);
      return;
    }
    if (health?.status === "error") throw new Error(health.error + " See kgl serve logs.");
    if (data.state === "connection_lost") throw new Error("Kaggle session unavailable. Reconnect kgl and check again.");
    if (data.state === "loading" || data.state === "starting") throw new Error("Loading model. Check again in a moment.");
    if (data.state === "down") throw new Error("Worker unavailable. Check kgl serve status and kgl serve logs.");
    if (!response.ok) {
      const detail = data.error?.message || "Worker identity or device does not match.";
      throw new Error(detail + " Check kgl status and reconnect if the Kaggle session ended.");
    }
    applyReadyHealth(health);
  } catch (error) {
    if (error instanceof TypeError) updateSessionTime();
    showError(error);
  } finally {
    refresh.disabled = false;
  }
}

// Coalesce SSE DOM paints via rAF + 60ms interval while preserving exact delta order.
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
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (timerId !== null) {
      clearTimeout(timerId);
      timerId = null;
    }
    if (dirty) {
      dirty = false;
      renderFn();
    }
  }

  return { schedule, flush };
}

async function send({ mode = "send", overridePrevious = null, priorVariants = null } = {}) {
  if (!state.canSend || state.running || refresh.disabled) return;
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
    byId("stream-phase").textContent = "Waiting for worker…";
    setStatus("Running inference…", "busy");
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
      const hint = response.status === 503 ? " Đang tự kiểm tra trạng thái worker…" : "";
      throw new Error((error?.message || "Inference failed.") + hint);
    }
    if (state.vision) {
      const { predictions } = await response.json();
      output.textContent = predictions.map(item => item.label + " · " + (item.score * 100).toFixed(1) + "%").join("\n")
        || "No predictions returned.";
      byId("result").hidden = false;
      setStatus("Ready", "ready");
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

    renderHistory(selectChat);
    const reply = renderTranscript(message);
    persist();

    let reasoningDirty = false;
    let contentDirty = false;
    painter = createStreamPainter(() => {
      const nearBottom = isNearBottom();
      if (reasoningDirty) {
        reasoningDirty = false;
        reply.thoughts.parentElement.hidden = false;
        reply.thoughts.textContent = message.reasoning;
      }
      if (contentDirty) {
        contentDirty = false;
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
        if (!message.reasoning) byId("stream-phase").textContent = "Thinking…";
        message.reasoning += chunk;
        reasoningDirty = true;
        painter.schedule();
      },
      onContent(chunk) {
        if (byId("stream-phase").textContent !== "Writing…") byId("stream-phase").textContent = "Writing…";
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
      const detail = message.content.trim() ? "Reply reached the token limit." : "Token limit reached before an answer.";
      const hint = thinking ? " Increase the limit or turn off Thinking before retrying."
        : " Use Continue or adjust the limit before retrying.";
      throw new Error(detail + hint);
    }
    if (finishReason !== "stop") throw new Error("Reply is incomplete. Check the connection before retrying.");
    if (!message.content.trim()) {
      message.partialReason = "empty";
      throw new Error("The worker ended the turn without an answer. Retry the request.");
    }
    message.partial = false;
    delete message.partialReason;
    setStatus("Ready", "ready");
  } catch (error) {
    painter?.flush();
    if (state.controller?.signal.aborted) {
      if (message) message.partialReason = "stopped";
      state.canSend = false;
      shouldAutoPoll = true;
      setStatus("Đã dừng cục bộ · đang tự kiểm tra khi worker rảnh…", "busy");
    } else {
      showError(error);
    }
  } finally {
    byId("stream-phase").textContent = "";
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
    renderHistory(selectChat);
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

// Events
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
thinkingField.addEventListener("change", renderBudget);
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

const compactWorkspace = window.matchMedia("(max-width: 52rem)");
function showWorkerRail(open) {
  byId("worker-rail").hidden = !open;
  byId("rail-toggle").setAttribute("aria-expanded", String(open));
}
showWorkerRail(!compactWorkspace.matches);
compactWorkspace.addEventListener("change", event => {
  showWorkerRail(!event.matches);
  byId("conversations").hidden = event.matches || byId("thread-head").hidden;
});
byId("rail-toggle").addEventListener("click", () => {
  if (compactWorkspace.matches) byId("conversations").hidden = true;
  showWorkerRail(byId("worker-rail").hidden);
});
document.querySelector(".thread-actions").addEventListener("click", event => {
  if (event.target.closest("button")) document.querySelector(".thread-menu").open = false;
});
byId("sidebar-toggle").addEventListener("click", () => {
  if (compactWorkspace.matches) showWorkerRail(false);
  byId("conversations").hidden = !byId("conversations").hidden;
});
byId("toggle-system").addEventListener("click", () => {
  const bar = byId("system-bar");
  bar.hidden = !bar.hidden;
  byId("toggle-system").setAttribute("aria-expanded", String(!bar.hidden));
  if (!bar.hidden) systemField.focus();
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

// Handover / summarize into a new linked chat
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
  if (!state.active || state.running) return;
  handoverStatus.hidden = true;
  handoverDraft.value = "";
  handoverModal.hidden = false;
  updateHandoverBudget();
}

byId("handover-chat").addEventListener("click", openHandoverModal);
byId("handover-hint").addEventListener("click", openHandoverModal);
handoverTurns.addEventListener("change", updateHandoverBudget);
byId("handover-cancel").addEventListener("click", () => { handoverModal.hidden = true; });

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
  } catch (error) {
    handoverStatus.textContent = `Không thể tạo bản bàn giao (${error.message}). Hội thoại gốc vẫn được giữ nguyên.`;
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
    system: orig?.system,
    sampling: orig?.sampling ? { ...orig.sampling } : undefined,
    sources: pinnedCopies,
    messages: [
      { role: "user", content: text },
      { role: "assistant", content: "Đã nhận bản ghi chú bàn giao. Bạn muốn tiếp tục công việc từ mục nào?" },
    ],
  };
  state.conversations.unshift(nextChat);
  handoverModal.hidden = true;
  selectChat(nextChat, true);
  historyNotice("Đã mở hội thoại mới từ bản bàn giao (hội thoại gốc vẫn được giữ nguyên).");
});

byId("toggle-action-items").addEventListener("click", () => {
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
  if (!state.active) return;
  const msgs = state.active.messages || [];
  const partialCount = msgs.filter(m => m.partial).length;
  byId("export-md-scope").textContent = `Hội thoại "${chatTitle(state.active)}" · ${msgs.length} tin nhắn trong nhánh hiện tại`
    + (partialCount ? ` (${partialCount} lượt dở).` : ".");
  byId("export-md-reasoning").checked = false;
  exportModal.hidden = false;
  byId("export-md-confirm").focus();
});
byId("export-md-cancel").addEventListener("click", () => {
  exportModal.hidden = true;
});
byId("export-md-confirm").addEventListener("click", () => {
  if (!state.active) return;
  const includeReasoning = byId("export-md-reasoning").checked;
  const md = exportThreadMarkdown(state.active, { includeReasoning });
  downloadMarkdown(chatTitle(state.active), md);
  exportModal.hidden = true;
});

// Global keyboard shortcuts (respect IME and menu priority)
document.addEventListener("keydown", event => {
  if (event.isComposing) return;
  const mod = event.ctrlKey || event.metaKey;
  if (mod && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (!state.vision) {
      byId("conversations").hidden = false;
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
    if (!byId("slash-menu").hidden) {
      promptLib.hideSlashMenu();
      return;
    }
    if (!byId("prompt-library").hidden) {
      promptLib.closeLibrary();
      return;
    }
    if (!exportModal.hidden) {
      exportModal.hidden = true;
      return;
    }
    if (!handoverModal.hidden) {
      handoverModal.hidden = true;
      return;
    }
    if (!exportJsonModal.hidden) {
      exportJsonModal.hidden = true;
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
byId("history-search").addEventListener("input", () => renderHistory(selectChat));
byId("history-date-filter").addEventListener("change", event => {
  state.dateFilter = event.target.value;
  renderHistory(selectChat);
});
byId("pin-chat").addEventListener("click", () => {
  if (!state.active || state.running) return;
  state.active.pinned = !state.active.pinned;
  state.active.updatedAt = new Date().toISOString();
  renderHistory(selectChat);
  renderTranscript();
  flushDraftAndPersist();
});
byId("edit-tags").addEventListener("click", () => {
  if (!state.active || state.running) return;
  const current = (state.active.tags || []).join(", ");
  const entered = window.prompt("Nhập các tag, phân cách bằng dấu phẩy (vd: họp, báo-cáo, qwen):", current);
  if (entered === null) return;
  const tags = normalizeTags(entered.split(","));
  if (tags.length) state.active.tags = tags;
  else delete state.active.tags;
  state.active.updatedAt = new Date().toISOString();
  renderHistory(selectChat);
  renderTranscript();
  flushDraftAndPersist();
});
byId("rename-chat").addEventListener("click", () => {
  const title = window.prompt("Conversation name", chatTitle(state.active));
  if (title === null || !title.trim()) return;
  state.active.title = title.trim();
  state.active.updatedAt = new Date().toISOString();
  renderHistory(selectChat);
  renderTranscript();
  flushDraftAndPersist();
});

const exportJsonModal = byId("export-json-modal");
byId("export-chat").addEventListener("click", () => {
  draftSaver.flush();
  byId("export-json-scope").textContent = `Hiện có ${state.conversations.length} hội thoại và ${state.customPrompts.length} mẫu prompt cá nhân.`;
  exportJsonModal.hidden = false;
});
byId("export-v2-btn").addEventListener("click", () => {
  exportHistory(snapshot(state), "kgl-conversations-v2.json");
  exportJsonModal.hidden = true;
});
byId("export-v1-btn").addEventListener("click", () => {
  exportHistory(snapshotV1(state), "kgl-conversations-v1.json");
  exportJsonModal.hidden = true;
});
byId("export-json-cancel").addEventListener("click", () => {
  exportJsonModal.hidden = true;
});

// Two-tab synchronization & conflict detection on localStorage
window.addEventListener("storage", event => {
  if (event.key !== STORAGE_KEY) return;
  try {
    const { merged, conflicts, nextBase } = mergeTabConversations(
      state.conversations, readSharedChats(), chatBase, state.running ? state.active?.id : undefined,
    );
    chatBase = nextBase;
    applySharedChats(merged, state.active?.id);
    reportChatConflicts(conflicts);
  } catch {
    // Ignore malformed external storage writes.
  }
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
  if (!state.active || !window.confirm("Delete this conversation and its draft?")) return;
  draftSaver.cancel();
  const kept = state.conversations.filter(chat => chat !== state.active);
  if (!persist({ ...snapshot(state), conversations: kept, active: null })) return;
  selectChat(null, false);
  historyNotice("Conversation deleted.");
});
byId("history-scope").addEventListener("change", renderBudget);
byId("max-tokens").addEventListener("input", renderBudget);
promptField.addEventListener("input", () => {
  if (state.active) state.active.draft = promptField.value;
  else state.newDraft = promptField.value;
  resizePrompt();
  draftSaver.schedule();
  renderBudget();
});
refresh.addEventListener("click", check);

if (state.conversations.length || state.newDraft) {
  for (const id of ["conversations", "sidebar-toggle", "thread-head", "chat", "generation", "panel"]) {
    byId(id).hidden = false;
  }
  renderHistory(selectChat);
  renderTranscript();
  resizePrompt();
}
check();
