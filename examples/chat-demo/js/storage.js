import { normalizeTags } from "./organize.js";
import { PARTIAL_LABELS, SCHEMA_VERSION, STORAGE_KEY } from "./state.js";
import { validateCustomPrompt } from "./tasks/prompts.js";
import { defaultSampling, sanitizeSampling } from "./tasks/sampling.js";

export const BACKUP_V1 = 1;
export const BACKUP_V2 = 2;
const VALID_TASKS = new Set(["action_items", "summary", "rewrite"]);
const utf8Encoder = new TextEncoder();

export const storageStats = {
  lastBytes: 0,
  lastWriteMs: 0,
  writeCount: 0,
};

export function snapshot(current) {
  const data = {
    schema_version: BACKUP_V2,
    conversations: current.conversations,
    active: current.active?.id,
    newDraft: current.newDraft,
    newSystem: current.newSystem,
  };
  if (current.newSampling && current.newSampling.mode !== "preset") {
    data.newSampling = sanitizeSampling(current.newSampling);
  }
  if (Array.isArray(current.newSources) && current.newSources.length) {
    data.newSources = current.newSources;
  }
  if (Array.isArray(current.customPrompts) && current.customPrompts.length) {
    data.prompts = current.customPrompts;
  }
  return data;
}

// Export v1-compatible backup (flat messages, titles, drafts, system instructions only).
export function snapshotV1(current) {
  return {
    schema_version: BACKUP_V1,
    conversations: current.conversations.map(chat => {
      const entry = {
        id: chat.id,
        title: chat.title,
        draft: chat.draft,
        messages: (chat.messages || []).map(message => {
          const clean = { role: message.role, content: message.content, reasoning: message.reasoning };
          if (message.role === "assistant" && message.partial) {
            clean.partial = true;
            if (Object.hasOwn(PARTIAL_LABELS, message.partialReason)) clean.partialReason = message.partialReason;
          }
          if (Number.isSafeInteger(message.usage?.prompt_tokens) && Number.isSafeInteger(message.usage?.completion_tokens)) {
            clean.usage = { prompt_tokens: message.usage.prompt_tokens, completion_tokens: message.usage.completion_tokens };
          }
          return clean;
        }),
      };
      if (chat.system !== undefined) entry.system = chat.system;
      return entry;
    }),
    active: current.active?.id,
    newDraft: current.newDraft,
    newSystem: current.newSystem,
  };
}

function sanitizeSource(src) {
  if (!src || typeof src !== "object" || typeof src.id !== "string" || !src.id
      || typeof src.name !== "string" || !src.name || typeof src.content !== "string") {
    throw new Error("Invalid source attachment in backup.");
  }
  const lineCount = src.content.split("\n").length;
  const fromLine = Number.isInteger(src.fromLine) ? Math.max(1, Math.min(lineCount, src.fromLine)) : 1;
  const toLine = Number.isInteger(src.toLine) ? Math.max(fromLine, Math.min(lineCount, src.toLine)) : lineCount;
  return {
    id: src.id,
    name: src.name,
    content: src.content,
    lineCount,
    fromLine,
    toLine,
    selected: src.selected !== false,
    pinned: Boolean(src.pinned),
  };
}

export function loadHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (Array.isArray(saved?.conversations)) {
      const active = saved.conversations.find(chat => chat.id === saved.active) || null;
      const customPrompts = Array.isArray(saved.prompts)
        ? saved.prompts.flatMap(entry => {
            try { return [validateCustomPrompt(entry)]; } catch { return []; }
          })
        : [];
      const newSources = Array.isArray(saved.newSources)
        ? saved.newSources.flatMap(src => {
            try { return [sanitizeSource(src)]; } catch { return []; }
          })
        : [];
      return {
        conversations: saved.conversations,
        active,
        newDraft: saved.newDraft ?? "",
        newSystem: typeof saved.newSystem === "string" ? saved.newSystem : "",
        newSampling: saved.newSampling ? sanitizeSampling(saved.newSampling) : defaultSampling(),
        newSources,
        customPrompts,
      };
    }
  } catch (error) {
    console.warn("Could not load chat history", error);
  }
  return null;
}

export function saveHistory(data, onNotice, start = performance.now()) {
  try {
    const serialized = JSON.stringify(data);
    localStorage.setItem(STORAGE_KEY, serialized);
    storageStats.lastWriteMs = Math.max(0, performance.now() - start);
    storageStats.lastBytes = utf8Encoder.encode(serialized).length;
    storageStats.writeCount++;
    return true;
  } catch (error) {
    if (onNotice) {
      onNotice(error.name === "QuotaExceededError"
        ? "Browser storage is full. Export your history, then delete an old conversation."
        : "History could not be saved. Export a backup before leaving this page.");
    }
    return false;
  }
}

export function createDraftSaver(persistFn, delayMs = 350, {
  setTimer = (fn, ms) => setTimeout(fn, ms),
  clearTimer = id => clearTimeout(id),
} = {}) {
  let timer = null;
  let dirty = false;

  function schedule() {
    dirty = true;
    if (timer !== null) clearTimer(timer);
    timer = setTimer(() => {
      timer = null;
      dirty = false;
      persistFn();
    }, delayMs);
  }

  function flush() {
    if (timer !== null) {
      clearTimer(timer);
      timer = null;
    }
    if (dirty) {
      dirty = false;
      return persistFn();
    }
    return true;
  }

  function cancel() {
    if (timer !== null) {
      clearTimer(timer);
      timer = null;
    }
    dirty = false;
  }

  return { schedule, flush, cancel, isDirty: () => dirty };
}

function sanitizeMessage(message) {
  if (!message || !["user", "assistant"].includes(message.role) || typeof message.content !== "string"
      || (message.reasoning !== undefined && typeof message.reasoning !== "string")
      || (message.partial !== undefined && typeof message.partial !== "boolean")
      || (message.task !== undefined && !VALID_TASKS.has(message.task))
      || (message.structured !== undefined && typeof message.structured !== "boolean")
      || (message.sourceText !== undefined && typeof message.sourceText !== "string")
      || (message.targetWords !== undefined && !Number.isSafeInteger(message.targetWords))) {
    throw new Error("Invalid message in backup.");
  }
  const { role, content, reasoning, usage } = message;
  const clean = { role, content, reasoning };
  if (VALID_TASKS.has(message.task)) {
    clean.task = message.task;
    if (typeof message.structured === "boolean") clean.structured = message.structured;
    if (typeof message.sourceText === "string") clean.sourceText = message.sourceText;
    if (Number.isSafeInteger(message.targetWords)) clean.targetWords = message.targetWords;
  }
  if (role === "assistant" && message.partial) {
    clean.partial = true;
    if (Object.hasOwn(PARTIAL_LABELS, message.partialReason)) clean.partialReason = message.partialReason;
  }
  if (Number.isSafeInteger(usage?.prompt_tokens) && Number.isSafeInteger(usage?.completion_tokens)) {
    clean.usage = { prompt_tokens: usage.prompt_tokens, completion_tokens: usage.completion_tokens };
  }
  if (message.metrics && typeof message.metrics === "object") {
    const m = {};
    for (const k of ["firstSseMs", "firstReasoningMs", "firstContentMs", "totalMs", "workerTokPerSec"]) {
      if (Number.isFinite(message.metrics[k]) && message.metrics[k] >= 0) m[k] = message.metrics[k];
    }
    if (Object.keys(m).length) clean.metrics = m;
  }
  if (Array.isArray(message.segments) && message.segments.every(s => typeof s === "string")) {
    clean.segments = [...message.segments];
  }
  if (role === "user" && Array.isArray(message.siblings) && message.siblings.length) {
    clean.siblings = message.siblings.map(branch => {
      if (!Array.isArray(branch) || !branch.length) throw new Error("Invalid branch in backup.");
      return branch.map(sanitizeMessage);
    });
    clean.activeSibling = Number.isInteger(message.activeSibling)
      && message.activeSibling >= 0 && message.activeSibling < clean.siblings.length
      ? message.activeSibling : 0;
  }
  if (role === "assistant" && Array.isArray(message.variants) && message.variants.length) {
    clean.variants = message.variants.map(sanitizeMessage);
    clean.activeVariant = Number.isInteger(message.activeVariant)
      && message.activeVariant >= 0 && message.activeVariant < clean.variants.length
      ? message.activeVariant : 0;
  }
  return clean;
}

export function readBackup(raw) {
  if (raw?.schema_version !== BACKUP_V1 && raw?.schema_version !== BACKUP_V2) {
    throw new Error("Unsupported backup version.");
  }
  if (!Array.isArray(raw.conversations)
      || (raw.newDraft !== undefined && typeof raw.newDraft !== "string")
      || (raw.newSystem !== undefined && typeof raw.newSystem !== "string")
      || (raw.prompts !== undefined && !Array.isArray(raw.prompts))) {
    throw new Error("Invalid conversation backup.");
  }
  const prompts = (raw.prompts || []).map(entry => validateCustomPrompt(entry));
  const ids = new Set();
  const chats = raw.conversations.map(chat => {
    if (!chat || typeof chat.id !== "string" || !chat.id || ids.has(chat.id)
        || !Array.isArray(chat.messages) || !chat.messages.length
        || (chat.title !== undefined && typeof chat.title !== "string")
        || (chat.draft !== undefined && typeof chat.draft !== "string")
        || (chat.system !== undefined && typeof chat.system !== "string")
        || (chat.pinned !== undefined && typeof chat.pinned !== "boolean")
        || (chat.tags !== undefined && !Array.isArray(chat.tags))
        || (chat.createdAt !== undefined && typeof chat.createdAt !== "string")
        || (chat.updatedAt !== undefined && typeof chat.updatedAt !== "string")
        || (chat.sampling !== undefined && (!chat.sampling || typeof chat.sampling !== "object"))) {
      throw new Error("Invalid or duplicate conversation in backup.");
    }
    ids.add(chat.id);
    const messages = chat.messages.map(sanitizeMessage);
    const entry = { id: chat.id, title: chat.title, draft: chat.draft, messages };
    if (chat.system !== undefined) entry.system = chat.system;
    if (chat.sampling !== undefined) entry.sampling = sanitizeSampling(chat.sampling);
    if (Array.isArray(chat.sources)) entry.sources = chat.sources.map(sanitizeSource);
    if (typeof chat.parentId === "string" && chat.parentId) entry.parentId = chat.parentId;
    if (chat.pinned) entry.pinned = true;
    const cleanTags = normalizeTags(chat.tags);
    if (cleanTags.length) entry.tags = cleanTags;
    if (chat.createdAt) entry.createdAt = chat.createdAt;
    if (chat.updatedAt) entry.updatedAt = chat.updatedAt;
    return entry;
  });
  return {
    version: raw.schema_version,
    chats,
    active: raw.active,
    newDraft: raw.newDraft ?? "",
    newSystem: raw.newSystem ?? "",
    newSampling: raw.newSampling ? sanitizeSampling(raw.newSampling) : undefined,
    newSources: Array.isArray(raw.newSources) ? raw.newSources.map(sanitizeSource) : [],
    prompts,
  };
}

export function mergeBackup(current, raw) {
  const backup = readBackup(raw);
  const existing = new Set(current.conversations.map(chat => chat.id));
  const added = backup.chats.filter(chat => !existing.has(chat.id));
  const merged = [...current.conversations, ...added];
  const existingPrompts = new Set((current.customPrompts || []).map(p => p.id));
  const mergedPrompts = [
    ...(current.customPrompts || []),
    ...backup.prompts.filter(p => !existingPrompts.has(p.id)),
  ];
  const selection = current.active?.id ?? backup.active;
  const data = {
    ...snapshot(current),
    conversations: merged,
    active: selection,
    newDraft: current.newDraft || backup.newDraft,
    newSystem: current.newSystem || backup.newSystem,
  };
  if (mergedPrompts.length) data.prompts = mergedPrompts;
  if (!data.newSampling && backup.newSampling) data.newSampling = backup.newSampling;
  if (!data.newSources && backup.newSources.length) data.newSources = backup.newSources;
  return {
    data,
    merged,
    mergedPrompts,
    selection,
    addedCount: added.length,
    skippedCount: backup.chats.length - added.length,
  };
}

export function exportHistory(data, filename = "kgl-conversations.json") {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
