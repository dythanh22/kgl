import { chatTitle } from "./state.js";

export function normalizeTag(raw) {
  return String(raw || "")
    .trim()
    .replace(/\s+/g, " ")
    .slice(0, 32);
}

export function normalizeTags(tags) {
  if (!Array.isArray(tags)) return [];
  const seen = new Set();
  const out = [];
  for (const raw of tags) {
    const clean = normalizeTag(raw);
    if (!clean) continue;
    const key = clean.toLocaleLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(clean);
  }
  return out;
}

export function collectAllTags(conversations = []) {
  const counts = new Map();
  for (const chat of conversations) {
    for (const tag of chat.tags || []) {
      const key = tag.toLocaleLowerCase();
      const prev = counts.get(key);
      if (prev) prev.count++;
      else counts.set(key, { label: tag, count: 1 });
    }
  }
  return [...counts.values()].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

export function sortConversations(conversations = []) {
  const pinned = [];
  const unpinned = [];
  for (const chat of conversations) {
    if (chat.pinned) pinned.push(chat);
    else unpinned.push(chat);
  }
  return [...pinned, ...unpinned];
}

export function matchesDateRange(chat, range, nowMs = Date.now()) {
  if (!range || range === "all") return true;
  const ts = Date.parse(chat.updatedAt || chat.createdAt || "");
  if (!Number.isFinite(ts)) return false;
  const startOfToday = new Date(nowMs);
  startOfToday.setHours(0, 0, 0, 0);
  if (range === "today") return ts >= startOfToday.getTime();
  if (range === "7d") return ts >= nowMs - 7 * 86400 * 1000;
  if (range === "30d") return ts >= nowMs - 30 * 86400 * 1000;
  return true;
}

function collectSearchTexts(chat) {
  const texts = [chatTitle(chat), ...(chat.tags || [])];
  for (const msg of chat.messages || []) {
    if (msg.content) texts.push(msg.content);
    if (Array.isArray(msg.siblings)) {
      for (const branch of msg.siblings) {
        for (const m of branch) {
          if (m?.content) texts.push(m.content);
        }
      }
    }
    if (Array.isArray(msg.variants)) {
      for (const v of msg.variants) {
        if (v?.content) texts.push(v.content);
      }
    }
  }
  return texts;
}

export function findMatchSnippet(chat, query, radius = 28) {
  const q = String(query || "").trim();
  if (!q) return null;
  const lowerQ = q.toLocaleLowerCase();
  for (const text of collectSearchTexts(chat)) {
    const lower = text.toLocaleLowerCase();
    const idx = lower.indexOf(lowerQ);
    if (idx === -1) continue;
    const start = Math.max(0, idx - radius);
    const end = Math.min(text.length, idx + q.length + radius);
    const prefix = start > 0 ? "…" : "";
    const suffix = end < text.length ? "…" : "";
    return {
      before: prefix + text.slice(start, idx).replace(/\s+/g, " "),
      match: text.slice(idx, idx + q.length),
      after: text.slice(idx + q.length, end).replace(/\s+/g, " ") + suffix,
    };
  }
  return null;
}

export function filterConversations(conversations, { query = "", tag = "", dateRange = "all", nowMs = Date.now() } = {}) {
  const sorted = sortConversations(conversations);
  const tagLower = String(tag || "").trim().toLocaleLowerCase();
  const q = String(query || "").trim();
  const results = [];
  for (const chat of sorted) {
    if (tagLower && !(chat.tags || []).some(t => t.toLocaleLowerCase() === tagLower)) continue;
    if (!matchesDateRange(chat, dateRange, nowMs)) continue;
    if (q) {
      const snippet = findMatchSnippet(chat, q);
      if (!snippet) continue;
      results.push({ chat, snippet });
    } else {
      results.push({ chat, snippet: null });
    }
  }
  return results;
}

// Compare each version with the last shared snapshot. Keep a conflicting local
// edit available for export while leaving the already saved version in storage.
export function mergeTabConversations(localConversations, remoteConversations, base, protectedId) {
  const localById = new Map(localConversations.map(chat => [chat.id, chat]));
  const remoteById = new Map(remoteConversations.map(chat => [chat.id, chat]));
  const newLocalIds = [...localById.keys()].filter(id => !base.has(id) && !remoteById.has(id));
  const ids = new Set([...newLocalIds, ...remoteById.keys(), ...localById.keys()]);
  const merged = [];
  const saved = [];
  const conflicts = [];
  const nextBase = new Map();

  for (const id of ids) {
    const local = localById.get(id);
    const remote = remoteById.get(id);
    const localJson = JSON.stringify(local);
    const remoteJson = JSON.stringify(remote);
    const original = base.get(id);
    const localChanged = localJson !== original;
    const remoteChanged = remoteJson !== original;
    const conflict = remoteChanged && localJson !== remoteJson && (localChanged || id === protectedId);
    const chosen = localChanged && !remoteChanged ? local : remote;
    const localVersion = conflict || localJson === remoteJson ? local : chosen;
    if (localVersion) merged.push(localVersion);
    if (chosen) saved.push(chosen);
    if (conflict) conflicts.push(id);
    const baseline = conflict ? original : remoteJson;
    if (baseline !== undefined) nextBase.set(id, baseline);
  }
  return { merged, saved, conflicts, nextBase };
}
