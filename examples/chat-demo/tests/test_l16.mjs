import assert from "node:assert/strict";
import test from "node:test";

import { forkAtUserMessage } from "../js/branches.js";
import {
  collectAllTags,
  filterConversations,
  findMatchSnippet,
  matchesDateRange,
  mergeTabConversations,
  normalizeTags,
  sortConversations,
} from "../js/organize.js";
import { BACKUP_V1, BACKUP_V2, mergeBackup, readBackup, snapshot, snapshotV1 } from "../js/storage.js";
import { defaultSampling } from "../js/tasks/sampling.js";
import { CONNECTION_MESSAGES, formatConnectionError, formatMessageStats } from "../js/ui.js";

test("organize: pinned chats float to top, tags normalize & filter, date ranges work", () => {
  const now = Date.parse("2026-11-25T12:00:00Z");
  const chats = [
    { id: "c1", title: "Cũ", updatedAt: "2026-10-01T10:00:00Z", tags: ["báo-cáo"], messages: [{ role: "user", content: "a" }] },
    { id: "c2", title: "Ghim", pinned: true, updatedAt: "2026-11-25T08:00:00Z", tags: ["họp", "qwen"], messages: [{ role: "user", content: "b" }] },
    { id: "c3", title: "Tuần này", updatedAt: "2026-11-22T08:00:00Z", tags: ["họp"], messages: [{ role: "user", content: "c" }] },
  ];

  const sorted = sortConversations(chats);
  assert.equal(sorted[0].id, "c2");
  assert.deepEqual(normalizeTags([" Họp ", "họp", "Qwen", ""]), ["Họp", "Qwen"]);
  assert.equal(collectAllTags(chats)[0].label, "họp");
  assert.equal(collectAllTags(chats)[0].count, 2);

  assert.equal(matchesDateRange(chats[1], "today", now), true);
  assert.equal(matchesDateRange(chats[2], "today", now), false);
  assert.equal(matchesDateRange(chats[2], "7d", now), true);
  assert.equal(matchesDateRange(chats[0], "30d", now), false);

  const byTag = filterConversations(chats, { tag: "họp", nowMs: now });
  assert.deepEqual(byTag.map(x => x.chat.id), ["c2", "c3"]);
});

test("organize: search matches Vietnamese keywords across branches and returns snippet", () => {
  const orig = [
    { role: "user", content: "Chốt ngân sách quý bốn với Lan." },
    { role: "assistant", content: "Đã ghi nhận." },
  ];
  const forked = forkAtUserMessage(orig, 0, "Đổi chủ đề khác hoàn toàn.");
  const chat = { id: "c-br", title: "Kế hoạch", messages: forked };

  const snip = findMatchSnippet(chat, "ngân sách");
  assert.ok(snip);
  assert.equal(snip.match, "ngân sách");
});

test("storage: v2 full backup and v1 compatible backup round-trip cleanly", () => {
  const stateLike = {
    conversations: [
      {
        id: "c-full",
        title: "Họp Q4",
        pinned: true,
        tags: ["họp", "qwen"],
        createdAt: "2026-11-25T08:00:00Z",
        updatedAt: "2026-11-25T09:00:00Z",
        parentId: "p-0",
        system: "Concise VI",
        sampling: { mode: "precise", temperature: "", top_p: "", presence_penalty: "", repeat_penalty: "" },
        sources: [{ id: "s1", name: "notes.txt", content: "abc", fromLine: 1, toLine: 1, selected: true, pinned: true }],
        messages: [{ role: "user", content: "Xin chào" }, { role: "assistant", content: "Chào bạn" }],
      },
    ],
    active: { id: "c-full" },
    newDraft: "nháp",
    newSystem: "sys",
    newSampling: defaultSampling(),
    newSources: [],
    customPrompts: [{ id: "p1", slash: "/baocao", name: "Báo cáo", description: "", template: "Ngày {{date}}" }],
  };

  const v2 = snapshot(stateLike);
  assert.equal(v2.schema_version, BACKUP_V2);
  const parsedV2 = readBackup(v2);
  assert.equal(parsedV2.version, BACKUP_V2);
  assert.equal(parsedV2.chats[0].pinned, true);
  assert.deepEqual(parsedV2.chats[0].tags, ["họp", "qwen"]);
  assert.equal(parsedV2.prompts.length, 1);

  const v1 = snapshotV1(stateLike);
  assert.equal(v1.schema_version, BACKUP_V1);
  assert.equal(v1.prompts, undefined);
  assert.equal(v1.conversations[0].pinned, undefined);
  const parsedV1 = readBackup(v1);
  assert.equal(parsedV1.version, BACKUP_V1);
  assert.equal(parsedV1.chats[0].title, "Họp Q4");
  assert.equal(parsedV1.chats[0].system, "Concise VI");

  assert.throws(() => readBackup({ schema_version: 99, conversations: [] }), /Unsupported/);
});

test("tab merge distinguishes remote changes, deletions, and genuine conflicts", () => {
  const initial = [{ id: "a", draft: "" }, { id: "b", draft: "" }];
  const base = new Map(initial.map(chat => [chat.id, JSON.stringify(chat)]));
  const remote = [{ ...initial[0], draft: "nháp A" }, initial[1]];
  const local = [initial[0], { ...initial[1], draft: "nháp B" }];
  const combined = mergeTabConversations(local, remote, base);
  assert.deepEqual(combined.conflicts, []);
  assert.deepEqual(combined.saved.map(chat => chat.draft), ["nháp A", "nháp B"]);
  assert.deepEqual(combined.merged, combined.saved);

  const deletion = mergeTabConversations(local, [initial[1]], base);
  assert.deepEqual(deletion.saved, [local[1]]);
  assert.deepEqual(deletion.merged, [local[1]]);
  assert.equal(deletion.nextBase.has("a"), false);

  const divergent = [{ ...initial[0], draft: "tab khác" }, local[1]];
  const conflict = mergeTabConversations(divergent, remote, base);
  assert.deepEqual(conflict.conflicts, ["a"]);
  assert.equal(conflict.saved[0].draft, "nháp A");
  assert.equal(conflict.merged[0].draft, "tab khác");
  assert.equal(conflict.nextBase.get("a"), base.get("a"));
  const deletedWhileEditing = mergeTabConversations(divergent, [initial[1]], base);
  assert.deepEqual(deletedWhileEditing.conflicts, ["a"]);
  assert.deepEqual(deletedWhileEditing.saved, [local[1]]);
  assert.equal(deletedWhileEditing.merged.find(chat => chat.id === "a").draft, "tab khác");
});

test("v2 import restores sources and sampling for an unsent draft", () => {
  const current = { conversations: [], newDraft: "", newSystem: "", newSampling: defaultSampling(), newSources: [] };
  const source = { id: "s", name: "notes.txt", content: "abc", selected: false, pinned: true };
  const backup = snapshot({ ...current, newDraft: "nháp", newSampling: { mode: "precise" }, newSources: [source] });
  const { data } = mergeBackup(current, backup);
  assert.equal(data.newDraft, "nháp");
  assert.equal(data.newSampling.mode, "precise");
  assert.equal(data.newSources[0].content, "abc");
  assert.equal(data.newSources[0].selected, false);
  const local = mergeBackup({ ...current, newSources: [{ ...source, id: "local" }], newSampling: { mode: "creative" } }, backup).data;
  assert.equal(local.newSources[0].id, "local");
  assert.equal(local.newSampling.mode, "creative");
});

test("budget: buildChatRequest returns accurate breakdown for context meter", async () => {
  const { buildChatRequest } = await import("../js/budget.js");
  const req = buildChatRequest({
    prompt: "Tóm tắt nội dung này",
    previous: [
      { role: "user", content: "Lượt trước" },
      { role: "assistant", content: "Trả lời trước" },
    ],
    systemText: "Trả lời ngắn gọn bằng tiếng Việt.",
    sources: [
      { id: "s1", name: "brief.md", content: "Dòng 1\nDòng 2\nDòng 3", fromLine: 1, toLine: 3, selected: true, pinned: true },
    ],
    contextLimit: 32768,
    limitValue: "1024",
    limitNumber: 1024,
    supportsThinking: true,
    thinkingValue: "off",
    scope: "all",
  });
  assert.ok(req.breakdown);
  assert.ok(req.breakdown.system > 0);
  assert.ok(req.breakdown.sources > 0);
  assert.ok(req.breakdown.messages > 0);
  assert.equal(req.breakdown.reply, 1024);
  assert.equal(req.breakdown.contextLimit, 32768);
  assert.equal(req.breakdown.total, req.estimate + req.replyTokens);
  assert.equal(req.breakdown.free, 32768 - req.breakdown.total);

  const cont = buildChatRequest({
    mode: "continue",
    previous: [
      { role: "user", content: "Viết báo cáo" },
      { role: "assistant", content: "Phần 1...", partial: true, partialReason: "length" },
    ],
    contextLimit: 32768,
    limitValue: "512",
    limitNumber: 512,
    supportsThinking: false,
  });
  assert.ok(cont.breakdown);
  assert.equal(cont.breakdown.sources, 0);
  assert.equal(cont.breakdown.reply, 512);
});


test("message stats show generation speed only for completed replies with worker timings", () => {
  const message = { usage: { completion_tokens: 20 }, metrics: { totalMs: 2000 } };
  assert.doesNotMatch(formatMessageStats(message), /tok\/s/);
  message.metrics.workerTokPerSec = 12.5;
  assert.match(formatMessageStats(message), /12\.5 tok\/s/);
  assert.doesNotMatch(formatMessageStats({ ...message, partial: true }), /tok\/s/);
});

test("connection error guidance points operator to KGL Desktop and preserves OOM hints", () => {
  assert.match(formatConnectionError(new TypeError("Failed to fetch")), /KGL Desktop/);
  assert.match(CONNECTION_MESSAGES.connectionLost, /KGL Desktop.*Chạy Qwen/);
  assert.match(formatConnectionError(new Error("CUDA out of memory")), /bộ nhớ GPU/);
});
