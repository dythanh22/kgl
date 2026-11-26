import assert from "node:assert/strict";
import test from "node:test";

import {
  attachVariant,
  exportThreadMarkdown,
  forkAtUserMessage,
  prepareRegenerate,
  switchAssistantVariant,
  switchUserSibling,
} from "../js/branches.js";
import { CONTINUE_PROMPT, buildChatRequest } from "../js/budget.js";
import { createReadinessPoller } from "../js/poller.js";
import { createDraftSaver, readBackup, snapshot } from "../js/storage.js";
import { defaultSampling } from "../js/tasks/sampling.js";

test("createDraftSaver debounces rapid writes and flushes immediately on demand", () => {
  let writes = 0;
  const scheduled = [];
  const saver = createDraftSaver(() => { writes++; return true; }, 350, {
    setTimer: (fn, ms) => {
      const id = scheduled.length + 1;
      scheduled.push({ id, fn, ms, cleared: false });
      return id;
    },
    clearTimer: id => {
      const item = scheduled.find(s => s.id === id);
      if (item) item.cleared = true;
    },
  });

  saver.schedule();
  saver.schedule();
  saver.schedule();
  assert.equal(writes, 0);
  assert.equal(saver.isDirty(), true);
  assert.equal(scheduled.filter(s => !s.cleared).length, 1);

  saver.flush();
  assert.equal(writes, 1);
  assert.equal(saver.isDirty(), false);
  // Second flush with no new changes does not write again.
  saver.flush();
  assert.equal(writes, 1);
});

test("branches: forkAtUserMessage and switchUserSibling preserve original path and follow-up turns", () => {
  const original = [
    { role: "user", content: "Q1" },
    { role: "assistant", content: "A1", reasoning: "r1" },
    { role: "user", content: "Q2" },
    { role: "assistant", content: "A2" },
  ];

  const forked = forkAtUserMessage(original, 0, "Q1 edited");
  assert.equal(forked.length, 1);
  assert.equal(forked[0].content, "Q1 edited");
  assert.equal(forked[0].siblings.length, 2);
  assert.equal(forked[0].activeSibling, 1);

  // Add a reply on the new branch
  const withReply = [...forked, { role: "assistant", content: "A1 new" }];
  // Switch back to original branch 0
  const backToOrig = switchUserSibling(withReply, 0, 0);
  assert.equal(backToOrig.length, 4);
  assert.equal(backToOrig[0].content, "Q1");
  assert.equal(backToOrig[3].content, "A2");
  assert.equal(backToOrig[0].activeSibling, 0);

  // Switch forward again to branch 1
  const backToNew = switchUserSibling(backToOrig, 0, 1);
  assert.equal(backToNew.length, 2);
  assert.equal(backToNew[0].content, "Q1 edited");
  assert.equal(backToNew[1].content, "A1 new");
});

test("branches: prepareRegenerate, attachVariant and switchAssistantVariant preserve variants", () => {
  const msgs = [
    { role: "user", content: "Explain GPU" },
    { role: "assistant", content: "Answer v1", reasoning: "r1" },
  ];
  const { prefix, variants } = prepareRegenerate(msgs);
  assert.equal(prefix.length, 1);
  assert.equal(variants.length, 1);
  const regenerate = buildChatRequest({ previous: prefix, mode: "regenerate", scope: "0" });
  assert.deepEqual(regenerate.payload.messages, [{ role: "user", content: "Explain GPU" }]);
  assert.deepEqual(regenerate.messages, prefix);

  const summary = buildChatRequest({
    previous: [{ role: "user", content: "Meeting notes", task: "summary", targetWords: 60 }],
    mode: "regenerate",
  });
  assert.equal(summary.task, "summary");
  assert.match(summary.payload.messages[0].content, /60 từ/);

  const v2 = { role: "assistant", content: "Answer v2", reasoning: "r2" };
  attachVariant(v2, variants);
  const updated = [...prefix, v2];
  assert.equal(updated[1].variants.length, 2);
  assert.equal(updated[1].activeVariant, 1);

  const switched = switchAssistantVariant(updated, 1, 0);
  assert.equal(switched[1].content, "Answer v1");
  assert.equal(switched[1].activeVariant, 0);
  assert.equal(switched[1].variants.length, 2);

  // buildChatRequest sends only active path without reasoning/variants
  const req = buildChatRequest({ previous: switched, prompt: "Follow up", mode: "send" });
  assert.deepEqual(req.sent.map(m => ({ role: m.role, content: m.content })), [
    { role: "user", content: "Explain GPU" },
    { role: "assistant", content: "Answer v1" },
    { role: "user", content: "Follow up" },
  ]);
});

test("budget: continue mode sends partial content plus active continue prompt", () => {
  const prev = [
    { role: "user", content: "Write a report" },
    { role: "assistant", content: "Part 1...", partial: true, partialReason: "length" },
  ];
  const req = buildChatRequest({ previous: prev, mode: "continue", contextLimit: 32768 });
  assert.equal(req.sent.length, 3);
  assert.equal(req.sent[1].role, "assistant");
  assert.equal(req.sent[1].content, "Part 1...");
  assert.equal(req.sent[2].role, "user");
  assert.equal(req.sent[2].content, CONTINUE_PROMPT);
});

test("poller: single-flight, respects Retry-After, stops on ready or terminal or 30s timeout", async () => {
  let clock = 0;
  const delays = [];
  let readyCalls = 0;
  let terminalCalls = 0;
  let timeoutCalls = 0;
  let probes = 0;

  const responses = [
    { kind: "busy", retryAfterSec: 4 },
    { kind: "ready", health: { status: "ready" } },
  ];

  const poller = createReadinessPoller({
    async probe() {
      return responses[probes++];
    },
    onReady() { readyCalls++; },
    onTerminal() { terminalCalls++; },
    onTimeout() { timeoutCalls++; },
    intervalMs: 1800,
    maxDurationMs: 30000,
    now: () => clock,
    setTimer: (fn, ms) => {
      delays.push(ms);
      clock += ms;
      Promise.resolve().then(fn);
      return delays.length;
    },
    clearTimer: () => {},
  });

  poller.start();
  await new Promise(r => setTimeout(r, 10));
  assert.equal(readyCalls, 1);
  assert.equal(terminalCalls, 0);
  assert.equal(timeoutCalls, 0);
  assert.equal(delays[0], 1800);
  assert.equal(delays[1], 4000); // respected Retry-After: 4s > 1.8s
});

test("exportThreadMarkdown omits reasoning by default and marks partial turns", () => {
  const chat = {
    id: "c1",
    title: "Họp tuần 47",
    messages: [
      { role: "user", content: "Tóm tắt giúp mình" },
      { role: "assistant", content: "Nội dung chính", reasoning: "suy nghĩ nội bộ", partial: true, partialReason: "length" },
    ],
  };
  const mdDefault = exportThreadMarkdown(chat);
  assert.ok(mdDefault.includes("# Họp tuần 47"));
  assert.ok(mdDefault.includes("## Assistant (Token limit)"));
  assert.equal(mdDefault.includes("suy nghĩ nội bộ"), false);

  const mdWithReasoning = exportThreadMarkdown(chat, { includeReasoning: true });
  assert.ok(mdWithReasoning.includes("suy nghĩ nội bộ"));
});

test("storage: backup round-trips siblings, variants, segments, and metrics", () => {
  const original = [
    { role: "user", content: "Q1" },
    { role: "assistant", content: "A1" },
  ];
  const forked = forkAtUserMessage(original, 0, "Q1 v2");
  const reply = {
    role: "assistant",
    content: "Part 1\n\nPart 2",
    segments: ["Part 1", "Part 2"],
    metrics: { firstContentMs: 1200, totalMs: 3400, workerTokPerSec: 68.5 },
    usage: { prompt_tokens: 100, completion_tokens: 200 },
  };
  const stateLike = {
    conversations: [{ id: "c-l14", messages: [...forked, reply] }],
    active: { id: "c-l14" },
    newDraft: "",
    newSystem: "",
    newSampling: defaultSampling(),
    customPrompts: [],
  };
  const parsed = readBackup(snapshot(stateLike));
  const m0 = parsed.chats[0].messages[0];
  const m1 = parsed.chats[0].messages[1];
  assert.equal(m0.siblings.length, 2);
  assert.equal(m0.activeSibling, 1);
  assert.deepEqual(m1.segments, ["Part 1", "Part 2"]);
  assert.equal(m1.metrics.workerTokPerSec, 68.5);
});
