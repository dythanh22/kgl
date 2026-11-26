import assert from "node:assert/strict";
import test from "node:test";

import { buildChatRequest } from "../js/budget.js";
import { mergeBackup, readBackup, snapshot } from "../js/storage.js";
import {
  DEFAULT_PROMPTS,
  detectSlashQuery,
  expandTemplate,
  filterPromptsBySlash,
  formatLocalDate,
  getAllPrompts,
  normalizeSlash,
  templateUsesVariable,
  validateCustomPrompt,
} from "../js/tasks/prompts.js";
import { buildSamplingOverrides, defaultSampling, sanitizeSampling } from "../js/tasks/sampling.js";
import { buildRewritePrompt, buildSummaryPrompt, compareFacts, countWords } from "../js/tasks/work_templates.js";

test("prompts: slash detection, filtering, normalization and variable expansion", () => {
  assert.equal(normalizeSlash("Bao Cao"), "/bao-cao");
  assert.equal(normalizeSlash("///review_1"), "/review_1");
  assert.equal(normalizeSlash("   "), "");

  const detected = detectSlashQuery("Xin chào /tom", 13);
  assert.deepEqual(detected, { query: "/tom", start: 9, end: 13 });
  assert.equal(detectSlashQuery("http://example.com/path", 23), null);

  const all = getAllPrompts([]);
  assert.ok(all.length >= DEFAULT_PROMPTS.length);
  const filtered = filterPromptsBySlash(all, "/tom");
  assert.equal(filtered[0].slash, "/tomtat");

  const tpl = "Ngày {{date}}\nChọn: {{selection}}\nDán: {{clipboard}}";
  assert.equal(templateUsesVariable(tpl, "clipboard"), true);
  assert.equal(templateUsesVariable(tpl, "selection"), true);
  const expanded = expandTemplate(tpl, {
    date: formatLocalDate(new Date(2026, 10, 25)),
    selection: "đoạn A",
    clipboard: "nội dung B",
  });
  assert.equal(expanded, "Ngày 2026-11-25\nChọn: đoạn A\nDán: nội dung B");

  const valid = validateCustomPrompt({ name: "Báo cáo tuần", slash: "tuan", template: "Tóm tắt {{date}}" });
  assert.equal(valid.slash, "/tuan");
  assert.throws(() => validateCustomPrompt({ name: "", slash: "/x", template: "a" }));
  assert.throws(() => validateCustomPrompt({ name: "X", slash: "!!!", template: "a" }));
});

test("work_templates: word count, summary/rewrite prompts and fact diff", () => {
  assert.equal(countWords("  Lan gửi báo cáo trước thứ Sáu.  "), 7);
  assert.equal(countWords(""), 0);

  const sumPrompt = buildSummaryPrompt("Ghi chú họp", { targetWords: 120 });
  assert.ok(sumPrompt.includes("### Quyết định"));
  assert.ok(sumPrompt.includes("### Đề xuất"));
  assert.ok(sumPrompt.includes("### Câu hỏi còn mở"));
  assert.ok(sumPrompt.includes("120 từ"));

  const rwPrompt = buildRewritePrompt("Orion is penciled in for May 14. Fee is $180.", {
    audience: "client",
    goal: "confirm",
    tone: "polite",
  });
  assert.ok(rwPrompt.includes("Khách hàng"));
  assert.ok(rwPrompt.includes("Xin xác nhận"));

  const same = compareFacts(
    "Orion is penciled in for May 14. The estimated venue fee is $180.",
    "Project Orion is tentatively scheduled for May 14 with an estimated fee of $180.",
  );
  assert.equal(same.numbers.missing.length, 0);
  assert.equal(same.dates.missing.length, 0);
  assert.equal(same.names.missing.length, 0);

  const changed = compareFacts(
    "Thảo dời họp sang 10:30 thứ Năm, ngân sách 350.000 đồng.",
    "Lan dời họp sang 11:00 thứ Sáu, ngân sách 400.000 đồng.",
  );
  assert.ok(changed.totalChanges > 0);
  assert.ok(changed.numbers.missing.includes("10:30"));
  assert.ok(changed.dates.missing.includes("thứ Năm"));
  assert.ok(changed.names.missing.includes("Thảo"));
});

test("sampling: preset omits overrides, custom validates protocol ranges and gates penalties", () => {
  const preset = buildSamplingOverrides(defaultSampling(), { presence_penalty: true, repeat_penalty: true });
  assert.deepEqual(preset.overrides, {});

  const precise = buildSamplingOverrides({ mode: "precise" }, {});
  assert.deepEqual(precise.overrides, { temperature: 0.2, top_p: 0.9 });

  const customFull = buildSamplingOverrides(
    { mode: "custom", temperature: "0.3", top_p: "0.85", presence_penalty: "0.5", repeat_penalty: "1.1" },
    { presence_penalty: true, repeat_penalty: true },
  );
  assert.deepEqual(customFull.overrides, {
    temperature: 0.3,
    top_p: 0.85,
    presence_penalty: 0.5,
    repeat_penalty: 1.1,
  });

  // Unsupported worker omits penalties cleanly
  const customLimited = buildSamplingOverrides(
    { mode: "custom", temperature: "0.3", presence_penalty: "0.5", repeat_penalty: "1.1" },
    { presence_penalty: false, repeat_penalty: false },
  );
  assert.deepEqual(customLimited.overrides, { temperature: 0.3 });

  // Protocol range violations throw
  assert.throws(() => buildSamplingOverrides({ mode: "custom", temperature: "2.5" }, {}));
  assert.throws(() => buildSamplingOverrides({ mode: "custom", top_p: "0" }, {}));
  assert.throws(() => buildSamplingOverrides({ mode: "custom", presence_penalty: "3" }, { presence_penalty: true }));
  assert.throws(() => buildSamplingOverrides({ mode: "custom", repeat_penalty: "0" }, { repeat_penalty: true }));
  assert.equal(sanitizeSampling({ mode: "unknown" }).mode, "preset");
});

test("budget & storage: summary/rewrite requests and v1 backup round-trip for prompts & sampling", () => {
  const req = buildChatRequest({
    prompt: "Nhóm xử lý 93/120 hồ sơ.",
    contextLimit: 32768,
    taskMode: "summary",
    taskOptions: { targetWords: "80" },
  });
  assert.equal(req.task, "summary");
  assert.equal(req.targetWords, 80);
  assert.equal(req.sent.length, 1);
  assert.ok(req.sent[0].content.includes("80 từ"));

  const stateLike = {
    conversations: [
      {
        id: "c-l13",
        sampling: { mode: "precise", temperature: "", top_p: "", presence_penalty: "", repeat_penalty: "" },
        messages: [
          { role: "user", content: "Bản gốc", task: "rewrite" },
          { role: "assistant", content: "Bản sửa", task: "rewrite", sourceText: "Bản gốc" },
        ],
      },
    ],
    active: { id: "c-l13" },
    newDraft: "",
    newSystem: "",
    newSampling: defaultSampling(),
    customPrompts: [{ id: "custom-1", slash: "/baocao", name: "Báo cáo", description: "", template: "Ngày {{date}}" }],
  };
  const snap = snapshot(stateLike);
  const parsed = readBackup(snap);
  assert.equal(parsed.chats[0].sampling.mode, "precise");
  assert.equal(parsed.prompts.length, 1);
  assert.equal(parsed.prompts[0].slash, "/baocao");

  const merged = mergeBackup({ conversations: [], customPrompts: [], newDraft: "", newSystem: "", newSampling: defaultSampling() }, snap);
  assert.equal(merged.addedCount, 1);
  assert.equal(merged.mergedPrompts.length, 1);
});

test("reply budgets for explanations and long reports on the 32K worker", () => {
  const chat = { contextLimit: 32768, supportsThinking: true, prompt: "Explain" };
  for (const thinkingValue of ["off", "on"]) {
    assert.equal(buildChatRequest({ ...chat, thinkingValue }).payload.max_tokens, 2048);
  }
  assert.equal(buildChatRequest({ ...chat, limitValue: "4096", limitNumber: 4096 }).payload.max_tokens, 4096);
  assert.equal(buildChatRequest({ ...chat, supportsThinking: false }).payload.max_tokens, 768);
});
