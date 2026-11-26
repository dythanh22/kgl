import assert from "node:assert/strict";
import test from "node:test";

import { buildChatRequest } from "../js/budget.js";
import { readBackup } from "../js/storage.js";
import {
  GUI_ACTION_ITEMS_FORMAT,
  analyzeActionItems,
  buildActionItemsPrompt,
  formatChecklist,
  formatCsv,
  formatJson,
  formatTsv,
  locateQuote,
  parseActionItems,
  sanitizeSpreadsheetCell,
  supportsActionItemsSchema,
} from "../js/tasks/action_items.js";

test("supportsActionItemsSchema requires llama backend and both capabilities", () => {
  assert.equal(supportsActionItemsSchema({
    backend: "llama",
    capabilities: { response_format: true, json_schema: true },
  }), true);
  assert.equal(supportsActionItemsSchema({
    backend: "llama",
    capabilities: { response_format: true, json_schema: false },
  }), false);
  assert.equal(supportsActionItemsSchema({
    backend: "transformers",
    capabilities: { response_format: true, json_schema: true },
  }), false);
  assert.equal(supportsActionItemsSchema(null), false);
});

test("parseActionItems accepts E1/E2, unassigned owner, null due, and empty list", () => {
  const e1 = JSON.stringify({
    items: [
      { task: "Gửi báo cáo", owner: "Lan", due: "trước thứ Sáu", source_quote: "Lan gửi báo cáo trước thứ Sáu." },
      { task: "Xác nhận phòng họp", owner: "Minh", due: null, source_quote: "Minh xác nhận phòng họp." },
      { task: "Cập nhật danh sách khách mời", owner: null, due: "thứ Hai", source_quote: "Cần cập nhật danh sách khách mời vào thứ Hai nhưng chưa giao ai." },
    ],
  });
  const res = parseActionItems(e1);
  assert.equal(res.ok, true);
  assert.equal(res.items.length, 3);
  assert.equal(res.items[0].due, "trước thứ Sáu");
  assert.equal(res.items[1].due, null);
  assert.equal(res.items[2].owner, null);

  const empty = parseActionItems(JSON.stringify({ items: [] }));
  assert.equal(empty.ok, true);
  assert.deepEqual(empty.items, []);
});

test("parseActionItems rejects malformed JSON, extra fields, and empty required strings", () => {
  assert.equal(parseActionItems("{not json").ok, false);
  assert.equal(parseActionItems(JSON.stringify({ items: [], extra: 1 })).ok, false);
  assert.equal(parseActionItems(JSON.stringify({
    items: [{ task: "Check", owner: "Ana", due: null, source_quote: "Ana checks", extra: true }],
  })).ok, false);
  assert.equal(parseActionItems(JSON.stringify({
    items: [{ task: "  ", owner: "Ana", due: null, source_quote: "Ana checks" }],
  })).ok, false);
  assert.equal(parseActionItems(JSON.stringify({
    items: [{ task: "Check", owner: "", due: null, source_quote: "Ana checks" }],
  })).ok, false);
  assert.equal(parseActionItems(JSON.stringify({
    items: [{ task: "Check", owner: "Ana", due: null, source_quote: "  " }],
  })).ok, false);
});

test("locateQuote matches verbatim, normalized whitespace/quotes, and flags missing quotes", () => {
  const source = "Lan gửi báo cáo “doanh thu”\ntrước thứ Sáu.\nMinh xác nhận phòng họp.\nQuân chỉ góp ý và không nhận việc.";

  const exact = locateQuote(source, "Minh xác nhận phòng họp.");
  assert.equal(exact.found, true);
  assert.equal(exact.exact, true);
  assert.equal(source.slice(exact.start, exact.end), "Minh xác nhận phòng họp.");

  const normalized = locateQuote(source, 'Lan gửi báo cáo "doanh thu" trước thứ Sáu.');
  assert.equal(normalized.found, true);
  assert.equal(normalized.exact, false);
  assert.equal(source.slice(normalized.start, normalized.end), "Lan gửi báo cáo “doanh thu”\ntrước thứ Sáu.");

  const missing = locateQuote(source, "Quân gửi báo cáo trước thứ Ba.");
  assert.equal(missing.found, false);

  const analyzed = analyzeActionItems([
    { task: "Gửi báo cáo", owner: "Lan", due: "trước thứ Sáu", source_quote: 'Lan gửi báo cáo "doanh thu" trước thứ Sáu.' },
    { task: "Việc bịa", owner: "Quân", due: null, source_quote: "Quân nhận việc này." },
  ], source);
  assert.equal(analyzed[0].needsReview, false);
  assert.equal(analyzed[1].needsReview, true);
});

test("CSV and TSV export preserves Unicode, delimiters, quotes, newlines and neutralizes formulas", () => {
  assert.equal(sanitizeSpreadsheetCell("=SUM(A1:A2)"), "'=SUM(A1:A2)");
  assert.equal(sanitizeSpreadsheetCell("  =HYPERLINK(\"http://x\")"), "'  =HYPERLINK(\"http://x\")");
  assert.equal(sanitizeSpreadsheetCell("+cmd"), "'+cmd");
  assert.equal(sanitizeSpreadsheetCell("-1+1"), "'-1+1");
  assert.equal(sanitizeSpreadsheetCell("@SUM(1)"), "'@SUM(1)");
  assert.equal(sanitizeSpreadsheetCell("\t=1"), "'\t=1");
  assert.equal(sanitizeSpreadsheetCell("Lan gửi báo cáo"), "Lan gửi báo cáo");

  const items = [
    {
      task: '=Gửi "báo cáo", bản 2',
      owner: null,
      due: "trước thứ Sáu\t17:00",
      source_quote: "Dòng 1\nDòng 2",
    },
  ];
  const csv = formatCsv(items);
  assert.ok(csv.startsWith("Việc,Người phụ trách,Hạn,Câu nguồn\r\n"));
  assert.ok(csv.includes('"\'=Gửi ""báo cáo"", bản 2",Chưa giao,'));
  assert.ok(csv.includes('"Dòng 1\nDòng 2"'));

  const tsv = formatTsv(items);
  assert.ok(tsv.startsWith("Việc\tNgười phụ trách\tHạn\tCâu nguồn\r\n"));
  assert.ok(tsv.includes('"\'=Gửi ""báo cáo"", bản 2"\tChưa giao\t"trước thứ Sáu\t17:00"\t"Dòng 1\nDòng 2"'));
});

test("formatChecklist and formatJson produce clean output", () => {
  const items = [
    { task: "Amend invoice", owner: "Jules", due: "by Tuesday", source_quote: "Jules will amend the invoice by Tuesday." },
    { task: "Check address", owner: null, due: null, source_quote: "Check delivery address." },
  ];
  const checklist = formatChecklist(items);
  assert.ok(checklist.includes("- [ ] Amend invoice — **Jules** · Hạn: by Tuesday"));
  assert.ok(checklist.includes("- [ ] Check address — **Chưa giao**"));
  assert.equal(formatChecklist([]), "Không có đầu việc được giao.");

  const json = JSON.parse(formatJson(items));
  assert.equal(json.items.length, 2);
  assert.equal(json.items[1].owner, null);
});

test("buildChatRequest attaches response_format only when capability is present", () => {
  const notes = "Lan gửi báo cáo trước thứ Sáu.";
  const withSchema = buildChatRequest({
    previous: [{ role: "user", content: "old chat" }, { role: "assistant", content: "old reply" }],
    prompt: notes,
    systemText: "",
    scope: "all",
    contextLimit: 32768,
    supportsThinking: true,
    supportsJsonSchema: true,
    thinkingValue: "off",
    taskMode: "action_items",
  });
  assert.equal(withSchema.task, "action_items");
  assert.equal(withSchema.structured, true);
  assert.deepEqual(withSchema.responseFormat, GUI_ACTION_ITEMS_FORMAT);
  assert.equal(withSchema.sent.length, 1);
  assert.equal(withSchema.sent[0].content, buildActionItemsPrompt(notes, true));

  const withoutSchema = buildChatRequest({
    prompt: notes,
    contextLimit: 32768,
    supportsJsonSchema: false,
    taskMode: "action_items",
  });
  assert.equal(withoutSchema.structured, false);
  assert.equal(withoutSchema.responseFormat, undefined);
  assert.equal(withoutSchema.sent[0].content, buildActionItemsPrompt(notes, false));
});

test("readBackup preserves action_items metadata while keeping v1 compatibility", () => {
  const raw = {
    schema_version: 1,
    conversations: [
      {
        id: "c-ai",
        messages: [
          { role: "user", content: "Lan gửi báo cáo.", task: "action_items", structured: true },
          {
            role: "assistant",
            content: '{"items":[]}',
            task: "action_items",
            structured: true,
            sourceText: "Lan gửi báo cáo.",
          },
        ],
      },
    ],
  };
  const parsed = readBackup(raw);
  assert.equal(parsed.chats[0].messages[1].task, "action_items");
  assert.equal(parsed.chats[0].messages[1].structured, true);
  assert.equal(parsed.chats[0].messages[1].sourceText, "Lan gửi báo cáo.");
});
