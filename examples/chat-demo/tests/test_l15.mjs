import assert from "node:assert/strict";
import test from "node:test";

import { buildChatRequest } from "../js/budget.js";
import { readBackup, snapshot } from "../js/storage.js";
import {
  DEFAULT_MAX_REQUEST_BYTES,
  buildHandoverMessages,
  collectActiveSourceBlocks,
  decodeTextBytes,
  formatBytes,
  formatSourceBlock,
  measurePayloadBytes,
  sliceSourceContent,
} from "../js/tasks/files.js";
import { defaultSampling } from "../js/tasks/sampling.js";

test("deselecting an attachment excludes old ranges from outgoing messages only", () => {
  const source = { id: "s", name: "notes (VI).txt", content: "đoạn riêng\npublic\nlast", fromLine: 1, toLine: 2 };
  const block = formatSourceBlock(source);
  const fullBlock = formatSourceBlock({ ...source, toLine: 3 });
  const unrelated = formatSourceBlock({ ...source, content: "unrelated\npublic\nlast" });
  const previous = [
    { role: "user", content: `Original prompt\n\n${block}\n\n${fullBlock}\n\n${unrelated}` },
    { role: "assistant", content: "Reply" },
  ];
  const sources = [{ ...source, selected: false, fromLine: 3, toLine: 3 }];
  const followup = buildChatRequest({ previous, prompt: "Next", sources });
  const regenerated = buildChatRequest({ previous: previous.slice(0, 1), mode: "regenerate", sources });
  for (const request of [followup, regenerated]) {
    const sent = request.payload.messages[0].content;
    assert.equal(sent, `Original prompt\n\n\n\n\n\n${unrelated}`);
    assert.ok(request.messages[0].content.includes(block));
  }
  const handover = buildHandoverMessages(previous, sources);
  assert.doesNotMatch(handover[0].content, /đoạn riêng/);
});

test("files: decodeTextBytes handles UTF-8 VI/EN, rejects empty, binary, invalid UTF-8, and PDF", () => {
  const enc = new TextEncoder();
  const vi = "Lan gửi báo cáo trước thứ Sáu.\nMinh xác nhận phòng họp.";
  assert.equal(decodeTextBytes(enc.encode(vi), "notes.txt"), vi);

  assert.throws(() => decodeTextBytes(new Uint8Array([]), "empty.txt"), /rỗng/);
  assert.throws(() => decodeTextBytes(new Uint8Array([0x25, 0x50, 0x44, 0x46]), "scan.pdf"), /PDF/);
  assert.throws(() => decodeTextBytes(new Uint8Array([0x41, 0x00, 0x42]), "bin.dat"), /nhị phân/);
  assert.throws(() => decodeTextBytes(new Uint8Array([0xff, 0xfe, 0xfd]), "bad.txt"), /UTF-8/);
});

test("files: sliceSourceContent, deduplication against history, and exact UTF-8 payload byte count", () => {
  const src = {
    id: "s1",
    name: "data.csv",
    content: "col1,col2\na,1\nb,2\nc,3",
    fromLine: 2,
    toLine: 3,
    selected: true,
    pinned: true,
  };
  const sliced = sliceSourceContent(src);
  assert.equal(sliced.text, "a,1\nb,2");
  assert.equal(sliced.from, 2);
  assert.equal(sliced.to, 3);
  assert.equal(sliced.total, 4);

  const block = formatSourceBlock(src);
  assert.ok(block.includes("=== NGUỒN: data.csv (dòng 2–3/4) ==="));
  assert.ok(block.includes("a,1\nb,2"));

  // First turn: source is included in payload
  const req1 = buildChatRequest({
    previous: [],
    prompt: "Tóm tắt",
    sources: [src],
    contextLimit: 32768,
  });
  assert.equal(req1.skippedDuplicates.length, 0);
  assert.ok(req1.sent[0].content.includes("a,1\nb,2"));
  assert.equal(req1.payloadBytes, measurePayloadBytes(req1.payload));
  assert.ok(req1.payloadBytes < DEFAULT_MAX_REQUEST_BYTES);

  // Second turn with same history: pinned source is NOT duplicated
  const historyAfterTurn1 = [
    req1.messages[0],
    { role: "assistant", content: "Đã tóm tắt." },
  ];
  const dedup = collectActiveSourceBlocks([src], historyAfterTurn1);
  assert.equal(dedup.included.length, 0);
  assert.deepEqual(dedup.skippedDuplicates, ["data.csv"]);

  // Unchecked source is omitted completely
  const reqUnchecked = buildChatRequest({
    previous: [],
    prompt: "Xin chào",
    sources: [{ ...src, selected: false }],
    contextLimit: 32768,
  });
  assert.equal(reqUnchecked.sent[0].content, "Xin chào");
  assert.equal(formatBytes(512 * 1024), "512 KiB");
});

test("files: JSON escaping and multi-byte UTF-8 are reflected accurately in payloadBytes", () => {
  const jsonText = '{"quote": "Tiếng Việt có dấu \\"nháy kép\\"\\nDòng 2"}';
  const req = buildChatRequest({
    prompt: jsonText,
    contextLimit: 32768,
  });
  const expected = new TextEncoder().encode(JSON.stringify(req.payload)).length;
  assert.equal(req.payloadBytes, expected);
  assert.ok(req.payloadBytes > jsonText.length);
});

test("files: buildHandoverMessages preserves revised deadline turns and pinned source names", () => {
  const history = [
    { role: "user", content: "Mai phụ trách gửi biên bản vào thứ Sáu." },
    { role: "assistant", content: "Đã ghi Mai gửi biên bản thứ Sáu." },
    { role: "user", content: "Dời hạn đó sang thứ Hai, người phụ trách giữ nguyên." },
    { role: "assistant", content: "Đã cập nhật hạn sang thứ Hai." },
  ];
  const pinned = [{ id: "p1", name: "brief.md", content: "Mục tiêu Q4", pinned: true, selected: true }];
  const msgs = buildHandoverMessages(history, pinned, 0);
  assert.equal(msgs.length, 1);
  assert.ok(msgs[0].content.includes("CHỈ giữ thông tin mới nhất"));
  assert.ok(msgs[0].content.includes("brief.md"));
  assert.ok(msgs[0].content.includes("thứ Hai"));
});

test("storage: backup round-trips chat.sources and chat.parentId cleanly", () => {
  const stateLike = {
    conversations: [
      {
        id: "child-1",
        parentId: "parent-1",
        title: "Bàn giao · Họp",
        sources: [{ id: "s1", name: "brief.md", content: "Line 1\nLine 2", fromLine: 1, toLine: 2, selected: true, pinned: true }],
        messages: [{ role: "user", content: "Handover" }, { role: "assistant", content: "Ready" }],
      },
    ],
    active: { id: "child-1" },
    newDraft: "",
    newSystem: "",
    newSampling: defaultSampling(),
    newSources: [],
    customPrompts: [],
  };
  const parsed = readBackup(snapshot(stateLike));
  assert.equal(parsed.chats[0].parentId, "parent-1");
  assert.equal(parsed.chats[0].sources.length, 1);
  assert.equal(parsed.chats[0].sources[0].pinned, true);
  assert.equal(parsed.chats[0].sources[0].name, "brief.md");
});
