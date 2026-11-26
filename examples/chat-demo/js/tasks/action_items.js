export const GUI_ITEM_FIELDS = {
  task: { type: "string" },
  owner: { type: ["string", "null"] },
  due: { type: ["string", "null"] },
  source_quote: { type: "string" },
};

export const GUI_ACTION_ITEMS_FORMAT = {
  type: "json_schema",
  json_schema: {
    name: "action_items_with_evidence",
    strict: true,
    schema: {
      type: "object",
      properties: {
        items: {
          type: "array",
          items: {
            type: "object",
            properties: GUI_ITEM_FIELDS,
            required: ["task", "owner", "due", "source_quote"],
            additionalProperties: false,
          },
        },
      },
      required: ["items"],
      additionalProperties: false,
    },
  },
};

export function supportsActionItemsSchema(health) {
  return Boolean(
    health
    && health.backend === "llama"
    && health.capabilities?.response_format === true
    && health.capabilities?.json_schema === true
  );
}

export function buildActionItemsPrompt(notes, structured = true) {
  const instructions = [
    "Chỉ liệt kê cam kết hoặc đầu việc đã được nêu rõ ràng (tiếng Việt hoặc tiếng Anh, giữ nguyên ngôn ngữ của ghi chú).",
    "Bỏ qua người chỉ hỏi, nêu ý kiến, góp ý, đề xuất chưa được chốt hoặc nói rõ họ không nhận việc.",
    "Nếu một đầu việc đã nêu rõ nhưng chưa giao cho ai, đặt owner là null (chưa giao).",
    "Chép nguyên cụm hạn trong ghi chú, kể cả các từ như 'trước', 'chậm nhất', 'by', 'before'. Nếu có hạn bị đính chính ở câu sau, lấy hạn mới nhất. Nếu thiếu hạn, đặt due là null.",
    "Trường source_quote phải chép nguyên văn câu hoặc mệnh đề ngắn từ ghi chú làm bằng chứng cho đầu việc và hạn đó. Không tự suy diễn hoặc bổ sung người/hạn.",
  ].join(" ");

  const output = structured
    ? "Trả về JSON đúng dạng {\"items\":[{\"task\":\"...\",\"owner\":\"...\"|null,\"due\":\"...\"|null,\"source_quote\":\"...\"}]}. Nếu ghi chú không có đầu việc nào, trả {\"items\":[]}."
    : "Trả về bảng Markdown gồm 4 cột: Việc | Người phụ trách | Hạn | Câu nguồn. Ghi 'Chưa giao' khi thiếu người, 'Chưa có hạn' khi thiếu hạn. Nếu không có đầu việc nào, ghi rõ 'Không có đầu việc được giao.'";

  return `${instructions}\n${output}\n\nGhi chú:\n${notes.trim()}`;
}

export function parseActionItems(raw) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ok: false, error: "Phản hồi không phải JSON hợp lệ." };
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { ok: false, error: "Cần đối tượng JSON có trường items." };
  }
  const topKeys = Object.keys(parsed);
  if (topKeys.length !== 1 || topKeys[0] !== "items" || !Array.isArray(parsed.items)) {
    return { ok: false, error: "Cần JSON có đúng một trường items dạng danh sách." };
  }
  const allowedKeys = new Set(["task", "owner", "due", "source_quote"]);
  const items = [];
  for (const item of parsed.items) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return { ok: false, error: "Mỗi phần tử trong items phải là một đối tượng." };
    }
    const keys = Object.keys(item);
    if (keys.length !== allowedKeys.size || keys.some(k => !allowedKeys.has(k))) {
      return { ok: false, error: "Mỗi đầu việc phải có đúng 4 trường: task, owner, due, source_quote." };
    }
    if (typeof item.task !== "string" || !item.task.trim()) {
      return { ok: false, error: "Trường task phải là chuỗi không rỗng." };
    }
    if (item.owner !== null && (typeof item.owner !== "string" || !item.owner.trim())) {
      return { ok: false, error: "Trường owner phải là chuỗi không rỗng hoặc null." };
    }
    if (item.due !== null && (typeof item.due !== "string" || !item.due.trim())) {
      return { ok: false, error: "Trường due phải là chuỗi không rỗng hoặc null." };
    }
    if (typeof item.source_quote !== "string" || !item.source_quote.trim()) {
      return { ok: false, error: "Trường source_quote phải là chuỗi không rỗng." };
    }
    items.push({
      task: item.task.trim(),
      owner: item.owner === null ? null : item.owner.trim(),
      due: item.due === null ? null : item.due.trim(),
      source_quote: item.source_quote.trim(),
    });
  }
  return { ok: true, items };
}

function normalizeChar(ch) {
  if (ch === "\u201c" || ch === "\u201d" || ch === "\u201e") return "\"";
  if (ch === "\u2018" || ch === "\u2019") return "'";
  if (ch === "\u2013" || ch === "\u2014") return "-";
  if (/\s/.test(ch)) return " ";
  return ch;
}

function normalizeWithMap(text) {
  let out = "";
  const map = [];
  let prevSpace = false;
  for (let i = 0; i < text.length; i++) {
    const ch = normalizeChar(text[i]);
    if (ch === " ") {
      if (prevSpace || out.length === 0) continue;
      prevSpace = true;
    } else {
      prevSpace = false;
    }
    out += ch;
    map.push(i);
  }
  if (out.endsWith(" ")) {
    out = out.slice(0, -1);
    map.pop();
  }
  return { text: out, map };
}

export function locateQuote(sourceText, quote) {
  if (typeof sourceText !== "string" || typeof quote !== "string") {
    return { found: false, exact: false, start: -1, end: -1 };
  }
  const trimmed = quote.trim();
  if (!trimmed) return { found: false, exact: false, start: -1, end: -1 };

  const exactIdx = sourceText.indexOf(trimmed);
  if (exactIdx !== -1) {
    return { found: true, exact: true, start: exactIdx, end: exactIdx + trimmed.length };
  }

  const normSource = normalizeWithMap(sourceText);
  const normQuote = normalizeWithMap(trimmed).text;
  if (!normQuote) return { found: false, exact: false, start: -1, end: -1 };

  const normIdx = normSource.text.indexOf(normQuote);
  if (normIdx !== -1) {
    const start = normSource.map[normIdx];
    const lastNormIdx = normIdx + normQuote.length - 1;
    const end = normSource.map[lastNormIdx] + 1;
    return { found: true, exact: false, start, end };
  }

  return { found: false, exact: false, start: -1, end: -1 };
}

export function analyzeActionItems(items, sourceText) {
  return items.map((item, index) => {
    const match = locateQuote(sourceText, item.source_quote);
    return {
      ...item,
      index,
      match,
      needsReview: !match.found,
    };
  });
}

export function displayOwner(owner) {
  return owner === null ? "Chưa giao" : owner;
}

export function displayDue(due) {
  return due === null ? "Chưa có hạn" : due;
}

// Prefix leading spreadsheet formula triggers so Excel/Sheets/Calc keep them as text.
export function sanitizeSpreadsheetCell(value) {
  const str = value == null ? "" : String(value);
  if (/^[\s\u0000-\u001f]*[=+\-@]/.test(str) || /^[\t\r]/.test(str)) {
    return "'" + str;
  }
  return str;
}

function escapeCsvCell(value) {
  const safe = sanitizeSpreadsheetCell(value);
  if (/[",\r\n]/.test(safe)) {
    return `"${safe.replace(/"/g, "\"\"")}"`;
  }
  return safe;
}

function escapeTsvCell(value) {
  const safe = sanitizeSpreadsheetCell(value);
  if (/[\t"\r\n]/.test(safe)) {
    return `"${safe.replace(/"/g, "\"\"")}"`;
  }
  return safe;
}

const EXPORT_HEADERS = ["Việc", "Người phụ trách", "Hạn", "Câu nguồn"];

function itemToCells(item) {
  return [
    item.task,
    item.owner === null ? "Chưa giao" : item.owner,
    item.due === null ? "" : item.due,
    item.source_quote,
  ];
}

export function formatCsv(items) {
  const rows = [EXPORT_HEADERS, ...items.map(itemToCells)];
  return rows.map(row => row.map(escapeCsvCell).join(",")).join("\r\n");
}

export function formatTsv(items) {
  const rows = [EXPORT_HEADERS, ...items.map(itemToCells)];
  return rows.map(row => row.map(escapeTsvCell).join("\t")).join("\r\n");
}

export function formatChecklist(items) {
  if (!items.length) return "Không có đầu việc được giao.";
  return items.map(item => {
    const owner = displayOwner(item.owner);
    const due = item.due ? ` · Hạn: ${item.due}` : "";
    return `- [ ] ${item.task} — **${owner}**${due}`;
  }).join("\n");
}

export function formatJson(items) {
  const clean = items.map(({ task, owner, due, source_quote }) => ({ task, owner, due, source_quote }));
  return JSON.stringify({ items: clean }, null, 2);
}
