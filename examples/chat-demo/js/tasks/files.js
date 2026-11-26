export const DEFAULT_MAX_REQUEST_BYTES = 512 * 1024;
export const MAX_READ_FILE_BYTES = 2 * 1024 * 1024;

export const ALLOWED_TEXT_EXTENSIONS = new Set([
  ".txt", ".md", ".csv", ".json", ".log", ".py",
  ".js", ".mjs", ".ts", ".jsx", ".tsx", ".yaml", ".yml",
  ".xml", ".sql", ".sh", ".html", ".css", ".toml", ".ini", ".env",
]);

const utf8Encoder = new TextEncoder();

export function getExtension(filename) {
  const name = String(filename || "").trim().toLowerCase();
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot) : "";
}

export function isLikelyBinary(bytes) {
  const len = Math.min(bytes.length, 8192);
  if (len === 0) return false;
  let control = 0;
  for (let i = 0; i < len; i++) {
    const b = bytes[i];
    if (b === 0) return true;
    if (b < 7 || (b > 13 && b < 32)) control++;
  }
  return control / len > 0.1;
}

export function decodeTextBytes(bytes, filename = "") {
  const ext = getExtension(filename);
  if (ext === ".pdf") {
    throw new Error("Tệp PDF chưa hỗ trợ trích xuất tự động (PDF scan cần OCR). Hãy xuất sang .txt/.md hoặc dán văn bản.");
  }
  if (!bytes || bytes.length === 0) {
    throw new Error(`Tệp "${filename || "đã chọn"}" rỗng (0 byte).`);
  }
  if (bytes.length > MAX_READ_FILE_BYTES) {
    throw new Error(`Tệp "${filename}" quá lớn (${formatBytes(bytes.length)}). Giới hạn đọc cục bộ là ${formatBytes(MAX_READ_FILE_BYTES)}.`);
  }
  if (isLikelyBinary(bytes)) {
    throw new Error(`Tệp "${filename}" có vẻ là tệp nhị phân, không phải văn bản thuần.`);
  }
  try {
    const decoder = new TextDecoder("utf-8", { fatal: true });
    return decoder.decode(bytes);
  } catch {
    throw new Error(`Không thể giải mã UTF-8 cho tệp "${filename}". Hãy chuyển tệp sang UTF-8 trước khi nạp.`);
  }
}

export async function readLocalTextFile(file) {
  if (!file) throw new Error("Chưa chọn tệp.");
  const ext = getExtension(file.name);
  if (ext === ".pdf") {
    throw new Error("Tệp PDF chưa hỗ trợ trích xuất tự động (PDF scan cần OCR). Hãy xuất sang .txt/.md hoặc dán văn bản.");
  }
  if (file.size === 0) {
    throw new Error(`Tệp "${file.name}" rỗng (0 byte).`);
  }
  if (file.size > MAX_READ_FILE_BYTES) {
    throw new Error(`Tệp "${file.name}" quá lớn (${formatBytes(file.size)}). Giới hạn đọc cục bộ là ${formatBytes(MAX_READ_FILE_BYTES)}.`);
  }
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const content = decodeTextBytes(bytes, file.name);
  const lineCount = content.split("\n").length;
  return {
    id: (globalThis.crypto?.randomUUID?.() || `src-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`),
    name: file.name,
    content,
    lineCount,
    fromLine: 1,
    toLine: lineCount,
    selected: true,
    pinned: false,
  };
}

export function sliceSourceContent(source) {
  const lines = String(source.content || "").split("\n");
  const total = lines.length;
  const from = Math.max(1, Math.min(total, Number(source.fromLine) || 1));
  const to = Math.max(from, Math.min(total, Number(source.toLine) || total));
  const text = lines.slice(from - 1, to).join("\n");
  return { text, from, to, total };
}

export function formatSourceBlock(source) {
  const { text, from, to, total } = sliceSourceContent(source);
  const rangeLabel = from === 1 && to === total ? `toàn bộ ${total} dòng` : `dòng ${from}–${to}/${total}`;
  return `=== NGUỒN: ${source.name} (${rangeLabel}) ===\n${text}\n=== HẾT NGUỒN ===`;
}

// Old conversations already store formatted attachments inline. Match the
// original source and range so unrelated prose and same-named files stay intact.
export function omitDeselectedSources(content, sources) {
  let result = content;
  for (const source of sources) {
    if (source?.selected !== false) continue;
    const name = source.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const header = new RegExp(String.raw`^=== NGUỒN: ${name} \((?:toàn bộ (\d+) dòng|dòng (\d+)–(\d+)/\d+)\) ===$`, "gm");
    for (const match of content.matchAll(header)) {
      const block = formatSourceBlock({
        ...source,
        fromLine: Number(match[2] || 1),
        toLine: Number(match[3] || match[1]),
      });
      if (content.startsWith(block, match.index)) result = result.replaceAll(block, "");
    }
  }
  return result === content ? content : (result.trim() || "[Nguồn đính kèm đã bỏ chọn]");
}

// Build combined source block while skipping sources whose exact formatted block
// already appears in the history messages that will be sent.
export function collectActiveSourceBlocks(sources = [], historyMessages = []) {
  const historyBlob = historyMessages.map(m => m.content || "").join("\n");
  const included = [];
  const skippedDuplicates = [];
  for (const src of sources) {
    if (!src || src.selected === false) continue;
    const block = formatSourceBlock(src);
    if (historyBlob.includes(block)) {
      skippedDuplicates.push(src.name);
      continue;
    }
    included.push({ source: src, block });
  }
  return { included, skippedDuplicates };
}

export function measurePayloadBytes(payloadObj) {
  return utf8Encoder.encode(JSON.stringify(payloadObj)).length;
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const kib = bytes / 1024;
  return `${kib >= 100 ? kib.toFixed(0) : kib.toFixed(1)} KiB`;
}

export function buildHandoverMessages(historyMessages = [], pinnedSources = [], maxTurns = 0) {
  const usable = historyMessages.filter((m, idx) =>
    m.content && !m.partial && !historyMessages[idx + 1]?.partial);
  const selected = maxTurns > 0 ? usable.slice(Math.max(0, usable.length - maxTurns * 2)) : usable;
  const transcript = selected.map(m =>
    `${m.role === "user" ? "Người dùng" : "Trợ lý"}:\n${omitDeselectedSources(m.content, pinnedSources)}`
  ).join("\n\n---\n\n");
  const pinnedNames = pinnedSources.filter(s => s.pinned && s.selected !== false).map(s => s.name);

  const instruction = [
    "Tạo bản ghi chú bàn giao ngắn gọn, chính xác để tiếp tục công việc trong một cuộc hội thoại mới (giữ ngôn ngữ chính của hội thoại).",
    "Nếu thông tin, hạn hoặc quyết định bị đính chính ở lượt sau, CHỈ giữ thông tin mới nhất và loại bỏ giá trị cũ đã bị thay thế.",
    "Không nhắc đến hạn cũ ở bất kỳ mục nào, kể cả trong cụm 'thay cho' hoặc khi thuật lại quyết định đính chính.",
    "Trình bày đúng 4 mục Markdown:",
    "### Dữ kiện hiện hành",
    "### Quyết định đã chốt",
    "### Việc đang mở",
    "### Nguồn ghim đang dùng",
    pinnedNames.length ? `(Các nguồn đang ghim: ${pinnedNames.join(", ")})` : "(Không có nguồn ghim kèm theo)",
    "Không bịa thêm chi tiết không có trong hội thoại.",
  ].join("\n");

  return [
    { role: "user", content: `${instruction}\n\nNội dung hội thoại cần bàn giao:\n\n${transcript}` },
  ];
}
