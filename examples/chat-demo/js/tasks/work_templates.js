export const REWRITE_AUDIENCES = [
  { value: "general", label: "Chung" },
  { value: "manager", label: "Quản lý / Sếp" },
  { value: "team", label: "Đồng nghiệp / Nhóm" },
  { value: "client", label: "Khách hàng / Đối tác" },
];

export const REWRITE_GOALS = [
  { value: "polish", label: "Chỉnh rõ & súc tích" },
  { value: "update", label: "Báo cáo tiến độ" },
  { value: "confirm", label: "Xin xác nhận / phê duyệt" },
  { value: "notice", label: "Thông báo chính thức" },
];

export const REWRITE_TONES = [
  { value: "clear", label: "Rõ ràng, trung lập" },
  { value: "polite", label: "Lịch sự, tôn trọng" },
  { value: "concise", label: "Ngắn gọn, trực diện" },
  { value: "formal", label: "Trang trọng" },
];

export function countWords(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).filter(Boolean).length;
}

export function buildSummaryPrompt(notes, { targetWords } = {}) {
  const parts = [
    "Tóm tắt văn bản dưới đây bằng đúng ngôn ngữ của bản gốc.",
    "Tách rõ 3 mục bằng tiêu đề Markdown: ### Quyết định, ### Đề xuất, ### Câu hỏi còn mở (nếu mục nào không có, ghi rõ 'Không có').",
    "Giữ nguyên mọi tên riêng, số liệu, phần trăm, tiền tệ và mốc thời gian. Không suy diễn hay nói là đã hoàn tất nếu bản gốc chưa chốt.",
  ];
  const limit = Number(targetWords);
  if (Number.isInteger(limit) && limit > 0) {
    parts.push(`Độ dài mục tiêu: khoảng ${limit} từ (đếm theo khoảng trắng).`);
  }
  return `${parts.join(" ")}\n\nVăn bản gốc:\n${notes.trim()}`;
}

export function buildRewritePrompt(draft, { audience = "general", goal = "polish", tone = "clear" } = {}) {
  const audienceLabel = REWRITE_AUDIENCES.find(a => a.value === audience)?.label || "Chung";
  const goalLabel = REWRITE_GOALS.find(g => g.value === goal)?.label || "Chỉnh rõ & súc tích";
  const toneLabel = REWRITE_TONES.find(t => t.value === tone)?.label || "Rõ ràng, trung lập";
  const instructions = [
    `Biên tập lại bản nháp dưới đây (giữ nguyên ngôn ngữ của bản gốc).`,
    `Người đọc: ${audienceLabel}. Mục đích: ${goalLabel}. Giọng văn: ${toneLabel}.`,
    "Bắt buộc giữ nguyên chính xác mọi tên riêng, mã định danh, số liệu, số tiền và mốc ngày/giờ/hạn. Không thêm dữ kiện mới. Chỉ trả về bản đã chỉnh sửa.",
  ].join(" ");
  return `${instructions}\n\nBản gốc:\n${draft.trim()}`;
}

// Extract factual tokens (numbers/money/times, dates/deadlines, capitalized proper names/IDs)
// so the rewrite review panel can flag any changed or dropped facts between original and edited text.
const NUMBER_RE = /(?:\$|€|£|₫)?\b\d+(?:[.,:/-]\d+)*(?:\s?(?:%|đồng|VND|USD|h|giờ|phút|ngày|tuần|tháng))?\b/gu;
const DATE_PHRASE_RE = /\b(?:thứ\s+(?:Hai|Ba|Tư|Năm|Sáu|Bảy)|Chủ\s+[Nn]hật|ngày\s+\d{1,2}(?:\/\d{1,2})?|hôm\s+nay|ngày\s+mai|sáng\s+mai|chiều\s+mai|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2})\b/gu;
const PROPER_NAME_RE = /\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ][\p{L}\d-]{1,}\b/gu;

const STOP_CAPS = new Set([
  "The", "This", "That", "These", "Those", "There", "Here", "When", "Where", "What", "Which",
  "With", "From", "Into", "About", "After", "Before", "Under", "Over", "Please", "Dear", "Hello",
  "Hi", "Thanks", "Best", "Regards", "Note", "Original", "Draft", "Summary", "Decision", "Decisions",
  "Xin", "Chào", "Kính", "Gửi", "Thân", "Cảm", "Nhóm", "Bên", "Theo", "Trong", "Ngoài", "Nếu",
  "Khi", "Cần", "Đã", "Đang", "Sẽ", "Không", "Chưa", "Một", "Các", "Những", "Bản", "Mục", "Ngày",
]);

function collectMatches(text, regex, filterFn) {
  const found = new Set();
  for (const match of String(text || "").matchAll(regex)) {
    const val = match[0].trim();
    if (!val) continue;
    if (filterFn && !filterFn(val)) continue;
    found.add(val);
  }
  return found;
}

export function extractFacts(text) {
  const dates = collectMatches(text, DATE_PHRASE_RE);
  const numbers = collectMatches(text, NUMBER_RE);
  const names = collectMatches(text, PROPER_NAME_RE, token => !STOP_CAPS.has(token) && !dates.has(token));
  return { numbers, dates, names };
}

export function compareFacts(originalText, revisedText) {
  const orig = extractFacts(originalText);
  const rev = extractFacts(revisedText);

  const diffSet = (a, b) => ({
    missing: [...a].filter(x => !b.has(x)),
    added: [...b].filter(x => !a.has(x)),
  });

  const numbers = diffSet(orig.numbers, rev.numbers);
  const dates = diffSet(orig.dates, rev.dates);
  const names = diffSet(orig.names, rev.names);

  const totalChanges = numbers.missing.length + numbers.added.length
    + dates.missing.length + dates.added.length
    + names.missing.length + names.added.length;

  return { numbers, dates, names, totalChanges };
}
