export const DEFAULT_PROMPTS = [
  {
    id: "builtin-tomtat",
    slash: "/tomtat",
    name: "Tóm tắt có cấu trúc",
    description: "Tách riêng Quyết định, Đề xuất và Câu hỏi còn mở.",
    builtin: true,
    task: "summary",
    template: "Tóm tắt văn bản sau, tách rõ 3 mục: (1) Quyết định đã chốt, (2) Đề xuất chưa chốt, (3) Câu hỏi còn mở. Giữ nguyên mọi tên, số liệu và mốc hạn, không suy diễn thêm.\n\n{{selection}}",
  },
  {
    id: "builtin-action",
    slash: "/action",
    name: "Rút đầu việc có bằng chứng",
    description: "Chuyển sang chế độ rút đầu việc (Bảng + Câu nguồn + CSV/TSV).",
    builtin: true,
    task: "action_items",
    template: "{{selection}}",
  },
  {
    id: "builtin-soan",
    slash: "/soan",
    name: "Soạn / sửa văn bản",
    description: "Chỉnh rõ ý, đối chiếu bản gốc cạnh bản sửa và kiểm tên/số/hạn.",
    builtin: true,
    task: "rewrite",
    template: "Biên tập và chỉnh sửa bản nháp sau sao cho rõ ràng, súc tích, đúng giọng văn nhưng giữ nguyên mọi tên riêng, số liệu và mốc hạn:\n\n{{selection}}",
  },
  {
    id: "builtin-email",
    slash: "/email",
    name: "Viết email công việc",
    description: "Soạn email có tiêu đề, rõ mục đích và lời đề nghị hành động.",
    builtin: true,
    template: "Soạn email công việc lịch sự, súc tích (có tiêu đề) dựa trên các ý sau. Ngày hôm nay: {{date}}.\n\n{{selection}}",
  },
  {
    id: "builtin-dich",
    slash: "/dich",
    name: "Dịch Việt ↔ Anh",
    description: "Dịch tự nhiên, giữ nguyên thuật ngữ, tên riêng và số liệu.",
    builtin: true,
    template: "Dịch văn bản sau (Việt sang Anh hoặc Anh sang Việt tùy ngôn ngữ đầu vào). Giữ nguyên tên riêng, mã định danh, số liệu và mốc thời gian:\n\n{{selection}}",
  },
  {
    id: "builtin-review",
    slash: "/review",
    name: "Rà soát lỗi & điểm mơ hồ",
    description: "Tìm mâu thuẫn, thiếu dữ kiện, sai số hoặc câu khó hiểu.",
    builtin: true,
    template: "Rà soát văn bản sau để tìm điểm mâu thuẫn, dữ kiện thiếu, hạn/người chưa rõ và đề xuất cách sửa ngắn gọn:\n\n{{selection}}",
  },
  {
    id: "builtin-sosanh",
    slash: "/sosanh",
    name: "So sánh & chọn phương án",
    description: "Lập bảng ưu/nhược điểm, rủi ro và đề xuất bước tiếp theo.",
    builtin: true,
    template: "So sánh các phương án sau, liệt kê ưu/nhược điểm, rủi ro, thông tin còn thiếu và đề xuất bước tiếp theo:\n\n{{selection}}",
  },
];

export function formatLocalDate(now = new Date()) {
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function normalizeSlash(raw) {
  const trimmed = String(raw || "").trim().toLowerCase().replace(/\s+/g, "-");
  if (!trimmed) return "";
  const cleaned = trimmed.replace(/^\/+/, "").replace(/[^a-z0-9_-]/g, "");
  return cleaned ? `/${cleaned}` : "";
}

export function getAllPrompts(customPrompts = []) {
  const customById = new Map();
  const extra = [];
  for (const item of customPrompts) {
    if (!item || typeof item.id !== "string") continue;
    if (DEFAULT_PROMPTS.some(d => d.id === item.id)) {
      customById.set(item.id, item);
    } else {
      extra.push(item);
    }
  }
  const mergedBuiltins = DEFAULT_PROMPTS.map(item =>
    customById.has(item.id) ? { ...item, ...customById.get(item.id), builtin: true } : item
  );
  return [...mergedBuiltins, ...extra];
}

export function filterPromptsBySlash(prompts, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q || q === "/") return prompts;
  const needle = q.startsWith("/") ? q.slice(1) : q;
  return prompts.filter(item =>
    item.slash.slice(1).toLowerCase().includes(needle)
    || item.name.toLowerCase().includes(needle)
  );
}

export function detectSlashQuery(text, cursorPos = text.length) {
  const before = text.slice(0, cursorPos);
  const match = before.match(/(?:^|\s)(\/[a-zA-Z0-9_-]*)$/);
  if (!match) return null;
  const token = match[1];
  const start = cursorPos - token.length;
  return { query: token, start, end: cursorPos };
}

export function templateUsesVariable(template, name) {
  return String(template || "").includes(`{{${name}}}`);
}

export function expandTemplate(template, { date, selection = "", clipboard = "" } = {}) {
  const dateStr = date || formatLocalDate();
  return String(template || "")
    .replaceAll("{{date}}", dateStr)
    .replaceAll("{{selection}}", selection)
    .replaceAll("{{clipboard}}", clipboard);
}

export function validateCustomPrompt(entry) {
  if (!entry || typeof entry !== "object") throw new Error("Prompt không hợp lệ.");
  const name = String(entry.name || "").trim();
  const slash = normalizeSlash(entry.slash);
  const template = String(entry.template || "").trim();
  const description = String(entry.description || "").trim();
  if (!name) throw new Error("Tên mẫu không được để trống.");
  if (!slash) throw new Error("Lệnh slash chỉ gồm chữ, số, gạch ngang hoặc gạch dưới (ví dụ /baocao).");
  if (!template) throw new Error("Nội dung mẫu không được để trống.");
  return {
    id: typeof entry.id === "string" && entry.id.trim() ? entry.id.trim() : `custom-${Date.now().toString(36)}`,
    slash,
    name,
    description,
    template,
  };
}
