import { compareFacts, countWords } from "./work_templates.js";

export function renderSummaryMeta(content, targetWords) {
  const bar = document.createElement("div");
  bar.className = "work-meta-bar";
  const words = countWords(content);
  const badge = document.createElement("span");
  badge.className = "badge-accent";
  badge.textContent = targetWords
    ? `${words} / ~${targetWords} từ`
    : `${words} từ`;
  const note = document.createElement("span");
  note.className = "work-meta-note";
  note.textContent = "Quy ước đếm: theo khoảng trắng (Việt–Anh).";
  bar.append(badge, note);
  return bar;
}

function appendFactLine(list, label, diff) {
  if (!diff.missing.length && !diff.added.length) return;
  const li = document.createElement("li");
  const parts = [];
  if (diff.missing.length) parts.push(`Thiếu so với bản gốc: ${diff.missing.join(", ")}`);
  if (diff.added.length) parts.push(`Xuất hiện mới: ${diff.added.join(", ")}`);
  li.textContent = `${label} — ${parts.join(" · ")}`;
  list.append(li);
}

export function renderRewriteReview(originalText, revisedText) {
  const details = document.createElement("details");
  details.className = "rewrite-review";

  const diff = compareFacts(originalText, revisedText);
  const summary = document.createElement("summary");
  summary.className = "rewrite-review-summary";
  const label = document.createElement("span");
  label.textContent = "Đối chiếu bản gốc ↔ bản sửa";
  const badge = document.createElement("span");
  badge.className = diff.totalChanges > 0 ? "badge-warn" : "badge-accent";
  badge.textContent = diff.totalChanges > 0
    ? `${diff.totalChanges} thay đổi ở tên/số/hạn cần duyệt`
    : "Tên, số liệu và hạn được giữ nguyên";
  summary.append(label, badge);
  details.append(summary);

  const body = document.createElement("div");
  body.className = "rewrite-review-body";

  if (diff.totalChanges > 0) {
    const ul = document.createElement("ul");
    ul.className = "rewrite-fact-list";
    appendFactLine(ul, "Số liệu", diff.numbers);
    appendFactLine(ul, "Ngày / hạn", diff.dates);
    appendFactLine(ul, "Tên riêng", diff.names);
    body.append(ul);
  }

  const grid = document.createElement("div");
  grid.className = "rewrite-compare";

  const left = document.createElement("div");
  left.className = "rewrite-col";
  const leftHead = document.createElement("h4");
  leftHead.textContent = `Bản gốc (${countWords(originalText)} từ)`;
  const leftPre = document.createElement("pre");
  leftPre.textContent = originalText;
  left.append(leftHead, leftPre);

  const right = document.createElement("div");
  right.className = "rewrite-col";
  const rightHead = document.createElement("h4");
  rightHead.textContent = `Bản sửa (${countWords(revisedText)} từ)`;
  const rightPre = document.createElement("pre");
  rightPre.textContent = revisedText;
  right.append(rightHead, rightPre);

  grid.append(left, right);
  body.append(grid);
  details.append(body);
  return details;
}
