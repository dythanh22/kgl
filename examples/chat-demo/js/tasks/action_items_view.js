import { copyButton, displayAnswer } from "../markdown.js";
import {
  analyzeActionItems,
  displayDue,
  displayOwner,
  formatChecklist,
  formatCsv,
  formatJson,
  formatTsv,
  parseActionItems,
} from "./action_items.js";

function makeBadge(text, className) {
  const span = document.createElement("span");
  span.className = className;
  span.textContent = text;
  return span;
}

function makeDisabledButton(label) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "copy";
  btn.textContent = label;
  btn.disabled = true;
  return btn;
}

function renderSourceText(container, sourceText, match, quote) {
  container.replaceChildren();
  const note = document.createElement("p");
  note.className = "ai-source-note";
  if (!sourceText) {
    note.textContent = "Không có bản ghi chú nguồn đi kèm.";
    container.append(note);
    return;
  }
  if (!match.found) {
    note.textContent = `Không tìm thấy nguyên văn câu nguồn trong bản nhập: "${quote}". Kiểm tra thủ công bên dưới:`;
    note.dataset.review = "true";
    const body = document.createElement("pre");
    body.className = "ai-source-body";
    body.textContent = sourceText;
    container.append(note, body);
    return;
  }
  note.textContent = match.exact
    ? "Đã khớp nguyên văn trong bản ghi chú nguồn:"
    : "Đã khớp sau khi chuẩn hóa khoảng trắng/dấu ngoặc kép:";
  const body = document.createElement("pre");
  body.className = "ai-source-body";
  const before = sourceText.slice(0, match.start);
  const hit = sourceText.slice(match.start, match.end);
  const after = sourceText.slice(match.end);
  const mark = document.createElement("mark");
  mark.className = "ai-highlight";
  mark.textContent = hit;
  body.append(document.createTextNode(before), mark, document.createTextNode(after));
  container.append(note, body);
  requestAnimationFrame(() => {
    mark.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
}

function renderInvalidState(reason, raw) {
  const wrap = document.createElement("div");
  wrap.className = "ai-card";
  const banner = document.createElement("div");
  banner.className = "ai-error-banner";
  const msg = document.createElement("span");
  msg.textContent = `${reason} Đã vô hiệu hóa xuất dữ liệu đã xác nhận.`;
  const actions = document.createElement("div");
  actions.className = "ai-export-actions";
  actions.append(makeDisabledButton("Copy CSV"), makeDisabledButton("Copy TSV"));
  banner.append(msg, actions);
  wrap.append(banner);
  if (raw) {
    const pre = document.createElement("pre");
    pre.className = "ai-raw";
    pre.textContent = raw;
    wrap.append(pre);
  }
  return wrap;
}

export function renderActionItemsCard({ raw, sourceText, partial, live, structured }) {
  const frag = document.createDocumentFragment();

  if (!structured) {
    const notice = document.createElement("div");
    notice.className = "ai-fallback-banner";
    notice.textContent = "Chế độ văn bản · Worker hiện tại chưa hỗ trợ JSON Schema.";
    frag.append(notice);
    if (raw) {
      frag.append(displayAnswer(raw));
    } else if (live) {
      const indicator = document.createElement("span");
      indicator.className = "thinking-indicator";
      indicator.textContent = "Đang trích xuất đầu việc…";
      frag.append(indicator);
    }
    return frag;
  }

  if (live) {
    const indicator = document.createElement("span");
    indicator.className = "thinking-indicator";
    indicator.textContent = "Đang trích xuất đầu việc (JSON Schema)…";
    frag.append(indicator);
    if (raw) {
      const pre = document.createElement("pre");
      pre.className = "ai-raw";
      pre.textContent = raw;
      frag.append(pre);
    }
    return frag;
  }

  if (partial) {
    frag.append(renderInvalidState("Phản hồi chưa hoàn tất (stream bị ngắt hoặc chạm giới hạn token).", raw));
    return frag;
  }

  const parsed = parseActionItems(raw);
  if (!parsed.ok) {
    frag.append(renderInvalidState(`JSON không đúng cấu trúc: ${parsed.error}`, raw));
    return frag;
  }

  const analyzed = analyzeActionItems(parsed.items, sourceText || "");
  const reviewCount = analyzed.filter(item => item.needsReview).length;

  const card = document.createElement("div");
  card.className = "ai-card";

  // Toolbar: summary + view switcher + CSV/TSV export
  const toolbar = document.createElement("div");
  toolbar.className = "ai-toolbar";

  const summary = document.createElement("div");
  summary.className = "ai-summary";
  summary.append(makeBadge(`${analyzed.length} đầu việc`, "badge-accent"));
  if (reviewCount > 0) {
    summary.append(makeBadge(`${reviewCount} cần duyệt`, "badge-warn"));
  }

  const tabs = document.createElement("div");
  tabs.className = "ai-tabs";
  tabs.setAttribute("role", "tablist");

  const exportGroup = document.createElement("div");
  exportGroup.className = "ai-export-actions";
  const csvBtn = copyButton(formatCsv(analyzed), "Copy CSV");
  const tsvBtn = copyButton(formatTsv(analyzed), "Copy TSV");
  exportGroup.append(csvBtn, tsvBtn);

  toolbar.append(summary, tabs, exportGroup);
  card.append(toolbar);

  // Views
  const tableView = document.createElement("div");
  tableView.className = "ai-view";

  const checklistView = document.createElement("div");
  checklistView.className = "ai-view";
  checklistView.hidden = true;

  const jsonView = document.createElement("div");
  jsonView.className = "ai-view";
  jsonView.hidden = true;

  // Source inspector (shared under table view)
  const sourcePanel = document.createElement("details");
  sourcePanel.className = "ai-source-panel";
  const sourceSummary = document.createElement("summary");
  sourceSummary.textContent = "Bản ghi chú nguồn (bấm vào một câu nguồn để đối chiếu)";
  const sourceContent = document.createElement("div");
  sourceContent.className = "ai-source-content";
  if (sourceText) {
    const initialBody = document.createElement("pre");
    initialBody.className = "ai-source-body";
    initialBody.textContent = sourceText;
    sourceContent.append(initialBody);
  }
  sourcePanel.append(sourceSummary, sourceContent);

  if (analyzed.length === 0) {
    const empty = document.createElement("p");
    empty.className = "ai-empty";
    empty.textContent = "Không có đầu việc được giao trong ghi chú.";
    tableView.append(empty);
  } else {
    const tableWrap = document.createElement("div");
    tableWrap.className = "table-wrap";
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const heading of ["Việc", "Người phụ trách", "Hạn", "Câu nguồn"]) {
      const th = document.createElement("th");
      th.textContent = heading;
      headRow.append(th);
    }
    thead.append(headRow);
    table.append(thead);

    const tbody = document.createElement("tbody");
    for (const item of analyzed) {
      const tr = document.createElement("tr");
      if (item.needsReview) tr.dataset.review = "true";

      const tdTask = document.createElement("td");
      tdTask.textContent = item.task;

      const tdOwner = document.createElement("td");
      if (item.owner === null) {
        tdOwner.append(makeBadge(displayOwner(item.owner), "badge-muted"));
      } else {
        tdOwner.textContent = item.owner;
      }

      const tdDue = document.createElement("td");
      if (item.due === null) {
        tdDue.append(makeBadge(displayDue(item.due), "badge-muted"));
      } else {
        tdDue.textContent = item.due;
      }

      const tdQuote = document.createElement("td");
      const quoteBtn = document.createElement("button");
      quoteBtn.type = "button";
      quoteBtn.className = "ai-quote-btn";
      quoteBtn.textContent = item.source_quote;
      quoteBtn.title = "Bấm để xem câu này trong bản ghi chú nguồn";
      quoteBtn.addEventListener("click", () => {
        sourcePanel.open = true;
        renderSourceText(sourceContent, sourceText, item.match, item.source_quote);
      });
      tdQuote.append(quoteBtn);
      if (item.needsReview) {
        tdQuote.append(makeBadge("Cần duyệt · không khớp nguyên văn", "badge-warn"));
      }

      tr.append(tdTask, tdOwner, tdDue, tdQuote);
      tbody.append(tr);
    }
    table.append(tbody);
    tableWrap.append(table);
    tableView.append(tableWrap);
  }
  if (sourceText) tableView.append(sourcePanel);

  // Checklist content
  const checklistText = formatChecklist(analyzed);
  const checklistHead = document.createElement("div");
  checklistHead.className = "ai-subbar";
  checklistHead.append(copyButton(checklistText, "Copy checklist"));
  const checklistBody = document.createElement("div");
  checklistBody.className = "answer";
  checklistBody.append(displayAnswer(checklistText));
  checklistView.append(checklistHead, checklistBody);

  // JSON content
  const jsonText = formatJson(analyzed);
  const jsonHead = document.createElement("div");
  jsonHead.className = "ai-subbar";
  jsonHead.append(copyButton(jsonText, "Copy JSON"));
  const jsonPre = document.createElement("pre");
  jsonPre.className = "ai-raw";
  const jsonCode = document.createElement("code");
  jsonCode.textContent = jsonText;
  jsonPre.append(jsonCode);
  jsonView.append(jsonHead, jsonPre);

  const views = [
    { id: "table", label: "Bảng", node: tableView },
    { id: "checklist", label: "Checklist", node: checklistView },
    { id: "json", label: "JSON", node: jsonView },
  ];

  const tabButtons = views.map((view, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ai-tab";
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", String(idx === 0));
    btn.textContent = view.label;
    btn.addEventListener("click", () => {
      views.forEach((v, i) => {
        v.node.hidden = i !== idx;
        tabButtons[i].setAttribute("aria-selected", String(i === idx));
      });
    });
    tabs.append(btn);
    return btn;
  });

  card.append(tableView, checklistView, jsonView);
  frag.append(card);
  return frag;
}
