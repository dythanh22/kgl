import { activeSources } from "../state.js";
import { formatBytes, readLocalTextFile, sliceSourceContent } from "./files.js";

export function initSourcesUI({
  state,
  onPersist,
  onBudgetChange,
  onError,
}) {
  const byId = id => document.getElementById(id);
  const attachBtn = byId("attach-file-btn");
  const attachInput = byId("attach-file-input");
  const pinNoteBtn = byId("pin-note-btn");
  const sourcesBar = byId("sources-bar");
  const sourcesList = byId("sources-list");
  const composerBox = document.querySelector(".composer-box");

  const expandedIds = new Set();

  function renderSources() {
    const list = activeSources();
    sourcesBar.hidden = state.vision || list.length === 0;
    sourcesList.replaceChildren();
    if (sourcesBar.hidden) return;

    for (const src of list) {
      const card = document.createElement("div");
      card.className = "source-item";

      const row = document.createElement("div");
      row.className = "source-row";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = src.selected !== false;
      cb.disabled = state.running;
      cb.title = "Bật/tắt gửi nguồn này trong yêu cầu";
      cb.addEventListener("change", () => {
        src.selected = cb.checked;
        onPersist();
        onBudgetChange();
      });

      const name = document.createElement("strong");
      name.className = "source-name";
      name.textContent = src.name;

      const { text, from, to, total } = sliceSourceContent(src);
      const meta = document.createElement("span");
      meta.className = "source-meta";
      const byteSize = new TextEncoder().encode(text).length;
      meta.textContent = `${from}–${to}/${total} dòng · ${formatBytes(byteSize)}`;

      const actions = document.createElement("div");
      actions.className = "source-actions";

      const pinBtn = document.createElement("button");
      pinBtn.type = "button";
      pinBtn.className = "copy";
      pinBtn.textContent = src.pinned ? "Đã ghim" : "Ghim";
      pinBtn.disabled = state.running;
      pinBtn.title = "Giữ nguồn này cho các lượt tiếp theo";
      if (src.pinned) pinBtn.dataset.pinned = "true";
      pinBtn.addEventListener("click", () => {
        src.pinned = !src.pinned;
        onPersist();
        renderSources();
      });

      const previewBtn = document.createElement("button");
      previewBtn.type = "button";
      previewBtn.className = "copy";
      previewBtn.textContent = expandedIds.has(src.id) ? "Ẩn đoạn" : "Xem / chọn đoạn";
      previewBtn.addEventListener("click", () => {
        if (expandedIds.has(src.id)) expandedIds.delete(src.id);
        else expandedIds.add(src.id);
        renderSources();
      });

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "btn-danger";
      removeBtn.textContent = "×";
      removeBtn.title = "Bỏ nguồn này";
      removeBtn.disabled = state.running;
      removeBtn.addEventListener("click", () => {
        const arr = activeSources();
        const idx = arr.indexOf(src);
        if (idx >= 0) arr.splice(idx, 1);
        expandedIds.delete(src.id);
        onPersist();
        renderSources();
        onBudgetChange();
      });

      actions.append(pinBtn, previewBtn, removeBtn);
      row.append(cb, name, meta, actions);
      card.append(row);

      if (expandedIds.has(src.id)) {
        const preview = document.createElement("div");
        preview.className = "source-preview";

        const rangeRow = document.createElement("div");
        rangeRow.className = "source-range";

        const fromLbl = document.createElement("label");
        fromLbl.textContent = "Từ dòng";
        const fromIn = document.createElement("input");
        fromIn.type = "number";
        fromIn.min = "1";
        fromIn.max = String(total);
        fromIn.value = String(from);
        fromIn.disabled = state.running;

        const toLbl = document.createElement("label");
        toLbl.textContent = "Đến dòng";
        const toIn = document.createElement("input");
        toIn.type = "number";
        toIn.min = "1";
        toIn.max = String(total);
        toIn.value = String(to);
        toIn.disabled = state.running;

        const allBtn = document.createElement("button");
        allBtn.type = "button";
        allBtn.className = "btn-ghost";
        allBtn.textContent = "Toàn bộ";
        allBtn.disabled = state.running;

        const applyRange = () => {
          const nextFrom = Math.max(1, Math.min(total, Number(fromIn.value) || 1));
          const nextTo = Math.max(nextFrom, Math.min(total, Number(toIn.value) || total));
          src.fromLine = nextFrom;
          src.toLine = nextTo;
          onPersist();
          renderSources();
          onBudgetChange();
        };
        fromIn.addEventListener("change", applyRange);
        toIn.addEventListener("change", applyRange);
        allBtn.addEventListener("click", () => {
          src.fromLine = 1;
          src.toLine = total;
          onPersist();
          renderSources();
          onBudgetChange();
        });

        rangeRow.append(fromLbl, fromIn, toLbl, toIn, allBtn);
        const pre = document.createElement("pre");
        pre.className = "source-preview-body";
        pre.textContent = text;
        preview.append(rangeRow, pre);
        card.append(preview);
      }

      sourcesList.append(card);
    }
  }

  async function addFiles(fileList) {
    if (state.running || state.vision) return;
    const arr = activeSources(true);
    for (const file of Array.from(fileList || [])) {
      try {
        const src = await readLocalTextFile(file);
        arr.push(src);
      } catch (error) {
        onError(error);
      }
    }
    onPersist();
    renderSources();
    onBudgetChange();
  }

  attachBtn.addEventListener("click", () => {
    if (!state.running) attachInput.click();
  });
  attachInput.addEventListener("change", async event => {
    const files = event.target.files;
    await addFiles(files);
    event.target.value = "";
  });

  pinNoteBtn.addEventListener("click", () => {
    if (state.running) return;
    const promptField = byId("prompt");
    const initial = promptField.value.trim();
    const content = initial || window.prompt("Dán đoạn brief / nguồn cần ghim trong hội thoại này:", "");
    if (!content || !content.trim()) return;
    const name = window.prompt("Tên nguồn ghim:", "Brief ghim") || "Brief ghim";
    const clean = content.trim();
    const lineCount = clean.split("\n").length;
    activeSources(true).push({
      id: (globalThis.crypto?.randomUUID?.() || `pin-${Date.now().toString(36)}`),
      name: name.trim() || "Brief ghim",
      content: clean,
      lineCount,
      fromLine: 1,
      toLine: lineCount,
      selected: true,
      pinned: true,
    });
    if (initial) {
      promptField.value = "";
      promptField.dispatchEvent(new Event("input"));
    }
    onPersist();
    renderSources();
    onBudgetChange();
  });

  if (composerBox) {
    composerBox.addEventListener("dragover", event => {
      if (state.running || state.vision) return;
      event.preventDefault();
      composerBox.dataset.dragover = "true";
    });
    composerBox.addEventListener("dragleave", () => {
      delete composerBox.dataset.dragover;
    });
    composerBox.addEventListener("drop", async event => {
      delete composerBox.dataset.dragover;
      if (state.running || state.vision) return;
      const files = event.dataTransfer?.files;
      if (files && files.length) {
        event.preventDefault();
        await addFiles(files);
      }
    });
  }

  return { renderSources };
}
