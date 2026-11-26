import {
  detectSlashQuery,
  expandTemplate,
  filterPromptsBySlash,
  getAllPrompts,
  templateUsesVariable,
  validateCustomPrompt,
} from "./prompts.js";

export function initPromptLibrary({
  promptField,
  state,
  onPersist,
  onSetTaskMode,
  onDraftChanged,
}) {
  const byId = id => document.getElementById(id);
  const panel = byId("prompt-library");
  const listEl = byId("prompt-list");
  const slashMenu = byId("slash-menu");
  const openBtn = byId("open-prompts");
  const closeBtn = byId("close-prompts");
  const form = byId("prompt-form");
  const nameInput = byId("pl-name");
  const slashInput = byId("pl-slash");
  const descInput = byId("pl-desc");
  const tplInput = byId("pl-template");
  const formError = byId("pl-error");
  const cancelEditBtn = byId("pl-cancel");
  const submitBtn = byId("pl-submit");

  const varBox = byId("pl-var-box");
  const varInfo = byId("pl-var-info");
  const clipReadBtn = byId("pl-read-clipboard");
  const clipFallback = byId("pl-clipboard-input");
  const varApplyBtn = byId("pl-var-apply");
  const varCancelBtn = byId("pl-var-cancel");

  let editingId = null;
  let slashMatches = [];
  let slashIndex = 0;
  let pendingInsert = null;

  function captureSelection() {
    if (promptField.selectionStart !== promptField.selectionEnd) {
      return promptField.value.slice(promptField.selectionStart, promptField.selectionEnd);
    }
    const winSel = window.getSelection?.()?.toString() || "";
    return winSel.trim();
  }

  function applyExpanded(item, text, slashRange) {
    if (item.task === "action_items") {
      onSetTaskMode("action_items");
      if (slashRange) {
        promptField.value = (
          promptField.value.slice(0, slashRange.start) + promptField.value.slice(slashRange.end)
        ).trimStart();
      }
      if (text.trim()) {
        promptField.value = text;
      }
      onDraftChanged();
      closeLibrary();
      hideSlashMenu();
      promptField.focus();
      return;
    }

    if (item.task === "summary" || item.task === "rewrite") {
      onSetTaskMode(item.task);
    }

    if (slashRange) {
      const before = promptField.value.slice(0, slashRange.start);
      const after = promptField.value.slice(slashRange.end);
      promptField.value = before + text + after;
      const pos = before.length + text.length;
      promptField.setSelectionRange(pos, pos);
    } else if (promptField.selectionStart !== promptField.selectionEnd) {
      const start = promptField.selectionStart;
      const end = promptField.selectionEnd;
      promptField.value = promptField.value.slice(0, start) + text + promptField.value.slice(end);
      const pos = start + text.length;
      promptField.setSelectionRange(pos, pos);
    } else {
      promptField.value = promptField.value.trim()
        ? `${promptField.value}\n\n${text}`
        : text;
      promptField.setSelectionRange(promptField.value.length, promptField.value.length);
    }

    onDraftChanged();
    closeLibrary();
    hideSlashMenu();
    promptField.focus();
  }

  function startInsert(item, slashRange = null) {
    const selection = captureSelection();
    const needsClipboard = templateUsesVariable(item.template, "clipboard");
    const usesSelection = templateUsesVariable(item.template, "selection");

    if (needsClipboard || (usesSelection && selection)) {
      pendingInsert = { item, slashRange, selection, clipboard: "" };
      panel.hidden = false;
      varBox.hidden = false;
      clipReadBtn.hidden = !needsClipboard;
      clipFallback.hidden = !needsClipboard;
      clipFallback.value = "";
      const notes = [];
      if (usesSelection && selection) {
        notes.push(`Đoạn đang chọn (${selection.length} ký tự): "${selection.slice(0, 120)}${selection.length > 120 ? "…" : ""}"`);
      }
      if (needsClipboard) {
        notes.push("Mẫu có {{clipboard}}. Bấm 'Đọc clipboard' hoặc dán thủ công bên dưới trước khi chèn.");
      }
      varInfo.textContent = notes.join(" · ");
      return;
    }

    const expanded = expandTemplate(item.template, { selection: "" });
    applyExpanded(item, expanded, slashRange);
  }

  clipReadBtn.addEventListener("click", async () => {
    try {
      if (!navigator.clipboard?.readText) throw new Error("Trình duyệt không hỗ trợ đọc clipboard.");
      const text = await navigator.clipboard.readText();
      clipFallback.value = text;
      varInfo.textContent = `Đã đọc ${text.length} ký tự từ clipboard. Kiểm tra rồi bấm 'Chèn vào ô soạn'.`;
    } catch {
      varInfo.textContent = "Trình duyệt từ chối quyền clipboard. Vui lòng dán tay (Ctrl+V) vào ô bên dưới.";
      clipFallback.focus();
    }
  });

  varApplyBtn.addEventListener("click", () => {
    if (!pendingInsert) return;
    const { item, slashRange, selection } = pendingInsert;
    const expanded = expandTemplate(item.template, {
      selection,
      clipboard: clipFallback.value,
    });
    pendingInsert = null;
    varBox.hidden = true;
    applyExpanded(item, expanded, slashRange);
  });

  varCancelBtn.addEventListener("click", () => {
    pendingInsert = null;
    varBox.hidden = true;
  });

  function resetForm() {
    editingId = null;
    nameInput.value = "";
    slashInput.value = "";
    descInput.value = "";
    tplInput.value = "";
    formError.hidden = true;
    submitBtn.textContent = "Lưu mẫu mới";
    cancelEditBtn.hidden = true;
  }

  function renderList() {
    listEl.replaceChildren();
    const all = getAllPrompts(state.customPrompts);
    for (const item of all) {
      const row = document.createElement("div");
      row.className = "pl-item";

      const meta = document.createElement("div");
      meta.className = "pl-meta";
      const titleRow = document.createElement("div");
      titleRow.className = "pl-title-row";
      const slashBadge = document.createElement("code");
      slashBadge.className = "pl-slash";
      slashBadge.textContent = item.slash;
      const name = document.createElement("strong");
      name.textContent = item.name;
      titleRow.append(slashBadge, name);
      meta.append(titleRow);
      if (item.description) {
        const desc = document.createElement("p");
        desc.className = "pl-desc";
        desc.textContent = item.description;
        meta.append(desc);
      }

      const actions = document.createElement("div");
      actions.className = "pl-actions";
      const useBtn = document.createElement("button");
      useBtn.type = "button";
      useBtn.className = "btn-secondary";
      useBtn.textContent = "Chèn";
      useBtn.addEventListener("click", () => startInsert(item));
      actions.append(useBtn);

      if (!item.builtin) {
        const editBtn = document.createElement("button");
        editBtn.type = "button";
        editBtn.className = "btn-ghost";
        editBtn.textContent = "Sửa";
        editBtn.addEventListener("click", () => {
          editingId = item.id;
          nameInput.value = item.name;
          slashInput.value = item.slash;
          descInput.value = item.description || "";
          tplInput.value = item.template;
          submitBtn.textContent = "Cập nhật mẫu";
          cancelEditBtn.hidden = false;
          nameInput.focus();
        });
        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "btn-danger";
        delBtn.textContent = "Xóa";
        delBtn.addEventListener("click", () => {
          state.customPrompts = state.customPrompts.filter(p => p.id !== item.id);
          onPersist();
          renderList();
        });
        actions.append(editBtn, delBtn);
      }

      row.append(meta, actions);
      listEl.append(row);
    }
  }

  function openLibrary() {
    renderList();
    panel.hidden = false;
    openBtn.setAttribute("aria-expanded", "true");
  }

  function closeLibrary() {
    panel.hidden = true;
    varBox.hidden = true;
    pendingInsert = null;
    openBtn.setAttribute("aria-expanded", "false");
  }

  function hideSlashMenu() {
    slashMenu.hidden = true;
    slashMatches = [];
    slashIndex = 0;
  }

  function renderSlashMenu(detected) {
    const all = getAllPrompts(state.customPrompts);
    slashMatches = filterPromptsBySlash(all, detected.query);
    if (!slashMatches.length) {
      hideSlashMenu();
      return;
    }
    slashIndex = Math.min(slashIndex, slashMatches.length - 1);
    slashMenu.replaceChildren();
    slashMatches.forEach((item, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "slash-option";
      btn.setAttribute("aria-selected", String(idx === slashIndex));
      const code = document.createElement("code");
      code.textContent = item.slash;
      const name = document.createElement("span");
      name.textContent = item.name;
      btn.append(code, name);
      btn.addEventListener("mousedown", event => {
        event.preventDefault();
        startInsert(item, detected);
      });
      slashMenu.append(btn);
    });
    slashMenu.hidden = false;
  }

  openBtn.addEventListener("click", () => {
    if (state.running) return;
    if (panel.hidden) openLibrary();
    else closeLibrary();
  });
  closeBtn.addEventListener("click", closeLibrary);
  cancelEditBtn.addEventListener("click", resetForm);

  submitBtn.addEventListener("click", () => {
    formError.hidden = true;
    try {
      const entry = validateCustomPrompt({
        id: editingId || undefined,
        name: nameInput.value,
        slash: slashInput.value,
        description: descInput.value,
        template: tplInput.value,
      });
      const existing = state.customPrompts.findIndex(p => p.id === entry.id);
      if (existing >= 0) state.customPrompts[existing] = entry;
      else state.customPrompts.push(entry);
      onPersist();
      resetForm();
      renderList();
    } catch (error) {
      formError.textContent = error.message;
      formError.hidden = false;
    }
  });

  promptField.addEventListener("input", () => {
    if (state.running) return hideSlashMenu();
    const detected = detectSlashQuery(promptField.value, promptField.selectionStart);
    if (detected) renderSlashMenu(detected);
    else hideSlashMenu();
  });

  promptField.addEventListener("keydown", event => {
    if (slashMenu.hidden || !slashMatches.length) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      slashIndex = (slashIndex + 1) % slashMatches.length;
      const detected = detectSlashQuery(promptField.value, promptField.selectionStart);
      if (detected) renderSlashMenu(detected);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      slashIndex = (slashIndex - 1 + slashMatches.length) % slashMatches.length;
      const detected = detectSlashQuery(promptField.value, promptField.selectionStart);
      if (detected) renderSlashMenu(detected);
    } else if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      event.stopImmediatePropagation();
      const detected = detectSlashQuery(promptField.value, promptField.selectionStart);
      startInsert(slashMatches[slashIndex], detected);
    } else if (event.key === "Escape") {
      event.preventDefault();
      hideSlashMenu();
    }
  }, true);

  return { renderList, closeLibrary, hideSlashMenu };
}
