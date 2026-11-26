export function copyButton(value, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy";
  button.textContent = label;
  let resetTimer;
  button.addEventListener("click", async () => {
    clearTimeout(resetTimer);
    try {
      await navigator.clipboard.writeText(value);
      button.textContent = "Copied";
      resetTimer = setTimeout(() => { button.textContent = label; }, 1800);
    } catch {
      button.textContent = "Copy unavailable. Select the text to copy.";
    }
  });
  return button;
}

export function inlineMarkdown(parent, raw) {
  let offset = 0;
  for (const match of raw.matchAll(/`([^`\n]+)`|\*\*([^*\n]+)\*\*|~~([^~\n]+)~~|\*([^*\n]+)\*/g)) {
    parent.append(raw.slice(offset, match.index));
    const tag = match[1] ? "code" : match[2] ? "strong" : match[3] ? "del" : "em";
    const node = document.createElement(tag);
    node.textContent = match[1] ?? match[2] ?? match[3] ?? match[4];
    parent.append(node);
    offset = match.index + match[0].length;
  }
  parent.append(raw.slice(offset));
}

export function splitTableRow(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map(cell => cell.trim());
}

export function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

// Build only known DOM elements. Raw HTML remains text.
export function displayAnswer(raw) {
  const fragment = document.createDocumentFragment();
  const lines = raw.split("\n");
  let code = null;
  let codeLang = "";
  let list;
  let quote = [];
  let prose = [];

  function flushProse() {
    if (!prose.length) return;
    const paragraph = document.createElement("p");
    inlineMarkdown(paragraph, prose.join("\n"));
    fragment.append(paragraph);
    prose = [];
  }
  function flushQuote() {
    if (!quote.length) return;
    const bq = document.createElement("blockquote");
    const p = document.createElement("p");
    inlineMarkdown(p, quote.join("\n"));
    bq.append(p);
    fragment.append(bq);
    quote = [];
  }
  function flushCode() {
    const value = code.join("\n");
    const wrap = document.createElement("div");
    wrap.className = "code-block";
    const head = document.createElement("div");
    head.className = "code-head";
    const langLabel = document.createElement("span");
    langLabel.textContent = codeLang || "code";
    head.append(langLabel, copyButton(value, "Copy code"));
    const pre = document.createElement("pre");
    const node = document.createElement("code");
    node.textContent = value;
    pre.append(node);
    wrap.append(head, pre);
    fragment.append(wrap);
    code = null;
    codeLang = "";
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (code !== null) {
      if (/^\s*```\s*$/.test(line)) flushCode();
      else code.push(line);
      continue;
    }

    const fence = line.match(/^\s*```(\S*)/);
    if (fence) {
      flushProse();
      flushQuote();
      list = undefined;
      code = [];
      codeLang = fence[1] || "";
      continue;
    }

    // Markdown table: header line + separator line
    if (line.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      flushProse();
      flushQuote();
      list = undefined;
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const cell of splitTableRow(line)) {
        const th = document.createElement("th");
        inlineMarkdown(th, cell);
        headerRow.append(th);
      }
      thead.append(headerRow);
      table.append(thead);
      const tbody = document.createElement("tbody");
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        const tr = document.createElement("tr");
        for (const cell of splitTableRow(lines[i])) {
          const td = document.createElement("td");
          inlineMarkdown(td, cell);
          tr.append(td);
        }
        tbody.append(tr);
        i++;
      }
      i--;
      table.append(tbody);
      wrap.append(table);
      fragment.append(wrap);
      continue;
    }

    const bq = line.match(/^ {0,3}>\s?(.*)/);
    if (bq) {
      flushProse();
      list = undefined;
      quote.push(bq[1]);
      continue;
    }
    flushQuote();

    const item = line.match(/^ {0,3}(?:([-+*])|(\d+)[.)])\s+(.+)/);
    if (item) {
      flushProse();
      const tag = item[2] ? "ol" : "ul";
      if (!list || list.localName !== tag) {
        list = document.createElement(tag);
        if (item[2]) list.start = Number(item[2]);
        fragment.append(list);
      }
      const li = document.createElement("li");
      const task = !item[2] && item[3].match(/^\[([ xX])\]\s+(.+)/);
      if (task) {
        li.className = "task-item";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.disabled = true;
        cb.checked = task[1].toLowerCase() === "x";
        const span = document.createElement("span");
        inlineMarkdown(span, task[2]);
        li.append(cb, span);
      } else {
        inlineMarkdown(li, item[3]);
      }
      list.append(li);
      continue;
    }
    list = undefined;

    if (/^ {0,3}(?:---+|\*\*\*+|___+)\s*$/.test(line)) {
      flushProse();
      fragment.append(document.createElement("hr"));
      continue;
    }

    const heading = line.match(/^ {0,3}(#{1,6})\s+(.+)/);
    if (heading) {
      flushProse();
      const node = document.createElement("h" + Math.min(6, heading[1].length + 2));
      inlineMarkdown(node, heading[2]);
      fragment.append(node);
    } else if (!line.trim()) {
      flushProse();
    } else {
      prose.push(line);
    }
  }
  if (code !== null) flushCode();
  flushQuote();
  flushProse();
  return fragment;
}
