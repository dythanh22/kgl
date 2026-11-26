import { PARTIAL_LABELS, chatTitle } from "./state.js";

function cloneMessage(msg) {
  return JSON.parse(JSON.stringify(msg));
}

// Edit any user message at `index`, preserving the previous path from `index` onward as a sibling branch.
export function forkAtUserMessage(messages, index, newContent) {
  if (!Array.isArray(messages) || index < 0 || index >= messages.length) {
    throw new Error("Invalid user message index.");
  }
  const target = messages[index];
  if (target.role !== "user") {
    throw new Error("Only user messages can be edited.");
  }

  const existingSiblings = Array.isArray(target.siblings) ? target.siblings.map(cloneMessage) : null;
  const activeIdx = Number.isInteger(target.activeSibling) ? target.activeSibling : 0;

  // Strip siblings metadata from the stored branch copy to avoid recursive nesting.
  const currentBranch = messages.slice(index).map((m, i) => {
    const copy = cloneMessage(m);
    if (i === 0) {
      delete copy.siblings;
      delete copy.activeSibling;
    }
    return copy;
  });

  const siblings = existingSiblings ? existingSiblings : [currentBranch];
  siblings[activeIdx] = currentBranch;

  const editedUser = { role: "user", content: newContent.trim() };
  if (target.task) editedUser.task = target.task;
  if (typeof target.structured === "boolean") editedUser.structured = target.structured;
  if (Number.isSafeInteger(target.targetWords)) editedUser.targetWords = target.targetWords;

  siblings.push([cloneMessage(editedUser)]);
  const nextActive = siblings.length - 1;
  editedUser.siblings = siblings;
  editedUser.activeSibling = nextActive;

  return [...messages.slice(0, index), editedUser];
}

export function switchUserSibling(messages, index, targetSiblingIndex) {
  const target = messages[index];
  if (!target || target.role !== "user" || !Array.isArray(target.siblings)) return messages;
  const count = target.siblings.length;
  if (targetSiblingIndex < 0 || targetSiblingIndex >= count) return messages;
  const currentIdx = Number.isInteger(target.activeSibling) ? target.activeSibling : 0;
  if (targetSiblingIndex === currentIdx) return messages;

  const siblings = target.siblings.map(cloneMessage);
  const currentBranch = messages.slice(index).map((m, i) => {
    const copy = cloneMessage(m);
    if (i === 0) {
      delete copy.siblings;
      delete copy.activeSibling;
    }
    return copy;
  });
  siblings[currentIdx] = currentBranch;

  const chosenBranch = siblings[targetSiblingIndex].map(cloneMessage);
  chosenBranch[0].siblings = siblings;
  chosenBranch[0].activeSibling = targetSiblingIndex;

  return [...messages.slice(0, index), ...chosenBranch];
}

// Prepare a completed last assistant message for regeneration while preserving prior variants.
export function prepareRegenerate(messages) {
  if (!Array.isArray(messages) || messages.length < 2) {
    throw new Error("No completed reply to regenerate.");
  }
  const last = messages.at(-1);
  if (last.role !== "assistant" || last.partial) {
    throw new Error("Regenerate applies only to a completed assistant reply.");
  }

  const existingVariants = Array.isArray(last.variants) ? last.variants.map(cloneMessage) : null;
  const activeIdx = Number.isInteger(last.activeVariant) ? last.activeVariant : 0;

  const currentClean = cloneMessage(last);
  delete currentClean.variants;
  delete currentClean.activeVariant;

  const variants = existingVariants ? existingVariants : [currentClean];
  variants[activeIdx] = currentClean;

  return {
    prefix: messages.slice(0, -1),
    variants,
  };
}

export function attachVariant(assistantMessage, priorVariants) {
  const clean = cloneMessage(assistantMessage);
  delete clean.variants;
  delete clean.activeVariant;
  const variants = [...priorVariants.map(cloneMessage), clean];
  assistantMessage.variants = variants;
  assistantMessage.activeVariant = variants.length - 1;
  return assistantMessage;
}

export function switchAssistantVariant(messages, index, targetVariantIndex) {
  const msg = messages[index];
  if (!msg || msg.role !== "assistant" || !Array.isArray(msg.variants)) return messages;
  const count = msg.variants.length;
  if (targetVariantIndex < 0 || targetVariantIndex >= count) return messages;
  const currentIdx = Number.isInteger(msg.activeVariant) ? msg.activeVariant : 0;
  if (targetVariantIndex === currentIdx) return messages;

  const variants = msg.variants.map(cloneMessage);
  const currentClean = cloneMessage(msg);
  delete currentClean.variants;
  delete currentClean.activeVariant;
  variants[currentIdx] = currentClean;

  const chosen = cloneMessage(variants[targetVariantIndex]);
  chosen.variants = variants;
  chosen.activeVariant = targetVariantIndex;

  const next = [...messages];
  next[index] = chosen;
  return next;
}

export function exportThreadMarkdown(chat, { includeReasoning = false } = {}) {
  if (!chat || !Array.isArray(chat.messages)) return "";
  const title = chatTitle(chat);
  const lines = [`# ${title}`, ""];
  for (const msg of chat.messages) {
    if (msg.role === "user") {
      lines.push("## You", "", msg.content || "", "");
    } else {
      const partialTag = msg.partial
        ? ` (${PARTIAL_LABELS[msg.partialReason] || "Incomplete"})`
        : "";
      lines.push(`## Assistant${partialTag}`, "");
      if (includeReasoning && msg.reasoning?.trim()) {
        lines.push("<details>", "<summary>Reasoning</summary>", "", msg.reasoning.trim(), "", "</details>", "");
      }
      lines.push(msg.content || "_(Empty reply)_", "");
    }
  }
  return lines.join("\n").trimEnd() + "\n";
}

export function downloadMarkdown(filename, text) {
  const safeName = String(filename || "conversation")
    .toLowerCase()
    .replace(/[^a-z0-9\u00c0-\u024f\u1e00-\u1eff_-]+/gi, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60) || "conversation";
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${safeName}.md`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
