import { GUI_ACTION_ITEMS_FORMAT, buildActionItemsPrompt } from "./tasks/action_items.js";
import { collectActiveSourceBlocks, measurePayloadBytes, omitDeselectedSources } from "./tasks/files.js";
import { buildRewritePrompt, buildSummaryPrompt } from "./tasks/work_templates.js";

const tokenEncoder = new TextEncoder();

export function estimateMessageTokens(content) {
  const bytes = tokenEncoder.encode(content).length;
  const units = content.match(/\p{L}+|\p{N}{1,3}|[^\s\p{L}\p{N}]/gu)?.length ?? 0;
  return 6 + Math.ceil(Math.max(bytes / 3.5, units * 1.1));
}

export const CONTINUE_PROMPT = "Tiếp tục viết ngay từ điểm đang dừng, giữ nguyên mạch và định dạng, không lặp lại phần đã viết.";

export function buildChatRequest({
  previous = [],
  prompt = "",
  systemText = "",
  scope = "all",
  limitValue = "",
  limitNumber = NaN,
  contextLimit,
  supportsThinking = false,
  supportsJsonSchema = false,
  thinkingValue = "off",
  taskMode = null,
  taskOptions = {},
  sources = [],
  samplingOverrides = {},
  retry = false,
  mode = null,
}) {
  const effectiveMode = mode || (retry ? "retry" : "send");
  const trimmedPrompt = prompt.trim();
  const trimmedSystem = systemText.trim();
  const outbound = message => ({
    role: message.role,
    content: omitDeselectedSources(message.content, sources),
  });

  function finalize({ messages, sent, task, structured, responseFormat, sourceText, targetWords, skippedDuplicates = [], includedSources = [] }) {
    const thinking = supportsThinking ? thinkingValue === "on" : undefined;
    const suggested = supportsThinking ? 2048 : 768;
    const defaultTokens = contextLimit && Math.min(suggested, contextLimit - 1);
    const replyTokens = limitValue ? limitNumber : defaultTokens;
    const estimate = 64 + sent.reduce((tokens, message) =>
      tokens + estimateMessageTokens(message.content), 0);
    const systemTokens = trimmedSystem ? estimateMessageTokens(trimmedSystem) : 0;
    const sourceTokens = includedSources.reduce((tokens, item) =>
      tokens + estimateMessageTokens(item.block), 0);
    const messageTokens = Math.max(0, estimate - systemTokens - sourceTokens);
    const reservedReply = Number.isFinite(replyTokens) && replyTokens > 0 ? replyTokens : 0;
    const breakdown = {
      system: systemTokens,
      sources: sourceTokens,
      messages: messageTokens,
      reply: reservedReply,
      total: estimate + reservedReply,
      contextLimit: contextLimit || 0,
      free: contextLimit ? Math.max(0, contextLimit - (estimate + reservedReply)) : 0,
    };
    const payload = {
      messages: sent.map(({ role, content }) => ({ role, content })),
      stream: true,
      stream_options: { include_usage: true },
      ...samplingOverrides,
    };
    if (replyTokens !== undefined) payload.max_tokens = replyTokens;
    if (thinking !== undefined) payload.chat_template_kwargs = { enable_thinking: thinking };
    if (responseFormat !== undefined) payload.response_format = responseFormat;
    const payloadBytes = measurePayloadBytes(payload);
    return {
      messages,
      sent,
      payload,
      payloadBytes,
      replyTokens,
      defaultTokens,
      estimate,
      breakdown,
      thinking,
      task,
      structured,
      responseFormat,
      sourceText,
      targetWords,
      skippedDuplicates,
    };
  }

  if (effectiveMode === "continue") {
    const partialMsg = previous.at(-1);
    const prior = previous.slice(0, -1);
    const usable = prior.filter((message, index) =>
      message.content && !message.partial && !prior[index + 1]?.partial);
    const count = 2 * Number(scope) + 1;
    const selected = scope === "all" ? usable : usable.slice(Math.max(0, usable.length - count));
    const continueSeq = [
      ...selected.map(outbound),
      outbound({ role: "assistant", content: partialMsg?.content || "" }),
      { role: "user", content: CONTINUE_PROMPT },
    ];
    const sent = trimmedSystem ? [{ role: "system", content: trimmedSystem }, ...continueSeq] : continueSeq;
    return finalize({
      messages: previous,
      sent,
      task: partialMsg?.task || null,
      structured: false,
      responseFormat: undefined,
      sourceText: partialMsg?.sourceText,
      targetWords: partialMsg?.targetWords,
    });
  }

  const reuseExistingUser = effectiveMode === "retry" || effectiveMode === "regenerate" || effectiveMode === "edit";
  const targetUser = effectiveMode === "edit" || effectiveMode === "regenerate"
    ? previous.at(-1)
    : (reuseExistingUser ? previous.at(-2) : null);
  const activeTask = reuseExistingUser ? (targetUser?.task || null) : taskMode;
  const isActionItems = activeTask === "action_items";
  const isSummary = activeTask === "summary";
  const isRewrite = activeTask === "rewrite";
  const isStructuredTask = isActionItems || isSummary || isRewrite;
  const structured = isActionItems && supportsJsonSchema;
  const rawTargetWords = Number(reuseExistingUser ? targetUser?.targetWords : taskOptions.targetWords);
  const targetWords = isSummary && Number.isInteger(rawTargetWords) && rawTargetWords > 0
    ? rawTargetWords : undefined;

  // Deduplicate active sources against prior history that will be sent.
  const priorUsable = previous.filter((message, index) =>
    message.content && !message.partial && !previous[index + 1]?.partial);
  const priorCount = 2 * Number(scope);
  const priorSelected = scope === "all" ? priorUsable : priorUsable.slice(Math.max(0, priorUsable.length - priorCount));
  const { included, skippedDuplicates } = reuseExistingUser
    ? { included: [], skippedDuplicates: [] }
    : collectActiveSourceBlocks(sources, isStructuredTask ? [] : priorSelected);

  const attachedText = included.map(x => x.block).join("\n\n");
  const combinedInput = attachedText
    ? (trimmedPrompt ? `${trimmedPrompt}\n\n${attachedText}` : attachedText)
    : trimmedPrompt;

  let messages;
  if (effectiveMode === "edit" || effectiveMode === "regenerate") {
    messages = previous;
  } else if (effectiveMode === "retry") {
    messages = previous.slice(0, -1);
  } else {
    const userMessage = { role: "user", content: combinedInput };
    if (isActionItems) {
      userMessage.task = "action_items";
      userMessage.structured = structured;
    } else if (isSummary) {
      userMessage.task = "summary";
      if (targetWords) userMessage.targetWords = targetWords;
    } else if (isRewrite) {
      userMessage.task = "rewrite";
    }
    messages = [...previous, userMessage];
  }

  const sourceText = isStructuredTask
    ? (reuseExistingUser ? outbound({ content: targetUser?.content || "" }).content : combinedInput)
    : undefined;
  let sent;
  if (isStructuredTask) {
    let taskPrompt;
    if (isActionItems) taskPrompt = buildActionItemsPrompt(sourceText, structured);
    else if (isSummary) taskPrompt = buildSummaryPrompt(sourceText, { targetWords });
    else taskPrompt = buildRewritePrompt(sourceText, taskOptions);
    const taskMsg = { role: "user", content: taskPrompt };
    sent = trimmedSystem ? [{ role: "system", content: trimmedSystem }, taskMsg] : [taskMsg];
  } else {
    const usable = messages.filter((message, index) =>
      message.content && !message.partial && !messages[index + 1]?.partial);
    const count = 2 * Number(scope) + (reuseExistingUser || combinedInput ? 1 : 0);
    const selected = (scope === "all" ? usable : usable.slice(Math.max(0, usable.length - count))).map(outbound);
    sent = trimmedSystem ? [{ role: "system", content: trimmedSystem }, ...selected] : selected;
  }

  const responseFormat = structured ? GUI_ACTION_ITEMS_FORMAT : undefined;
  return finalize({
    messages,
    sent,
    task: activeTask,
    structured,
    responseFormat,
    sourceText,
    targetWords,
    skippedDuplicates,
    includedSources: included,
  });
}
