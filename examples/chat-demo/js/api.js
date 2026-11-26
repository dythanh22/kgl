export function authHeaders(key) {
  return key ? { Authorization: "Bearer " + key } : {};
}

export function fetchReady(key) {
  return fetch("/ready", { headers: authHeaders(key) });
}

export function readImage(file) {
  if (!file || file.size === 0 || file.size > 350 * 1024) {
    throw new Error("Choose a JPEG or PNG between 1 byte and 350 KiB.");
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",", 2)[1]);
    reader.onerror = () => reject(new Error("Could not read the selected file."));
    reader.readAsDataURL(file);
  });
}

export function postInference(vision, payload, key, signal) {
  return fetch(vision ? "/handle" : "/v1/chat/completions", {
    method: "POST",
    headers: { ...authHeaders(key), "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export function parseRetryAfter(response) {
  const raw = response?.headers?.get?.("Retry-After");
  if (!raw) return 0;
  const num = Number(raw);
  return Number.isFinite(num) && num > 0 ? num : 0;
}

export async function readChatStream(
  response,
  { startTime = performance.now(), onReasoning, onContent, onUsage, onSaveCheckpoint },
) {
  let finishReason;
  let lastSaved = performance.now();
  let firstSseMs = null;
  let firstReasoningMs = null;
  let firstContentMs = null;
  let workerTokPerSec = null;
  let sawNonEmptyContent = false;
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) throw new Error("The reply ended early. Check the connection before retrying.");
      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const text = line.slice(5).trim();
        if (text === "[DONE]") {
          return {
            finishReason,
            timings: {
              firstSseMs,
              firstReasoningMs,
              firstContentMs,
              totalMs: Math.max(0, performance.now() - startTime),
              workerTokPerSec,
            },
          };
        }
        const now = performance.now();
        if (firstSseMs === null) firstSseMs = Math.max(0, now - startTime);
        const event = JSON.parse(text);
        if (event.error) throw new Error(event.error.message);
        if (Number.isFinite(event.timings?.predicted_per_second) && event.timings.predicted_per_second > 0) {
          workerTokPerSec = event.timings.predicted_per_second;
        }
        if (Number.isInteger(event.usage?.prompt_tokens) && Number.isInteger(event.usage?.completion_tokens)) {
          onUsage(event.usage);
        }
        const choice = event.choices?.[0];
        const delta = choice?.delta;
        if (delta?.reasoning_content) {
          if (firstReasoningMs === null) firstReasoningMs = Math.max(0, now - startTime);
          onReasoning(delta.reasoning_content);
        }
        if (delta?.content) {
          if (!sawNonEmptyContent && delta.content.trim().length > 0) {
            sawNonEmptyContent = true;
            firstContentMs = Math.max(0, now - startTime);
          }
          onContent(delta.content);
        }
        finishReason = choice?.finish_reason || finishReason;
        if (performance.now() - lastSaved >= 400) {
          onSaveCheckpoint();
          lastSaved = performance.now();
        }
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
}
