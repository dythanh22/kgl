# Browser demo

Start a model, then run `kgl serve proxy --static examples/chat-demo` from the repository root. Open the loopback URL printed by the proxy. The page supports text chat for daily personal work and the example image classifier. [Setup and usage](../demos.md) · [Qwen GGUF operating guide](../qwen35b-llama/QWEN35B-KAGGLE.md)

## Daily workflow features

- **Full-height Graphite / Fog workspace and Pixel Cat mascot.** The 3-column layout puts history, collapsible search, and Settings on the left, chat in the center, and Environment (GPU VRAM / layers / context), multicolor Context breakdown, Sources, Generation, and Backup cards in a collapsible right Session Rail. The original 2D Canvas Pixel Cat perched on the composer reflects connection, reading, thinking, writing, and error states without injecting prompts into the model. Click the thread title in the top bar for thread actions, collapse either side panel when reading long outputs, and switch between Graphite (dark), Fog (light), or System themes from the sidebar or Settings.
- **Quick-start templates.** Empty conversations provide starter prompts for summarizing text, extracting action items, drafting or polishing writing, and comparing options.
- **Action items with evidence.** Use the **Action items** button in the thread header or the **Extract action items** starter to extract tasks into a verified **Task · Owner · Due · Source quote** table when the worker advertises `response_format` and `json_schema`. Click any source quote to highlight the matching passage in the original note, flag unmatched quotes for manual review, switch between Table / Checklist / JSON views, and copy spreadsheet-safe CSV/TSV. Workers without JSON Schema support automatically fall back to plain-text mode with a clear label.
- **Prompt library & slash commands.** Open **/ Prompts** from the composer or type `/` to insert built-in or custom templates (`/tomtat`, `/action`, `/soan`, `/email`, `/dich`, `/review`, `/sosanh`) with `{{date}}`, `{{selection}}`, and explicit-consent `{{clipboard}}` variables.
- **Structured summary & rewrite review.** Summary mode separates Decisions, Proposals, and Open questions and displays a whitespace word count; Rewrite mode lets you set audience, goal, and tone and provides a side-by-side comparison flagging changes to names, numbers, and deadlines.
- **Per-conversation sampling.** Choose **Theo preset** (leaves worker defaults untouched), **Chính xác**, **Sáng tạo**, or **Tùy chỉnh** (`temperature`, `top_p`, and capability-gated `presence_penalty` / `repeat_penalty`), with a live readout of the effective parameters sent.
- **Branching, Continue, auto-poll & Markdown export.** Edit any user turn to fork a branch (preserving the original path), **Regenerate** the last completed answer with variant switching, **Continue** non-JSON replies that hit `length` with a visible boundary, auto-poll `/ready` (respecting `Retry-After`, up to 30 s) after Stop or `503 worker_busy`, export a single thread to `.md`, and use shortcuts (`Ctrl/Cmd+K` search, `Ctrl/Cmd+Shift+O` new chat, `Esc` close the active dialog, menu, or floating panel before stopping a reply, `↑` edit last turn when empty).
- **Local text files, pinned sources & handover.** Attach or drag-and-drop local text/code files (`.txt`, `.md`, `.csv`, `.json`, `.log`, `.py`, etc.), preview and select line ranges, pin briefs/sources across turns with automatic deduplication, track both token budget and exact UTF-8 JSON payload bytes against the 512 KiB proxy default, and generate an editable handover note into a new linked conversation when context fills up.
- **Optional system instructions.** Open **Instructions** in the thread header to attach a per-conversation system prompt (or pick a quick preset). When non-empty, it is sent as a `system` message and included in the token budget estimate.
- **Composer ergonomics.** Press `Enter` to send and `Shift+Enter` for a new line (IME composition is respected). The input expands automatically as you type or paste text.
- **Safe Markdown rendering.** Assistant replies render headings, lists, checklists, tables, blockquotes, horizontal rules, and fenced code blocks with language headers and one-click copy buttons, while keeping raw HTML as plain text.

## Conversation history

Drafts, titles, optional system instructions, and replies are saved for this browser profile and origin. Use **Search history** (matches titles, content, branches, and tags with highlighted snippets), **Ghim chat**, **Tags**, date filters, **Rename**, **Export** (full **v2** or v1-compatible **v1** JSON), and **Import** (accepts both v1 and v2 backups and keeps the local copy when a conversation ID already exists) to manage your history. A v2 import can also restore the unsent draft's sources and sampling.

Tabs on the same origin sync updates to different conversations. If two tabs make competing changes to one conversation, the already saved version stays in storage and the local edits remain available for JSON export before reloading. The sidebar footer displays storage size and write latency. If storage is full, export before deleting an old conversation.

Uncheck a pinned source to omit its matching attachment blocks from later requests, including blocks carried in conversation history. The original transcript still shows the attachment and any text quoted separately in a message can still be sent as history.

**Stop** ends the stream in the browser and preserves received text. Use **Thử kết nối lại** beside the composer on connection failure or **Kiểm tra kết nối** in the worker panel, then use **Thử lại** to replace the incomplete answer. A reload keeps the last saved text and the reason the reply is incomplete. You can change Thinking or the token limit before retrying the same user turn.

The connection area displays the estimated session time when the proxy supplies it. If the proxy requires a bearer key, enter it and select **Check connection**. The key is stored for the tab and cleared when the proxy rejects it.

When the worker reports its context, the composer shows a usage bar along with the selected message count, estimated input tokens, and reply budget. When the worker advertises Thinking, the chat starts with Thinking Off and a 2,048-token generation limit. Choose 4,096 in Max generation tokens for a longer report. With Thinking On, reasoning and the answer share the selected limit. Older workers keep the 768-token default. An explicit generation limit takes precedence. [Qwen API capabilities](../qwen35b-llama/API.md) Reduce the history scope or reply budget if the estimate exceeds the context. The confirmation allows sending when the estimate is conservative. Actual usage appears after the reply.

## File layout

The UI uses native browser ES modules and plain CSS with no build step:

- `demo.html` — page markup and module entry point.
- `styles/theme.css`, `styles/app.css` — light/dark tokens, shared controls, and workspace layout.
- `js/state.js`, `js/storage.js`, `js/budget.js`, `js/mascot.js`, `js/markdown.js`, `js/api.js`, `js/ui.js`, `js/main.js` — state, persistence/backup, token budget & breakdown, original 2D Canvas Pixel Cat mascot & context donut ring, safe Markdown rendering, proxy/SSE client, DOM rendering, and event wiring.
