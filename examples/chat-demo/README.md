# Browser demo

Start a model, then run `kgl serve proxy --static examples/chat-demo` from the repository root. Open the loopback URL printed by the proxy. The page supports text chat for daily personal work and the example image classifier. [Setup and usage](../demos.md)

## Daily workflow features

- **Full-height workspace and theme toggle.** Collapse the conversation sidebar when reading long outputs, and switch between light and dark themes from the top bar.
- **Quick-start templates.** Empty conversations provide starter prompts for summarizing text, extracting action items, drafting or polishing writing, and comparing options.
- **Optional system instructions.** Open **Instructions** in the thread header to attach a per-conversation system prompt (or pick a quick preset). When non-empty, it is sent as a `system` message and included in the token budget estimate.
- **Composer ergonomics.** Press `Enter` to send and `Shift+Enter` for a new line (IME composition is respected). The input expands automatically as you type or paste text.
- **Safe Markdown rendering.** Assistant replies render headings, lists, checklists, tables, blockquotes, horizontal rules, and fenced code blocks with language headers and one-click copy buttons, while keeping raw HTML as plain text.

## Conversation history

Drafts, titles, optional system instructions, and replies are saved for this browser profile and origin. Use **Search history**, **Rename**, **Export** and **Import** to manage your history. Export contains conversation text, instructions, and drafts. Import accepts version 1 backups and keeps the local copy when a conversation ID already exists. If storage is full, export before deleting an old conversation.

**Stop** ends the stream in the browser and preserves received text. Use **Check connection** to verify that the worker is ready, then use **Retry reply** to replace the incomplete answer. A reload keeps the last saved text from an interrupted reply.

The connection area displays the estimated session time when the proxy supplies it. If the proxy requires a bearer key, enter it and select **Check connection**. The key is stored for the tab and cleared when the proxy rejects it.

When the worker reports its context, the composer shows a usage bar along with the selected message count, estimated input tokens, and reply budget. The initial reply limit is 768 tokens and can be edited per request. Reduce the history scope or reply budget if the estimate exceeds the context. The confirmation allows sending when the estimate is conservative. Actual usage appears after the reply.
