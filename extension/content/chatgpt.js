/**
 * content/chatgpt.js
 * ───────────────────
 * Captures conversations on chat.openai.com and chatgpt.com.
 *
 * Strategy:
 *  1. MutationObserver watches the conversation container for new nodes.
 *  2. When a complete assistant turn appears (streaming is done), we
 *     pair it with the most recent user turn and send STORE.
 *  3. "Streaming done" is detected by the absence of the streaming
 *     indicator cursor (the blinking | character / result-streaming class).
 *
 * Selectors (accurate as of April 2026 — update if ChatGPT redesigns):
 *   User messages   : [data-message-author-role="user"]
 *   Assistant msgs  : [data-message-author-role="assistant"]
 *   Streaming flag  : .result-streaming  (present while generating)
 *   Input textarea  : #prompt-textarea
 */

(async () => {
  const PLATFORM = "chatgpt";

  // ── Load prefs and get session ID ─────────────────────────────────────────
  const prefs = await loadPrefs();
  if (!prefs.captures[PLATFORM]) {
    console.log("[ContextRot/ChatGPT] Capture disabled in prefs.");
    return;
  }

  const sessionId = await getOrCreateSessionId(PLATFORM);
  console.log(`[ContextRot/ChatGPT] Active. Session: ${sessionId}`);

  // ── Debounce helper ───────────────────────────────────────────────────────
  let debounceTimer = null;
  function debounce(fn, ms) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fn, ms);
  }

  // ── Track which assistant turns we've already stored ─────────────────────
  const storedTurnIds = new Set();

  // ── Extract text from a message element ──────────────────────────────────
  function extractText(el) {
    return (el?.innerText || el?.textContent || "").trim();
  }

  // ── Check if streaming is still in progress ───────────────────────────────
  function isStreaming() {
    return document.querySelector(".result-streaming") !== null;
  }

  // ── Attempt to capture the latest complete exchange ───────────────────────
  function tryCaptureLatestExchange() {
    if (isStreaming()) return; // Wait until generation is done

    const userMsgs      = [...document.querySelectorAll('[data-message-author-role="user"]')];
    const assistantMsgs = [...document.querySelectorAll('[data-message-author-role="assistant"]')];

    if (!userMsgs.length || !assistantMsgs.length) return;

    const lastUser      = userMsgs[userMsgs.length - 1];
    const lastAssistant = assistantMsgs[assistantMsgs.length - 1];

    // Use a stable ID: the assistant element's data-message-id or its index
    const turnId = lastAssistant.getAttribute("data-message-id") ||
                   `turn-${assistantMsgs.length}`;

    if (storedTurnIds.has(turnId)) return; // Already stored this turn
    storedTurnIds.add(turnId);

    const prompt   = extractText(lastUser);
    const response = extractText(lastAssistant);

    if (!prompt || !response) return;

    // Send to background → backend
    chrome.runtime.sendMessage({
      type: "STORE",
      payload: { platform: PLATFORM, sessionId, prompt, response },
    }, (reply) => {
      if (reply?.ok) {
        console.log(`[ContextRot/ChatGPT] Stored turn ${turnId} (${reply.data?.chunk_count} chunks)`);
      } else {
        console.warn("[ContextRot/ChatGPT] Store failed:", reply?.error);
      }
    });
  }

  // ── MutationObserver ──────────────────────────────────────────────────────
  // Watch the full document body — ChatGPT's conversation container
  // is dynamically created and may not exist at script load time.
  const observer = new MutationObserver(() => {
    debounce(tryCaptureLatestExchange, 800);
  });

  observer.observe(document.body, { childList: true, subtree: true });

  // ── Inject context when user focuses the textarea ─────────────────────────
  // Use event delegation on document so it works even if textarea
  // is re-created by React.
  document.addEventListener("focusin", async (e) => {
    const target = e.target;
    const isInput = target.id === "prompt-textarea" ||
                    target.getAttribute("contenteditable") === "true";
    if (!isInput) return;

    const currentPrefs = await loadPrefs();
    if (!currentPrefs.injectEnabled) return;

    await injectContext(target, PLATFORM, currentPrefs.similarityThreshold);
  });

  console.log("[ContextRot/ChatGPT] Observer and focus listener attached.");
})();
