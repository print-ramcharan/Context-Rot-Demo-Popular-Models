/**
 * content/gemini.js
 * Captures conversations on gemini.google.com.
 * Selectors (April 2026): .query-text, .response-content, .ql-editor
 */
(async () => {
  const PLATFORM = "gemini";

  const prefs = await loadPrefs();
  if (!prefs.captures[PLATFORM]) { console.log("[ContextRot/Gemini] Disabled."); return; }

  const sessionId = await getOrCreateSessionId(PLATFORM);
  console.log(`[ContextRot/Gemini] Active. Session: ${sessionId}`);

  // Proactive injection — don't wait for focusin, Quill editor is unreliable
  if (prefs.injectEnabled) {
    tryInjectWhenReady(PLATFORM, prefs.similarityThreshold);
  }

  // focusin fallback
  document.addEventListener("focusin", async (e) => {
    const t = e.target;
    if (!t.classList.contains("ql-editor") && t.getAttribute("contenteditable") !== "true") return;
    const p = await loadPrefs();
    if (p.injectEnabled) await injectContext(t, PLATFORM, p.similarityThreshold);
  });

  // Capture observer
  let debounceTimer = null;
  const storedTurnIds = new Set();

  function isStreaming() {
    return document.querySelector('model-response[data-is-generating="true"]') !== null ||
           document.querySelector(".loading-indicator") !== null;
  }

  function tryCaptureLatestExchange() {
    if (isStreaming()) return;
    const userEls      = [...document.querySelectorAll(".query-text")];
    const assistantEls = [...document.querySelectorAll(".response-content")];
    if (!userEls.length || !assistantEls.length) return;

    const turnId = `turn-${assistantEls.length}`;
    if (storedTurnIds.has(turnId)) return;
    storedTurnIds.add(turnId);

    const prompt   = (userEls[userEls.length - 1]?.innerText   || "").trim();
    const response = (assistantEls[assistantEls.length - 1]?.innerText || "").trim();
    if (!prompt || !response) return;

    chrome.runtime.sendMessage(
      { type: "STORE", payload: { platform: PLATFORM, sessionId, prompt, response } },
      (reply) => {
        if (reply?.ok) console.log(`[ContextRot/Gemini] Stored ${turnId} (${reply.data?.chunk_count} chunks)`);
        else console.warn("[ContextRot/Gemini] Store failed:", reply?.error);
      }
    );
  }

  new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(tryCaptureLatestExchange, 1000);
  }).observe(document.body, { childList: true, subtree: true });

  console.log("[ContextRot/Gemini] Ready.");
})();
