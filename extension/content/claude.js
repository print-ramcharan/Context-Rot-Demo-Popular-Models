/**
 * content/claude.js
 * Captures conversations on claude.ai.
 * Selectors (April 2026): [data-testid="user-message"], [data-testid="assistant-message"], .ProseMirror
 */
(async () => {
  const PLATFORM = "claude";

  const prefs = await loadPrefs();
  if (!prefs.captures[PLATFORM]) { console.log("[ContextRot/Claude] Disabled."); return; }

  const sessionId = await getOrCreateSessionId(PLATFORM);
  console.log(`[ContextRot/Claude] Active. Session: ${sessionId}`);

  if (prefs.injectEnabled) {
    tryInjectWhenReady(PLATFORM, prefs.similarityThreshold);
  }

  document.addEventListener("focusin", async (e) => {
    const t = e.target;
    const isInput = (t.getAttribute("contenteditable") === "true" && t.classList.contains("ProseMirror")) || t.tagName === "TEXTAREA";
    if (!isInput) return;
    const p = await loadPrefs();
    if (p.injectEnabled) await injectContext(t, PLATFORM, p.similarityThreshold);
  });

  let debounceTimer = null;
  const storedTurnIds = new Set();

  function isStreaming() {
    return document.querySelector('[data-is-streaming="true"]') !== null ||
           document.querySelector(".cursor-blink") !== null;
  }

  function tryCaptureLatestExchange() {
    if (isStreaming()) return;
    const userEls = [...document.querySelectorAll('[data-testid="user-message"]')];
    let assistantEls = [...document.querySelectorAll('[data-testid="assistant-message"]')];
    if (!assistantEls.length) assistantEls = [...document.querySelectorAll("div.font-claude-message")];
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
        if (reply?.ok) console.log(`[ContextRot/Claude] Stored ${turnId} (${reply.data?.chunk_count} chunks)`);
        else console.warn("[ContextRot/Claude] Store failed:", reply?.error);
      }
    );
  }

  new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(tryCaptureLatestExchange, 800);
  }).observe(document.body, { childList: true, subtree: true });

  console.log("[ContextRot/Claude] Ready.");
})();
