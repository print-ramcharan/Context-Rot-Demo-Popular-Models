/**
 * content/injector.js
 * ────────────────────
 * Shared utility — loaded before every platform content script.
 * All functions are plain globals (no export) so content scripts
 * can call them directly after this file is injected first.
 *
 * Exposes:
 *   injectContext(inputEl, platform, threshold)
 *   tryInjectWhenReady(platform, threshold, maxWaitMs)
 */

let activeBanner = null;
let injectionDone = false; // inject once per page load

// ── Main entry ────────────────────────────────────────────────────────────────

async function injectContext(inputEl, platform, similarityThreshold = 0.3) {
  if (injectionDone) return;
  if (inputEl.dataset.contextInjected === "true") return;

  const currentText = getInputText(inputEl).trim();
  const query = currentText.length > 8
    ? currentText
    : await getRecentTopicQuery();

  if (!query || query.length < 3) {
    console.log("[ContextRot/Injector] No query to retrieve with.");
    return;
  }

  console.log(`[ContextRot/Injector] Retrieving for query: "${query.slice(0, 60)}"`);

  const reply = await sendMessage({
    type: "RETRIEVE",
    payload: { query, topK: 5, platformFilter: null, similarityThreshold },
  });

  if (!reply?.ok || !reply.data?.results?.length) {
    console.log("[ContextRot/Injector] No relevant memories found. threshold =", similarityThreshold);
    return;
  }

  const results = reply.data.results;
  const contextBlock = formatContextBlock(results);

  prependToInput(inputEl, contextBlock);
  inputEl.dataset.contextInjected = "true";
  injectionDone = true;

  showBanner(inputEl, results.length, () => {
    removeInjectedContext(inputEl, contextBlock);
    delete inputEl.dataset.contextInjected;
    injectionDone = false;
  });
}

// ── Proactive: poll for input then inject ─────────────────────────────────────

async function tryInjectWhenReady(platform, similarityThreshold = 0.3, maxWaitMs = 8000) {
  if (injectionDone) return;

  const selectorMap = {
    chatgpt: ["#prompt-textarea", "textarea", "[contenteditable='true']"],
    gemini: [".ql-editor", "[contenteditable='true']"],
    claude: [".ProseMirror", "[contenteditable='true']"],
  };
  const candidates = selectorMap[platform] || ["textarea", "[contenteditable='true']"];
  const start = Date.now();

  return new Promise((resolve) => {
    const poll = setInterval(async () => {
      let inputEl = null;
      for (const sel of candidates) {
        const el = document.querySelector(sel);
        if (el && isVisible(el)) { inputEl = el; break; }
      }

      if (inputEl) {
        clearInterval(poll);
        await sleep(400); // let page settle
        await injectContext(inputEl, platform, similarityThreshold);
        resolve();
      } else if (Date.now() - start > maxWaitMs) {
        clearInterval(poll);
        console.log("[ContextRot/Injector] Timed out waiting for input.");
        resolve();
      }
    }, 300);
  });
}

// ── Fallback query when input is empty ───────────────────────────────────────

async function getRecentTopicQuery() {
  try {
    const reply = await sendMessage({ type: "LIST" });
    if (reply?.data?.sessions?.length) {
      // Use a broad generic query — real embeddings will still surface relevant chunks
      return "previous conversation memory important user facts ongoing project discussion";
    }
  } catch (_) { }
  return null;
}

// ── Format context block ──────────────────────────────────────────────────────

function formatContextBlock(results) {
  const snippets = results.slice(0, 3).map((r, i) => {
    const date = r.timestamp ? new Date(r.timestamp).toLocaleDateString() : "unknown";
    const preview = r.text.slice(0, 400) + (r.text.length > 400 ? "…" : "");
    return `[Memory ${i + 1} | ${r.platform} | ${date} | score: ${r.score.toFixed(2)}]\n${preview}`;
  }).join("\n\n");

  return `[CONTEXT ROT MEMORY — relevant past conversations]\n${snippets}\n[END MEMORY]\n\n`;
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function getInputText(el) {
  if (!el) return "";
  return el.tagName === "TEXTAREA" ? (el.value || "") : (el.innerText || el.textContent || "");
}

function prependToInput(el, text) {
  if (!el) return;
  if (el.tagName === "TEXTAREA") {
    el.value = text + el.value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  } else {
    el.innerText = text + (el.innerText || "");
    el.dispatchEvent(new InputEvent("input", { bubbles: true }));
    moveCursorAfter(el, text.length);
  }
}

function removeInjectedContext(el, contextBlock) {
  if (!el) return;
  if (el.tagName === "TEXTAREA") {
    if (el.value.startsWith(contextBlock)) {
      el.value = el.value.slice(contextBlock.length);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }
  } else {
    const cur = el.innerText || "";
    if (cur.startsWith(contextBlock)) {
      el.innerText = cur.slice(contextBlock.length);
      el.dispatchEvent(new InputEvent("input", { bubbles: true }));
    }
  }
}

function moveCursorAfter(el, offset) {
  try {
    const range = document.createRange();
    const sel = window.getSelection();
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let chars = 0, node;
    while ((node = walker.nextNode())) {
      if (chars + node.length >= offset) {
        range.setStart(node, offset - chars);
        range.collapse(true);
        sel.removeAllRanges();
        sel.addRange(range);
        return;
      }
      chars += node.length;
    }
    range.selectNodeContents(el);
    range.collapse(false);
    sel.removeAllRanges();
    sel.addRange(range);
  } catch (_) { }
}

function isVisible(el) {
  if (!el) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0 &&
    getComputedStyle(el).visibility !== "hidden" &&
    getComputedStyle(el).display !== "none";
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Banner ────────────────────────────────────────────────────────────────────

function showBanner(inputEl, count, onDismiss) {
  if (activeBanner) activeBanner.remove();

  const banner = document.createElement("div");
  banner.id = "contextrot-banner";
  banner.style.cssText = `
    position:fixed; bottom:80px; left:50%; transform:translateX(-50%);
    z-index:999999; display:flex; align-items:center; gap:10px;
    padding:10px 16px; background:#1e1e2e; color:#cdd6f4;
    border:1px solid #6c7086; border-radius:10px;
    font-family:system-ui,sans-serif; font-size:13px;
    box-shadow:0 4px 24px rgba(0,0,0,0.4); max-width:440px; pointer-events:all;
  `;

  const icon = Object.assign(document.createElement("span"), { textContent: "🧠" });
  icon.style.fontSize = "16px";

  const msg = Object.assign(document.createElement("span"), {
    textContent: `Memory: ${count} snippet${count !== 1 ? "s" : ""} injected from past sessions`
  });

  const dismiss = Object.assign(document.createElement("button"), { textContent: "✕ Undo" });
  dismiss.style.cssText = `
    margin-left:auto; padding:3px 10px; background:#313244; color:#cdd6f4;
    border:1px solid #6c7086; border-radius:6px; cursor:pointer;
    font-size:12px; font-family:inherit; white-space:nowrap;
  `;
  dismiss.onclick = () => { banner.remove(); activeBanner = null; onDismiss(); };

  banner.append(icon, msg, dismiss);
  document.body.appendChild(banner);
  activeBanner = banner;

  setTimeout(() => {
    if (document.body.contains(banner)) banner.remove();
    if (activeBanner === banner) activeBanner = null;
  }, 10000);
}

// ── Message helper ────────────────────────────────────────────────────────────

function sendMessage(msg) {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(msg, (res) => {
        if (chrome.runtime.lastError) {
          const errMsg = chrome.runtime.lastError.message;
          if (!errMsg.includes("context invalidated")) {
            console.warn("[ContextRot/Injector]", errMsg);
          }
          resolve(null);
        } else {
          resolve(res);
        }
      });
    } catch (e) {
      if (!e.message.includes("context invalidated")) {
        console.warn("[ContextRot/Injector] sendMessage threw:", e.message);
      }
      resolve(null);
    }
  });
}
