/**
 * background.js
 * ──────────────
 * Manifest V3 service worker.
 *
 * api.js is loaded via importScripts so all its functions are
 * available as globals here. No ES module import needed.
 */

importScripts('./utils/api.js');

// ── Message router ────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message)
    .then(sendResponse)
    .catch((err) => {
      console.error(`[ContextRot BG] Error handling ${message.type}:`, err.message);
      sendResponse({ ok: false, error: err.message });
    });
  return true; // keep channel open for async response
});

async function handleMessage(message) {
  const { type, payload = {} } = message;

  switch (type) {

    case "STORE": {
      const { platform, sessionId, prompt, response } = payload;
      if (!prompt?.trim() && !response?.trim()) {
        return { ok: false, error: "Empty exchange — nothing to store" };
      }
      const result = await storeConversation(platform, sessionId, prompt, response);
      console.log(`[ContextRot] Stored: platform=${platform} chunks=${result.chunk_count}`);
      return { ok: true, data: result };
    }

    case "RETRIEVE": {
      const { query, topK, platformFilter, similarityThreshold } = payload;
      if (!query?.trim()) return { ok: true, data: { results: [], count: 0 } };
      const result = await retrieveContext(query, topK, platformFilter, similarityThreshold);
      return { ok: true, data: result };
    }

    case "LIST": {
      const result = await listSessions();
      return { ok: true, data: result };
    }

    case "DELETE": {
      const result = await deleteSession(payload.sessionId);
      return { ok: true, data: result };
    }

    case "HEALTH": {
      const alive = await checkHealth();
      return { ok: true, data: { alive } };
    }

    case "STATS": {
      const result = await getConvStats();
      return { ok: true, data: result };
    }

    default:
      return { ok: false, error: `Unknown message type: ${type}` };
  }
}

console.log("[ContextRot] Background service worker started.");