/**
 * utils/api.js
 * ─────────────
 * fetch() wrappers for every backend endpoint.
 *
 * Loaded two ways:
 *   - background.js  → ES module import (background has type:module)
 *   - content scripts → plain script tag injection (no module support)
 *
 * Therefore: NO export keywords. Functions are plain globals.
 * background.js imports this file and the functions are available
 * because the module scope shares them via the import binding.
 */

const API_BASE = "http://127.0.0.1:8000";
const TIMEOUT_MS = 5000;

// ── Internal fetch helper ─────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    clearTimeout(timer);

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const body = await res.json(); detail = body.detail || detail; } catch (_) { }
      throw new Error(detail);
    }

    return await res.json();
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError") throw new Error("Backend timeout — is the server running?");
    throw err;
  }
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

async function checkHealth() {
  try { await apiFetch("/health"); return true; }
  catch (_) { return false; }
}

async function storeConversation(platform, sessionId, prompt, response) {
  return apiFetch("/store-conversation", {
    method: "POST",
    body: JSON.stringify({ platform, session_id: sessionId, prompt, response }),
  });
}

async function retrieveContext(query, topK = 5, platformFilter = null, similarityThreshold = 0.3) {
  return apiFetch("/retrieve-context", {
    method: "POST",
    body: JSON.stringify({
      query,
      top_k: topK,
      platform_filter: platformFilter,
      similarity_threshold: similarityThreshold,
    }),
  });
}

async function listSessions() {
  return apiFetch("/list-sessions");
}

async function deleteSession(sessionId) {
  return apiFetch(`/delete-session/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

async function getConvStats() {
  return apiFetch("/conv-stats");
}