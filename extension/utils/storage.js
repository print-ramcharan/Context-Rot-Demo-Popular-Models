/**
 * utils/storage.js
 * ─────────────────
 * chrome.storage.local helpers for persisting user preferences.
 */

// ── Constants first — must be declared before any function uses them ──────────

const PREFS_KEY = "contextrot_prefs";

const DEFAULT_PREFS = {
  captures: {
    chatgpt: true,
    gemini:  true,
    claude:  true,
  },
  similarityThreshold: 0.3,
  injectEnabled: true,
  sessionIds: {
    chatgpt: null,
    gemini:  null,
    claude:  null,
  },
};

// ── Guard: check extension context is still valid before any chrome API call ──

function isExtensionValid() {
  try {
    // Accessing chrome.runtime.id throws if context is invalidated
    return typeof chrome !== "undefined" && !!chrome.runtime?.id;
  } catch (_) {
    return false;
  }
}

// ── Functions ─────────────────────────────────────────────────────────────────

async function loadPrefs() {
  if (!isExtensionValid()) return { ...DEFAULT_PREFS };

  return new Promise((resolve) => {
    try {
      chrome.storage.local.get(PREFS_KEY, (result) => {
        if (chrome.runtime.lastError) {
          resolve({ ...DEFAULT_PREFS });
          return;
        }
        const stored = result[PREFS_KEY] || {};
        resolve(deepMerge(DEFAULT_PREFS, stored));
      });
    } catch (_) {
      resolve({ ...DEFAULT_PREFS });
    }
  });
}

async function savePrefs(updates) {
  if (!isExtensionValid()) return { ...DEFAULT_PREFS };

  const current = await loadPrefs();
  const merged  = deepMerge(current, updates);

  return new Promise((resolve) => {
    try {
      chrome.storage.local.set({ [PREFS_KEY]: merged }, () => {
        if (chrome.runtime.lastError) { resolve(merged); return; }
        resolve(merged);
      });
    } catch (_) {
      resolve(merged);
    }
  });
}

async function getOrCreateSessionId(platform) {
  const prefs = await loadPrefs();
  let sessionId = prefs.sessionIds[platform];

  if (!sessionId) {
    sessionId = `${platform}-${generateId()}`;
    await savePrefs({ sessionIds: { ...prefs.sessionIds, [platform]: sessionId } });
  }

  return sessionId;
}

async function rotateSessionId(platform) {
  const prefs = await loadPrefs();
  const newId = `${platform}-${generateId()}`;
  await savePrefs({ sessionIds: { ...prefs.sessionIds, [platform]: newId } });
  return newId;
}

async function clearAll() {
  if (!isExtensionValid()) return;
  return new Promise((resolve) => {
    try {
      chrome.storage.local.remove(PREFS_KEY, resolve);
    } catch (_) {
      resolve();
    }
  });
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function deepMerge(target, source) {
  const out = { ...target };
  for (const key of Object.keys(source)) {
    if (
      source[key] !== null &&
      typeof source[key] === "object" &&
      !Array.isArray(source[key]) &&
      typeof target[key] === "object" &&
      target[key] !== null
    ) {
      out[key] = deepMerge(target[key], source[key]);
    } else {
      out[key] = source[key];
    }
  }
  return out;
}