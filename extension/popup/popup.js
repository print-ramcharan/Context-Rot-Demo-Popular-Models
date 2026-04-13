/**
 * popup/popup.js
 * ───────────────
 * Drives the extension popup UI.
 * Talks to background.js via chrome.runtime.sendMessage.
 */

// ── Message helper ────────────────────────────────────────────────────────────
function sendMessage(msg) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(msg, (res) => {
      if (chrome.runtime.lastError) resolve(null);
      else resolve(res);
    });
  });
}

// ── Storage helpers (inline — popup can't import utils/storage.js directly) ──
const PREFS_KEY = "contextrot_prefs";
const DEFAULT_PREFS = {
  captures: { chatgpt: true, gemini: true, claude: true },
  similarityThreshold: 0.3,
  injectEnabled: true,
  sessionIds: { chatgpt: null, gemini: null, claude: null },
};

function loadPrefs() {
  return new Promise((resolve) => {
    chrome.storage.local.get(PREFS_KEY, (r) => {
      const stored = r[PREFS_KEY] || {};
      resolve(deepMerge(DEFAULT_PREFS, stored));
    });
  });
}

function savePrefs(updates) {
  return loadPrefs().then((current) => {
    const merged = deepMerge(current, updates);
    return new Promise((resolve) => {
      chrome.storage.local.set({ [PREFS_KEY]: merged }, () => resolve(merged));
    });
  });
}

function deepMerge(target, source) {
  const out = { ...target };
  for (const key of Object.keys(source)) {
    if (source[key] !== null && typeof source[key] === "object" && !Array.isArray(source[key]) &&
        typeof target[key] === "object" && target[key] !== null) {
      out[key] = deepMerge(target[key], source[key]);
    } else {
      out[key] = source[key];
    }
  }
  return out;
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const statusDot      = $("status-dot");
const statSessions   = $("stat-sessions");
const statChunks     = $("stat-chunks");
const toggleChatGPT  = $("toggle-chatgpt");
const toggleGemini   = $("toggle-gemini");
const toggleClaude   = $("toggle-claude");
const toggleInject   = $("toggle-inject");
const thresholdSlider= $("threshold-slider");
const thresholdVal   = $("threshold-val");
const sessionsList   = $("sessions-list");
const btnRefresh     = $("btn-refresh");
const btnClearAll    = $("btn-clear-all");
const loadingEl      = $("loading");
const mainEl         = $("main");

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  const [prefs, healthReply, statsReply, sessionsReply] = await Promise.all([
    loadPrefs(),
    sendMessage({ type: "HEALTH" }),
    sendMessage({ type: "STATS"  }),
    sendMessage({ type: "LIST"   }),
  ]);

  // Backend status
  const alive = healthReply?.data?.alive ?? false;
  statusDot.classList.toggle("online",  alive);
  statusDot.classList.toggle("offline", !alive);
  statusDot.title = alive ? "Backend online" : "Backend offline — start uvicorn";

  // Stats
  if (statsReply?.ok) {
    statSessions.textContent = statsReply.data.total_sessions ?? "–";
    statChunks.textContent   = statsReply.data.total_chunks   ?? "–";
  }

  // Toggles from prefs
  toggleChatGPT.checked = prefs.captures.chatgpt;
  toggleGemini.checked  = prefs.captures.gemini;
  toggleClaude.checked  = prefs.captures.claude;
  toggleInject.checked  = prefs.injectEnabled;
  thresholdSlider.value = prefs.similarityThreshold;
  thresholdVal.textContent = Number(prefs.similarityThreshold).toFixed(2);

  // Sessions list
  renderSessions(sessionsReply?.data?.sessions ?? []);

  // Show main
  loadingEl.style.display = "none";
  mainEl.style.display    = "block";
}

// ── Render sessions ───────────────────────────────────────────────────────────
function renderSessions(sessions) {
  if (!sessions.length) {
    sessionsList.innerHTML = `<div class="empty-state">No sessions yet — start chatting!</div>`;
    return;
  }

  sessionsList.innerHTML = "";

  sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = "session-item";

    const date = s.last_updated || s.created_at
      ? new Date(s.last_updated || s.created_at).toLocaleDateString()
      : "–";

    item.innerHTML = `
      <span class="session-platform">${escHtml(s.platform)}</span>
      <span class="session-meta">
        <strong>${s.message_count}</strong> msg · <strong>${s.chunk_count}</strong> chunks · ${date}
      </span>
      <button class="delete-btn" title="Delete session" data-id="${escHtml(s.session_id)}">✕</button>
    `;

    item.querySelector(".delete-btn").addEventListener("click", async (e) => {
      const id = e.currentTarget.dataset.id;
      const reply = await sendMessage({ type: "DELETE", payload: { sessionId: id } });
      if (reply?.ok) {
        item.style.opacity = "0.4";
        item.style.pointerEvents = "none";
        setTimeout(() => refreshAll(), 400);
      }
    });

    sessionsList.appendChild(item);
  });
}

// ── Toggle change handlers ────────────────────────────────────────────────────
toggleChatGPT.addEventListener("change", () =>
  savePrefs({ captures: { chatgpt: toggleChatGPT.checked } })
);
toggleGemini.addEventListener("change", () =>
  savePrefs({ captures: { gemini: toggleGemini.checked } })
);
toggleClaude.addEventListener("change", () =>
  savePrefs({ captures: { claude: toggleClaude.checked } })
);
toggleInject.addEventListener("change", () =>
  savePrefs({ injectEnabled: toggleInject.checked })
);

// ── Threshold slider ──────────────────────────────────────────────────────────
thresholdSlider.addEventListener("input", () => {
  const v = parseFloat(thresholdSlider.value);
  thresholdVal.textContent = v.toFixed(2);
});
thresholdSlider.addEventListener("change", () => {
  savePrefs({ similarityThreshold: parseFloat(thresholdSlider.value) });
});

// ── Footer buttons ────────────────────────────────────────────────────────────
btnRefresh.addEventListener("click", refreshAll);

btnClearAll.addEventListener("click", async () => {
  const confirmed = confirm("Delete ALL stored conversation memory? This cannot be undone.");
  if (!confirmed) return;

  const sessionsReply = await sendMessage({ type: "LIST" });
  const sessions = sessionsReply?.data?.sessions ?? [];

  // Delete all sessions sequentially
  for (const s of sessions) {
    await sendMessage({ type: "DELETE", payload: { sessionId: s.session_id } });
  }

  await refreshAll();
});

// ── Refresh ───────────────────────────────────────────────────────────────────
async function refreshAll() {
  const [statsReply, sessionsReply] = await Promise.all([
    sendMessage({ type: "STATS" }),
    sendMessage({ type: "LIST"  }),
  ]);

  if (statsReply?.ok) {
    statSessions.textContent = statsReply.data.total_sessions ?? "–";
    statChunks.textContent   = statsReply.data.total_chunks   ?? "–";
  }

  renderSessions(sessionsReply?.data?.sessions ?? []);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Boot ──────────────────────────────────────────────────────────────────────
init().catch(console.error);
