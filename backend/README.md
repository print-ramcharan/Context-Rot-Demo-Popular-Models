# LLM Memory System

This project implements an external memory system for Large Language Models to mitigate context degradation.

## Project Structure

- `backend/`: Python-based external memory system using FAISS and Ollama/Gemini.
  - `api.py`: FastAPI entrypoint.
  - `chunker.py`: Document chunking utilities.
  - `vector_store.py`: Vector store helpers (updated with new method).
- `frontend/`: Web interface for interacting with the memory system (updated UI).

## Getting Started

### Backend

```bash
cd backend
# install dependencies per backend/README.md
fastapi dev api.py
```

### Environment Variables

Make sure required API keys are set in your environment (see backend/README.md for exact names).



````markdown id="readmefinal"
# Context Rot Demo — Cross-Platform Memory Extension

## Project Overview
This project extends the original Context Rot Demo into a cross-platform AI memory system that preserves conversational context across multiple LLM platforms.

The system now allows conversations from ChatGPT, Gemini, and Claude to be:
- captured automatically from browser sessions,
- stored in vector memory using FAISS,
- semantically retrieved later,
- injected into future conversations across platforms.

Example:
A discussion started in ChatGPT can later be recalled automatically in Gemini without the user repeating prior context.

---

## What Has Been Implemented

### Backend Integration (FastAPI + FAISS)
The backend is fully connected with the browser extension and supports:

- `POST /store-conversation`
- `POST /retrieve-context`
- `GET /list-sessions`
- `DELETE /delete-session/{session_id}`
- `GET /conv-stats`

Implemented capabilities:
- semantic embedding generation
- vector storage using FAISS
- similarity-based retrieval
- persistent cross-session memory indexing

---

### Chrome Extension Added
A Manifest V3 Chrome extension has been developed under:

```bash
extension/
````

This includes:

#### Core Files

* `manifest.json`
* `background.js`

#### Content Scripts

* `content/chatgpt.js`
* `content/gemini.js`
* `content/claude.js`
* `content/injector.js`

#### Utility Modules

* `utils/api.js`
* `utils/storage.js`

#### Popup UI

* `popup/popup.html`
* `popup/popup.js`

---

## Working Features Confirmed

### ✅ ChatGPT Conversation Capture

Prompt-response pairs are captured and stored automatically.

### ✅ Gemini Conversation Capture

Gemini chats are also stored successfully.

### ✅ Cross-Platform Memory Transfer

Memory created in ChatGPT can be retrieved in Gemini.

### ✅ Automatic Context Injection

Relevant memory snippets are injected into new prompt inputs.

### ✅ Session Tracking

Stored sessions are visible and deletable via backend APIs.

---

## Testing Status

### Verified Successfully

* ChatGPT → Gemini memory transfer works
* Retrieval and semantic injection works
* Backend stores and retrieves sessions correctly

### Pending / Optional Further Testing

* Claude end-to-end full validation
* Additional UI polishing

---

## Known Development Notes

### Chrome Reload Warning

Sometimes browser console may show:

```text
Extension context invalidated
```

This happens when:

* extension is reloaded during development,
* old browser tabs still contain stale content scripts.

This is a Chrome extension development artifact, not a functional bug.

Fix:

1. Close AI tabs
2. Reload extension
3. Reopen fresh tabs

---

### Gemini UI Instability

Gemini frequently changes DOM structure, so selectors may need updates if Google changes layout.

---

## How To Run

### Start Backend

```bash
cd backend
uvicorn api:app --reload
or
venv\Scripts\python -m uvicorn api:app --reload
```

### Load Extension

1. Open Chrome:

```bash
chrome://extensions
```

2. Enable Developer Mode

3. Click:

```bash
Load unpacked
```

4. Select:

```bash
extension/
```

---

## Current Branch

Development branch for this implementation:

```bash
feature/browser-memory-extension
```

---

## Recommended Next Steps

* Complete Claude platform validation
* Improve Gemini rich-editor injection reliability
* Add smarter topic-aware memory ranking
* Add memory summarization before injection

---

## Team Notes

The browser extension and backend integration are now functional end-to-end.

Major milestone achieved:

### Cross-platform persistent AI memory is successfully working.

## Example Test Prompts

Use the following prompts to validate cross-platform memory capture, retrieval, and context injection.

### Test 1 — Technical Memory Recall

**In ChatGPT:**
```text
Remember this:
Our hybrid retrieval system uses BM25, FAISS, alpha=0.65 and RRF constant K=60.
```

**Then in Gemini/Claude:**
```text
What retrieval configuration did I mention earlier?
```

Expected:
Memory should be retrieved and referenced in the response.

---

### Test 2 — Cross-Platform Discussion Continuation

**In ChatGPT:**
```text
We discussed attention dilution and lost-in-the-middle effects in large language models.
```

**Then in Gemini:**
```text
Continue the earlier discussion about context degradation.
```

Expected:
Previous discussion should be injected and used.

---

### Test 3 — Personal Preference Memory

**In ChatGPT:**
```text
Remember that I prefer concise technical explanations.
```

**Then in Claude:**
```text
Answer in the style I mentioned earlier.
```

Expected:
Response should reflect stored preference.

---

### Test 4 — Secret Phrase Recall

**In ChatGPT:**
```text
Remember this phrase:
The purple elephant uses hybrid retrieval.
```

**Then in Gemini:**
```text
What secret phrase did I mention earlier?
```

Expected:
Phrase should be recalled correctly.

---

### Test 5 — Meeting Notes Recall

**In ChatGPT:**
```text
Remember these meeting notes:
- Deadline Friday
- Use BM25 module
- Prepare demo slides
```

**Then in Claude:**
```text
Can you recall the meeting notes I shared earlier?
```

Expected:
Stored notes should be retrieved.

---

### Test 6 — Relevance Filtering (Negative Test)

Set similarity threshold high and ask:

```text
What is the capital of Peru?
```

Expected:
No unrelated memory should be injected.


```
```

