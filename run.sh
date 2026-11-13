#!/usr/bin/env bash
#
# run.sh — One-command launcher for the Context Rot Demo.
#
#   1. Creates & activates the Python venv (backend)
#   2. Installs Python + Node dependencies (only if missing)
#   3. Makes sure an OpenRouter API key is configured
#   4. Starts the backend (FastAPI) AND frontend (Vite) together
#
# Usage:
#   ./run.sh              # setup (if needed) + run everything
#   ./run.sh --install    # force re-install of dependencies
#   ./run.sh --no-frontend# backend only
#   ./run.sh --help       # show options
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/venv"
ENV_FILE="$BACKEND_DIR/.env"
BACKEND_PORT="${BACKEND_PORT:-8001}"

# Default flags
FORCE_INSTALL=0
NO_FRONTEND=0

# --- Parse args ---------------------------------------------------------
for arg in "$@"; do
  case "$arg" in
    --install) FORCE_INSTALL=1 ;;
    --no-frontend) NO_FRONTEND=1 ;;
    --help|-h)
      sed -n '1,15p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

echo ""
echo "=============================================="
echo "  Context Rot Demo  —  one-click launcher"
echo "=============================================="

# --- 1. Ensure OpenRouter API key ---------------------------------------
echo ""
echo ">> [1/4] Checking OpenRouter API key..."
if [ ! -f "$ENV_FILE" ] || ! grep -q "^OPENROUTER_API_KEY=.\+" "$ENV_FILE" 2>/dev/null || grep -q "^OPENROUTER_API_KEY=your_openrouter_key_here$" "$ENV_FILE" 2>/dev/null; then
  echo "    No OpenRouter API key found (needed by the LLM)."
  read -r -p "    Paste your OpenRouter API key (or press Enter to skip): " KEY
  if [ -n "$KEY" ]; then
    mkdir -p "$(dirname "$ENV_FILE")"
    # Preserve an existing file; just add/replace the key line
    [ -f "$ENV_FILE" ] && sed -i '' '/^OPENROUTER_API_KEY=/d' "$ENV_FILE"
    echo "OPENROUTER_API_KEY=$KEY" >> "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "    Saved OPENROUTER_API_KEY to $ENV_FILE"
  else
    echo "    !! Skipped. Queries will fail until you set OPENROUTER_API_KEY in $ENV_FILE"
    echo "       Get a key: https://openrouter.ai/keys"
  fi
else
  echo "    Key already configured in $ENV_FILE"
fi

# --- 2. Backend: venv + dependencies ------------------------------------
echo ""
echo ">> [2/4] Setting up backend..."
if [ ! -d "$VENV_DIR/bin" ]; then
  echo "    Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
elif [ "$FORCE_INSTALL" -eq 1 ]; then
  echo "    (re-installing backend deps)"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ "$FORCE_INSTALL" -eq 1 ] || ! python -c "import fastapi, uvicorn, dotenv, faiss" >/dev/null 2>&1; then
  echo "    Installing backend dependencies (this may take a while)..."
  pip install --quiet --upgrade pip
  pip install --quiet -r "$BACKEND_DIR/requirements.txt"
else
  echo "    Backend dependencies already installed."
fi

# --- 3. Frontend: node_modules ------------------------------------------
FRONTEND_READY=0
if [ "$NO_FRONTEND" -ne 1 ]; then
  echo ""
  echo ">> [3/4] Setting up frontend..."
  if [ ! -d "$FRONTEND_DIR/node_modules" ] || [ "$FORCE_INSTALL" -eq 1 ]; then
    echo "    Installing frontend dependencies (npm install)..."
    ( cd "$FRONTEND_DIR" && npm install --no-fund --no-audit )
  else
    echo "    Frontend dependencies already installed."
  fi
  FRONTEND_READY=1
else
  echo ""
  echo ">> [3/4] Skipping frontend (--no-frontend)."
fi

# --- 4. Launch -----------------------------------------------------------
echo ""
echo ">> [4/4] Launching services..."
echo "    Backend : http://127.0.0.1:$BACKEND_PORT  (API docs at /docs)"
[ "$FRONTEND_READY" -eq 1 ] && echo "    Frontend: http://localhost:5173"
if [ "$FRONTEND_READY" -eq 1 ]; then
  echo ""
  echo "    Starting BOTH services. Press Ctrl+C to stop them together."
fi
echo ""

# Kill both processes on Ctrl+C / exit
BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
  trap - INT TERM EXIT
  echo ""
  echo "Shutting down..."
  [ -n "$BACKEND_PID"  ] && kill "$BACKEND_PID"  2>/dev/null || true
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "Done. See you next time!"
}
trap cleanup INT TERM EXIT

# Start backend in the background
( cd "$BACKEND_DIR" && exec uvicorn api:app --reload --host 127.0.0.1 --port "$BACKEND_PORT" ) &
BACKEND_PID=$!

# Start frontend in the background (if requested)
if [ "$FRONTEND_READY" -eq 1 ]; then
  ( cd "$FRONTEND_DIR" && VITE_API_BASE_URL="http://127.0.0.1:$BACKEND_PORT" exec npm run dev ) &
  FRONTEND_PID=$!
fi

# Wait for Ctrl+C
wait