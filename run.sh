
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/venv"

BACKEND_PORT="${BACKEND_PORT:-8001}"

# Limit numerical library threading
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Find existing npm installation
NPM_BIN=""

find_npm() {
    if command -v npm >/dev/null 2>&1; then
        NPM_BIN="$(command -v npm)"
        return 0
    fi

    for candidate in \
        "$HOME/.volta/bin/npm" \
        "$HOME/.asdf/shims/npm" \
        "$HOME"/.nvm/versions/node/*/bin/npm \
        "/opt/homebrew/bin/npm" \
        "/usr/local/bin/npm"
    do
        if [[ -x "$candidate" ]]; then
            NPM_BIN="$candidate"
            return 0
        fi
    done

    return 1
}

# Validate everything before launching
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: Existing backend venv not found."
    exit 1
fi

if [[ ! -x "$FRONTEND_DIR/node_modules/.bin/vite" ]]; then
    echo "ERROR: Frontend node_modules missing."
    exit 1
fi

if ! find_npm; then
    echo "ERROR: npm not found."
    echo "Check your Node.js installation."
    exit 1
fi

# Activate existing environment
source "$VENV_DIR/bin/activate"

echo ""
echo "======================================"
echo "       CONTEXT ROT DEMO"
echo "======================================"
echo "Backend : http://127.0.0.1:$BACKEND_PORT"
echo "Frontend: http://localhost:5173"
echo "Python  : $(python --version)"
echo "Node    : $("$NPM_BIN" --version)"
echo "npm     : $NPM_BIN"
echo "======================================"
echo ""

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    trap - EXIT INT TERM

    echo ""
    echo "Stopping services..."

    if [[ -n "$FRONTEND_PID" ]]; then
        pkill -TERM -P "$FRONTEND_PID" 2>/dev/null || true
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi

    if [[ -n "$BACKEND_PID" ]]; then
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
    fi

    wait 2>/dev/null || true

    echo "Services stopped."
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Start one backend process, no reload
echo "Starting backend..."

(
    cd "$BACKEND_DIR"
    exec "$VENV_DIR/bin/python" -m uvicorn \
        api:app \
        --host 127.0.0.1 \
        --port "$BACKEND_PORT" \
        --workers 1
) &

BACKEND_PID=$!

# Start frontend using resolved npm path
echo "Starting frontend..."

(
    cd "$FRONTEND_DIR"
    exec "$NPM_BIN" run dev -- --host 127.0.0.1
) &

FRONTEND_PID=$!

echo ""
echo "Both services launched."
echo "Press Ctrl+C to stop."
echo ""

wait
