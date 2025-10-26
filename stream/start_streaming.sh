#!/bin/bash
# Quick start script for IndexTTS Streaming Server

echo "========================================"
echo "  IndexTTS Streaming Server Launcher"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of stream/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if streaming dependencies are installed
echo "Checking dependencies..."
if ! uv run python -c "import fastapi" 2>/dev/null; then
    echo "Installing streaming dependencies..."
    uv sync --extra streaming
fi

echo "✓ Dependencies OK"
echo ""

# Configuration
export TTS_CONFIG="${TTS_CONFIG:-checkpoints/config.yaml}"
export TTS_MODEL_DIR="${TTS_MODEL_DIR:-checkpoints}"
export TTS_FP16="${TTS_FP16:-true}"
export TTS_CUDA_KERNEL="${TTS_CUDA_KERNEL:-true}"
export TTS_DEEPSPEED="${TTS_DEEPSPEED:-false}"
export TTS_CHUNK_SIZE="${TTS_CHUNK_SIZE:-4096}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  Model config: $TTS_CONFIG"
echo "  Model dir: $TTS_MODEL_DIR"
echo "  FP16: $TTS_FP16"
echo "  CUDA kernels: $TTS_CUDA_KERNEL"
echo "  DeepSpeed: $TTS_DEEPSPEED"
echo "  Chunk size: $TTS_CHUNK_SIZE"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""

# Check if model exists
if [ ! -f "$TTS_CONFIG" ]; then
    echo "❌ Error: Model config not found: $TTS_CONFIG"
    echo "   Please download models first:"
    echo "   hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints"
    exit 1
fi

echo "Starting server..."
echo "================================"
echo ""
echo "🌐 Web UI: http://localhost:$PORT"
echo "🔌 WebSocket: ws://localhost:$PORT/ws/stream"
echo "💚 Health check: http://localhost:$PORT/health"
echo ""
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

# Start server from project root with stream.streaming_server module
exec uv run uvicorn stream.streaming_server:app --host "$HOST" --port "$PORT" --log-level info
