#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"

PORT=${PORT:-${VLLM_PORT:-30000}}
HOST=${HOST:-0.0.0.0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder}

exec /data/temp/vllm/.venv/bin/vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL_NAME" \
  --reasoning-parser "$REASONING_PARSER" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  "$@"
