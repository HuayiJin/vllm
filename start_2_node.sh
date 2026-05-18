#!/bin/bash
set -euo pipefail

export MODEL_PATH=${MODEL_PATH:-/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/kimi_k26_sft_0515_mgt_hf}
export MODEL_NAME=${MODEL_NAME:-default}

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"

# System-provided distributed training parameters (from K8s/SLURM/etc.)
: "${MASTER_ADDR:?MASTER_ADDR is required}"
# Resolve MASTER_ADDR to IP in case it is a hostname (vLLM requires an IP)
_resolve_to_ip() {
  local addr="$1"
  if [[ "$addr" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf '%s' "$addr"
    return
  fi
  local ip
  ip=$(getent hosts "$addr" 2>/dev/null | awk '{print $1; exit}') || true
  if [[ -z "$ip" ]]; then
    ip=$(python3 -c "import socket; print(socket.gethostbyname('$addr'))" 2>/dev/null) || true
  fi
  if [[ -z "$ip" ]]; then
    printf '[warn] cannot resolve %s to IP, using as-is\n' "$addr" >&2
    printf '%s' "$addr"
  else
    printf '%s' "$ip"
  fi
}
export MASTER_ADDR=$(_resolve_to_ip "$MASTER_ADDR")
export MASTER_PORT=${MASTER_PORT:-23456}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-2}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-}

export PORT=${PORT:-${VLLM_PORT:-30000}}
export HOST=${HOST:-0.0.0.0}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
export PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-2}
export REASONING_PARSER=${REASONING_PARSER:-qwen3}
export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder}

export VLLM_IMAGE_FETCH_TIMEOUT=${VLLM_IMAGE_FETCH_TIMEOUT:-3}
export VLLM_VIDEO_FETCH_TIMEOUT=${VLLM_VIDEO_FETCH_TIMEOUT:-10}
export VLLM_AUDIO_FETCH_TIMEOUT=${VLLM_AUDIO_FETCH_TIMEOUT:-5}
export VLLM_MEDIA_FETCH_MAX_RETRIES=${VLLM_MEDIA_FETCH_MAX_RETRIES:-1}
export VLLM_MEDIA_LOADING_THREAD_COUNT=${VLLM_MEDIA_LOADING_THREAD_COUNT:-4}

export VLLM_MEDIA_CACHE=${VLLM_MEDIA_CACHE:-/tmp/vllm_media_cache}
export VLLM_MEDIA_CACHE_MAX_SIZE_MB=${VLLM_MEDIA_CACHE_MAX_SIZE_MB:-10240}
export VLLM_MEDIA_CACHE_TTL_HOURS=${VLLM_MEDIA_CACHE_TTL_HOURS:-24}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

WATCHDOG_ENABLE=${WATCHDOG_ENABLE:-1}
WATCHDOG_INTERVAL_SECONDS=${WATCHDOG_INTERVAL_SECONDS:-30}
WATCHDOG_PROBE_TIMEOUT_SECONDS=${WATCHDOG_PROBE_TIMEOUT_SECONDS:-20}
WATCHDOG_STARTUP_GRACE_SECONDS=${WATCHDOG_STARTUP_GRACE_SECONDS:-600}
WATCHDOG_FAILURE_THRESHOLD=${WATCHDOG_FAILURE_THRESHOLD:-2}
WATCHDOG_RESTART_COOLDOWN_SECONDS=${WATCHDOG_RESTART_COOLDOWN_SECONDS:-30}
WATCHDOG_LOG_DIR=${WATCHDOG_LOG_DIR:-/tmp/vllm_watchdog}
WATCHDOG_MAX_RESTARTS=${WATCHDOG_MAX_RESTARTS:-0}
WATCHDOG_PROBE_MAX_TOKENS=${WATCHDOG_PROBE_MAX_TOKENS:-1}
WATCHDOG_CURL_CONNECT_TIMEOUT_SECONDS=${WATCHDOG_CURL_CONNECT_TIMEOUT_SECONDS:-2}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.82}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}
VLLM_LONG_PREFILL_TOKEN_THRESHOLD=${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-8192}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-1}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-262144}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1}

mkdir -p "$WATCHDOG_LOG_DIR"

VLLM_PID=""
WATCHDOG_FAILURES=0
WATCHDOG_RESTARTS=0
VLLM_LOG_FILE=""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

local_url_host() {
  if [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]]; then
    printf '127.0.0.1'
  else
    printf '%s' "$HOST"
  fi
}

vllm_cmd() {
  vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --reasoning-parser "$REASONING_PARSER" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --nnodes "$WORLD_SIZE" \
    --node-rank "$RANK" \
    --disable-custom-all-reduce \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --disable-chunked-mm-input \
    --max-num-partial-prefills 1 \
    --max-long-partial-prefills 1 \
    --long-prefill-token-threshold "$VLLM_LONG_PREFILL_TOKEN_THRESHOLD" \
    --mm-processor-cache-gb "$VLLM_MM_PROCESSOR_CACHE_GB" \
    --trust-remote-code \
    "$@"
}

start_vllm() {
  local ts
  ts=$(date '+%Y%m%d_%H%M%S')
  VLLM_LOG_FILE="$WATCHDOG_LOG_DIR/vllm_${ts}.log"
  log "starting vLLM, log_file=$VLLM_LOG_FILE"
  vllm_cmd "$@" > >(tee -a "$VLLM_LOG_FILE") 2>&1 &
  VLLM_PID=$!
  WATCHDOG_FAILURES=0
  log "vLLM pid=$VLLM_PID"
}

stop_vllm() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    log "stopping vLLM pid=$VLLM_PID"
    kill -TERM "$VLLM_PID" 2>/dev/null || true

    for _ in $(seq 1 30); do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done

    if kill -0 "$VLLM_PID" 2>/dev/null; then
      log "vLLM pid=$VLLM_PID did not exit after SIGTERM, sending SIGKILL"
      kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi

    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

print_diagnostics() {
  local diag_file="$WATCHDOG_LOG_DIR/diagnostics_$(date '+%Y%m%d_%H%M%S').log"
  {
    echo "========== watchdog diagnostics =========="
    date '+%Y-%m-%d %H:%M:%S'
    echo
    echo "========== config =========="
    echo "MODEL_PATH=$MODEL_PATH"
    echo "MODEL_NAME=$MODEL_NAME"
    echo "HOST=$HOST"
    echo "PORT=$PORT"
    echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
    echo "PIPELINE_PARALLEL_SIZE=$PIPELINE_PARALLEL_SIZE"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"
    echo "RANK=$RANK"
    echo "WORLD_SIZE=$WORLD_SIZE"
    echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    echo "NCCL_IB_HCA=$NCCL_IB_HCA"
    echo "VLLM_PID=${VLLM_PID:-}"
    echo "VLLM_LOG_FILE=${VLLM_LOG_FILE:-}"
    echo "VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION"
    echo "VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN"
    echo "VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS"
    echo "VLLM_MAX_NUM_BATCHED_TOKENS=$VLLM_MAX_NUM_BATCHED_TOKENS"
    echo "VLLM_LONG_PREFILL_TOKEN_THRESHOLD=$VLLM_LONG_PREFILL_TOKEN_THRESHOLD"
    echo "VLLM_MM_PROCESSOR_CACHE_GB=$VLLM_MM_PROCESSOR_CACHE_GB"
    echo
    echo "========== health =========="
    curl -sS --max-time 5 "http://$(local_url_host):$PORT/health" || true
    echo
    echo
    echo "========== models =========="
    curl -sS --max-time 5 "http://$(local_url_host):$PORT/v1/models" || true
    echo
    echo
    echo "========== nvidia-smi =========="
    nvidia-smi || true
    echo
    echo "========== nvidia-smi compute apps =========="
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv || true
    echo
    echo "========== process tree =========="
    if [[ -n "${VLLM_PID:-}" ]]; then
      ps -o pid,ppid,pgid,stat,%cpu,%mem,etime,rss,command -p "$VLLM_PID" || true
      pgrep -P "$VLLM_PID" | xargs -r ps -o pid,ppid,pgid,stat,%cpu,%mem,etime,rss,command -p || true
    fi
    echo
    echo "========== listening ports =========="
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN || true
    echo
    echo "========== recent vLLM log =========="
    if [[ -n "${VLLM_LOG_FILE:-}" && -f "$VLLM_LOG_FILE" ]]; then
      tail -n 300 "$VLLM_LOG_FILE" || true
    fi
  } > "$diag_file" 2>&1

  log "diagnostics saved to $diag_file"
}

probe_generation() {
  local host_for_probe
  host_for_probe=$(local_url_host)
  curl -sS -N \
    --connect-timeout "$WATCHDOG_CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time "$WATCHDOG_PROBE_TIMEOUT_SECONDS" \
    "http://$host_for_probe:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":$WATCHDOG_PROBE_MAX_TOKENS,\"temperature\":0}" \
    >/dev/null
}

watchdog_loop() {
  log "watchdog enabled: interval=${WATCHDOG_INTERVAL_SECONDS}s timeout=${WATCHDOG_PROBE_TIMEOUT_SECONDS}s threshold=$WATCHDOG_FAILURE_THRESHOLD startup_grace=${WATCHDOG_STARTUP_GRACE_SECONDS}s"
  sleep "$WATCHDOG_STARTUP_GRACE_SECONDS"

  while true; do
    if [[ -z "${VLLM_PID:-}" ]] || ! kill -0 "$VLLM_PID" 2>/dev/null; then
      log "vLLM process exited unexpectedly"
      print_diagnostics
      if [[ "$WATCHDOG_MAX_RESTARTS" != "0" && "$WATCHDOG_RESTARTS" -ge "$WATCHDOG_MAX_RESTARTS" ]]; then
        log "max restarts reached: $WATCHDOG_MAX_RESTARTS"
        exit 1
      fi
      WATCHDOG_RESTARTS=$((WATCHDOG_RESTARTS + 1))
      start_vllm "$@"
      sleep "$WATCHDOG_STARTUP_GRACE_SECONDS"
      continue
    fi

    if probe_generation; then
      if [[ "$WATCHDOG_FAILURES" -ne 0 ]]; then
        log "probe recovered after $WATCHDOG_FAILURES failure(s)"
      fi
      WATCHDOG_FAILURES=0
    else
      WATCHDOG_FAILURES=$((WATCHDOG_FAILURES + 1))
      log "probe failed: failures=$WATCHDOG_FAILURES/$WATCHDOG_FAILURE_THRESHOLD"
    fi

    if [[ "$WATCHDOG_FAILURES" -ge "$WATCHDOG_FAILURE_THRESHOLD" ]]; then
      log "hang detected by generation probe; collecting diagnostics and restarting"
      print_diagnostics
      if [[ "$WATCHDOG_MAX_RESTARTS" != "0" && "$WATCHDOG_RESTARTS" -ge "$WATCHDOG_MAX_RESTARTS" ]]; then
        log "max restarts reached: $WATCHDOG_MAX_RESTARTS"
        exit 1
      fi
      WATCHDOG_RESTARTS=$((WATCHDOG_RESTARTS + 1))
      stop_vllm
      sleep "$WATCHDOG_RESTART_COOLDOWN_SECONDS"
      start_vllm "$@"
      sleep "$WATCHDOG_STARTUP_GRACE_SECONDS"
    else
      sleep "$WATCHDOG_INTERVAL_SECONDS"
    fi
  done
}

cleanup() {
  stop_vllm
}
trap cleanup EXIT INT TERM

# Worker nodes don't expose an API endpoint, skip watchdog
if [[ "$RANK" != "0" ]]; then
  log "node-rank=$RANK (worker): watchdog disabled"
  WATCHDOG_ENABLE=0
fi

if [[ "$WATCHDOG_ENABLE" == "1" ]]; then
  start_vllm "$@"
  watchdog_loop "$@"
else
  exec vllm_cmd "$@"
fi
