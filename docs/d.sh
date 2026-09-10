#!/bin/bash
set -euo pipefail

# Bailing 3.0 Flash - PD disaggregated decode, 1P1D on one machine, TP8.
# Usage:
#   bash bailing3.0_flash_start_decode_1p1d_tp8.sh
#   DECODE_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 bash bailing3.0_flash_start_decode_1p1d_tp8.sh

NODE_RANK="${1:-0}"
if [ "${NODE_RANK}" != "0" ]; then
    echo "Only node_rank=0 is supported for this single-machine 1P1D script."
    exit 1
fi

current_dir=$(cd "$(dirname "$0")" && pwd)

# ----------------------------- model / ports ----------------------------- #
export DECODE_PORT="${DECODE_PORT:-8200}"
export DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"
export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# A TP8 decode instance needs 8 visible NPUs. For same-machine 1P1D with both
# sides TP8, prefill and decode must use non-overlapping device lists.
export DECODE_VISIBLE_DEVICES="${DECODE_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export ASCEND_RT_VISIBLE_DEVICES="${DECODE_VISIBLE_DEVICES}"

# Parallel config.
PREFILL_DP_SIZE=2
PREFILL_TP_SIZE=4
DECODE_DP_SIZE=2
DECODE_TP_SIZE=4

# ------------------------------ environment ------------------------------ #
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64/python3.11/site-packages/mooncake:${LD_LIBRARY_PATH:-}
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-600}"
export HCCL_EVENT_TIMEOUT="${HCCL_EVENT_TIMEOUT:-600}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-120}"
export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-10000}"
export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-10000}"

export VLLM_USE_V1=1
export VLLM_VERSION="${VLLM_VERSION:-0.20.2}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-380}"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export PROMETHEUS_MULTIPROC_DIR=/tmp/

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

export NET_CARD_NAME="${NET_CARD_NAME:-eth0}"
export GLOO_SOCKET_IFNAME="${NET_CARD_NAME}"
export TP_SOCKET_IFNAME="${NET_CARD_NAME}"
export HCCL_SOCKET_IFNAME="${NET_CARD_NAME}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-0}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"

export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}:/usr/include/c++/12/:/usr/include/c++/12/x86_64-openEuler-linux/"

export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-800}"
export CAM_BUFFSIZE="${CAM_BUFFSIZE:-${HCCL_BUFFSIZE}}"
export ASCEND_AGGREGATE_ENABLE="${ASCEND_AGGREGATE_ENABLE:-1}"
export ASCEND_TRANSPORT_PRINT="${ASCEND_TRANSPORT_PRINT:-1}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
export ASCEND_A3_ENABLE="${ASCEND_A3_ENABLE:-1}"
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT="${VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:-1800}"
export MC_LOG_LEVEL="${MC_LOG_LEVEL:-ERROR}"
export GLOG_alsologtostderr="${GLOG_alsologtostderr:-1}"

# ------------------------------- logging --------------------------------- #
export LOG_PATH="${LOG_PATH:-${current_dir}/logs}"
mkdir -p "${LOG_PATH}"

export ASCEND_PROCESS_LOG_PATH="${LOG_PATH}/bailing_flash_decode/ascend"
export MC_LOG_DIR="${LOG_PATH}/bailing_flash_decode/mooncake"
rm -rf "${ASCEND_PROCESS_LOG_PATH}" "${MC_LOG_DIR}" 2>/dev/null || true
mkdir -p "${ASCEND_PROCESS_LOG_PATH}" "${MC_LOG_DIR}"

export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-/tmp/theta_workdir/profiles_bailing_flash_decode}"
export VLLM_WORKDIR="${VLLM_WORKDIR:-/tmp/theta_workdir/bailing_flash_decode}"
rm -rf "${VLLM_WORKDIR}" "${VLLM_TORCH_PROFILER_DIR}" 2>/dev/null || true
mkdir -p "${VLLM_WORKDIR}" "${VLLM_TORCH_PROFILER_DIR}"

rm -rf /root/.triton/cache/* 2>/dev/null || true

cd "${VLLM_WORKDIR}"
ulimit -n 1048576
ulimit -c unlimited

# ---------------------------- vLLM arguments ----------------------------- #
COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --distributed-executor-backend mp
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --enable-log-requests
    --enable-prompt-tokens-details
    --otlp-traces-endpoint https://antcollector.alipay.com/namespace/aicloud/task/otlptrace/otlp/api/v1/traces
"

ADDITIONAL_CONFIG='{"enable_cpu_binding": true}'

kv_transfer_config=$(cat <<EOF
{
    "kv_connector": "MooncakeHybridConnector",
    "kv_buffer_device": "npu",
    "kv_role": "kv_consumer",
    "kv_parallel_size": 2,
    "kv_port": "${DECODE_KV_PORT}",
    "engine_id": "bailing-flash-decode-0",
    "kv_rank": 1,
    "kv_connector_extra_config": {
        "use_ascend_direct": true,
        "prefill": {
            "dp_size": ${PREFILL_DP_SIZE},
            "tp_size": ${PREFILL_TP_SIZE}
        },
        "decode": {
            "dp_size": ${DECODE_DP_SIZE},
            "tp_size": ${DECODE_TP_SIZE}
        }
    }
}
EOF
)

echo "Starting Bailing 3.0 Flash decode:"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "  HTTP port=${DECODE_PORT}, KV port=${DECODE_KV_PORT}"
echo "  log=${LOG_PATH}/vllm_bailing_flash_decode.log"

set -x
vllm serve /a3_inference/itask/workdir/models/ling_flash_v3_ifrl_0617 \
    --host "${SERVER_HOST}" \
    --port 8200 \
    --api-server-count 1 \
    ${COMMON_ARGS} \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 16384 \
    --block-size 128 \
    --gpu-memory-utilization 0.85 \
    --mamba_cache_dtype "float32" \
    --chat-template /a3_inference/itask/workdir/models/ling_flash_v3_ifrl_0617/chat_template.jinja \
    --data-parallel-size "${DECODE_DP_SIZE}" \
    --tensor-parallel-size "${DECODE_TP_SIZE}" \
    --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --no-disable-hybrid-kv-cache-manager \
    --enable-expert-parallel \
    --additional-config "${ADDITIONAL_CONFIG}" \
    --kv-transfer-config "${kv_transfer_config}" \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir="${VLLM_TORCH_PROFILER_DIR}"
