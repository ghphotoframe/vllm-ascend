#!/bin/bash
set -e
# export ASCEND_GLOBAL_LOG_LEVEL=1
# export ASCEND_HOST_LOG_FILE_NUM=2000
# ==========================================
# 1. 基础环境变量配置
# ==========================================
export MODEL_PATH="/a3_inference/itask/workdir/models/ling-3.0-flash"
export API_SERVER_COUNT=1
export PORT=8000
export HCCL_ENTRY_LOG_ENABLE=1
# vLLM 核心配置
export VLLM_USE_V1=1
export VLLM_VERSION="0.20.2"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_IMAGE_FETCH_TIMEOUT=30
export VLLM_VIDEO_FETCH_TIMEOUT=60

# 昇腾 (Ascend) / HCCL 通信配置
export HCCL_BUFFSIZE=800
export CAM_BUFFSIZE=800
export ASCEND_PROCESS_LOG_PATH="/home/admin/logs/ascend/"
rm -rf $ASCEND_PROCESS_LOG_PATH/*
# 网络与分布式通信接口
export NET_CARD_NAME="eth0"
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"

# 性能与系统调优
export OMP_NUM_THREADS=6
export TASK_QUEUE_ENABLE=0
export PROMETHEUS_MULTIPROC_DIR="/tmp/"

# OpenTelemetry 链路追踪
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf"

# ==========================================
# 2. vLLM 引擎启动参数 (ENGINE_EXTRA_ARGS)
# ==========================================
ENGINE_ARGS=(
    # --- 模型与加载 ---
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}'
    --chat-template ${MODEL_PATH}/chat_template.jinja

    # --- 并行策略 (TP=8, DP=1, EP) ---
    --tensor-parallel-size 8
    --data-parallel-size 2
    --enable-expert-parallel
    --distributed-executor-backend mp

    # --- 内存与序列控制 ---
    --max-num-seqs 64
    --max-model-len 131072
    --max-num-batched-tokens 16384
    --block-size 128
    --gpu-memory-utilization 0.85
    --enable-prefix-caching

    # --- Mamba / SSM 模型专用 ---
    --mamba-cache-mode align
    --mamba_cache_dtype float32

    # --- 推测解码 (Speculative Decoding) ---
#    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

    # --- 编译与调度优化 ---
    --async-scheduling
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --additional-config '{"enable_cpu_binding":true}'

    # --- Tool Calling & Reasoning ---
    --enable-auto-tool-choice
    --tool-call-parser ling3
    --reasoning-parser ling3

)

# ==========================================
# 3. 启动 vLLM 服务
# ==========================================
echo "============================================"
echo " Starting vLLM Ascend Server"
echo " Model: ${MODEL_PATH}"
echo " Port:  ${PORT}"
echo " TP:    8 | Max Len: 131072"
echo "============================================"

nohup vllm serve \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    "${ENGINE_ARGS[@]}" > ./vllm.log 2>&1 &
