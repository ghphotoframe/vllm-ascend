# BailingMoeV3 (Bailing 3.0) 模型架构深度解析

本文档基于 BailingMoeV3 的 HuggingFace modeling 文件和 vLLM 推理实现，深度剖析 Bailing 3.0 模型的完整架构，包括注意力机制（MLA + KDA 混合）、位置编码（RoPE）、归一化操作、MoE 路由、MTP 预测等，以及每一步的数学计算和 Tensor Shape 变化。

---

## 一、模型全局架构概览

### 1.1 核心配置参数

| 参数 | 值 | 说明 |
|------|------|------|
| `hidden_size` | 2560 | 隐藏层维度 |
| `num_hidden_layers` | 42 | 总层数 |
| `num_attention_heads` | 32 | 注意力头数 |
| `num_key_value_heads` | 32 | KV 头数（MLA 中不直接使用） |
| `head_dim` | 128 | KDA 中每个头的维度 |
| `vocab_size` | 157184 | 词表大小 |
| `intermediate_size` | 6144 | Dense MLP 中间维度 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `rope_theta` | 10000 | RoPE 基频 |
| `rope_interleave` | true | 使用交错式 RoPE |
| `layer_group_size` | 6 | 混合注意力分组大小 |
| `first_k_dense_replace` | 2 | 前 N 层使用 Dense MLP |
| `num_experts` | 512 | MoE 专家总数 |
| `num_experts_per_tok` | 8 | 每个 token 激活专家数 |
| `n_group` | 8 | 专家分组数 |
| `topk_group` | 4 | 选中的专家组数 |
| `moe_intermediate_size` | 768 | MoE 每个专家的中间维度 |
| `moe_shared_expert_intermediate_size` | 768 | 共享专家中间维度 |
| `num_shared_experts` | 1 | 共享专家数 |
| `routed_scaling_factor` | 2.5 | 路由专家缩放因子 |
| `scoring_func` | sigmoid | 路由评分函数 |
| `num_nextn_predict_layers` | 1 | MTP 预测层数 |
| `mtp_loss_scaling_factor` | 0 | MTP 损失缩放（推理时不使用） |

### 1.2 混合注意力层分布

Bailing 3.0 的核心创新是 **MLA（Multi-head Latent Attention）+ KDA（Kimi Delta Attention）混合架构**。每 `layer_group_size=6` 层为一组，其中最后 1 层是 MLA（Softmax 注意力），其余 5 层是 KDA（线性注意力/Delta Rule 注意力）。

```
层分布判断逻辑:
  is_kda_layer = NOT ( (layer_idx + 1) % 6 == 0 )
```

| 层索引 | 类型 | FFN 类型 |
|--------|------|---------|
| 0, 1 | KDA | Dense MLP |
| 2, 3, 4 | KDA | MoE |
| 5 | MLA | MoE |
| 6, 7, 8, 9, 10 | KDA | MoE |
| 11 | MLA | MoE |
| ... | ... | ... |
| 36, 37, 38, 39, 40 | KDA | MoE |
| 41 | MLA | MoE |

**统计**：
- KDA 层：35 层（占总层数 83.3%）
- MLA 层：7 层（占总层数 16.7%）
- Dense MLP 层：2 层（Layer 0、1）
- MoE 层：40 层（Layer 2~41）

### 1.3 整体前向流程

```
输入 token_ids: [batch_size, seq_len]
       │
       ▼
┌──────────────────────────┐
│  Word Embedding          │  [batch, seq_len] → [batch, seq_len, 2560]
│  (vocab_size=157184,     │
│   hidden_size=2560)      │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Rotary Embedding        │  生成 cos/sin 位置编码（仅 MLA 层使用）
│  (qk_rope_head_dim=64)   │
└──────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────┐
  │  42 x DecoderLayer                   │
  │  ┌────────────────────────────────┐  │
  │  │  Input RMSNorm                 │  │
  │  │  ├─ KDA层 → KimiDeltaAttention │  │
  │  │  └─ MLA层 → MultiLatentAttn   │  │
  │  │  Post-Attention RMSNorm        │  │
  │  │  ├─ Layer 0,1 → Dense MLP     │  │
  │  │  └─ Layer 2+  → MoE (含共享专家)│  │
  │  └────────────────────────────────┘  │
  └──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Final RMSNorm           │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  LM Head                 │  [batch, seq_len, 2560] → [batch, seq_len, 157184]
│  (linear, no bias)       │
└──────────────────────────┘
       │
       ▼
  logits (float32)
```

---

## 二、MLA（Multi-head Latent Attention）详解

MLA 是 DeepSeek-V2/V3 提出的低秩 KV 压缩注意力机制，Bailing 3.0 在每组的最后 1 层使用它。核心思想是将 KV 压缩到低秩隐空间，大幅降低 KV Cache 内存。

### 2.1 MLA 关键参数

| 参数 | 值 | 说明 |
|------|------|------|
| `q_lora_rank` | None | Q 不使用 LoRA 压缩，直接投影 |
| `kv_lora_rank` | 512 | KV 压缩隐空间维度 |
| `qk_nope_head_dim` | 128 | Q/K 中不参与 RoPE 的维度 |
| `qk_rope_head_dim` | 64 | Q/K 中参与 RoPE 的维度 |
| `qk_head_dim` | 192 | Q/K 总维度 = 128 + 64 |
| `v_head_dim` | 128 | Value 头维度 |

### 2.2 MLA 前向计算流程

#### 2.2.1 MLA 权重矩阵与参数对应表

Bailing 3.0 的 MLA 前 2 层使用 `q_lora_rank=None`（即 Q 不经过 LoRA 压缩），以下是此配置下的权重矩阵：

| 权重名 | Shape | 计算公式 | 对应 config 参数 |
|--------|-------|----------|-----------------|
| `q_proj.weight` | `[6144, 2560]` | `num_heads × qk_head_dim = 32 × 192` | `num_attention_heads=32, qk_head_dim=qk_nope_head_dim+qk_rope_head_dim=128+64` |
| `kv_a_proj_with_mqa.weight` | `[576, 2560]` | `kv_lora_rank + qk_rope_head_dim = 512 + 64` | `kv_lora_rank=512, qk_rope_head_dim=64` |
| `kv_a_layernorm.weight` | `[512]` | `kv_lora_rank` | `kv_lora_rank=512` |
| `kv_b_proj.weight` | `[8192, 512]` | `num_heads × (qk_nope_head_dim + v_head_dim) = 32 × (128 + 128)` | `num_attention_heads=32, qk_nope_head_dim=128, v_head_dim=128` |
| `g_proj.weight` | `[32, 2560]` | `num_heads = 32`（head_wise 粒度） | `gated_attention_proj_granularity_type="head_wise", num_attention_heads=32` |
| `dense.weight` (o_proj) | `[2560, 4096]` | `hidden_size, num_heads × v_head_dim = 32 × 128` | `hidden_size=2560, num_attention_heads=32, v_head_dim=128` |

> **设计说明**：
>
> 1. **`kv_a_proj_with_mqa` 中的 "MQA"**：MQA（Multi-Query Attention）意为"多查询注意力"，其核心特点是所有 Query 头共享同一组 K、V。在 MLA 中，KV 被压缩到一个低秩隐空间（kv_lora_rank=512 维），这个压缩向量对所有 32 个 Query 头共享，类似于 MQA 中所有头共享 KV 的思想——因此命名为 "with_mqa"。与标准 MHA（每头独立 KV 投影）不同，MLA 的 KV 投影只产生 1 份共享的低秩表示，这也是 KV Cache 大幅压缩的关键。
>
> 2. **`no bias`（`bias=False`）**：MLA 中所有 Linear 投影（q_proj、kv_a_proj_with_mqa、kv_b_proj、g_proj、dense）均不使用偏置项。这是现代 LLM 的普遍设计选择（LLaMA、PaLM 等均采用），主要原因为：(a) 减少 parameter count 和显存占用；(b) 简化量化（quantization），带 bias 的 Linear 在 INT8/INT4 量化时需额外处理；(c) 实验表明对大规模 Transformer 性能影响可忽略。唯一例外：HuggingFace 实现中 `kv_a_proj_with_mqa` 和 `dense` 使用 `config.use_qkv_bias`（可配置 bias），而 vLLM 实现始终 `bias=False`。
>
> 3. **`q_lora_rank=None`**：Bailing 3.0 配置中 Q 不使用 LoRA 压缩，直接从 hidden_size(2560) 投影到 num_heads×qk_head_dim(6144)。当 `q_lora_rank` 不为 None 时（如 DeepSeek-V2），Q 侧使用两步压缩：`q_a_proj`（2560→q_lora_rank）→ `q_a_layernorm` → `q_b_proj`（q_lora_rank→6144），且 `q_a_proj` 与 `kv_a_proj_with_mqa` 会融合为 `fused_qkv_a_proj` 以提升计算效率。Bailing 3.0 选择 `q_lora_rank=None` 可能是为了减少 Q 侧的信息损失——直接投影保留了 Q 的完整表达力。
>
> 4. **`gated_attention_proj_granularity_type="head_wise"`**：g_proj 的输出粒度。"head_wise" 表示每个头输出 1 个标量门控值，g_proj 输出 shape 为 `[batch, seq, num_heads=32]`；"element_wise" 则表示每个头的每个维度输出 1 个门控值，输出 shape 为 `[batch, seq, num_heads×v_head_dim=4096]`。Bailing 3.0 选择 head_wise，参数量仅 2560×32=82K vs element_wise 的 2560×4096=10.5M，是更参数高效的选择。若设为 None 则不使用门控。
>
> 5. **Flash Attention 的 V padding**：当 `qk_head_dim(192) ≠ v_head_dim(128)` 时，Flash Attention 2 要求 Q、K、V 维度一致，因此需要对 V 做 zero-padding：`value_states = F.pad(value_states, [0, 192-128=64])`，注意力计算后再截回 `v_head_dim=128`。

#### 2.2.2 MLA 前向计算流程图

```
输入 hidden_states: [B, S, 2560]  (hidden_size=2560)
       │
       ├─────────────────────────────────────────────────────────────────┐
       │                                                                 │
       ▼                                                                 ▼
┌───────────────────────────┐                     ┌──────────────────────────────────────────┐
│  q_proj (ColumnParallelLinear, no bias)    │                     │  kv_a_proj_with_mqa (ReplicatedLinear, no bias) │
│  ※ q_lora_rank=None, 直接全维投影            │                     │  ※ "with_mqa": KV压缩后所有Query头共享          │
│                           │                     │  ※ ReplicatedLinear: 不TP切分, 保持KV共享语义   │
│  W_q: [6144, 2560]        │                     │  W_kv_a: [576, 2560]                      │
│    = [num_heads×qk_head_dim,│                   │    = [kv_lora_rank + qk_rope_head_dim,    │
│       hidden_size]         │                     │       hidden_size]                        │
│    = [32×192, 2560]        │                     │    = [512+64, 2560]                       │
│                           │                     │                                          │
│  计算: q_states = hidden_states @ W_q.T         │  计算: compressed_kv = hidden_states @     │
│        = [B,S,2560] × [2560,6144]              │          W_kv_a.T                          │
│        → [B, S, 6144]     │                     │        = [B,S,2560] × [2560,576]          │
│                           │                     │        → [B, S, 576]                      │
└───────────┬───────────────┘                     └──────────────┬───────────────────────────┘
            │                                                    │
            ▼                                                    ├──────────────────────┐
       reshape & transpose                                       ▼                      ▼
       [B,S,6144] → [B,32,S,192]                          k_pass                k_rot
       拆分 num_heads=32 和 qk_head_dim=192                [B,S,512]             [B,S,64]
            │                                              kv_lora_rank部分       qk_rope_head_dim部分
            │                                                    │                      │
            ├──────────────────┐                                 ▼                      │
            │                  │                        ┌─────────────────┐              │
            ▼                  ▼                        │ kv_a_layernorm  │              │
       q_pass             q_rot                        │ (RMSNorm)       │              │
       [B,32,S,128]      [B,32,S,64]                   │                 │              │
       qk_nope_head_dim  qk_rope_head_dim              │ W_norm: [512]   │              │
            │                  │                        │ 计算:           │              │
            │                  │                        │  x = x /        │              │
            │                  │                        │   sqrt(mean(x²)+│              │
            │                  │                        │   eps) × W_norm │              │
            │                  │                        └───────┬─────────┘              │
            │                  │                                │                        │
            │                  │                                ▼                        │
            │                  │                     ┌──────────────────────────────────┐ │
            │                  │                     │  kv_b_proj (Linear, no bias)      │ │
            │                  │                     │                                  │ │
            │                  │                     │  W_kv_b: [8192, 512]             │ │
            │                  │                     │    = [num_heads×(qk_nope_head_dim │ │
            │                  │                     │       + v_head_dim), kv_lora_rank]│ │
            │                  │                     │    = [32×(128+128), 512]         │ │
            │                  │                     │                                  │ │
            │                  │                     │  计算: kv_b = k_pass_norm @       │ │
            │                  │                     │          W_kv_b.T                 │ │
            │                  │                     │        = [B,S,512] × [512,8192]  │ │
            │                  │                     │        → [B, S, 8192]            │ │
            │                  │                     └──────────────┬───────────────────┘ │
            │                  │                                    │                     │
            │                  │                         reshape & transpose               │
            │                  │                         [B,S,8192] → [B,32,S,256]         │
            │                  │                                    │                     │
            │                  │                         ┌──────────┴──────────┐          │
            │                  │                         ▼                     ▼          │
            │                  │                   k_pass_new           value_states     │
            │                  │                   [B,32,S,128]         [B,32,S,128]    │
            │                  │                   qk_nope_head_dim      v_head_dim      │
            │                  │                                                          │
            │                  ▼                                                          │
            │          ┌──────────────────┐                                               │
            │          │   交错式 RoPE     │                                               │
            │          │                  │                                               │
            │          │ 仅作用于 q_rot   │                                               │
            │          │ 和 k_rot 部分    │                                               │
            │          │ (各 64 维)       │                                               │
            │          │                  │                                               │
            │          │ inv_freq:        │                                               │
            │          │ [qk_rope_head_dim│                                               │
            │          │  /2] = [32]      │                                               │
            │          │                  │                                               │
            │          │ q_rot:           │                                               │
            │          │ [B,32,S,64] →    │                                               │
            │          │ [B,32,S,64]      │                                               │
            │          │ (旋转后不变维度)  │                                               │
            │          └────────┬─────────┘                                               │
            │                   │                                                         │
            │            k_rot (MQA共享)                                                   │
            │            [B,1,S,64]                                                       │
            │                   │                                                         │
            │            expand 到 num_heads                                              │
            │            → [B,32,S,64]                                                    │
            │                   │                                                         │
            ▼                   ▼                                                         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  拼接 Q 和 K 的非旋转部分与旋转部分                                                        │
  │                                                                                          │
  │  query_states = cat(q_pass, q_rot, dim=-1) → [B, 32, S, 192]                            │
  │                   q_pass [B,32,S,128] + q_rot [B,32,S,64] = 128+64=192                  │
  │                                                                                          │
  │  key_states   = cat(k_pass_new, k_rot, dim=-1) → [B, 32, S, 192]                        │
  │                   k_pass_new [B,32,S,128] + k_rot [B,32,S,64] = 128+64=192              │
  │                                                                                          │
  │  value_states = [B, 32, S, 128]   (v_head_dim=128)                                      │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  标准 Multi-Head Attention 计算                                                           │
  │                                                                                          │
  │  缩放因子: scaling = qk_head_dim^(-0.5) = 192^(-0.5) ≈ 0.0722                           │
  │                                                                                          │
  │  attn_weights = Q × K^T × scaling                                                       │
  │               = [B,32,S,192] × [B,32,192,S] × 0.0722                                   │
  │               → [B, 32, S, S]                                                            │
  │                                                                                          │
  │  (如 flash_attention_2 且 qk_head_dim≠v_head_dim，需对 V 做 padding:                    │
  │    value_states = pad(value_states, [0, 192-128=64])  → [B,32,S,192])                   │
  │                                                                                          │
  │  attn_output = softmax(attn_weights) × V                                                │
  │              → [B, 32, S, 128]  (pad后截回 v_head_dim)                                   │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  Gated Attention 门控 (head_wise 粒度)                                                    │
  │                                                                                          │
  │  g_proj (Linear, no bias):                                                               │
  │    W_g: [32, 2560]  = [num_heads, hidden_size]                                          │
  │    计算: gate_logits = hidden_states @ W_g.T                                             │
  │         = [B,S,2560] × [2560,32] → [B, S, 32]                                          │
  │    gate = sigmoid(gate_logits.float()).to(hidden_states.dtype)  → [B, S, 32]            │
  │                                                                                          │
  │  门控乘法 (head_wise):                                                                    │
  │    attn_output = attn_output * gate.unsqueeze(-1)                                        │
  │               = [B,32,S,128] × [B,S,32,1]  (广播乘法，per-head 标量门控)                 │
  │               → [B, 32, S, 128]                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  Output Projection (o_proj / dense)                                                      │
  │                                                                                          │
  │  reshape: [B, 32, S, 128] → [B, S, 32×128] = [B, S, 4096]                              │
  │                                                                                          │
  │  dense (RowParallelLinear, no bias):                                                     │
  │    W_o: [2560, 4096]  = [hidden_size, num_heads × v_head_dim]                            │
  │    计算: output = concat_attn @ W_o.T                                                   │
  │         = [B,S,4096] × [4096,2560] → [B, S, 2560]                                      │
  └──────────────────────────────────────────────────────────────────────────────────────────┘

最终输出: [B, S, 2560]
```

### 2.3 MLA 的 KV Cache 优化

MLA 的核心优势在于 KV Cache 压缩：

| 方案 | Per-token KV Cache 大小 |
|------|------------------------|
| 标准 MHA (32头, head_dim=128) | 2 × 32 × 128 = 8192 floats |
| MLA (kv_lora_rank=512 + rope_dim=64) | 512 + 64 = 576 floats |

**压缩比**: 8192 / 576 ≈ **14.2x**，MLA 将 KV Cache 压缩到原来的 ~7%。

KV Cache 存储的是 `kv_c`（512 维）和 `k_pe`（64 维）的低秩向量，而非完整的 K、V。在 decode 阶段，只需要从 KV Cache 取出 `kv_c`，通过 `kv_b_proj` 展开恢复 K 和 V，再与 Q 做注意力计算。

---

## 三、KDA（Kimi Delta Attention）详解

KDA 是一种基于 Delta Rule（增量规则）的线性注意力机制，源自 Kimi 的研究。它用递归更新替代 Softmax 注意力，实现线性复杂度的长序列建模。Bailing 3.0 中 83.3% 的层使用 KDA。

### 3.1 KDA 关键参数

| 参数 | 值 | 说明 |
|------|------|------|
| `head_dim` | 128 | 每个注意力头的维度 |
| `num_heads` | 32 | 注意力头数 |
| `short_conv_kernel_size` | 4 | 短卷积核大小 |
| `kda_safe_gate` | true | 是否使用安全门控（下界保护） |
| `kda_lower_bound` | -5.0 | 安全门控的下界 |
| `no_kda_lora` | true | KDA 不使用 LoRA（直接 f_proj） |

> **设计说明**：
>
> 1. **`kda_safe_gate=True` 与 `kda_lower_bound=-5.0`**：Safe Gate 机制是 KDA 衰减门控的数值稳定性保障。标准模式下，衰减因子 `g1 = -exp(A) × softplus(x)` 可以趋近于 `-∞`，导致 `exp(g1) → 0`，即记忆被"完全清空"（状态坍缩）。Safe Gate 改用 `g1 = lower_bound × sigmoid(exp(A) × x)`，由于 sigmoid 值域为 (0,1)，g1 被限制在 `(lower_bound, 0) = (-5.0, 0)`，因此衰减因子 `exp(g1) ∈ (exp(-5), 1) ≈ (0.0067, 1)`，永远保留至少 ~0.67% 的历史记忆，防止状态坍缩。
>
> 2. **`no_kda_lora=True`**：KDA 中的 f_proj 和 g_proj 使用直接线性投影（`f_proj: Linear(2560, 4096)`），而非 LoRA 两步分解。当 `no_kda_lora=False` 时，f_proj 会拆分为 `f_a_proj: Linear(2560, 128)` + `f_b_proj: Linear(128, 4096)`（先降维再升维，类似 MLA 的 q_a/q_b 分解），g_proj 同理。Bailing 3.0 仅支持 `no_kda_lora=True`（代码中若设置 False 会抛出 `ValueError`），选择直接投影以保证 KDA 门控信号的表达力。
>
> 3. **`b_proj` 输出 per-head 标量**：与 KDA 其他投影（q/k/v/f/g_proj 输出 num_heads×head_dim=4096 维）不同，b_proj 仅输出 `num_heads=32` 维，即每个注意力头 1 个标量值 β。写入门控 β 控制新 key-value 对写入递归记忆矩阵的强度——per-head 粒度是"粗粒度"控制，足够调节记忆写入，同时大幅减少参数量（32 vs 4096，减少 99.2%）。相比之下，衰减门控 g1 是 per-head-per-dim 粒度（4096 维），因为不同维度需要不同的遗忘速率。
>
> 4. **`use_qk_l2norm_in_kernel=True`**：在 KDA 核心 kernel 内部对 q 和 k 做 L2 归一化（`q̂ = q / ||q||`），使得注意力得分 `o = q̂^T × h` 只受 q/k 方向影响，不受幅度影响。这提升了训练稳定性，防止梯度爆炸/消失，尤其在使用外积更新 `h_t = exp(g1) ⊙ h_{t-1} + β × (k ⊗ v)` 时，归一化后的 k 使记忆矩阵的数值范围更可控。

### 3.2 KDA 权重矩阵与参数对应表

| 权重名 | Shape | 计算公式 | 对应 config 参数 |
|--------|-------|----------|-----------------|
| `q_proj.weight` | `[4096, 2560]` | `num_heads × head_dim, hidden_size = 32×128, 2560` | `num_attention_heads=32, head_dim=128, hidden_size=2560` |
| `k_proj.weight` | `[4096, 2560]` | 同 q_proj | 同上 |
| `v_proj.weight` | `[4096, 2560]` | 同 q_proj | 同上 |
| `f_proj.weight` | `[4096, 2560]` | `num_heads × head_dim, hidden_size`（衰减门控输入） | 同上 |
| `g_proj.weight` | `[4096, 2560]` | `num_heads × head_dim, hidden_size`（输出门控） | 同上 |
| `b_proj.weight` | `[32, 2560]` | `num_heads, hidden_size`（写入门控 beta） | `num_attention_heads=32, hidden_size=2560` |
| `q_conv1d.weight` | `[4096, 1, 4]` | `num_heads×head_dim, 1, conv_kernel_size` | `num_attention_heads=32, head_dim=128, short_conv_kernel_size=4` |
| `k_conv1d.weight` | `[4096, 1, 4]` | 同 q_conv1d | 同上 |
| `v_conv1d.weight` | `[4096, 1, 4]` | 同 q_conv1d | 同上 |
| `A_log` | `[1, 1, 32, 1]` | `[1, 1, num_heads, 1]` 可学习衰减参数 | `num_attention_heads=32` |
| `dt_bias` | `[4096]` | `num_heads × head_dim` 可学习偏置 | `num_attention_heads=32, head_dim=128` |
| `o_proj.weight` | `[2560, 4096]` | `hidden_size, num_heads×head_dim` | `hidden_size=2560, num_attention_heads=32, head_dim=128` |
| `o_norm.weight` | `[128]` | `head_dim` (FusedRMSNormGated 权重) | `head_dim=128` |

### 3.3 KDA 前向计算流程

```
输入 hidden_states: [S, 2560]  (vLLM 推理, 1D token 序列)
                    或 [B, S, 2560]  (HF training 中)
       │
       ├───────┬───────┬───────┬───────┬───────┐
       ▼       ▼       ▼       ▼       ▼       │ (g_proj 稍后使用)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  6 路并行线性投影                                             │
│  q/k/v/f/g_proj: ColumnParallelLinear, no bias              │
│  b_proj: ColumnParallelLinear, no bias (输出仅num_heads=32)  │
  │                                                                             │
  │  q_proj:  W_q: [4096, 2560]  = [num_heads×head_dim, hidden_size]           │
  │    计算: q_proj_states = hidden_states @ W_q.T                              │
  │         = [S, 2560] × [2560, 4096] → [S, 4096]                            │
  │                                                                             │
  │  k_proj:  W_k: [4096, 2560]  (同 q_proj)                                   │
  │    计算: k_proj_states = hidden_states @ W_k.T                              │
  │         = [S, 2560] × [2560, 4096] → [S, 4096]                            │
  │                                                                             │
  │  v_proj:  W_v: [4096, 2560]  (同 q_proj)                                   │
  │    计算: v_proj_states = hidden_states @ W_v.T                              │
  │         = [S, 2560] × [2560, 4096] → [S, 4096]                            │
  │                                                                             │
  │  f_proj:  W_f: [4096, 2560]  （衰减门控输入投影）                             │
  │    计算: f_states = hidden_states @ W_f.T                                   │
  │         = [S, 2560] × [2560, 4096] → [S, 4096]                            │
  │                                                                             │
  │  b_proj:  W_b: [32, 2560]  = [num_heads, hidden_size]                      │
  │    计算: beta_logits = hidden_states @ W_b.T                                │
  │         = [S, 2560] × [2560, 32] → [S, 32]                                │
  │    beta = sigmoid(beta_logits.float())  → [S, 32]                          │
  │    beta = beta.unsqueeze(0)  → [1, S, 32, 1]                               │
  │    (每个头一个标量，控制新信息写入记忆的强度)                                   │
  └─────────────────────────────────────────────────────────────────────────────┘
       │         │         │         │
       ▼         ▼         ▼         ▼ (仅 q, k, v 三路进入 ShortConv)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  ShortConv (因果1D卷积 + SiLU 激活)                                         │
  │                                                                             │
  │  对 q_proj_states, k_proj_states, v_proj_states 分别做因果1D卷积:             │
  │                                                                             │
  │  q_conv1d: W_qconv: [4096, 1, 4]                                           │
  │    = [num_heads×head_dim, 1, conv_kernel_size]                              │
  │    = [32×128, 1, 4]                                                         │
  │    输入: q_proj_states → reshape → [1, 4096, S]  (batch=1, channels, len)  │
  │    卷积: causal_conv1d_fn(x, W_qconv, bias, activation='silu')             │
  │      q[t] = SiLU( Σ_{i=0}^{3} q_proj_states[t-i] × W_qconv[:, :, i] )    │
  │    输出: transpose → [S, 4096]                                              │
  │                                                                             │
  │  k_conv1d: W_kconv: [4096, 1, 4]  (同 q_conv1d)                            │
  │    卷积 + SiLU → [S, 4096]                                                 │
  │                                                                             │
  │  v_conv1d: W_vconv: [4096, 1, 4]  (同 q_conv1d)                            │
  │    卷积 + SiLU → [S, 4096]                                                 │
  │                                                                             │
  │  Prefill: causal_conv1d_fn (并行卷积整个序列)                                │
  │  Decode:  causal_conv1d_update (增量卷积：读conv_state，卷1步，更新state)     │
  └─────────────────────────────────────────────────────────────────────────────┘
       │         │         │
       ▼         ▼         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  reshape 为多头形式                                                          │
  │                                                                             │
  │  q: [S, 4096] → [1, S, H, K] = [1, S, 32, 128]                            │
  │     rearrange("n (h d) -> 1 n h d", h=num_heads=32, d=head_dim=128)        │
  │     4096 = 32 × 128                                                        │
  │                                                                             │
  │  k: [S, 4096] → [1, S, 32, 128]  (同上)                                    │
  │  v: [S, 4096] → [1, S, H, V] = [1, S, 32, 128]                            │
  │     (K=V=head_dim=128 在 KDA 中)                                            │
  └─────────────────────────────────────────────────────────────────────────────┘
       │         │         │              │
       ▼         ▼         ▼              ▼ (f_states 参与计算)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  衰减门控 g1 计算 (fused_kda_gate)                                           │
  │                                                                             │
  │  输入: f_states = f_proj(hidden_states)  → [S, 4096]                        │
  │                                                                             │
  │  Step 1: reshape                                                            │
  │    f_states: [S, 4096] → [..., H=32, D=128]                                │
  │    = [S, num_heads, head_dim]                                               │
  │                                                                             │
  │  Step 2: 加偏置                                                             │
  │    f = f_states + dt_bias   (dt_bias: [4096] → reshape→ [32, 128])          │
  │    dt_bias 按 head 维度索引: bias[h, d] 对应 f[h, d]                         │
  │                                                                             │
  │  Step 3: 计算衰减因子                                                         │
  │    A_log: [1, 1, 32, 1]  可学习参数 (per-head 标量)                          │
  │    A_h = exp(A_log[0,0,h,0])  → per-head 衰减率                            │
  │                                                                             │
  │    ● Safe Gate 模式 (kda_safe_gate=True, kda_lower_bound=-5.0):             │
  │      g1[h,d] = lower_bound × sigmoid(exp(A_log[h]) × (f[h,d] + dt_bias[h,d]))│
  │             = -5.0 × sigmoid(A_h × f[h,d])                                 │
  │      g1 范围: (-5.0, 0)                                                     │
  │      对应衰减因子 exp(g1) 范围: (exp(-5), 1) ≈ (0.0067, 1)                  │
  │                                                                             │
  │    ● 标准 模式 (kda_safe_gate=False):                                       │
  │      g1[h,d] = -exp(A_log[h]) × softplus(f[h,d] + dt_bias[h,d])           │
  │      g1 范围: (-∞, 0)                                                       │
  │                                                                             │
  │  输出: g1 → [1, S, 32, 128]  (per-head, per-dim 衰减因子)                   │
  │                                                                             │
  │  beta (写入门控):                                                            │
  │    W_b: [32, 2560]  = [num_heads, hidden_size]                              │
  │    beta = sigmoid(hidden_states @ W_b.T)  → [S, 32]                        │
  │    beta = beta.unsqueeze(0) → [1, S, 32, 1]                                │
  │    (每个头一个标量，控制新 kv 写入记忆的强度)                                   │
  └─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  KDA 核心计算 (chunk_kda / fused_recurrent_kda)                               │
  │                                                                             │
  │  Delta Rule 递归更新公式:                                                    │
  │                                                                             │
  │  对序列位置 t = 1,...,S:                                                     │
  │                                                                             │
  │    h_t = exp(g1_t) ⊙ h_{t-1} + beta_t × (k_t ⊗ v_t)                      │
  │                                                                             │
  │  其中:                                                                      │
  │    h_t: [H, V, K] = [32, 128, 128] 递归隐状态 (记忆矩阵)                    │
  │    g1_t: [H, D] = [32, 128] 衰减门控                                        │
  │    exp(g1_t): [32, 128] 衰减因子 (0.0067~1)，⊙ 为逐元素乘                   │
  │    beta_t: [H, 1] = [32, 1] 写入门控                                        │
  │    k_t: [H, K] = [32, 128] 当前 key (L2归一化后)                            │
  │    v_t: [H, V] = [32, 128] 当前 value                                       │
  │    ⊗: 外积 (k_t ⊗ v_t → [H, V, K] = [32, 128, 128])                       │
  │                                                                             │
  │  查询输出:                                                                   │
  │    o_t = q_t @ h_t  (matmul: [H,K] × [H,V,K] → [H,V])                     │
  │    即 o_t[h,v] = Σ_k q_t[h,k] × h_t[h,v,k]                                │
  │    (q_t 在 kernel 内先做 L2 归一化: q_hat = q / ||q||)                      │
  │    → o: [1, S, H, V] = [1, S, 32, 128]                                     │
  │                                                                             │
  │  Prefill: chunk_kda (并行分块计算，chunk_size=64)                            │
  │  Decode:  fused_recurrent_kda (递归逐步计算，每步1个token)                   │
  └─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Output Gating (o_norm + g_proj)                                             │
  │                                                                             │
  │  g_proj:  W_g: [4096, 2560]  = [num_heads×head_dim, hidden_size]           │
  │    计算: g2 = hidden_states @ W_g.T                                         │
  │         = [S, 2560] × [2560, 4096] → [S, 4096]                            │
  │    reshape: [S, 4096] → [..., H=32, D=128]  = [1, S, 32, 128]             │
  │                                                                             │
  │  o_norm: FusedRMSNormGated(head_dim=128, activation='sigmoid')               │
  │    W_norm: [128]  = [head_dim]  可学习缩放权重                               │
  │    输入: core_attn_out [1,S,32,128], g2 [1,S,32,128]                        │
  │                                                                             │
  │    数学:                                                                      │
  │      x_float = core_attn_out.float()              → [1,S,32,128]           │
  │      variance = mean(x_float², dim=-1, keepdim=True)  → [1,S,32,1]        │
  │      x_normed = x_float / sqrt(variance + eps)    → [1,S,32,128]           │
  │      x_normed = x_normed × W_norm                → [1,S,32,128]            │
  │      out = x_normed × sigmoid(g2.float())         → [1,S,32,128]           │
  │      return out.to(core_attn_out.dtype)                                    │
  │                                                                             │
  │    总公式: output = RMSNorm(core_attn_out) × σ(g2)                          │
  │             = (core_attn_out / √(mean(core_attn_out²)+ε) × w) × σ(g2)     │
  └─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Output Projection                                                           │
  │                                                                             │
  │  o_proj (RowParallelLinear, no bias):                                        │
  │    W_o: [2560, 4096]  = [hidden_size, num_heads × head_dim]                 │
  │                                                                             │
  │  reshape: [1, S, 32, 128] → [S, 32×128] = [S, 4096]                       │
  │  计算: output = o_flat @ W_o.T                                              │
  │       = [S, 4096] × [4096, 2560] → [S, 2560]                              │
  └─────────────────────────────────────────────────────────────────────────────┘

最终输出: [S, 2560]
```

### 3.3 KDA 的状态缓存

KDA 层需要维护两类状态（不同于标准 KV Cache）：

| 状态名 | Shape | 说明 |
|--------|-------|------|
| `conv_state_q/k/v` | `[batch, head_dim×num_heads, conv_kernel_size]` | 3 个因果1D卷积状态，用于 decode 阶段增量更新卷积 |
| `recurrent_state` | `[batch, num_heads, head_dim, head_dim]` | 递归隐状态 h_t，即 Delta Rule 的累积记忆矩阵 |

在 vLLM 中，KDA 状态由 `MambaStateShapeCalculator.kda_state_shape` 计算，存储在 Mamba Cache 中。

### 3.4 Causal Conv1d（短卷积）

KDA 中的短卷积（kernel_size=4）对 q, k, v 做因果1D卷积：

```
输入: x_proj [num_tokens, 4096]  (已经过线性投影)
        │
        ▼  reshape → [1, 4096, num_tokens]
  causal_conv1d_fn(
      x,
      conv_weight,    # [4096, 1, 4]  (out_channels, 1, kernel_size)
      conv_bias,
      activation='silu',
      conv_states=conv_state,  # decode时增量更新
      has_initial_state=has_initial_state,
      cache_indices=state_indices,
      query_start_loc=query_start_loc,
  )
        │
        ▼  transpose → [num_tokens, 4096]
```

- **Prefill 阶段**: `causal_conv1d_fn` 对完整序列做因果卷积，同时保存最终卷积状态
- **Decode 阶段**: `causal_conv1d_update` 对单个 token 做增量卷积更新

卷积操作增加了局部信息交互，使 KDA 具备比纯线性注意力更好的短程建模能力。

### 3.5 chunk_kda vs fused_recurrent_kda

| 特性 | chunk_kda (Prefill) | fused_recurrent_kda (Decode) |
|------|---------------------|------------------------------|
| 输入长度 | 长序列（Prefill） | 短序列（Decode，单 token） |
| 计算方式 | 并行分块（Triton kernel） | 逐步递归（Triton kernel） |
| 时间复杂度 | O(S × chunk_size) | O(1) per token |
| 状态管理 | 输出 final_state 供后续使用 | 原地更新 recurrent_state |

---

## 四、RoPE（Rotary Position Embedding）详解

### 4.1 Bailing 3.0 的 RoPE 配置

| 参数 | 值 | 说明 |
|------|------|------|
| `rope_theta` | 10000 | 基频 θ |
| `qk_rope_head_dim` | 64 | 参与 RoPE 的维度（Q/K 的后 64 维） |
| `rope_interleave` | true | 使用交错式（interleaved）RoPE |
| `partial_rotary_factor` | 0.5 (config中) | 部分旋转因子（vLLM中不使用，被忽略） |
| `max_position_embeddings` | 8192 | 最大位置编码长度 |

> **设计说明**：
>
> 1. **`rope_interleave=true`：交错式 vs 半区分式 RoPE**：RoPE 对 Q/K 的维度两两配对施加旋转，配对方式有两种：
>    - **交错式（interleaved）**：相邻维度配对，即 (d0, d1), (d2, d3), (d4, d5), ...，旋转后维度顺序变为 [d0, d2, d4, ..., d1, d3, d5, ...]。实现时先用 `view(B,H,S,32,2).transpose(3,4)` 重排，再做旋转。
>    - **半区分式（half-split / GPT-NeoX 风格）**：前半与后半配对，即 (d0, d32), (d1, d33), (d2, d34), ...，旋转直接在前 32 维和后 32 维之间进行，无需重排。
>    - Bailing 3.0 使用交错式，这与原始 RoFormer 论文的配对方式一致。交错式在硬件上可能对某些 kernel 实现更友好，但功能上两者等价（只是 inv_freq 的排列不同）。
>
> 2. **`partial_rotary_factor=0.5`**：HuggingFace config 中保留了此参数，表示"仅对 50% 的 head_dim 维度施加 RoPE"。但在 MLA 架构中，RoPE 实际作用维度由 `qk_rope_head_dim=64` 决定，占总 QK 维度 `qk_head_dim=192` 的 64/192 ≈ 33.3%，而非 50%。vLLM 实现中完全忽略此参数，直接使用 `qk_rope_head_dim` 构建 RoPE。

### 4.2 RoPE 数学原理

RoPE 通过旋转矩阵对 Q 和 K 的部分维度施加位置编码：

**频率计算**（逆频率）：

```
inv_freq[i] = 1 / (theta ^ (2i / dim))    其中 dim = qk_rope_head_dim = 64
                                              i = 0, 1, ..., dim/2 - 1 = 31
```

即 `inv_freq: [32]`，对应 32 对旋转角度。

**旋转角度计算**：

```
freqs = inv_freq[None, :, None] @ position_ids[:, None, :].float()
      = [batch, 32, 1] × [batch, 1, seq_len]
      = [batch, 32, seq_len]

freqs = freqs.transpose(1, 2)  → [batch, seq_len, 32]
emb = cat(freqs, freqs, dim=-1)  → [batch, seq_len, 64]   # 复制拼接
cos = emb.cos() * attention_scaling
sin = emb.sin() * attention_scaling
```

**交错式 RoPE 应用** (`apply_rotary_pos_emb_interleave`):

Bailing 3.0 使用交错式 RoPE（`rope_interleave=true`），区别于半区分式（half-split）RoPE。

```
输入 q: [B, H, S, 64]  (仅 q_rot 部分)

Step 1: 将相邻两维重排为 (rot, rot) 对
  q = q.view(B, H, S, 32, 2).transpose(3, 4).reshape(B, H, S, 64)
  # 将 [d0,d1,d2,d3,...] 重排为 [d0,d2,d4,...,d1,d3,d5,...]

Step 2: 旋转
  rotate_half(q) = cat(-q[..., 32:], q[..., :32], dim=-1)
  q_embed = q * cos + rotate_half(q) * sin

Step 3: 同样对 k_rot 做旋转
```

**数学意义**：对于位置 pos 的第 i 对特征 (q_{2i}, q_{2i+1})，旋转后为：

```
q'_{2i} = q_{2i} × cos(pos × θ_i) - q_{2i+1} × sin(pos × θ_i)
q'_{2i+1} = q_{2i} × sin(pos × θ_i) + q_{2i+1} × cos(pos × θ_i)
```

这确保了内积 `q'·k'` 只依赖于相对位置 (pos_q - pos_k)，实现位置信息的隐式编码。

### 4.3 RoPE 仅作用于部分维度

在 MLA 中，RoPE 只作用于 Q 和 K 的后 `qk_rope_head_dim=64` 维：

```
Q: [q_pass(128维) | q_rot(64维)]  ← 仅 q_rot 参与旋转
K: [k_pass(128维) | k_rot(64维)]  ← 仅 k_rot 参与旋转
```

**原因**：MLA 的 `k_pass` 来自低秩压缩空间 `kv_lora_rank`，如果对 `k_pass` 施加 RoPE，会导致 KV Cache 中的压缩向量与位置耦合，无法在 decode 时从 KV Cache 恢复出位置编码后的 K。因此 RoPE 只作用于不经过压缩的 `k_rot` 部分。

---

## 五、归一化操作详解

Bailing 3.0 使用三种归一化操作：

### 5.1 RMSNorm（最常用）

用于所有 DecoderLayer 的 `input_layernorm` 和 `post_attention_layernorm`，以及 MLA 中的 `kv_a_layernorm`、`q_a_layernorm`。

```python
# 数学公式
variance = mean(x², dim=-1, keepdim=True)     # [B, S, 1]
x_normed = x / sqrt(variance + eps)            # [B, S, D]
output = weight * x_normed                      # [B, S, D], weight: [D]
```

**实现细节**（`BailingMoeV3RMSNorm`）：
```python
hidden_states = hidden_states.to(torch.float32)          # 转为 float32 计算方差
variance = hidden_states.pow(2).mean(-1, keepdim=True)   # [B, S, 1]
hidden_states = hidden_states * torch.rsqrt(variance + eps)
return self.weight * hidden_states.to(input_dtype)        # 转回原精度
```

### 5.2 GroupRMSNorm（配置支持但未使用）

`BailingMoeV3GroupRMSNorm` 是分组 RMSNorm：

```python
# group_norm_size = 1 时退化为标准 RMSNorm
# 将最后一维拆分为 (group_norm_size, D // group_norm_size)
hidden_states = hidden_states.view(*input_shape[:-1], group_norm_size, D // group_norm_size)
variance = mean(x², dim=-1, keepdim=True)  # 在 D//group_norm_size 维度上计算方差
x_normed = x / sqrt(variance + eps)
output = weight * x_normed.view(input_shape)
```

### 5.3 FusedRMSNormGated（KDA 输出门控归一化）

用于 KDA 层的输出归一化，结合了 RMSNorm 和门控机制：

```python
# 数学公式
x_rms = x / sqrt(mean(x², dim=-1) + eps)    # RMS 归一化
x_normed = x_rms * weight                    # 可学习缩放

# 门控（activation='sigmoid' 时）
output = x_normed × sigmoid(g)                # g 是门控信号
```

**Triton 融合 kernel 实现**（`layer_norm_gated_fwd_kernel`）：
```
1. 加载 x 和 residual（如有）
2. 计算 RMSNorm:
   variance = sum(x²) / D
   rstd = 1 / sqrt(variance + eps)
   x_hat = x * rstd
3. 应用可学习权重: y = x_hat * weight
4. 加载门控 g
5. 应用门控: y = y * sigmoid(g)   (sigmoid 模式)
6. 写出结果
```

该融合算子将 RMSNorm 和 Gating 合并在一个 kernel 中，减少 GPU 内存访问次数。

---

## 六、MoE（Mixture of Experts）详解

### 6.1 MoE 架构参数

| 参数 | 值 | 说明 |
|------|------|------|
| `num_experts` | 512 | 总专家数 |
| `num_experts_per_tok` | 8 | 每个 token 激活专家数 |
| `n_group` | 8 | 专家分组数 |
| `topk_group` | 4 | 选中的专家组数 |
| `moe_intermediate_size` | 768 | 每个路由专家的中间维度 |
| `num_shared_experts` | 1 | 共享专家数 |
| `moe_shared_expert_intermediate_size` | 768 | 共享专家中间维度 |
| `routed_scaling_factor` | 2.5 | 路由专家输出缩放因子 |
| `scoring_func` | sigmoid | 路由评分函数 |
| `topk_method` | noaux_tc | TopK 选择方法 |
| `norm_topk_prob` | true | 归一化 TopK 概率 |

> **设计说明**：
>
> 1. **`scoring_func=sigmoid` vs `softmax`**：MoE Router 传统上使用 softmax 将所有专家的得分归一化为概率分布（各项之和为1），但 Bailing 3.0 选择 sigmoid 对每个专家独立打分。sigmoid 的优势在于：(a) 每个专家的评分是独立的，不受其他专家得分的影响，避免了 softmax 中"一个专家得分高会压低其他专家"的竞争效应；(b) 配合 Group-Limited TopK，只有部分组内的专家参与选择，sigmoid 的独立性更合理——未选中组的专家不应影响选中组专家的相对权重；(c) sigmoid 输出恒正（0,1），天然适合作为权重。
>
> 2. **`topk_method=noaux_tc`**：意为"No Auxiliary loss, Throughput-Centric"——不使用辅助负载均衡损失（auxiliary loss），优先保证吞吐量。传统 MoE（如 Switch Transformer）使用 auxiliary loss 来平衡各专家的负载，但这会损害模型质量。Bailing 3.0 改用可训练的 `expert_bias` 来影响专家选择（不通过梯度惩罚），既实现了负载均衡，又不影响主损失函数。
>
> 3. **`expert_bias` 的设计角色**：`expert_bias` 是一个 `[512]` 的可训练参数，在路由决策时被加到 sigmoid 得分上（`scores_for_routing = scores + expert_bias`），但在计算最终的 topk_weight 时使用的是未加 bias 的原始 sigmoid 得分（`scores = torch.gather(scores, dim=1, index=topk_idx)`）。这种"只影响选择、不影响权重"的设计使得 expert_bias 可以引导负载均衡，而不会改变被选中专家的相对权重比例。
>
> 4. **`routed_scaling_factor=2.5`**：路由专家的输出在加权求和后乘以 2.5，再与共享专家输出相加。最终输出 = `2.5 × routed_output + shared_output`。这个缩放因子让路由专家的贡献大于共享专家，原因是：每个 token 只激活 8/512 = 1.56% 的路由专家，路由专家的输出经过稀疏选择归一化后数值偏小；2.5x 缩放补偿了这种"稀疏衰减"，使路由专家和共享专家对最终输出的贡献更加平衡。
>
> 5. **`norm_topk_prob=true`（renormalize）**：选中 8 个专家后，将它们的 sigmoid 得分归一化为和为1（`topk_weight = scores / (sum(scores) + 1e-20)`），再乘以 routed_scaling_factor。这确保了每个 token 的路由专家权重总和恒定（2.5），不受选中专家具体得分的影响，使训练更稳定。
>
> 6. **`expert_swiglu_limit` 与 `share_expert_swiglu_limit`**：部分层配置了 `expert_swiglu_limit`（通过 `expert_swiglu_limit_list` 逐层指定），对这些专家使用 `swiglustep` 激活：在 SwiGLU 激活（SiLU(gate) × up）之后做 clamp 限幅（`clamp(..., -limit, limit)`），防止专家输出值过大导致数值不稳定。限幅版激活函数名为 `swiglustep`，在 vLLM 中通过 `activation="swiglustep"` 和 `activation_limit` 参数传入 SharedFusedMoE。

### 6.2 MoE 权重矩阵与参数对应表

| 权重名 | Shape | 计算公式 | 对应 config 参数 |
|--------|-------|----------|-----------------|
| `gate.weight` (Router) | `[512, 2560]` | `num_experts, hidden_size` | `num_experts=512, hidden_size=2560` |
| `gate.expert_bias` | `[512]` | `num_experts` 可训练偏置 | `num_experts=512` |
| 专家 `gate_proj.weight` (×512) | `[768, 2560]` ×512 | `moe_intermediate_size, hidden_size` | `moe_intermediate_size=768, hidden_size=2560` |
| 专家 `up_proj.weight` (×512) | `[768, 2560]` ×512 | `moe_intermediate_size, hidden_size` | 同上 |
| 专家 `down_proj.weight` (×512) | `[2560, 768]` ×512 | `hidden_size, moe_intermediate_size` | 同上 |
| 共享专家 `gate_proj.weight` | `[768, 2560]` | `moe_shared_expert_intermediate_size × num_shared_experts, hidden_size` | `moe_shared_expert_intermediate_size=768, num_shared_experts=1, hidden_size=2560` |
| 共享专家 `up_proj.weight` | `[768, 2560]` | 同上 | 同上 |
| 共享专家 `down_proj.weight` | `[2560, 768]` | `hidden_size, shared_intermediate` | 同上 |

> 注意：部分层配置了 `expert_swiglu_limit` 和 `share_expert_swiglu_limit`，这些层使用 `swiglustep` 激活替代 `silu`，对 SwiGLU 输出做 clamp 限幅。

### 6.3 Router（门控网络）

```
输入 hidden_states: [num_tokens, 2560]
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Gate 计算                                                            │
│                                                                      │
│  Step 1: 线性变换                                                     │
│    W_gate: [512, 2560]  = [num_experts, hidden_size]  (fp32)       │
│    logits = hidden_states @ W_gate.T                                │
│           = [num_tokens, 2560] × [2560, 512] → [num_tokens, 512]  │
│                                                                      │
│  Step 2: Sigmoid 评分                                                 │
│    scores = sigmoid(logits.float())        → [num_tokens, 512]      │
│                                                                      │
│  Step 3: 专家偏置 (expert_bias)                                       │
│    scores_for_routing = scores + expert_bias   (可训练偏置, [512])   │
│                                                                      │
│  Step 4: Group-Limited TopK 选择                                     │
│    ├── 512个专家分为 8 组 (n_group=8), 每组 64 个专家                 │
│    ├── 对每组取 top-2 评分求和: group_scores [num_tokens, 8]         │
│    ├── 选择 top-4 组 (topk_group=4): group_idx [num_tokens, 4]      │
│    ├── 构建组掩码，屏蔽未选中组的专家                                  │
│    ├── 在选中组的专家中取 top-8 (num_experts_per_tok=8)              │
│    └── topk_idx: [num_tokens, 8], topk_probs: [num_tokens, 8]       │
│                                                                      │
│  Step 5: 归一化权重                                                   │
│    topk_weight = topk_probs / (sum(topk_probs) + 1e-20)             │
│    topk_weight = topk_weight × routed_scaling_factor (2.5)          │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.4 Expert 网络

每个专家是一个标准 MLP（SwiGLU 激活）：

```
输入 x: [num_tokens_for_expert, 2560]
       │
       ├───────────────────┐
       ▼                   ▼
┌──────────────┐  ┌──────────────┐
│  gate_proj   │  │  up_proj     │
│  W_gate:     │  │  W_up:       │
│  [768, 2560] │  │  [768, 2560] │
│  计算:       │  │  计算:       │
│  = x @ W_g.T │  │  = x @ W_u.T│
│  [n,2560]    │  │  [n,2560]   │
│  ×[2560,768] │  │  ×[2560,768]│
│  →[n, 768]   │  │  →[n, 768]  │
└──────┬───────┘  └──────┬───────┘
       │                   │
       ▼                   │
  SiLU(gate)               │
  = gate × sigmoid(gate)   │
       │                   │
       ▼                   ▼
  gate_activated * up  (element-wise)
  [n, 768] × [n, 768] → [n, 768]
       │
       ▼
┌──────────────┐
│  down_proj   │
│  W_down:     │
│  [2560, 768] │
│  计算:       │
│  = x @ W_d.T │
│  [n,768]     │
│  ×[768,2560] │
│  →[n, 2560]  │
└──────┬───────┘
       │
       ▼
  expert_output: [num_tokens_for_expert, 2560]
```

**SwiGLU 激活**：
```
SwiGLU(x) = (SiLU(gate_proj(x)) ⊙ up_proj(x))
output = down_proj(SwiGLU(x))
```

**SwiGLU 限幅变体** (`swiglustep`):
部分层配置了 `expert_swiglu_limit`，在 SwiGLU 激活后对输出做 step 限幅：

```
限幅版: output = down_proj(clamp(SwiLU(gate) * up, -limit, limit))
```

### 6.4 共享专家 (Shared Expert)

共享专家是一个与路由专家结构相同的 MLP，但所有 token 都会经过它（无需路由）。

| 权重名 | Shape | 计算公式 | 对应 config 参数 |
|--------|-------|----------|-----------------|
| `shared_experts.gate_proj.weight` | `[768, 2560]` | `shared_intermediate, hidden_size` | `moe_shared_expert_intermediate_size=768, num_shared_experts=1, hidden_size=2560` |
| `shared_experts.up_proj.weight` | `[768, 2560]` | 同上 | 同上 |
| `shared_experts.down_proj.weight` | `[2560, 768]` | `hidden_size, shared_intermediate` | 同上 |

其中 `shared_intermediate = moe_shared_expert_intermediate_size × num_shared_experts = 768 × 1 = 768`。

### 6.5 MoE 整体输出

```
routed_output = Σ(topk_weight_i × expert_i(x))    # 路由专家加权求和
shared_output = shared_expert(x)                    # 共享专家
final_output = routed_scaling_factor × routed_output + shared_output
             = 2.5 × routed_output + shared_output
```

### 6.6 vLLM 中的融合 MoE

在 vLLM 推理实现中，使用 `SharedFusedMoE` 将路由专家和共享专家融合计算：

```python
# vLLM: SharedFusedMoE.forward()
shared_output, hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits
)
# experts 内部已经包含了共享专家的计算
# shared_output: [num_tokens, 2560]  (共享专家输出)
# hidden_states: [num_tokens, 2560]  (路由专家加权输出)

hidden_states = hidden_states * routed_scaling_factor + shared_output
```

---

## 七、Dense MLP 详解

前 2 层（Layer 0, 1）使用 Dense MLP 而非 MoE：

### 7.1 Dense MLP 权重矩阵

| 权重名 | Shape | 计算公式 | 对应 config 参数 |
|--------|-------|----------|-----------------|
| `gate_proj.weight` | `[6144, 2560]` | `intermediate_size, hidden_size` | `intermediate_size=6144, hidden_size=2560` |
| `up_proj.weight` | `[6144, 2560]` | `intermediate_size, hidden_size` | 同上 |
| `down_proj.weight` | `[2560, 6144]` | `hidden_size, intermediate_size` | 同上 |

### 7.2 Dense MLP 前向流程

```
输入 x: [batch, seq_len, 2560]    (hidden_size=2560)
       │
       ├───────────────────┐
       ▼                   ▼
┌──────────────┐  ┌──────────────┐
│  gate_proj   │  │  up_proj     │
│  W_gate:     │  │  W_up:       │
│  [6144,2560] │  │  [6144,2560] │
│  计算:       │  │  计算:       │
│  gate = x @  │  │  up = x @    │
│  W_g.T       │  │  W_u.T       │
│  [B,S,2560]  │  │  [B,S,2560]  │
│  ×[2560,6144]│  │  ×[2560,6144]│
│  →[B,S,6144] │  │  →[B,S,6144] │
└──────┬───────┘  └──────┬───────┘
       │                   │
       ▼                   │
  SiLU(gate)               │
  = gate × sigmoid(gate)   │
       │                   │
       ▼                   ▼
  gate_activated * up  (element-wise)
  [B,S,6144] × [B,S,6144] → [B,S,6144]
       │
       ▼
┌──────────────┐
│  down_proj   │
│  W_down:     │
│  [2560,6144] │
│  计算:       │
│  = x @ W_d.T │
│  [B,S,6144]  │
│  ×[6144,2560]│
│  →[B,S,2560] │
└──────┬───────┘
       │
       ▼
  output: [batch, seq_len, 2560]
```

---

## 八、DecoderLayer 前向流程

### 8.1 标准 Residual 连接

```
输入: hidden_states, residual (初始 residual=None)
       │
       ├─ if residual is None:
       │    residual = hidden_states
       │    hidden_states = input_layernorm(hidden_states)
       │
       └─ else (vLLM 融合版本):
            hidden_states, residual = input_layernorm(hidden_states, residual)
            # 融合: norm_output = norm(x) + res  (一次kernel完成)
       │
       ▼
  Attention (MLA 或 KDA)
       │
       │  attn_output
       ▼
  hidden_states, residual = post_attention_layernorm(attn_output, residual)
       │
       ▼
  MLP / MoE
       │
       ▼
  return hidden_states, residual
```

**vLLM 中的融合 RMSNorm**：在推理时，`input_layernorm` 和 `post_attention_layernorm` 可以融合 residual 加法，减少一次 kernel launch：

```python
# 融合模式 (residual is not None):
# norm_output, new_residual = RMSNorm(x, residual)
#   = (RMSNorm(x) + residual, x)  或
#   = (x * rstd * weight + residual, x)
```

### 8.2 层类型判断

```python
def _is_kda_layer(layer_idx, layer_group_size, num_hidden_layers):
    """判断是否为 KDA 层"""
    return not (
        (layer_idx + 1) % layer_group_size == 0       # 每组最后一层是 MLA
        or layer_idx >= num_hidden_layers // layer_group_size * layer_group_size
    )
```

对于 Bailing 3.0 (layer_group_size=6, num_hidden_layers=42):
- KDA 层: (layer_idx+1) % 6 != 0 → layer 0,1,2,3,4, 6,7,8,9,10, ...
- MLA 层: (layer_idx+1) % 6 == 0 → layer 5, 11, 17, 23, 29, 35, 41

---

## 九、MTP（Multi-Token Prediction）详解

Bailing 3.0 配置了 `num_nextn_predict_layers=1`，即 1 个 MTP 预测层（用于投机解码）。

### 9.1 MTP 层权重矩阵

| 权重名 | Shape | 计算公式 | 说明 |
|--------|-------|----------|------|
| `enorm.weight` | `[2560]` | `hidden_size` | 输入 embedding 的 RMSNorm |
| `hnorm.weight` | `[2560]` | `hidden_size` | 隐状态的 RMSNorm |
| `eh_proj.weight` | `[2560, 5120]` | `hidden_size, 2×hidden_size` | embedding+hidden 拼接投影 |
| `input_layernorm.weight` | `[2560]` | `hidden_size` | MTP 层内输入 RMSNorm |
| `post_attention_layernorm.weight` | `[2560]` | `hidden_size` | MTP 层内注意力后 RMSNorm |
| `hnorm.weight` (final) | `[2560]` | `hidden_size` | MTP 层内最终 RMSNorm |

### 9.2 MTP 层结构

```
输入:
  - input_embeds: 下一个 token 的 embedding  [B, S, 2560]
  - hidden_states: 上一层 Decoder 的输出      [B, S, 2560]
       │
       ├──────────────────────────────┐
       ▼                              ▼
  ┌──────────────┐            ┌──────────────┐
  │  enorm       │            │  hnorm       │
  │  (RMSNorm)   │            │  (RMSNorm)   │
  │  W: [2560]   │            │  W: [2560]   │
  │  计算:       │            │  计算:       │
  │  norm(x) =   │            │  norm(x) =   │
  │  x/√(mean(x²)│            │  x/√(mean(x²)│
  │  +ε) × W     │            │  +ε) × W     │
  └──────┬───────┘            └──────┬───────┘
         │                            │
         ▼                            ▼
  normed_embeds: [B,S,2560]      normed_hidden: [B,S,2560]
         │                            │
         └──────── concat(dim=-1) ────┘
                      │
                      ▼
             [B, S, 5120]  (2 × hidden_size)
                      │
                      ▼
             ┌──────────────┐
             │  eh_proj     │
             │  W: [2560,   │
             │     5120]    │
             │  计算:       │
             │  x @ W.T     │
             │  [B,S,5120]  │
             │  ×[5120,2560]│
             │  →[B,S,2560] │
             └──────┬───────┘
                    │
                    ▼
             residual = eh_proj_output  (不用hidden_states)
                    │
                    ▼
             ┌──────────────┐
             │input_layernorm│
             │  W: [2560]    │
             └──────┬───────┘
                    │
                    ▼
             MLA Attention (始终用 MLA 层)
                    │
                    ▼
             residual = attn_output + residual
                    │
                    ▼
             ┌───────────────────┐
             │post_attention_norm │
             │  W: [2560]         │
             └──────┬────────────┘
                    │
                    ▼
             MoE SparseMoeBlock
                    │
                    ▼
             residual = mlp_output + residual
                    │
                    ▼
             ┌──────────────┐
             │final_layernorm│
             │  W: [2560]    │
             └──────┬───────┘
                    │
                    ▼
             mtp_hidden_states: [B, S, 2560]
                    │
                    ▼
             ┌──────────────┐
             │  lm_head     │
             │  W: [157184, │
             │     2560]    │
             │  计算:       │
             │  x @ W.T     │
             │  [B,S,2560]  │
             │  ×[2560,     │
             │   157184]    │
             │  →[B,S,157184│
             └──────┬───────┘
                    │
                    ▼
             mtp_logits: [B, S, 157184]
```

### 9.2 MTP 工作方式

MTP 层将当前 hidden states 和下一个 token 的 embedding 拼接投影后，经过一个完整的 Transformer 层，预测下一个 token 的 logits。在训练时，MTP 损失乘以 `mtp_loss_scaling_factor` 加到总损失中。推理时 `mtp_loss_scaling_factor=0`，MTP 层可用于投机解码加速。

---

## 十、完整前向流程中 Tensor Shape 变化汇总

### 10.1 Embedding 阶段

| 操作 | Shape 变化 |
|------|-----------|
| input_ids → word_embeddings | `[B, S]` → `[B, S, 2560]` |

### 10.2 MLA 层

| 操作 | 权重 Shape | Shape 变化 |
|------|------------|-----------|
| hidden_states → q_proj | `W_q: [6144, 2560]` | `[B,S,2560] × [2560,6144]` → `[B,S,6144]` |
| q_states reshape | — | `[B,S,6144]` → `[B,32,S,192]` |
| q_pass, q_rot split | — | `[B,32,S,192]` → `[B,32,S,128]` + `[B,32,S,64]` |
| hidden_states → kv_a_proj_with_mqa | `W_kv_a: [576, 2560]` | `[B,S,2560] × [2560,576]` → `[B,S,576]` |
| k_pass, k_rot split | — | `[B,S,576]` → `[B,S,512]` + `[B,S,64]` |
| k_pass → kv_a_layernorm | `W_norm: [512]` | `[B,S,512]` → `[B,S,512]` |
| k_pass → kv_b_proj | `W_kv_b: [8192, 512]` | `[B,S,512] × [512,8192]` → `[B,S,8192]` → reshape `[B,32,S,256]` |
| k_pass_new, value_states split | — | `[B,32,S,256]` → `[B,32,S,128]` + `[B,32,S,128]` |
| k_rot reshape | — | `[B,S,64]` → `[B,1,S,64]` → expand `[B,32,S,64]` |
| RoPE(q_rot, k_rot) | `inv_freq: [32]` | 不改变 shape |
| query_states = cat(q_pass, q_rot) | — | `[B,32,S,192]` |
| key_states = cat(k_pass_new, k_rot) | — | `[B,32,S,192]` |
| attention (Q×K^T/√d × V) | — | `[B,32,S,128]` |
| g_proj (head_wise) | `W_g: [32, 2560]` | `[B,S,2560] × [2560,32]` → `[B,S,32]` → sigmoid |
| gated output | — | `[B,32,S,128]` × gate → `[B,32,S,128]` |
| reshape + dense (o_proj) | `W_o: [2560, 4096]` | `[B,S,4096] × [4096,2560]` → `[B,S,2560]` |

### 10.3 KDA 层

| 操作 | 权重 Shape | Shape 变化 |
|------|------------|-----------|
| hidden_states → q_proj | `W_q: [4096, 2560]` | `[S,2560] × [2560,4096]` → `[S,4096]` |
| hidden_states → k_proj | `W_k: [4096, 2560]` | `[S,2560] × [2560,4096]` → `[S,4096]` |
| hidden_states → v_proj | `W_v: [4096, 2560]` | `[S,2560] × [2560,4096]` → `[S,4096]` |
| hidden_states → f_proj | `W_f: [4096, 2560]` | `[S,2560] × [2560,4096]` → `[S,4096]` |
| hidden_states → b_proj | `W_b: [32, 2560]` | `[S,2560] × [2560,32]` → `[S,32]` → sigmoid → `[1,S,32,1]` |
| hidden_states → g_proj | `W_g: [4096, 2560]` | `[S,2560] × [2560,4096]` → `[S,4096]` (稍后使用) |
| causal_conv1d(q) | `W_qconv: [4096,1,4]` | `[S,4096]` → `[S,4096]` |
| causal_conv1d(k) | `W_kconv: [4096,1,4]` | `[S,4096]` → `[S,4096]` |
| causal_conv1d(v) | `W_vconv: [4096,1,4]` | `[S,4096]` → `[S,4096]` |
| reshape q/k/v | — | `[S,4096]` → `[1,S,32,128]` |
| fused_kda_gate(f, A_log, dt_bias) | `A_log: [1,1,32,1]`, `dt_bias: [4096]` | `[S,4096]` → `[1,S,32,128]` (衰减门控 g1) |
| chunk_kda / fused_recurrent_kda | `h_state: [B,32,128,128]` | `[1,S,32,128]` → core_attn_out `[1,S,32,128]` |
| o_norm(core_attn_out, g2) | `W_norm: [128]` | RMSNorm + sigmoid(g2): `[1,S,32,128]` |
| reshape → o_proj | `W_o: [2560, 4096]` | `[1,S,32,128]` → `[S,4096]` × `[4096,2560]` → `[S,2560]` |

### 10.4 MoE 层

| 操作 | 权重 Shape | Shape 变化 |
|------|------------|-----------|
| hidden_states → gate | `W_gate: [512,2560]` + `expert_bias: [512]` | `[S,2560] × [2560,512]` → logits `[S,512]` → sigmoid → scores `[S,512]` |
| Group-Limited TopK | — | → topk_idx `[S,8]`, topk_weight `[S,8]` |
| 每个路由专家 gate_proj | `W_gate_e: [768,2560]` ×512 | `[tokens_i,2560] × [2560,768]` → `[tokens_i,768]` |
| 每个路由专家 up_proj | `W_up_e: [768,2560]` ×512 | `[tokens_i,2560] × [2560,768]` → `[tokens_i,768]` |
| 每个路由专家 down_proj | `W_down_e: [2560,768]` ×512 | `[tokens_i,768] × [768,2560]` → `[tokens_i,2560]` |
| 路由专家加权求和 | — | `[S,2560]` |
| 共享专家 gate_proj | `W_gate_s: [768,2560]` | `[S,2560] × [2560,768]` → `[S,768]` |
| 共享专家 up_proj | `W_up_s: [768,2560]` | `[S,2560] × [2560,768]` → `[S,768]` |
| 共享专家 down_proj | `W_down_s: [2560,768]` | `[S,768] × [768,2560]` → `[S,2560]` |
| final = 2.5×routed + shared | — | `[S,2560]` |

### 10.5 LM Head

| 操作 | 权重 Shape | Shape 变化 |
|------|------------|-----------|
| hidden_states → lm_head | `W_lm: [157184, 2560]` | `[B,S,2560] × [2560,157184]` → `[B,S,157184]` |
| logits = output.float() | — | `[B,S,157184]` (转为 float32) |

---

## 十一、KDA 递归状态更新的数学推导

KDA 的核心是 Delta Rule 递归更新，以下详细推导其数学原理。

### 11.1 标准 Delta Rule

Delta Rule 是在线学习规则的一种，在注意力语境下可理解为：

```
h_t = A_t ⊙ h_{t-1} + β_t (k_t ⊗ v_t)
o_t = q_t^T h_t
```

其中：
- `h_t ∈ R^{H×V×K}`: 递归隐状态（记忆矩阵）
- `A_t ∈ R^{H×D}`: 衰减因子（门控，控制历史遗忘速度）
- `β_t ∈ R^{H×1}`: 写入门控（控制新信息写入强度）
- `k_t ∈ R^{H×K}`: Key（来自短卷积后的特征）
- `v_t ∈ R^{H×V}`: Value（来自短卷积后的特征）
- `q_t ∈ R^{H×K}`: Query（来自短卷积后的特征）

### 11.2 Bailing 3.0 的 KDA 具体实现

**衰减门控 g1 计算**：

```python
# f_proj: Linear(2560, 4096)
f = f_proj(hidden_states)   # [B, S, 4096]

# fused_kda_gate: 将 f 转换为 g1 (head-wise 衰减因子)
# A_log: [1, 1, H, 1] 可学习参数，H=32
# dt_bias: [4096] 可学习偏置

# Safe Gate 模式 (kda_safe_gate=True):
g1 = lower_bound × sigmoid(exp(A_log) × (f + dt_bias))
# = -5.0 × sigmoid(exp(A_log_h) × (f_{h,d} + dt_bias_{h,d}))

# 每个头 h 和维度 d:
# g1[h, d] = -5.0 × σ(exp(A_log[h]) × (f[h, d] + dt_bias[h*d + d']))
# 由于 sigmoid 值域 (0, 1)，g1 的范围是 (-5.0, 0)
# 这意味着记忆衰减因子 exp(g1) 在 (exp(-5), 1) ≈ (0.0067, 1) 之间
```

**写入门控 β 计算**：

```python
# b_proj: Linear(2560, 32)
beta = sigmoid(b_proj(hidden_states).float())
# [B, S, 32] → unsqueeze → [1, S, 32, 1]
# 每个 head 一个标量值，控制 "新 key-value 对" 写入记忆的强度
```

**递归更新**：

```
对每个序列位置 t：

1. 衰减旧记忆:
   h_t = exp(g1_t) ⊙ h_{t-1}
   其中 exp(g1_t) ∈ (0.0067, 1)，实现"软遗忘"

2. 写入新信息:
   h_t = h_t + β_t × (k_t ⊗ v_t)
   其中 k_t ∈ R^{K}, v_t ∈ R^{V}
   k_t ⊗ v_t 是外积，产生 R^{V×K} 的矩阵

3. 查询:
   o_t = h_t @ q_t    (matmul: [V, K] × [K] → [V])
   或 qk_l2norm 后: o_t = (h_t @ q̂_t) 其中 q̂_t = q_t / ||q_t||
```

**use_qk_l2norm_in_kernel=True**：在 KDA kernel 内部对 q 和 k 做 L2 归一化，使得注意力 score 受方向而非幅度影响，提升训练稳定性。

### 11.3 chunk_kda 并行分块算法

Prefill 阶段使用 `chunk_kda`，将序列分成固定大小的 chunks 并行计算：

```
将序列 S 分成 chunks，每 chunk 大小 BT
对每个 chunk:
  1. 计算 chunk 内的 intra-chunk attention
  2. 使用初始状态 h_0 (来自前一个 chunk 的最终状态)
  3. 输出 chunk 内每个位置的 o_t
  4. 输出 chunk 的 final_state 供下一个 chunk 使用
```

### 11.4 fused_recurrent_kda 递归算法

Decode 阶段每次处理 1 个 token，使用 `fused_recurrent_kda`：

```
对每个 token t:
  1. q, k, v 来自 causal_conv1d_update (增量卷积)
  2. g1 = fused_kda_gate(f_proj_t, A_log, dt_bias)
  3. h_t = exp(g1_t) * h_{t-1} + beta_t * (k_t ⊗ v_t)
  4. o_t = h_t @ q_t
  原地更新 recurrent_state
```

---

## 十二、模型参数量估算

| 组件 | 参数量估算 | 说明 |
|------|-----------|------|
| Word Embedding | 157184 × 2560 ≈ 402M | |
| LM Head | 157184 × 2560 ≈ 402M | |
| **MLA 层 (×7)** | | 每个 MLA 层 |
| - q_proj | 2560 × 6144 ≈ 15.7M | |
| - kv_a_proj_with_mqa | 2560 × 576 ≈ 1.5M | |
| - kv_a_layernorm | 512 | |
| - kv_b_proj | 512 × 8192 ≈ 4.2M | |
| - g_proj (head_wise) | 2560 × 32 ≈ 82K | |
| - dense (o_proj) | 4096 × 2560 ≈ 10.5M | |
| **KDA 层 (×35)** | | 每个 KDA 层 |
| - q/k/v/f/g_proj | 5 × 2560 × 4096 ≈ 52.4M | |
| - q/k/v_conv1d | 3 × 4096 × 4 ≈ 49K | |
| - b_proj | 2560 × 32 ≈ 82K | |
| - A_log | 32 | |
| - dt_bias | 4096 | |
| - o_proj | 4096 × 2560 ≈ 10.5M | |
| **Dense MLP (×2)** | 2 × 3 × 2560 × 6144 ≈ 94.4M | |
| **MoE 层 (×40)** | | 每个 MoE 层 |
| - Router | 512 × 2560 ≈ 1.3M | |
| - 路由专家 (×512) | 512 × 3 × 2560 × 768 ≈ 3.0B | |
| - 共享专家 | 3 × 2560 × 768 ≈ 5.9M | |
| **Final RMSNorm** | 2560 | |

**总参数量** ≈ **数十B 级别**（取决于 MoE 激活参数：每个 token 只激活 8/512 的路由专家 + 共享专家，激活参数量远小于总参数量）。

---

## 十三、vLLM 推理实现与 HuggingFace 的差异

| 方面 | HuggingFace | vLLM |
|------|------------|------|
| Token 形状 | `[B, S, D]` (batch 优先) | `[num_tokens, D]` (token 优先，1D flatten) |
| 注意力实现 | 标准 MHA (eager/flash/sdpa) | MLA wrapper + 自定义 kernel |
| KDA 状态管理 | `DynamicCache` (layers 列表) | `MambaCache` (conv_state + recurrent_state) |
| KDA 计算模式 | chunk (训练) / fused_recurrent (短序列) | chunk (prefill) / fused_recurrent (decode) |
| MoE 实现 | 逐专家循环 | `SharedFusedMoE` 融合 kernel |
| RMSNorm | `BailingMoeV3RMSNorm` | `vLLM RMSNorm` (融合 res add) |
| Gated RMSNorm | `FusedRMSNormGated` (fla) | `vLLM FusedRMSNormGated` (Triton) |
| Causal Conv1d | `ShortConvolution` (fla) | `causal_conv1d_fn` / `causal_conv1d_update` (vLLM) |
| TP 并行 | 不支持 | ColumnParallelLinear / RowParallelLinear |
| KDA gate 计算 | Python 端 `chunk_kda(fused_gate=True)` | `fused_kda_gate` (Triton) + `chunk_kda` 分离 |
| RoPE | HF `BailingMoeV3RotaryEmbedding` | vLLM `get_rope` (is_neox_style=False) |
| Linear bias | `kv_a_proj_with_mqa` 和 `dense` 可配置 `use_qkv_bias` | 所有 Linear 层均 `bias=False` |

> **设计说明**：
>
> 1. **`bias=False` 的通用设计原则**：Bailing 3.0 中绝大多数 Linear 投影层不使用偏置项（`bias=False`）。这是现代 LLM 的主流设计（LLaMA、Mistral、Qwen 等均采用），主要原因包括：(a) **参数效率**：每个 Linear 去掉 bias 可减少 hidden_size 个参数，对于 embedding 维度后的投影和 MoE 的 512×3 个专家累积效果显著；(b) **量化友好**：INT8/INT4 量化时，bias 需要特殊处理（反量化后加回），去掉 bias 简化了量化流程；(c) **与 LayerNorm/RMSNorm 配合**：Linear 后通常紧跟 RMSNorm，RMSNorm 本身有可学习的缩放因子 γ weight，可以隐式吸收 bias 的平移效果，因此 bias 冗余。例外：HuggingFace 实现中 `kv_a_proj_with_mqa` 和 `dense` 可通过 `config.use_qkv_bias=True` 启用 bias，但 vLLM 实现中始终为 `bias=False`。
>
> 2. **vLLM 中的 Tensor Parallelism（TP）并行策略**：vLLM 实现中使用了 `ColumnParallelLinear` 和 `RowParallelLinear` 两种并行投影策略：
>    - **ColumnParallelLinear**（列并行）：按输出维度切分权重矩阵。每个 TP rank 持有权重的一部分列，计算输出的一个分片，获得 shard 后的输出。典型用于 **Q/K/V/g_proj/f_proj** 等投影（按 num_heads 切分头），以及 **MoE gate_proj/up_proj**（按 intermediate_size 切分）。
>    - **RowParallelLinear**（行并行）：按输入维度切分权重矩阵。每个 TP rank 用部分输入计算部分结果，然后通过 AllReduce 求和获得完整输出。典型用于 **o_proj/dense**（将多个头的输出合并回 hidden_size），以及 **MoE down_proj**。
>    - 对于 `kv_a_proj_with_mqa`，由于 KV 压缩后的维度共享于所有头，使用 **ReplicatedLinear**（不分片，每个 rank 复制完整权重），避免 TP 切分破坏共享语义。
>    - `A_log` 和 `dt_bias` 在 KDA 中按 head 维度分片（`A_log: [1,1,num_local_heads,1]`，`dt_bias: [num_local_heads×head_dim]`），与 TP 并行对齐。

---

## 十四、关键代码路径索引

### 14.1 HuggingFace 实现

| 模块 | 文件路径 | 关键类/函数 |
|------|---------|-----------|
| 模型入口 | `modeling_bailing_moe_v3.py` | `BailingMoeV3ForCausalLM` |
| Model | `modeling_bailing_moe_v3.py` | `BailingMoeV3Model` |
| DecoderLayer | `modeling_bailing_moe_v3.py` | `BailingMoeV3DecoderLayer` |
| MLA 注意力 | `modeling_bailing_moe_v3.py` | `BailingMoeV3MultiLatentAttention` |
| KDA 注意力 | `modeling_bailing_moe_v3.py` | `BailingMoeV3KimiDeltaAttention` |
| MoE | `modeling_bailing_moe_v3.py` | `BailingMoeV3SparseMoeBlock` |
| MoE Router | `modeling_bailing_moe_v3.py` | `BailingMoeV3Gate` |
| MLP | `modeling_bailing_moe_v3.py` | `BailingMoeV3MLP` |
| MTP | `modeling_bailing_moe_v3.py` | `BailingMoeV3MTPLayer` |
| RMSNorm | `modeling_bailing_moe_v3.py` | `BailingMoeV3RMSNorm` |
| RoPE | `modeling_bailing_moe_v3.py` | `BailingMoeV3RotaryEmbedding` |
| RoPE Apply | `modeling_bailing_moe_v3.py` | `apply_rotary_pos_emb_interleave` |
| KDA chunk | `fla/ops/kda/` | `chunk_kda` |
| KDA recurrent | `fla/ops/kda/` | `fused_recurrent_kda` |
| ShortConv | `fla/modules/` | `ShortConvolution` |

### 14.2 vLLM 推理实现

| 模块 | 文件路径 | 关键类/函数 |
|------|---------|-----------|
| 模型入口 | `vllm/model_executor/models/bailing_moe_v3.py` | `BailingMoeV3ForCausalLM` |
| Model | `vllm/model_executor/models/bailing_moe_v3.py` | `BailingMoeV3Model` |
| DecoderLayer | `vllm/model_executor/models/bailing_moe_v3.py` | `BailingMoeV3DecoderLayer` |
| MLA 注意力 | `vllm/model_executor/models/bailing_moe_v3.py` | `BailingMoeV3MLAAttention` |
| KDA 注意力 | `vllm/model_executor/models/bailing_moe_v3.py` | `BailingMoeV3KimiDeltaAttention` |
| MLA Wrapper | `vllm/model_executor/layers/mla.py` | `MultiHeadLatentAttentionWrapper` |
| MoE | `vllm/model_executor/models/bailing_moe_v3.py` | `BailingMoeV3MoE` |
| MoE Router | `vllm/model_executor/models/bailing_moe_v3.py` | `BailingMoeV3Gate` |
| KDA gate | `vllm/model_executor/layers/fla/ops/kda.py` | `fused_kda_gate` |
| KDA chunk | `vllm/model_executor/layers/fla/ops/kda.py` | `chunk_kda` |
| KDA recurrent | `vllm/model_executor/layers/fla/ops/kda.py` | `fused_recurrent_kda` |
| KDA attention op | `vllm/model_executor/layers/fla/ops/kda.py` | `torch.ops.vllm.kda_attention` |
| FusedRMSNormGated | `vllm/model_executor/layers/fla/ops/kda.py` | `FusedRMSNormGated` |
| Causal Conv1d | `vllm/model_executor/layers/mamba/ops/` | `causal_conv1d_fn` / `causal_conv1d_update` |

---

## 十五、总结

Bailing 3.0 的架构设计核心思想：

1. **混合注意力架构**：结合 MLA（Softmax 注意力，长距离全局建模）和 KDA（Delta Rule 线性注意力，线性复杂度高效推理），在 6 层一个周期中 5 层 KDA + 1 层 MLA，兼顾效率与效果。

2. **MLA 的 KV Cache 压缩**：将 KV 压缩到 512+64=576 维的低秩空间，相比标准 MHA 的 8192 维，压缩比约 14x，极大降低推理内存。

3. **KDA 的 Delta Rule 机制**：通过可学习的衰减门控 `g1` 和写入门控 `β`，实现自适应的记忆更新；短卷积增强局部信息交互；Safe Gate 机制（`lower_bound × sigmoid(exp(A) × x)`）保证衰减因子的数值稳定性。

4. **RoPE 仅作用于非压缩维度**：MLA 中 RoPE 只应用于 `k_rot`（64维），避免 KV Cache 中压缩向量与位置编码耦合。

5. **大规模 MoE**：512 个专家中每个 token 只激活 8 个（Group-Limited TopK 策略），配合共享专家和 2.5x 缩放因子，实现稀疏而高效的前馈计算。

6. **MTP 投机解码**：1 层 MTP 预测层可用于投机解码，通过嵌入投影和额外的 Transformer 层预测后续 token。