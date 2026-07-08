# YaRN RoPE 外推与百灵 3.0 Ascend 问题说明

这篇文档面向刚接触 RoPE / YaRN / vLLM 的读者。目标不是把所有数学细节讲完，而是建立足够清晰的直觉，帮助理解这次报告的问题：

> 百灵 3.0 在 YaRN RoPE 外推场景下，`AscendYaRNRotaryEmbedding` 初始化时没有调用 `_record_cos_and_sin_cache_interleaved`，导致 Ascend MLA 路径使用的 cos/sin 缓存不正确。

## 1. 为什么 Transformer 需要位置信息

普通 self-attention 看的是一组 token 之间的相似度。只看 token embedding 的话，模型本身并不知道谁在前、谁在后。

比如：

```text
我 喜欢 你
你 喜欢 我
```

这两句话 token 集合很像，但顺序不同，意思也不同。所以模型需要某种“位置编码”告诉它每个 token 在第几个位置。

常见位置编码方式有：

- 绝对位置编码：给第 0、1、2、... 个位置各加一个向量。
- 相对位置编码：attention 时显式考虑两个 token 的距离。
- RoPE：把位置信息注入到 query/key 的旋转角度里。

百灵 3.0 的 MLA attention 使用的是 RoPE。

## 2. RoPE 是什么：从公式到直觉

RoPE 全称是 Rotary Position Embedding，旋转位置编码。

它的核心思想是：不要直接给 hidden states 加一个位置向量，而是在 attention 计算前，把 query 和 key 的一部分维度按照当前位置旋转一个角度。

最小例子可以只看两个维度。把两个数看成二维平面上的一个点：

```text
(x0, x1)
```

旋转角度为 `theta` 时，二维旋转公式是：

```text
new_x0 = x0 * cos(theta) - x1 * sin(theta)
new_x1 = x0 * sin(theta) + x1 * cos(theta)
```

RoPE 把 `theta` 设计成和 token 位置相关：

```text
theta = position * inv_freq
```

所以第 0 个 token 旋转角度小，第 1000 个 token 旋转角度大。到这里还只是“知道怎么算”，更重要的是：为什么这样能表达位置？

关键在 attention 的相似度计算。attention 会拿某个位置的 query 和另一个位置的 key 做点积：

```text
score = dot(query, key)
```

RoPE 不是直接比较原始的 query/key，而是先旋转：

```text
第 m 个 token 的 query: q_m = rotate(q, m)
第 n 个 token 的 key:   k_n = rotate(k, n)
score = dot(q_m, k_n)
```

这里的 `rotate(q, m)` 不是另一个新公式，它就是前面的二维旋转公式，只是把 `position` 换成了 `m`。

假设 query 向量 `q` 的前两个维度是：

```text
q = [q0, q1, ...]
```

第 `m` 个 token 的旋转角度是：

```text
theta_m = m * inv_freq
```

那么：

```text
q_m[0] = q0 * cos(theta_m) - q1 * sin(theta_m)
q_m[1] = q0 * sin(theta_m) + q1 * cos(theta_m)
```

这和前面的：

```text
new_x0 = x0 * cos(theta) - x1 * sin(theta)
new_x1 = x0 * sin(theta) + x1 * cos(theta)
```

是同一个计算，只是把变量名从 `(x0, x1)` 换成了 `(q0, q1)`，把 `theta` 换成了 `theta_m`。

如果 query 有很多维度，RoPE 会每两个维度一组重复这个过程：

```text
(q0, q1) 使用第 1 个 inv_freq 旋转
(q2, q3) 使用第 2 个 inv_freq 旋转
(q4, q5) 使用第 3 个 inv_freq 旋转
...
```

key 也是同样的逻辑：

```text
k_n[0] = k0 * cos(theta_n) - k1 * sin(theta_n)
k_n[1] = k0 * sin(theta_n) + k1 * cos(theta_n)
theta_n = n * inv_freq
```

二维旋转有一个非常好的性质：两个都旋转以后，它们的点积只和“相对旋转角度”有关。直观地说：

```text
dot(rotate(q, m), rotate(k, n))
```

可以等价理解为：

```text
拿 q 和一个被旋转了 n - m 距离的 k 做比较
```

也就是说，attention 分数里自然出现了 `n - m`，也就是两个 token 的相对距离。

这就是 RoPE 的直觉优势：

- 它没有粗暴地给 token embedding 加一个位置标签。
- 它让 query/key 的相似度计算本身感知相对位置。
- 同样两个词，距离近和距离远时，点积会不一样。
- 位置差相同的 token 对，会以相似的旋转关系参与 attention。

这也是为什么 RoPE 很适合自回归语言模型：生成时，模型关心的不只是“这是第几个 token”，还关心“当前 token 和前面某个 token 隔了多远”。

## 3. `inv_freq` 从哪里来，RoPE 的“频率”是什么意思

在 RoPE 中，每一对维度都有自己的旋转速度。这个旋转速度通常叫 `inv_freq`，可以粗略理解成“每前进一步，角度增加多少”。

普通 RoPE 中，`inv_freq` 不是模型训练出来的权重，也不是 checkpoint 里的一组 learned parameter。它通常由配置和公式确定。

vLLM 里的普通 RoPE，也就是 Llama 等很多模型使用的默认 RoPE，公式可以简化成：

```text
inv_freq[i] = 1 / rope_theta ** (i / rotary_dim)
```

其中：

- `i` 取 `0, 2, 4, ...`，每两个维度共用一个频率。
- `rope_theta` 通常来自模型 `config.json` 里的 `rope_theta`，如果没写，vLLM 默认用 `10000`。
- `rotary_dim` 是真正参与 RoPE 旋转的维度数。

但这不代表“所有模型都一定按这个方式加位置编码”。不同模型可能使用不同的位置编码方案：

- 有的模型使用 RoPE。
- 有的模型使用绝对位置编码。
- 有的模型使用 ALiBi 或其他相对位置方法。
- 有的模型使用 RoPE，但会叠加 YaRN、linear scaling、NTK scaling、LongRoPE、M-RoPE 等变体。

所以更准确的说法是：

```text
普通 RoPE 的默认频率公式通常是这个；
具体模型是否使用它，以及是否使用 scaling，要看 config.json 和模型实现。
```

所以 `inv_freq` 的来源可以理解为：

```text
config.json 里的 rope_theta / rope_scaling 等参数
+ 模型结构决定的 rotary_dim
+ RoPE 类型对应的公式
=> 推导出 inv_freq
```

它不是直接“查 config 里的 inv_freq 数组”。通常 config 不会逐项写出 `inv_freq`，而是写 `rope_theta`、`rope_scaling`、`factor`、`original_max_position_embeddings` 这类高层参数。

这不会天然造成“训练和推理行为不一致”的问题，因为训练时和推理时都应该使用同一套 RoPE 规则。

可以把 RoPE 理解成模型结构的一部分，而不是一个临时推理技巧。训练时，模型的 query/key projection 权重就是在这套固定位置变换下学出来的；推理时只要使用同样的 `rope_theta`、同样的 RoPE 类型、同样的维度布局，行为就是一致的。

真正会出问题的是这些情况：

- 推理时 `rope_theta` 和训练配置不一致。
- 模型需要 YaRN，但推理框架按普通 RoPE 算。
- 模型使用 interleaved 布局，但推理框架按 Neox-style 布局拆 cache。
- RoPE cache 长度够了，但里面的 `inv_freq/cos/sin` 不是正确 scaling 后的结果。

这次 Ascend YaRN 问题就属于最后两类：不是 RoPE 固定公式本身有问题，而是正确的 YaRN cache 没有进入 Ascend MLA 快路径需要的 interleaved cache。

有了 `inv_freq` 后，框架会为每个位置预计算：

```text
cos(position * inv_freq)
sin(position * inv_freq)
```

推理时不需要每次重新算三角函数，只要根据 `positions` 查表即可。这个表通常叫：

```text
cos_sin_cache
```

在 vLLM 和 vLLM-Ascend 里，RoPE 的本质也是先生成这样一张 cos/sin 表，然后 attention 里根据 positions 去查。

## 4. “部分维度”是多大，由什么决定

attention 里的每个 head 都有一个向量维度，通常叫 `head_size` 或 `head_dim`。RoPE 不一定作用在整个 head 上，而是作用在其中的 `rotary_dim` 个维度上。

可以把一个 head 想成：

```text
[参与 RoPE 的维度 | 不参与 RoPE 的维度]
```

代码上常见逻辑是：

```python
query_rot = query[..., :rotary_dim]
query_pass = query[..., rotary_dim:]
```

`query_rot` 会做 RoPE 旋转，`query_pass` 原样保留。

`rotary_dim` 一般由模型结构和 config 决定。在 vLLM 的 `get_rope()` 中，大致优先级是：

```text
如果 rope_parameters 里有 rope_dim，就用 rope_dim
否则如果有 partial_rotary_factor，就用 head_size * partial_rotary_factor
否则默认 rotary_dim = head_size
```

百灵 3.0 的 MLA 比较特殊。它的 attention head 被拆成两部分：

```text
qk_nope_head_dim: 不做 RoPE 的 q/k 维度
qk_rope_head_dim: 做 RoPE 的 q/k 维度
```

在 `bailing_moe_v3.py` 里创建 RoPE 时，传进去的是：

```python
head_size=self.qk_rope_head_dim
```

并且代码特意去掉了 `partial_rotary_factor`，避免把 `qk_rope_head_dim` 再缩小一次。所以对百灵 3.0 MLA 来说，可以把 `qk_rope_head_dim` 理解为 RoPE 实际处理的宽度。

还有一个细节：标准 RoPE 是两两维度配对做二维旋转，所以 `rotary_dim` 通常必须是偶数。

如果你问“维度是 3 的话，公式会变吗？”答案是：标准 RoPE 不会把 3 个维度一起做一个三维旋转。它是按二维平面成对旋转：

```text
(第 0, 1 维) 一对
(第 2, 3 维) 一对
(第 4, 5 维) 一对
...
```

所以真正合法、常见的 `rotary_dim` 是偶数。公式不会因为“3 维”变成三维旋转公式；如果出现奇数维，通常说明模型配置或实现路径不符合标准 RoPE 的假设。

## 5. `max_position_embeddings`、`max_model_len` 和实际 position 的关系

这里最容易混，因为有几个“最大长度”长得很像，但含义不同。

先给结论：

> 你实际测试里 `prompt 长度 + max_tokens > 8192` 会报错，大概率是 vLLM 的运行时请求长度上限 `max_model_len` 在拦截请求。这个报错不等价于“RoPE 数学上不能算超过 8192 的 position”。

几个概念分开看：

| 名称 | 含义 |
| --- | --- |
| `max_position_embeddings` | 模型配置里的位置长度参数，很多模型把它当作原始训练上下文长度或 RoPE cache 的基础长度。 |
| `original_max_position_embeddings` | RoPE scaling 中常见字段，表示缩放前的原始长度，比如 8192。 |
| `factor` | RoPE scaling 扩展倍数，比如 4 表示希望扩到约 32768。 |
| `max_model_len` | vLLM 运行时允许一个请求使用的最大总长度，通常约束 `prompt_len + max_tokens`。 |
| RoPE cache 长度 | 实际构造出来的 cos/sin 表长度。没有 scaling 时常等于 `max_position_embeddings`；有 YaRN 时可能是 `original_max_position_embeddings * factor`。 |

实际 position 是怎么超过 8192 的？

自回归生成时，一个完整序列由两部分组成：

```text
输入 prompt token + 模型继续生成的 output token
```

position 是按“完整序列里的 token 下标”递增的，不会因为进入生成阶段就重新从 0 开始。

举一个不太好的极限例子，如果 prompt 已经有 8192 个 token：

```text
prompt 的第 1 个 token: position = 0
prompt 的第 8192 个 token: position = 8191
output 的第 1 个 token: position = 8192，也就是完整序列的第 8193 个 token
output 的第 1000 个 token: position = 9191，也就是完整序列的第 9192 个 token
```

所以只要：

```text
prompt_len + 已生成 token 数 > 8192
```

模型 forward 里就可能出现大于等于 8192 的 position。

这里容易误会的一点是：如果当前运行时 `max_model_len = 8192`，上面这个“prompt 8192 后再继续生成”的请求通常不会真的进入模型 forward。它会先被服务层拒绝。

在 vLLM 服务层，请求通常会先被 `max_model_len` 校验。如果当前 `max_model_len = 8192`，那么：

```text
prompt_len + max_tokens > 8192
```

请求会在进入真正长 position 计算前就被拒绝。这就是你实际看到“超过 8192 就报错”的原因。

更现实的例子是：

```text
prompt_len = 7000
max_tokens = 2000
总长度上限需求 = 9000
```

如果 `max_model_len = 8192`，这个请求会被拒绝，因为 9000 超过了运行时允许的总上下文长度。

如果通过 YaRN / rope scaling 让运行时 `max_model_len` 变成 32768，那么这个请求可以进入推理。此时：

```text
prompt 的最后一个 token: position = 6999
output 的第 1 个 token: position = 7000
output 的第 1193 个 token: position = 8192
output 的第 2000 个 token: position = 8999
```

这样就真的出现了超过 8192 的 position。

什么时候可以真的超过 8192？

需要同时满足几件事：

1. 模型 config 或运行参数让 vLLM 推导出的 `max_model_len` 大于 8192。
2. RoPE scaling 配置存在，例如 YaRN 的 `factor` 和 `original_max_position_embeddings`。
3. vLLM 构造了足够长的 RoPE cache。
4. KV cache / 显存 / NPU 内存允许这么长的上下文。
5. 请求里的 `prompt_len + max_tokens` 不超过运行时 `max_model_len`。

例如：

```text
original_max_position_embeddings = 8192
factor = 4
derived max_model_len ≈ 32768
RoPE cache 长度 ≈ 32768
```

这时请求长度 12000、16000、24000 才有机会通过服务层校验，并在 forward 中产生超过 8192 的 positions。

所以你刚才的理解是对的：YaRN RoPE 外推讨论的正是“运行时允许设置或推导出一个超过原始 `max_position_embeddings` 的 `max_model_len`，然后模型真的会看到更大的 position”这种场景。

所以，“超过 8192 会不会报错”取决于你说的是哪一层：

- 如果运行时 `max_model_len` 还是 8192，请求会直接报错。
- 如果 `max_model_len` 已经扩到 32768，但 RoPE cache 只有 8192，查表会越界。
- 如果 `max_model_len` 和 cache 都扩到 32768，但 cache 的频率不是 YaRN 算出来的，可能不报错但结果不正确。

## 6. 什么是 RoPE 外推

“外推”可以理解为：

> 模型训练时主要见过 8K 长度，现在希望推理时跑 32K、64K，如何让位置编码在更长范围内仍然合理？

从公式上看，RoPE 当然可以继续算：

```text
theta = position * inv_freq
```

即使 `position = 20000`，三角函数也能算出 `cos(theta)` 和 `sin(theta)`。问题不是“数学函数算不出来”，而是模型训练时没有充分见过这么大的 position。

如果什么都不做，直接把原始 RoPE 用到更长位置，旋转角度会继续按原来的速度增长。对某些高频维度来说，角度可能在长上下文里绕了很多圈；对模型来说，这些 query/key 点积模式可能已经偏离训练时熟悉的分布。

这就是长上下文 RoPE 外推的问题：

```text
能算 ≠ 模型能稳定理解
```

所以各种 RoPE scaling 方法会调整 `inv_freq` 或 position 的使用方式，让更长位置范围里的旋转模式更接近模型能接受的分布。

可以把它粗略理解成：

```text
原始 RoPE: 8192 以内角度分布正常，超过后继续原速旋转
RoPE scaling: 让 32768 这类更长范围里的旋转变化更平滑、更可用
```

常见方法包括：

- linear scaling
- dynamic NTK scaling
- YaRN
- DeepSeek YaRN / DeepSeek scaling

## 7. YaRN 是什么

YaRN 是一种 RoPE scaling 方法，用于长上下文扩展。

它可以理解为：在普通 RoPE 的 `inv_freq` 基础上，重新生成一套更适合长上下文的 `inv_freq`。

最朴素的长上下文想法是把 position 压缩一下。例如要从 8K 扩到 32K，可以近似想成：

```text
用 position / 4 参与 RoPE
```

这样 32K 的位置范围会被压进原来 8K 的角度变化范围里。这类思路叫 interpolation，直觉上是“把长序列压缩到模型熟悉的长度里”。

但只做这种压缩也有问题：所有频率都被同等压慢，可能损失模型原本在某些频率上学到的细粒度位置特征。

YaRN 的做法更折中。它不是简单把所有 position 除以同一个系数，而是把 RoPE 频率分成不同区间处理：

- 一部分频率更偏向 interpolation，适合把原始上下文压缩到更长范围。
- 一部分频率更偏向 extrapolation，保留某些原始频率特征。
- 中间用平滑 mask 过渡。
- 最后还有 `mscale` 对 attention 数值幅度做修正。

在代码直觉上，YaRN 会构造两套候选频率：

```text
inv_freq_extrapolation = 原始 RoPE 频率
inv_freq_interpolation = 原始 RoPE 频率 / factor
```

然后根据维度所在的频率区间，在两者之间做平滑混合：

```text
新的 inv_freq = interpolation 部分 + extrapolation 部分
```

所以 YaRN 改的不是 token 本身，也不是 attention 结构，而是 RoPE 查表前那套 `inv_freq`。

在 vLLM 代码里，YaRN 的核心逻辑在：

```text
vllm/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py
```

它会重新计算 `inv_freq`，然后生成新的：

```text
cos_sin_cache = concat(cos, sin)
```

同时，YaRN 会把 cache 长度扩到类似：

```text
original_max_position_embeddings * factor
```

所以 YaRN 同时做了两件事：

1. 让 RoPE cache 有机会覆盖更长的 positions。
2. 让这些更长 positions 对应的 cos/sin 频率更合理。

这就是为什么 YaRN 的关键不是“让索引不越界”这么简单，而是“生成一张适合长上下文的 RoPE cos/sin 表”。

## 8. vLLM 里如何选择 YaRN

vLLM 会根据模型 config 里的 `rope_parameters` 或 `rope_scaling` 选择不同 RoPE 类型。

在 vLLM 的 `get_rope()` 里，几个关键字段大致这样生效：

```text
rope_theta -> 决定普通 RoPE 频率公式里的 base，默认 10000
rope_type  -> 决定使用 default / yarn / linear / dynamic 等哪种 RoPE
factor     -> 决定长上下文扩展倍数
rope_dim / partial_rotary_factor -> 决定 rotary_dim
```

百灵 3.0 模型里，RoPE 参数通过 `bailing_moe_v3.py` 的 `_build_rope_parameters` 传给 `get_rope()`：

```python
self.rotary_emb = get_rope(
    head_size=self.qk_rope_head_dim,
    max_position=getattr(config, "max_position_embeddings", 8192),
    is_neox_style=False,
    rope_parameters=_build_rope_parameters(config),
)
```

如果配置里包含：

```python
rope_type = "yarn"
```

vLLM 就会构造：

```python
YaRNScalingRotaryEmbedding
```

它会用 YaRN 规则重新计算 `inv_freq`，再生成更长的 `cos_sin_cache`。

也就是说，百灵 3.0 在 vLLM 侧本身是支持 YaRN RoPE 外推的。Ascend 侧的问题不是“不认识 yarn 这个类型”，而是 YaRN 初始化后的 cache 没有被额外拆成 Ascend MLA 快路径需要的 interleaved cos/sin 全局 cache。

## 9. Neox-style 和 interleaved 是什么

前面说过，RoPE 是“两两维度组成一个二维平面，然后旋转”。但有一个实现细节还没说清楚：

> 哪两个维度算一对？

不同模型约定不一样。vLLM 里常见两种：

```text
is_neox_style=True   -> Neox-style
is_neox_style=False  -> GPT-J / interleaved style
```

它们使用的是同一个二维旋转公式，差异不是数学原理，而是“维度配对方式”和“旋转后怎么放回原向量”。

先假设 `rotary_dim = 8`，一个 head 里参与 RoPE 的部分是：

```text
x = [x0, x1, x2, x3, x4, x5, x6, x7]
```

### Neox-style

Neox-style 把前半部分和后半部分配对：

```text
x1 = [x0, x1, x2, x3]
x2 = [x4, x5, x6, x7]
```

所以二维旋转的配对是：

```text
(x0, x4)
(x1, x5)
(x2, x6)
(x3, x7)
```

对每一对套用二维旋转公式。比如第一对：

```text
new_x0 = x0 * cos - x4 * sin
new_x4 = x4 * cos + x0 * sin
```

旋转后仍然按“前半 + 后半”的方式放回：

```text
[new_x0, new_x1, new_x2, new_x3, new_x4, new_x5, new_x6, new_x7]
```

vLLM 里对应的直觉代码是：

```python
x1, x2 = torch.chunk(x, 2, dim=-1)
o1 = x1 * cos - x2 * sin
o2 = x2 * cos + x1 * sin
output = torch.cat((o1, o2), dim=-1)
```

### GPT-J / interleaved style

GPT-J / interleaved style 把相邻的偶数/奇数维度配对：

```text
x1 = [x0, x2, x4, x6]
x2 = [x1, x3, x5, x7]
```

所以二维旋转的配对是：

```text
(x0, x1)
(x2, x3)
(x4, x5)
(x6, x7)
```

对第一对来说：

```text
new_x0 = x0 * cos - x1 * sin
new_x1 = x1 * cos + x0 * sin
```

旋转后按相邻交错方式放回：

```text
[new_x0, new_x1, new_x2, new_x3, new_x4, new_x5, new_x6, new_x7]
```

vLLM 里对应的直觉代码是：

```python
x1 = x[..., ::2]
x2 = x[..., 1::2]
o1 = x1 * cos - x2 * sin
o2 = x2 * cos + x1 * sin
output = torch.stack((o1, o2), dim=-1).flatten(-2)
```

### 两者为什么容易混

这两种布局的输入输出 shape 通常完全一样：

```text
[num_tokens, num_heads, rotary_dim]
```

所以配错时不一定会立刻 shape 报错。代码还能跑，但 RoPE 语义错了。

例如模型训练时使用 interleaved，也就是它认为 `(x0, x1)` 是一对。如果推理时误按 Neox-style 处理，就会把 `(x0, x4)` 当一对去旋转。这样相当于把模型训练时学到的位置几何关系打乱了。

可以粗略理解为：

```text
同样一副扑克牌，牌数没少，但你把配对规则换了。
```

数量、shape、dtype 都可能对，语义却不对。

### cos/sin cache 和布局有什么关系

普通 `cos_sin_cache` 通常按“每个 position、每个频率”存：

```text
cos_sin_cache[position] = [cos_0, cos_1, ..., sin_0, sin_1, ...]
```

这里的 `cos_0/sin_0` 对应第 0 个二维旋转平面，`cos_1/sin_1` 对应第 1 个二维旋转平面。

对 Neox-style 来说，这些频率应用到：

```text
(x0, x4), (x1, x5), ...
```

对 interleaved 来说，这些频率应用到：

```text
(x0, x1), (x2, x3), ...
```

所以 cache 本身的数值可以来自同一个 RoPE / YaRN 公式，但把 cache 拆成算子需要的 `cos` 和 `sin` 时，必须匹配模型的维度布局。

这就是为什么 vLLM-Ascend 里有一个名字很长但很关键的函数：

```python
_record_cos_and_sin_cache_interleaved(...)
```

它不是重新发明 RoPE，而是把完整的 `cos_sin_cache` 整理成 Ascend MLA 快路径使用 interleaved RoPE 时要查的 `_cos_cache/_sin_cache`。

### 百灵 3.0 用哪一种

百灵 3.0 MLA 创建 RoPE 时：

```python
is_neox_style=False
```

这表示它用的是 GPT-J / interleaved 风格，也就是相邻维度配对：

```text
(x0, x1), (x2, x3), ...
```

所以百灵 3.0 + MLA + YaRN 在 Ascend 上需要的链路是：

```text
YaRN 生成正确的长上下文 cos_sin_cache
-> 按 interleaved 布局拆成 _cos_cache/_sin_cache
-> Ascend MLA 快路径按 positions 查表
```

这点非常关键，因为如果只记录完整 `cos_sin_cache`，但没有生成 interleaved 布局的 `_cos_cache/_sin_cache`，Ascend MLA 快路径就拿不到正确的 RoPE cache。

## 10. vLLM-Ascend 为什么要记录全局 cos/sin cache

普通 vLLM 的 RoPE 可以直接把 `cos_sin_cache` 放在 rotary embedding 对象里，用的时候调用对象 forward。

但是 vLLM-Ascend 的 MLA / SFA 快路径里，为了配合 NPU 算子，会提前把 cos/sin 拆出来，存在全局变量里：

```python
_cos_cache
_sin_cache
```

MLA 预处理时会调用：

```python
get_cos_and_sin_mla(positions)
```

内部逻辑类似：

```python
cos = _cos_cache[positions]
sin = _sin_cache[positions]
```

这里有一个容易误解的地方：`cos_sin_cache` 和 `_cos_cache/_sin_cache` 不是简单的两个名字。

普通 vLLM 的 `cos_sin_cache` 更像一张总表：

```text
每个 position -> [cos 部分, sin 部分]
```

而 Ascend MLA 快路径希望直接拿到已经拆好的：

```text
每个 position -> cos
每个 position -> sin
```

并且百灵 3.0 使用 `is_neox_style=False`，也就是 interleaved 维度布局。这样拆 cache 时不能只做普通的 `chunk`，还要把 cos/sin 整理成和 interleaved RoPE 输入匹配的形状。

所以 vLLM-Ascend 里会有两个动作：

```python
_record_cos_sin_cache(...)
_record_cos_and_sin_cache_interleaved(...)
```

第一步保留完整表，第二步准备 MLA 快路径真正要查的 `_cos_cache/_sin_cache`。

所以，如果 `_cos_cache/_sin_cache` 没有被正确记录，MLA 使用的 RoPE 就会错。

## 11. 这次报告的具体问题

在 vLLM-Ascend 的普通 RoPE 类里，初始化时会调用两步：

```python
_record_cos_sin_cache(self.cos_sin_cache)
_record_cos_and_sin_cache_interleaved(self.cos_sin_cache)
```

第一步记录完整的 `cos_sin_cache`。

第二步把完整 cache 拆成 MLA 快路径需要的 interleaved 版：

```python
_cos_cache
_sin_cache
```

但是 `AscendYaRNRotaryEmbedding` 里目前只调用了：

```python
_record_cos_sin_cache(self.cos_sin_cache)
```

缺了：

```python
_record_cos_and_sin_cache_interleaved(self.cos_sin_cache)
```

这就是报告里说的：

> AscendYaRNRotaryEmbedding init 的时候没调用 `_record_cos_and_sin_cache_interleaved`

用前面几节的概念串起来，这个问题可以这样理解：

```text
YaRN 已经算出了正确的长上下文 inv_freq
YaRN 也已经生成了正确的长上下文 cos_sin_cache
但是 Ascend MLA 快路径没有拿这张 YaRN 表去生成 _cos_cache/_sin_cache
```

因此，问题不在 YaRN 数学本身，而在 Ascend 侧缓存记录少了一步。

## 12. 会造成什么现象

这个问题不一定表现为“超过 `max_position_embeddings` 就报错”。

可能有几种情况：

### 情况 1：`_cos_cache/_sin_cache` 还是 None

MLA 调用：

```python
_cos_cache[positions]
```

如果 `_cos_cache` 没有初始化，就会报 NoneType 相关错误。

### 情况 2：`_cos_cache/_sin_cache` 被其他 RoPE 初始化过

这更隐蔽。

程序可能不报错，但百灵 3.0 YaRN 路径实际用到的是别的 RoPE cache，或者是没有 YaRN scaling 的 cache。

结果是：

- 短上下文可能看不出明显问题；
- 长上下文超过原始训练长度后，位置编码错误更明显；
- 输出质量可能下降；
- 和 GPU / vLLM 结果对不齐；
- 某些 batch 或图捕获场景下表现不稳定。

### 情况 3：cache 长度够，但频率不对

这正是初学者容易困惑的地方。

只要 RoPE cache 长度足够，position 查表就不会越界。

但如果查到的 cos/sin 表不是 YaRN 算出来的那张表，长上下文虽然不会报错，语义仍然是错的。

所以这个 bug 的核心不是“越界”，而是：

```text
Ascend MLA 快路径没有使用 YaRN 生成的 interleaved cos/sin cache。
```

## 13. 为什么百灵 3.0 更容易暴露这个问题

百灵 3.0 同时满足几个条件：

1. 使用 MLA。
2. MLA 路径会通过 `get_cos_and_sin_mla()` 取全局 cos/sin cache。
3. RoPE 使用 `is_neox_style=False`，需要 interleaved 布局。
4. 如果配置启用 `rope_type=yarn`，会走 `AscendYaRNRotaryEmbedding`。

这四个条件叠在一起，就会踩到：

```text
YaRN 初始化了完整 cos_sin_cache，但没有初始化 MLA 快路径需要的 interleaved cos/sin cache。
```

## 14. 推荐修复

最小修复是在 `AscendYaRNRotaryEmbedding.__init__` 中补上：

```python
_record_cos_and_sin_cache_interleaved(self.cos_sin_cache)
```

修复后应类似：

```python
class AscendYaRNRotaryEmbedding(YaRNScalingRotaryEmbedding):
    def __init__(...):
        ...
        super().__init__(...)
        vllm_config = get_current_vllm_config()
        self.use_mtp = (
            vllm_config.speculative_config
            and vllm_config.speculative_config.method == "mtp"
        )
        _record_cos_sin_cache(self.cos_sin_cache)
        _record_cos_and_sin_cache_interleaved(self.cos_sin_cache)
```

这样 YaRN 生成的 cos/sin 表就会被拆成 Ascend MLA 需要的格式。

## 15. 修复后应该怎么验证

建议按层次验证。

### 1. 初始化检查

确认启用 YaRN 后：

```python
_cos_cache is not None
_sin_cache is not None
```

并且 shape 中最后一维等于百灵 MLA 的：

```text
qk_rope_head_dim
```

### 2. 短上下文一致性

用短输入比较 GPU vLLM 和 Ascend 输出。

短上下文不一定能暴露全部问题，但可以保证基础路径没有坏。

### 3. 长上下文验证

构造超过 `original_max_position_embeddings` 的输入，但不要超过 YaRN 扩展后的最大长度。

例如：

```text
original_max_position_embeddings = 8192
factor = 4
测试 12K / 16K / 24K tokens
```

观察：

- 是否报错；
- logits 是否稳定；
- 和 GPU/vLLM 是否接近；
- 长文本续写质量是否明显异常。

### 4. MTP / graph capture 验证

百灵 3.0 还涉及 MTP 和 Ascend graph/profile 路径，所以最好额外验证：

- profile run；
- TP4；
- MTP speculative decoding；
- 长上下文 + MTP。

## 16. 一句话总结

YaRN 是一种 RoPE 长上下文扩展方法，它会生成一张经过频率调整的 cos/sin 表。百灵 3.0 在 Ascend MLA 路径中需要使用 interleaved 格式的 cos/sin cache，但 `AscendYaRNRotaryEmbedding` 目前只记录了完整 `cos_sin_cache`，没有拆出 interleaved `_cos_cache/_sin_cache`，所以长上下文 YaRN RoPE 路径可能使用错误的位置编码表。

最小修复就是让 `AscendYaRNRotaryEmbedding` 初始化时也调用：

```python
_record_cos_and_sin_cache_interleaved(self.cos_sin_cache)
```
