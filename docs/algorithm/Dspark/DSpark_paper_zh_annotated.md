# DSpark：结合半自回归生成的置信度调度式投机解码

> 中文译注版。原文来自当前目录的 `DSpark_paper.pdf`。公式符号基本沿用原文；图像内容以图注和文字重述；参考文献保留为关键链接与说明。  
> 译注重点放在 DSpark 与 MTP、EAGLE、EAGLE3、DFlash 等投机解码路线的差异。

作者：Xin Cheng、Xingkai Yu、Chenze Shao、Jiashi Li、Yunfan Xiong 等  
机构：Peking University、DeepSeek-AI

## 译注总览：DSpark 相比 MTP / EAGLE / EAGLE3 的核心差异

DSpark 不是简单的“再训练一个更强 draft model”。它把投机解码拆成两个层面同时优化：

1. **草稿生成层面**：用 DFlash 风格的并行 backbone 一次性生成一段草稿，再接一个很轻的顺序 head，让块内 token 能感知已经采样出来的前缀，从而缓解纯并行草稿的 suffix decay。
2. **验证调度层面**：不再固定验证全部草稿 token，而是用置信度头估计每个位置的前缀存活概率，再结合当前引擎的硬件吞吐曲线，动态决定每个请求应该验证多长。

和几条常见路线的差异如下。

| 方法 | 草稿生成方式 | 主要瓶颈 | 是否动态调度验证长度 | DSpark 的区别 |
|---|---|---|---|---|
| MTP / MTP-1 | 通常是目标模型附带的多 token 预测层，常按位置/深度逐步预测未来 token | token 数一多，固定验证会吃掉高并发 batch 容量；生产里常退化到 MTP-1 | 通常没有本文这种硬件感知全局调度 | DSpark 允许生成更长草稿，但按负载和置信度剪掉低收益后缀；重点解决“多草稿在 serving 中是否划算” |
| EAGLE | 自回归式草稿，偏“特征外推 + token 采样”，逐步生成草稿 token | 草稿延迟随草稿长度增长，长块成本高 | 通常靠固定长度或启发式 | DSpark 把主要计算放在一次并行 forward 中，只把很轻的转移 bias 放到顺序环节 |
| EAGLE3 | 进一步用多层目标特征和 Training-Time Test 改进草稿质量，仍属于自回归 drafter | 质量强，但 draft latency 仍随步数增长 | 不是本文的核心贡献 | DSpark 论文实验中在 Qwen3 系列上 macro accepted length 比 Eagle3 高约 26.7% 到 30.9% |
| DFlash | 纯并行 block drafter，一次 forward 产生多个位置 | 位置间独立，后缀接受率快速衰减 | 无硬件感知调度 | DSpark 基本继承 DFlash 的并行吞吐优势，再加 Markov/RNN head 建模块内依赖 |

> 译注：可以把 DSpark 理解成“并行草稿模型 + 很薄的自回归修正层 + serving 侧预算分配器”。它同时回答两个问题：草稿怎么更可信，以及在当前 batch 压力下哪些草稿值得送给 target model 验证。

## 摘要

投机解码通过把草稿生成与目标模型验证解耦来加速大语言模型推理。近期的并行 drafter 可以在一次前向中提出较长的 token 序列，但由于缺少 token 之间的依赖建模，后续位置的接受率会快速下降。另一方面，无差别地验证这些较长草稿块，会把宝贵 batch 容量浪费在高拒绝风险 token 上，在高并发 serving 系统中严重损害吞吐。

本文提出 **DSpark**，一个把高吞吐并行生成与自适应、负载感知验证结合起来的投机解码框架。为了保持草稿质量，DSpark 使用半自回归架构：并行 backbone 负责主要计算，轻量顺序模块负责引入块内依赖，从而缓解后缀衰减。为了优化系统效率，DSpark 使用置信度调度式验证，根据估计的前缀存活概率和具体推理引擎的吞吐曲线，为每个请求动态裁剪验证长度。

离线 benchmark 中，DSpark 在多个领域上相较当前先进的自回归与并行 drafter 都显著提升 accepted length。在线部署到 DeepSeek-V4 serving 系统并承载真实用户流量时，DSpark 能减少验证浪费。相比此前生产基线 MTP-1，在相同吞吐水平下，DSpark 将单用户生成速度提升 60% 到 85%。更重要的是，它能在严格交互约束下避免严重吞吐退化，使系统达到过去不可行的性能档位，推动 serving 系统的 Pareto frontier 外移。作者还开源 DSpark checkpoint 和 DeepSpec 训练仓库。

> 注解：摘要中最关键的指标不是单纯“每轮接受更多 token”，而是“在真实高并发 serving 下，不让低置信后缀挤占 batch”。这正是 DSpark 与普通 MTP-N 或固定长度 speculative decoding 的主要分界。

## 1. 引言

LLM 以自回归方式生成文本：每产生一个新 token，都需要基于之前所有 token 做一次完整前向。因此推理延迟与输出长度成正比。这会导致 GPU 利用率偏低、用户等待感明显，在实时对话助手和多轮 agent 工作流等延迟敏感场景中尤其突出。

投机解码提供了一种原则性的解决方案：轻量的 draft model 先提出一段候选 token，完整的 target model 再通过一次并行前向和拒绝采样验证整段候选，接受与目标分布一致的最长前缀，并追加一个 bonus token。由于验证是并行的，且接受规则严格保持目标模型分布，投机解码可以在不损失质量的情况下加速生成。

draft model 的设计决定了 draft 延迟与接受率之间的权衡。早期 drafter 多为自回归结构，每个位置都依赖之前采样出来的 token，因此 draft 延迟随 block size 线性增长。为了打破这个顺序瓶颈，近期出现了并行 drafter：所有草稿位置在一次 forward 中产生，draft 延迟基本不随 block size 增长，因此理论上能高效生成更长的草稿块。

但要真正释放长并行草稿块的潜力，会遇到两个瓶颈：

- **生成质量瓶颈**：纯并行 drafter 独立预测每个位置，无法在块内建模 token 依赖，容易产生多模态碰撞。例如上下文可能允许 “of course” 或 “no problem”，纯并行模型可能拼出 “of problem” 或 “no course”。这会导致后续位置接受率快速下降。
- **系统效率瓶颈**：即便能轻松生成长草稿，也不应无脑验证所有 token。代码等结构化请求天然更容易被接受，而开放式聊天接受率更低；轻负载时多验证一个 token 代价很小，高负载时低置信 token 会占用本可服务其他请求的 batch 容量。

DSpark 通过两个互补机制解决这些问题：

1. **半自回归生成**：保持计算昂贵的 draft backbone 完全并行，只在输出侧加一个轻量顺序 head 注入局部转移信息，在基本保留并行速度的同时缓解 suffix decay。
2. **置信度调度式验证**：置信度头估计每个位置的前缀存活概率；硬件感知调度器根据实时吞吐曲线，为每个请求分配验证长度，把 target verification budget 用到预期收益最高的 token 上。

实验中，DSpark 在数学、代码、日常聊天等领域都超过 Eagle3 和 DFlash。Qwen3-4B、8B、14B 上，DSpark 相比自回归 Eagle3 的 macro-average accepted length 分别提升 30.9%、26.7%、30.0%；相比并行 DFlash 分别提升 16.3%、18.4%、18.3%。在 DeepSeek-V4 真实流量 serving 中，DSpark 相比 MTP-1 将 V4-Flash 单用户速度提升 60% 到 85%，V4-Pro 提升 57% 到 78%。

> 注解：论文对 MTP-1 的定位很重要。MTP-1 是生产中可控的单 token baseline。静态 MTP-3/MTP-5 虽然能给出更多候选，但在高并发下验证开销会压垮吞吐。DSpark 的价值在于：可以拥有长草稿的潜力，同时通过调度避免长草稿的系统副作用。

## 2. 背景

### 2.1 投机解码

自回归语言模型每次 forward 只生成一个 token，推理延迟与输出长度成正比。投机解码用轻量 draft model `M_d` 加速目标模型 `M_t`。每轮解码中，draft model 提出 `gamma` 个候选 token：

```text
x_1, ..., x_gamma
```

target model 在一次 forward 中验证全部候选，并接受与自身分布一致的最长前缀。

在第 `k` 个草稿位置，target model 计算自己的分布 `p^t_k`，并与 draft 分布 `p^d_k` 比较。token `x_k` 的接受概率为：

```text
min(1, p^t_k(x_k) / p^d_k(x_k))
```

验证从左到右进行；一旦第 `k` 个位置被拒绝，后续 `x_{k+1}, ..., x_gamma` 都会被丢弃，不管它们本身质量如何。

令 `tau` 表示每轮接受 token 数，`T_draft` 和 `T_verify` 分别表示 draft 与 verify 的 wall-clock 时间，则平均每个生成 token 的延迟为：

```text
L = (T_draft + T_verify) / tau
```

因此，加速投机解码有三个杠杆：

- 降低 `T_draft`：draft 更快。
- 提高 `tau`：draft 更准。
- 降低有效 `T_verify`：验证更聪明。

### 2.2 Drafter 架构

draft model 的设计决定 `T_draft` 与 `tau` 如何权衡。现有方法大致分为两类。

**自回归 drafter** 逐步生成草稿 token，每个位置都依赖之前采样出的 token。这种显式依赖有强建模能力，但 draft 成本随 block size 线性增长，即 `T_draft ∝ gamma`。因此它们通常只能用短 block 和浅层架构。为了弥补短 block，一些方法使用 tree-based verification，通过 tree attention 同时验证多条路径，但大量验证 token 会降低整体 serving 吞吐。

**并行 drafter** 在一次 forward 中产生全部 `gamma` 个草稿 token，使 `T_draft` 基本不依赖 block size。这样可以在相同延迟预算下使用更长 block，例如 `gamma = 16`。

DFlash 是一种先进并行 drafter。它从 target model 的若干层抽取上下文特征，并通过 KV injection 条件化 draft model。预填充阶段，目标模型层 `{l_1, ..., l_m}` 的 hidden states 被拼接并投影到 draft hidden space：

```text
H_ctx = RMSNorm(W_c [H^(l_1); ...; H^(l_m)])
```

这些上下文特征会被注入每个 draft layer 的 key/value：

```text
K_i = [W^K_i H_ctx; W^K_i H_d]
V_i = [W^V_i H_ctx; W^V_i H_d]
```

block 内所有位置双向关注彼此和注入的目标上下文。draft model 共享 target model 的 embedding 层和 LM head，并冻结二者。由于 draft 无论 block 多长都只需一次 forward，DFlash 可以在同等延迟预算下使用比自回归 drafter 更深的架构和更长的 block。

> 注解：DFlash 的强项是“重计算并行化”，弱点是块内位置缺少采样后前缀依赖。DSpark 正是在 DFlash 的并行骨架上增加一个很轻的顺序校正层。

## 3. 架构

DSpark 的总体思路是：

- 并行 backbone 承担大部分 draft 计算，使 `T_draft` 近似不随 `gamma` 增长。
- 轻量 sequential block 在草稿 token 之间注入依赖，提高 `tau`。
- confidence head 估计每个位置的接受概率。
- hardware-aware scheduler 剪掉低置信后缀，减少不必要的 target verification。

图 1 的解码流程可以用文字描述如下：给定 prompt token `A B C`，target model 先执行一步生成 `D`，`D` 作为下一轮草稿阶段的 anchor。DSpark 以 `D` 为输入，用较重的并行 backbone 和轻量顺序 head 生成 `E F G H` 及对应置信度 `c_1` 到 `c_4`。硬件感知前缀调度器保留 `E F G`，丢弃低置信的 `H`。最后 target model 并行验证保留前缀；若 `E` 和 `F` 被接受而 `G` 被拒绝，target model 生成修正 token `G*` 完成本轮。

> 注解：这里的 anchor token 与 speculative decoding 中常说的 bonus token 在本文中可互换，指上一轮由 target model 生成的最后一个 token。

### 3.1 半自回归生成

纯并行 drafter 在一次 forward 中产生所有 draft logits，因此每个位置无法依赖同一 block 中其他位置实际采样出的 token。上下文有多个合理续写时，纯并行模型会对所有可能前驱做边缘化，而不是条件化在真实采样出的前缀上，于是容易发生多模态碰撞，接受率沿 block 快速衰减。

DSpark 将草稿生成拆为两阶段。

**并行阶段**：并行 backbone 一次 forward 覆盖整个 block，产生 hidden states `h_1, ..., h_gamma` 和 base logits `U_1, ..., U_gamma`。作者采用 DFlash 作为实现基础，并做一个小改动：原始 DFlash 输入为 anchor token 加 `gamma` 个 mask token，只预测 mask 位置；DSpark 把 anchor 本身也作为第一个预测位置，因此 `gamma` 个输入 token（anchor 加 `gamma - 1` 个 mask）就能产生 `gamma` 个 draft logits，减少 draft 计算并保持相近质量。

**顺序阶段**：顺序阶段为 base logits 增加前缀相关的 transition bias：

```text
B_k(x_0, x_<k, x_k)
```

这样第 `k` 个草稿位置就能依赖 block 内先前已经采样出的 token。顺序阶段通过自回归分解定义 causal block distribution：

```text
P(X | x_0) = product_{k=1}^{gamma} p_k(x_k | x_0, x_<k)

p_k(v | x_0, x_<k)
  = exp(U_k(v) + B_k(x_0, x_<k, v))
    / sum_{u in V} exp(U_k(u) + B_k(x_0, x_<k, u))
```

其中 `x_0` 是上一轮验证周期的 anchor token，`U_k` 是并行 backbone 在位置 `k` 输出的 base logit vector，`V` 是词表。推理时，顺序 block 从左到右按 `p_k` 采样。由于这个采样过程本身是顺序的，该模块必须非常轻，让整体 draft 延迟仍主要由并行阶段决定。

#### Markov head

最简单的实现是让 `B_k` 只依赖紧邻前一个 token，即一阶转移 `B(x_{k-1}, x_k)`。完整转移矩阵大小为 `V x V`，过大，因此用低秩分解近似：

```text
B = W_1 W_2
W_1 in R^{V x r}, W_2 in R^{r x V}
```

给定前一个 token `x_{k-1}`，第 `k` 个位置的转移 bias 为：

```text
B(x_{k-1}, .) = W_1[x_{k-1}] W_2
```

默认秩 `r = 256`。`W_1` 相当于 embedding lookup table，`W_2` 相当于 logit projection。回到 “of course” / “no problem” 的例子，一旦位置 1 采样出 “of”，Markov head 会提升 “course” 并压低 “problem”，从而减少跨模态拼接。

#### RNN head

Markov head 只能看到前一个 token。RNN head 进一步维护 recurrent state `s_k`，在 block 内累积完整前缀历史。每一步将上一状态、前一 token embedding 和 backbone hidden 拼接：

```text
z_k = [s_{k-1}; W_1[x_{k-1}]; h_k]
```

再做一次门控更新：

```text
s_k = sigma(W_g z_k) * s_{k-1}
      + (1 - sigma(W_g z_k)) * tanh(W_c z_k)

B_k(x_<k, .) = W_2^T tanh(W_o z_k)
```

`s_0` 初始化为 0。实验表明，RNN head 在更长 proposal length 下有一点额外收益，但部署复杂度更高；论文默认使用 Markov head。

> 注解：DSpark 的“半自回归”与 EAGLE / MTP 的关键区别在这里。EAGLE 和常见 MTP 路线的主要 draft 过程本身就是逐步的，长草稿会带来顺序 draft 延迟；DSpark 的重 backbone 只跑一次，顺序部分只是非常薄的 logit bias/状态更新，所以更接近“并行生成 + 顺序校正”。

### 3.2 置信度调度式验证

半自回归架构让 DSpark 能高效生成较长草稿块，但生成更多 draft token 并不必然带来端到端加速。无差别验证整段草稿，在高并发场景甚至会降低系统吞吐。

原因有两个：

- 数据侧：代码等结构化文本接受率高，开放式聊天接受率低。
- 系统侧：轻负载时额外验证一个 token 代价小；高负载时，每个不必要的验证 token 都会占用 target model batch 容量。

因此，长草稿块需要一个统一机制，只把 target compute 分配给预期收益为正的 token。DSpark 用 confidence head 预测前缀存活概率，再用 hardware-aware prefix scheduler 根据当前负载决定验证长度。

#### 3.2.1 Confidence Head

confidence head 为每个 draft 位置 `k` 输出标量估计：

```text
c_k in (0, 1)
```

`c_k` 表示条件概率：在该 block 中所有前序 token 都已被接受的前提下，第 `k` 个草稿 token 能通过 target verification 的概率。

模型结构是轻量线性投影加 sigmoid：

```text
c_k = sigma(w^T [h_k; W_1[x_{k-1}]])
```

其中 `h_k` 是 backbone hidden state，`W_1[x_{k-1}]` 是前一 draft token 的 Markov embedding。

训练监督使用每一步的解析接受率 `c*_k`。它由 draft 分布 `p^d_k` 与 target 分布 `p^t_k` 的 total variation distance 决定：

```text
c*_k = 1 - 1/2 * ||p^d_k - p^t_k||_1
```

#### 后验校准：Sequential Temperature Scaling

硬件感知调度不仅需要置信度排序正确，还需要累计接受概率的绝对数值可靠。神经网络置信度常常过度自信，如果直接使用原始 confidence，会扭曲吞吐估计，导致调度次优。

DSpark 因此引入 Sequential Temperature Scaling（STS）。因为每个 `c_i` 是条件概率，前缀被接受的联合概率按链式法则分解为：

```text
product_{i <= k} c_i
```

STS 在 held-out validation set 上从左到右校准这个累计乘积：对每个位置 `k`，在保持前面位置已校准分数不变的情况下，用一维 grid search 找到最优 temperature scalar，以最小化 Expected Calibration Error（ECE）。温度缩放保持排序不变，只校正概率幅度，使其匹配经验接受率。

> 注解：很多 confidence-based speculative 方法只要“知道哪个 token 更可靠”就够了；DSpark 的 scheduler 需要估算 `tau * SPS(B)`，所以概率数值本身必须接近真实接受率。这就是 STS 在本文中的必要性。

#### 3.2.2 硬件感知前缀调度器

以一批 `R` 个活跃请求为例。对请求 `r`，confidence 序列为：

```text
c_{r,1}, ..., c_{r,gamma}
```

调度器为每个请求选择验证长度：

```text
l_r in {0, ..., gamma}
```

由于 speculative decoding 只接受连续前缀，第 `j` 个位置的存活概率是累计乘积：

```text
a_{r,j} = product_{i <= j} c_{r,i}
```

一次验证 step 中，送入 target model 的总 batch size（按 token 计）为：

```text
B = sum_{r=1}^R (1 + l_r)
```

预期成功接受 token 数为：

```text
tau = sum_{r=1}^R (1 + sum_{j=1}^{l_r} a_{r,j})
```

令 `SPS(B)` 表示给定 forward-pass batch size `B` 时引擎的 steps per second。这个 capacity curve 在引擎初始化时 profile 一次并保存为轻量 cost table。调度器目标是选择 `l_1, ..., l_R`，最大化系统级预期 token 吞吐：

```text
Theta = tau * SPS(B)
```

虽然看起来是组合搜索，但目标结构支持高效贪心。因为 `a_{r,j}` 随 `j` 单调不增，把所有候选前缀扩展 `(r, j)` 按 `a_{r,j}` 全局降序排序，就自然满足块内前缀依赖。如果总 batch size 固定，最优策略就是从全局池中挑选存活概率最高的 draft token。

算法 1 的思想如下：

```text
输入：
  活跃请求 r = 1..R
  每个请求的 confidence 序列 c_{r,1}..c_{r,gamma}
  预先 profile 的 SPS(B)

步骤：
1. 对每个请求计算前缀存活概率 a_{r,j} = product_{i<=j} c_{r,i}
2. 构造候选空间 E = {(r,j) | a_{r,j} > 0}，按 a_{r,j} 降序排序
3. 初始 l_r = 0，B = R，tau* = R
4. 初始最佳吞吐 Theta_best = R * SPS(R)
5. 沿排序后的候选逐个尝试纳入：
   - 把请求 r 的长度扩展到 j
   - B 加 1
   - tau* 加 a_{r,j}
   - 计算 Theta = tau* * SPS(B)
   - 若 Theta 更好，记录当前 l_r
   - 否则提前停止
6. 返回达到 Theta_best 的每请求前缀长度
```

静态 threshold 方法在单请求或隔离假设下有效，但在高并发生产环境中可能次优，因为一个 draft token 是否值得验证取决于当前系统负载。DSpark 将验证长度选择表述为全局吞吐最大化问题。

#### 无损性与 early stopping

无损 speculative decoding 要求 **non-anticipating property**：接纳决策不能依赖未来候选 token。由于 DSpark 的 confidence head 使用前一个已采样 token 的 Markov feature，计算下一位置的 `a_{r,k+1}` 需要知道当前候选 `x_{r,k}`。如果做回溯式全局搜索，就可能让 `x_{r,k}` 泄漏进第 `k` 步的接纳决策，产生 selection bias。

算法 1 因此加入 early stopping：一旦贪心搜索中吞吐不再提升（`Theta <= Theta_best`），立刻停止。这样截断决策只依赖已经处理到当前 step 的前缀信息，与未来 token 隔离，从而保持精确目标分布恢复。

> 注解：这是论文里很细但很重要的理论点。动态调度如果“看到了未来 token 后再决定要不要验证前面的 token”，就会偏向那些导向高置信后续的 token，输出分布会偏离 target model。DSpark 在理论算法里用 early stopping 保证因果性；在生产系统里则通过异步两步前预测形成因果屏障。

### 3.3 训练

训练时，作者从每个目标序列中随机采样多个 anchor position，构造长度为 `gamma` 的 token block 作为训练数据。target model 全程冻结；draft model 共享并冻结 target model 的 embedding 层和 LM head，只更新 backbone drafter、sequential block 和 confidence head。

训练目标由三部分组成：

- cross-entropy loss `L_ce`
- distribution-matching loss `L_tv`
- confidence loss `L_conf`

三者都按位置加权：

```text
w_k = exp(-(k - 1) / gamma)
```

这样更重视 block 中较早的位置，因为前缀验证中早期 token 对期望接受长度影响更大。

交叉熵损失训练 drafter 预测正确 next token：

```text
L_ce = - sum_{k=1}^{gamma} w_k log p^d_k(x*_k)
```

distribution-matching loss 惩罚 draft 与 target 分布之间的 total variation distance：

```text
L_tv = sum_{k=1}^{gamma} w_k ||p^d_k - p^t_k||_1
```

由于每步接受概率等于：

```text
1 - 1/2 * ||p^d - p^t||_1
```

最小化 `L_tv` 直接最大化期望接受率。

confidence loss 是 binary cross entropy，用于训练 confidence head 预测公式中的软接受标签 `c*_k`：

```text
L_conf = - sum_k w_k [c*_k log c_k + (1-c*_k) log(1-c_k)]
```

总目标为：

```text
L = alpha_ce L_ce + alpha_tv L_tv + alpha_conf L_conf
```

默认权重：

```text
alpha_ce = 0.1
alpha_tv = 0.9
alpha_conf = 1.0
```

## 4. 实验

本节用离线 benchmark 验证 DSpark 的 draft quality，并在第 5 节报告 confidence scheduler 在在线生产流量下的效果。

### 4.1 实验设置

**Target 与 draft models**：作者在四个 target model 上评估 DSpark：Qwen3-4B、Qwen3-8B、Qwen3-14B、Gemma4-12B。draft baseline 包括：

- DFlash：先进并行 drafter。
- Eagle3：基于 Training-Time Test 的自回归 drafter。

为公平比较，所有 drafter 使用同一训练框架和同一数据重新训练。Eagle3 的 TTT horizon 设为 7，与 DFlash 和 DSpark 的 block size 7 对齐。所有 drafter 使用相同目标模型特征层。draft model 层数方面，Eagle3 用 1 层，DSpark 和 DFlash 用 5 层。除非特别说明，DSpark 指 Markov-head 版本。

**训练数据**：使用 Open-PerfectBlend，共 130 万样本，包含 chat、math、code、instruction-following 数据。作者只使用 prompt，response 由每个 target model 按推荐采样参数重新生成。每个 drafter 训练 10 个 epoch。数据生成和评估采用 non-thinking mode。

**评估协议**：评估三个领域：

- 数学推理：GSM8K、MATH500、AIME25。
- 代码生成：MBPP、HumanEval、Live-CodeBench。
- 日常聊天：MT-Bench、Alpaca、Arena-Hard。

所有 benchmark 使用标准 speculative decoding，sampling temperature 为 1.0。报告每轮 accepted length `tau`，且指标包含 target-generated bonus token。

### 4.2 主实验结果

离线评估禁用 confidence scheduler，强制所有 drafter 提出固定长度 token，以隔离原始 draft quality。表 1 报告每轮平均 accepted length，越高越好。

| Target | Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | Eagle3 | 5.14 | 4.62 | 3.92 | 3.69 | 4.16 | 3.77 | 2.39 | 2.26 | 2.55 |
| Qwen3-4B | DFlash | 5.40 | 4.85 | 4.15 | 4.40 | 4.74 | 4.18 | 3.07 | 2.96 | 2.83 |
| Qwen3-4B | DSpark | 6.11 | 5.70 | 4.89 | 5.13 | 5.38 | 4.86 | 3.64 | 3.54 | 3.29 |
| Qwen3-8B | Eagle3 | 5.30 | 4.77 | 3.91 | 3.96 | 4.33 | 4.17 | 2.66 | 2.54 | 2.54 |
| Qwen3-8B | DFlash | 5.33 | 4.91 | 4.07 | 4.36 | 4.64 | 4.39 | 3.11 | 2.98 | 2.81 |
| Qwen3-8B | DSpark | 6.17 | 5.78 | 5.01 | 5.16 | 5.52 | 5.17 | 3.72 | 3.58 | 3.21 |
| Qwen3-14B | Eagle3 | 5.24 | 4.60 | 3.71 | 3.81 | 4.14 | 4.01 | 2.62 | 2.47 | 2.48 |
| Qwen3-14B | DFlash | 5.41 | 4.84 | 3.98 | 4.44 | 4.59 | 4.33 | 3.10 | 2.94 | 2.72 |
| Qwen3-14B | DSpark | 6.21 | 5.74 | 4.94 | 5.26 | 5.43 | 5.02 | 3.70 | 3.58 | 3.13 |
| Gemma4-12B | Eagle3 | 5.87 | 5.46 | 4.83 | 4.72 | 5.37 | 4.16 | 3.19 | 3.06 | 2.72 |
| Gemma4-12B | DFlash | 5.45 | 5.04 | 4.22 | 4.39 | 4.95 | 3.70 | 2.98 | 2.84 | 2.59 |
| Gemma4-12B | DSpark | 6.05 | 5.78 | 5.12 | 5.11 | 5.64 | 4.51 | 3.49 | 3.35 | 2.92 |

DSpark 在所有 target model 和 benchmark domain 上都超过 Eagle3 与 DFlash。在 Qwen3-4B、8B、14B 上，DSpark 相比 Eagle3 的 macro-average accepted length 分别提升 30.9%、26.7%、30.0%；相比 DFlash 分别提升 16.3%、18.4%、18.3%。Gemma4-12B 上也有稳定收益，说明优势能跨模型家族泛化。

表中也显示明显 domain effect：结构化任务接受长度天然更高，开放式聊天更低。这说明固定 verification length 常会把算力浪费在很可能被拒绝的尾部 token 上，直接支持了 confidence-scheduled verification 的动机。

### 4.3 实验分析

#### 4.3.1 为什么并行生成能超过自回归？

表 1 有一个反直觉现象：并行 DFlash 和半自回归 DSpark 常常比完全自回归 Eagle3 获得更长 accepted length。通常我们会预期逐步自回归模型生成质量更高。作者用 position-wise conditional acceptance 分析原因。

对于 draft position `k`，只统计前面 `1..k-1` 都已被 target model 成功验证并接受的实例，然后计算位置 `k` 也被接受的比例。这样可以去掉早期前缀错误的惩罚，单独观察每个位置的预测质量。

**位置 1 的容量优势**：第一个 draft 位置上，自回归和并行模型都只基于 target context 预测 next token。差异来自模型容量：Eagle3 这类自回归模型由于 `O(gamma)` 延迟限制，通常较浅；`O(1)` 并行 drafter 则能用更深网络。DFlash 在第一个位置明显高于 Eagle3，例如 Math 上 0.88 对 0.81，Chat 上 0.72 对 0.53。由于 speculative decoding 是严格前缀存活过程，第一个 token 权重最大，位置 1 拒绝会使整段 block 作废，因此并行 drafter 的首 token 容量优势会显著提升全局 accepted length。

**后续位置的独立性限制**：位置 2 到 7 暴露了纯并行生成的弱点。前面的 token 确定具体语义路径后，后续 token 本应更容易预测。Eagle3 能利用这种条件确定性，因此接受率稳定甚至上升；DFlash 因每个位置对所有可能前缀做边缘化，后缀接受率快速下降，形成 suffix decay。

**DSpark 的折中**：DSpark 的目标正是结合两者：用深并行 backbone 保留第一个 token 的高容量优势，用轻量 sequential head 为后续 token 注入依赖，缓解纯并行的后缀衰减。实验曲线显示 DSpark 在整段 block 中保持较高且稳定的 conditional acceptance。

> 注解：这解释了 DSpark 为什么不是“自回归更强”或“并行更强”的单选题。它把大算力花在并行 backbone 上，把块内依赖交给便宜的 Markov/RNN head，是一种性价比导向的混合设计。

#### 4.3.2 少量自回归就足够

作者从 drafter depth 和 proposal length 两个维度探索 DSpark。

**Drafter depth**：固定 block size 为 7，将 DSpark 层数从 1 增加到 5，并与 5 层 DFlash 比较。DSpark 随深度单调提升，且从 1 层到 2 层的边际收益最大。值得注意的是，2 层 DSpark 在所有领域都超过 5 层 DFlash，说明轻量局部自回归在参数效率上优于单纯堆叠更深并行层。

**Proposal length**：固定 drafter depth 为 5，将 draft length 扩展到 `{4, 8, 12, 16}`。DSpark 在每个 proposal length 上都超过 DFlash，且 block 越长，优势越大。原因是纯并行 DFlash 后缀接受率快速衰减，长 block 的边际收益变小；DSpark 缓解衰减后，在长 block 上相对收益更高。例如 `gamma=7` 时，DSpark 相比 DFlash 在 math、code、chat 上分别提升 16%、15%、18%；`gamma=15` 时提升扩大到 30%、26%、22%。

**延迟开销**：作者在 batch size 128 下测量每轮 engine latency，包括一次 target verification、并行 draft block forward 和 serial sampling loop。由于该 batch size 下 target model verification 占主导，sequential block 的延迟开销很小。draft length 从 4 扩到 16，相比 DFlash 只增加 0.2% 到 1.3% 的整轮延迟，却带来最高 30% 的 accepted length 提升。

#### 4.3.3 验证要更聪明，而不是更长

即使 DSpark 在长草稿上保持较高接受率，验证完整 proposal 仍可能低效，尤其在开放聊天中尾部 token 拒绝风险高。作者用 Qwen3-4B 做离线 threshold sweep 来单独验证 confidence head。

随着 confidence threshold 升高，整体 acceptance rate 稳步上升，因为 estimator 能过滤掉最终会被拒绝的 token。这个剪枝在 chat workload 上最明显：acceptance rate 从 45.7% 提升到 95.7%，同时拒绝 token 显著减少。结构化任务剪枝更温和，Math 从 76.9% 提升到 92.5%，Code 从 67.6% 提升到 92.0%。

静态 threshold 只能用于诊断，在动态 serving 中不是最优，因为它忽略系统负载：低并发时验证低置信 token 机会成本小，高并发时则会浪费关键 batch 容量。硬件感知 scheduler 因此需要 confidence model 既有区分能力，也有精确校准。可靠性图显示，原始模型 ROC-AUC 约 0.81 到 0.90，区分能力强，但 ECE 为 3% 到 8%，偏过度自信。应用 STS 后平均 ECE 降到约 1%，得到更可靠的前缀存活估计。

## 5. DSpark 的真实部署

离线实验证明了 DSpark 的算法收益，但部署到 DeepSeek-V4 这样的规模模型时，还需要解决训练和推理系统挑战。本节介绍端到端生产流水线，包括可扩展训练、硬件感知 scheduler 的工程改造，以及真实用户流量下的端到端性能。

### 5.1 可扩展且灵活的训练

DSpark draft models 与 DeepSeek-V4-Flash preview 和 DeepSeek-V4-Pro preview 共同部署。并行 backbone 由三层 MoE layer 组成，使用 mHC 和窗口大小为 128 的 sliding window attention。最大 block size 设置为 `gamma = 5`，顺序建模使用 Markov head。confidence head 与 draft model 端到端训练，然后通过 STS 校准。

训练 draft model 需要 target model 的输出分布作为监督。若在完整文档上下文上同时评估两个模型，会产生巨大显存占用和跨 worker 通信开销。作者在内部训练框架 HAI-LLM 中做了两项系统优化：

**Hidden state communication**：跨并行 worker 传输 target model 的全词表 logits 会造成显著带宽瓶颈，因为 `V` 约为 `10^5`。作者改为临时缓存 target model forward activation，只通信 LM head 前的 hidden states。LM head projection 在 draft model worker 本地、只针对采样的目标位置执行。这样每 token 通信复杂度降为 `O(d)`，其中 `d` 是 hidden dimension。

**Anchor-bounded sequence packing**：为了解耦 draft model 计算成本与 target model 上下文长度，作者从训练序列中采样固定数量 draft anchors，并把这些孤立预测 block 打包成 dense training batches。packing 通过 token-level attention indices 管理，而不是标准 2D mask。这样可以在多个独立序列和 anchor 上保持精确 causal masking，同时避免标准 padding 的计算和显存开销。

### 5.2 生产中的硬件感知前缀调度器

算法 1 在理论上无损且合理，但直接部署到生产环境会遇到两个现实冲突：

1. 算法假设 capacity curve 平滑且单峰，而真实硬件的 `SPS(B)` 是离散、锯齿状、阶梯下降的。
2. 算法每 step 都要动态调度 draft token，这与连续 CUDA graph replay 和 Zero-Overhead Scheduling（ZOS）冲突。

为了兼顾系统兼容性、吞吐和算法正确性，作者把 scheduler 改造成异步运行。因为 ZOS 要求下一步 batch size 在当前 step 完成前就已知，同步调度会让 GPU pipeline 停顿。DSpark 用两步之前 confidence head 的输出近似即将到来的 verification capacity。当前 step 的候选 token 仍按最新累计 confidence 严格排序；两步前的历史预测只用于决定动态截断长度，也就是 batch capacity limit `K`。这相当于动态 top-K 选择。

这个异步设计隐藏了调度延迟，并自然形成因果屏障。理论算法中，为避免锯齿状 SPS 造成局部最小值，生产实现移除了 early-stopping break，允许无约束全局搜索。通常这会有回溯搜索泄漏未来 token 的风险，但在异步实现中，无约束搜索只评估两步前的历史预测，当前 token `x_{r,k}` 的实现不会影响当前接纳决策，因此仍保持目标分布精确。

> 注解：这部分是 DSpark 工程贡献的核心。理论算法用 early stopping 保证无损，生产系统为了适配锯齿硬件曲线和 CUDA graph/ZOS，把“容量决定”延后/前移到历史预测，让全局 top-K 搜索不再偷看当前 token。它不是单纯改 threshold，而是把 speculative decoding 当成 serving 调度问题。

### 5.3 高吞吐、低延迟推理

生产 serving 同时优化两个目标：

- 单请求延迟 / 单用户生成速度。
- 聚合吞吐 / 并发服务能力。

投机解码不可避免会产生浪费验证计算，因此它本质上是在用额外系统计算换取单请求更快生成。

在作者部署场景中，每 step 请求数经常受资源上限限制，例如每请求固定 KV-cache 容量，以及可用用户流量池。因此有效 batch size 长期低于 GPU compute-saturating threshold。在这种 regime 下，传统 latency-throughput trade-off 简化为：给定固定并发上限，最大化每 GPU token 吞吐和最大化单用户 token/s 高度相关。

异步 scheduler 会把空闲算力路由给最有希望的 draft token。难点在于物理执行层必须高效支持同一 batch 中不同请求的 variable-length queries。标准 decode kernels 通常针对固定 query length 优化，直接 padding 会造成 GPU 利用率下降。

DSpark 的解决方式是解耦物理执行与逻辑序列追踪：kernel 中将不同请求的 token flatten 成独立元素统一处理；序列内部复杂依赖通过 marker tensor 传给 sparse attention 实现。在 DeepSeek-V4 架构上，只需要修改 index-attention 和 compress kernels，就能支持 variable-length routing，而不会引入底层执行开销。

### 5.4 真实用户流量下的性能

作者用 DSpark-5（最大 draft length `gamma=5`）对比生产 serving engines 中的 MTP-1 baseline，目标模型为 DeepSeek-V4-Flash preview 和 DeepSeek-V4-Pro preview。MTP-1 是此前生产配置，在 DeepSeek-V4-preview 发布两周后被 DSpark 替代。生产中长期保留单 token MTP 的原因是，静态多 token drafter（如 MTP-3/5）在高并发下会因过多验证开销严格降低聚合吞吐。

**Serving Pareto Frontier**：图 7 展示聚合系统吞吐与单用户生成速度之间的权衡。V4-Flash 在 80 tok/s/user SLA 下，DSpark 相比 MTP-1 聚合吞吐提升 51%。在更严格的 120 tok/s/user SLA 下，MTP-1 已接近运行边界，只能支持很小并发 batch；DSpark 名义吞吐提升达到 661%。作者强调，这个高 SLA 点主要说明 DSpark 扩展了可行交互性边界，而不是代表常规稳态下的倍数加速。在匹配实际吞吐水平时，DSpark 将单用户生成速度提升 60% 到 85%。

V4-Pro 呈现同样模式。35 tok/s/user SLA 下，DSpark 聚合吞吐提升 52%；50 tok/s/user SLA 下，MTP-1 进入低并发区域，DSpark 名义吞吐优势为 406%。匹配系统容量时，DSpark 带来 57% 到 78% 的单用户速度提升。

**负载下的吞吐动态**：图 8 分析了收益机制。中等并发时，硬件感知 scheduler 利用可用 target compute，把每请求 verification budget 从 MTP-1 的静态 2 token 扩展到约 4 到 6 token，从而每次 forward 接受更多 token。随着并发升高并接近 target capacity 饱和，scheduler 会动态收缩预算，平均验证长度随负载平滑下降，在低置信 token 消耗关键 batch 容量之前剪掉它们。

**局限性**：即便 prefix scheduler 最小化 target verification 浪费，DSpark 仍需要固定 draft-side 成本，用并行 backbone 生成初始 `gamma` token block。对天然低接受率的复杂请求，这部分 upfront drafting compute 无法回收。未来可以引入 difficulty-aware early exiting，让这类请求跳过完整 block generation。

> 注解：这解释了为什么 DSpark 不只是“更多 token 更快”。它在轻载时吃满空闲算力，在重载时主动缩短验证前缀；而静态 MTP-N 在重载时无法自我收缩，容易把吞吐打下去。

## 6. 相关工作

### 投机解码算法

投机解码通过解耦 token proposal 与 verification 加速自回归生成。早期 blockwise 方法之后，现代 speculative decoding 使用拒绝采样精确保持 target model 分布。由于速度提升直接取决于 drafter 的效率和准确度，大量研究集中在优化 drafter 架构。

除了独立小语言模型，后续工作也把 multi-token heads 或 feature extrapolators 直接集成到 target model 中，例如 MTP、Medusa、EAGLE 系列等。其他策略包括早退式 self-speculation、动态词表压缩、prompt lookup、suffix automata 和 retrieval。为去除 draft 本身的顺序瓶颈，近期方法提出并行或 blockwise generation：P-EAGLE 并行化 EAGLE-style drafting，PARD、DART、DFlash 使用 diffusion-inspired prediction 一次 forward 生成整个 block，DDTree 再扩展成可验证 draft tree。也有并发工作改进 DFlash，例如 Domino 的 CausalEncoder 与 DSpark 的 RNN Head 概念相近，DFlare 通过 layer-wise fusion 处理 conditioning bottleneck。

### 系统感知投机解码调度

除了 drafter 架构，另一条工作线关注每轮应生成或验证多少 speculative token。已有方法会用 confidence heuristics、learned acceptance predictors 或 bandit-style policy 动态调整 draft length。更系统导向的工作则把 speculative decoding 视为 serving 调度问题，根据实时系统负载和请求优先级调整 speculation budget，以优化 goodput 和 latency。

DSpark 属于这条系统化路线，但它把 calibrated prefix survival probability、硬件 profile `SPS(B)`、因果无损调度和生产异步 top-K 结合在同一框架中。

### 并行生成

并行生成模型的 decoding latency 近似不依赖输出长度，因此是自回归 decoding 的有吸引力替代。Non-Autoregressive Transformers 最早推动了这个方向，但一次性独立预测所有位置会迫使模型平均多个合理模式，产生混合不同有效序列片段的输出。

已有两类解决方式：

- 保留 single-pass 架构，但改变模型可见信息或训练方式，例如引入 latent variables，或调整训练目标让模型关注单个连贯输出。
- 重新引入有限顺序依赖，例如 iterative re-prediction、block-level autoregression，或 CRF、CTC、HMM、PCFG 等结构化输出层。

speculative decoding 还额外要求 drafter 提供精确 per-token probabilities 以支持 rejection sampling。许多并行生成技术由于 iterative refinement、latent marginalization 或 global normalization，无法直接提供这种概率。例如 CRF-NAT 也在并行 hidden states 上叠顺序模块，但全局归一化 partition function 阻碍了精确 per-token probability 计算；CTC-drafter 由于 alignment paths 的 latent marginalization，被限制为 greedy verification。DSpark 通过保持顺序校正的局部性，使每个 token 概率仍是精确 softmax 计算。

> 注解：这是 DSpark 架构边界的另一个关键点。它不能随意加一个复杂全局结构层，因为 speculative decoding 需要知道 draft token 的概率来做无损拒绝采样。Markov/RNN head 是“局部修正”，不会破坏每步 softmax 概率。

## 7. 结论

本文提出 DSpark，一个面向高并发生产环境的 speculative decoding 框架，用于克服 LLM 推理中的结构性和系统性瓶颈。

算法上，DSpark 引入半自回归生成范式：计算重的并行 backbone 与轻量 sequential head 结合，缓解独立并行 drafter 的后缀接受率快速衰减。系统上，DSpark 把验证长度选择表述为全局吞吐最大化问题，使用硬件感知前缀调度器，根据校准后的存活概率和实时引擎负载动态分配 target model 验证预算。

离线评估表明，DSpark 在多个领域显著超过先进自回归和并行 baseline。DeepSeek-V4 的真实生产部署验证了其实用价值：通过智能管理验证开销，DSpark 能在重载下保持稳健并发能力，持续提升单用户生成速度，并将 LLM serving 的 Pareto frontier 向外推进。

## 附录 A：没有 early-stopping 时的选择偏差反例

作者给出一个简单反例，说明如果 scheduler 做离线全局搜索，即不使用算法 1 中的 break condition，会违反无损 speculative decoding 所需的 non-anticipating property。

形式上，第 `k` 个 draft token 是否被纳入验证前缀，必须由采样该 token 之前 scheduler 可见的信息决定，不能依赖 `x_{r,k}` 本身的实际取值。

考虑单请求 `R=1`，最大 draft length `gamma=2`。假设第一个位置的预 token 置信度为：

```text
a_1 = 0.8
```

profile 出的 capacity curve 为：

```text
SPS(1)=1.0, SPS(2)=0.5, SPS(3)=0.45
```

验证 0 和 1 个 draft token 的期望吞吐为：

```text
Theta_0 = 1 * SPS(1) = 1.0
Theta_1 = (1 + 0.8) * SPS(2) = 0.9
```

如果没有 early-stopping，scheduler 会继续评估 `Theta_2`。由于 Markov confidence head 使用前一个采样 token，下一置信度 `c_2` 明确依赖 `x_1` 的实现，因此第二前缀存活概率：

```text
a_2 = a_1 c_2
```

也依赖 `x_1`。

两种可能情况：

- 若 `x_1` 导致高 `c_2=0.9`，则 `a_2=0.72`，`Theta_2=(1+0.8+0.72)*0.45=1.134`，全局最大为 `Theta_2`，scheduler 返回 `l=2`，第一个 token 被纳入验证前缀。
- 若 `x_1` 导致低 `c_2=0`，则 `a_2=0`，`Theta_2=(1+0.8+0)*0.45=0.81`，全局最大仍为 `Theta_0=1.0`，scheduler 返回 `l=0`，第一个 token 不被纳入验证前缀。

于是，第一个 draft token 是否被接纳，动态依赖第一个 draft token 自己的值。这种回溯依赖会产生 selection bias：scheduler 偏好那些导向高置信后续的 token，尽管 `x_1` 的接纳决策本应在观察 `x_1` 前就决定。

作者进一步用二元词表 `{A, B}` 展示分布偏差。设第一个位置：

```text
p_t(A)=0.7, p_t(B)=0.3
p_d(A)=0.5, p_d(B)=0.5
```

标准 speculative acceptance probability 为：

```text
min(0.7,0.5) + min(0.3,0.5) = 0.8
```

假设回溯 scheduler 的行为如上：`x_1=A` 导致高后续置信度并返回 `l=2`；`x_1=B` 导致低后续置信度并返回 `l=0`。若 `x_1=A`，draft token 被纳入并以概率 1 接受，因为：

```text
min(1, p_t(A)/p_d(A)) = min(1, 0.7/0.5) = 1
```

若 `x_1=B`，draft token 不被纳入，target model 从 `p_t` 重新生成 fresh token。因此输出为 `A` 的概率为：

```text
Pr(Y=A) = Pr(x_1=A)*1 + Pr(x_1=B)*p_t(A)
        = 0.5 + 0.5*0.7
        = 0.85
```

输出分布变成 `(0.85, 0.15)`，不同于目标分布 `(0.7, 0.3)`，证明回溯 scheduler 不是无损的。

early-stopping 可以阻止这个问题：由于 `Theta_1 < Theta_0`，scheduler 在评估任何依赖后续 token 的量（如 `c_2`）之前立即停止并返回 `l=0`。因此第一个位置的接纳决策只依赖 pre-token information，不会被 `x_1` 的实现污染，从而恢复无损性所需的 non-anticipating property。

## 额外译注：从实现视角看 DSpark 与 MTP / EAGLE / EAGLE3

### 与 MTP

MTP 的典型目标是在模型训练或模型结构中加入未来多个 token 的预测能力。DeepSeek-V3 技术报告中的 MTP 模块会维护完整 causal chain，用于提升训练信号，也可以用于 speculative decoding。生产中的 MTP-1 则是只预测/验证一个额外 token 的稳健配置。

DSpark 与 MTP 的主要区别：

- MTP 更像“目标模型旁路上的未来 token 预测层”，DSpark 更像“独立 draft framework + serving scheduler”。
- MTP-N 的验证长度通常是静态的，负载升高时会把很多低收益 token 塞进 target verification；DSpark 动态决定每个请求的验证长度。
- MTP 的多步预测若按深度/位置顺序展开，draft 成本会随 token 数增加；DSpark 的重计算是并行 backbone 一次 forward，顺序 Markov/RNN head 很薄。
- DSpark 额外训练 confidence head，并用 STS 校准前缀存活概率，用于系统级吞吐最大化。

### 与 EAGLE

EAGLE 的核心思想是让轻量 autoregressive head 在 feature level 外推目标模型的下一步特征，再用目标 LM head 得到 token。它比小模型 drafter 更贴近 target model，但草稿生成仍是逐步进行。

DSpark 与 EAGLE 的主要区别：

- EAGLE 的依赖建模天然来自自回归逐步生成；DSpark 只让轻量 head 顺序化，主体计算并行。
- EAGLE 更关注 draft quality 与 target feature 对齐；DSpark 同时关注 draft quality 和 target verification budget。
- EAGLE 生成长草稿时 draft latency 随步数增长；DSpark 更适合长 block，因为 backbone forward 次数不随 block 长度增长。

### 与 EAGLE3

EAGLE3 相比 EAGLE 进一步加强 draft model：它弱化/放弃直接预测下一层 feature 的路线，更多利用目标模型多层特征和 Training-Time Test 来改善 token 预测质量。论文把 Eagle3 作为强自回归 baseline。

DSpark 与 EAGLE3 的主要区别：

- Eagle3 是强自回归 drafter，优势在每步条件化质量；DSpark 是半自回归并行 drafter，优势在首 token 容量、长 block 成本和系统调度。
- Eagle3 的生成路径仍受顺序 draft latency 限制；DSpark 的重 backbone 并行，Markov head 成本极低。
- DSpark 的在线收益主要来自“动态验证预算”，这不是 Eagle3 的核心贡献点。

### 与 DFlash

DSpark 可以看作站在 DFlash 之上的扩展：

- 继承 DFlash 的并行 block backbone 和 target feature injection。
- 修改输入预测方式，减少 draft 计算。
- 增加 Markov/RNN sequential head 缓解 suffix decay。
- 增加 confidence head + STS + hardware-aware scheduler，解决 serving 中“该验证多少”的问题。

## 关键参考链接

- DSpark / DeepSpec 官方仓库：https://github.com/deepseek-ai/DeepSpec
- DeepSeek-V3 Technical Report（MTP 相关）：https://arxiv.org/abs/2412.19437
- Better & Faster Large Language Models via Multi-token Prediction：https://proceedings.mlr.press/v235/gloeckle24a.html
- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty：https://arxiv.org/abs/2401.15077
- EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test：https://arxiv.org/abs/2503.01840

