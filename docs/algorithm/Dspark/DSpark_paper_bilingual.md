# DSpark 英中对照逐段译文

> 本文档是 `DSpark_paper.pdf` 的英中对照阅读版。  
> 已保留原有 `DSpark_paper_zh_annotated.md` 和 `DSpark_paper_zh_detailed.md` 不变。  
> 排版规则：先给出原始英文段落，再给出对应中文翻译；中文段落末尾偶尔加入“译注”，帮助理解术语和算法意图。  
> 覆盖范围：论文标题、摘要、正文第 1 到第 7 节、附录 A。参考文献列表主要是书目信息，未逐条翻译。

## Title

**English**

> DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

**中文翻译**

DSpark：结合半自回归生成的置信度调度式投机解码。  
译注：标题里的两个关键词是 `Confidence-Scheduled` 和 `Semi-Autoregressive`。前者对应“验证多长由置信度和系统负载决定”，后者对应“草稿生成主体并行，但输出处带一点顺序依赖”。

---

## Abstract

**English**

> Speculative decoding accelerates Large Language Model (LLM) inference by decoupling draft generation from target verification. While recent parallel drafters efficiently propose long token sequences in a single forward pass, they suffer from rapid acceptance decay due to a lack of inter-token dependencies. Furthermore, indiscriminately verifying these extended blocks wastes critical batch capacity on tokens with high rejection risks, severely degrading throughput in high-concurrency serving systems.

**中文翻译**

投机解码通过将草稿生成与目标模型验证解耦，来加速大语言模型推理。近期的并行草稿模型可以在一次前向传播中高效提出较长的 token 序列，但由于缺少 token 之间的依赖关系，它们会出现接受率快速衰减的问题。此外，如果不加区分地验证这些扩展后的草稿块，就会把关键 batch 容量浪费在高拒绝风险的 token 上，在高并发 serving 系统中严重降低吞吐。  
译注：这里的“接受率衰减”主要指越靠后的草稿 token 越容易被 target model 拒绝；“batch 容量”指一次 target forward 能并行处理的 token 预算。

补充辨析：这句话里的“并行草稿模型”主要指 DFlash、PARD、DART 这类一次 forward 预测整段 block 的 drafter，而不是 EAGLE/EAGLE3。EAGLE/EAGLE3 在本文实验中被归为自回归 drafter：后一个草稿 token 会依赖前面已经采样出的草稿 token，因此块内依赖更强，但草稿长度增加时 draft latency 也更容易增长。MTP 这类多 token 预测在很多实现中也更接近保留因果链/逐步预测的一侧，所以“token 之间有依赖但长草稿或固定多 token 验证成本较高”这个理解大体是对的；不过 DSpark 论文中与“并行缺少依赖、后缀衰减”直接对照的主要对象是 DFlash，而不是 EAGLE/EAGLE3。

**English**

> We introduce DSpark, a speculative decoding framework that unifies high-throughput parallel generation with adaptive, load-aware verification. To maintain draft quality, DSpark utilizes a semi-autoregressive architecture—coupling a parallel backbone with a lightweight sequential module—to introduce intra-block dependency modeling and mitigate suffix decay. To optimize system efficiency, DSpark employs confidence-scheduled verification, dynamically tailoring the verification length for each request based on estimated prefix survival probabilities and engine-specific throughput profiles.

**中文翻译**

我们提出 DSpark，这是一个将高吞吐并行生成与自适应、负载感知验证统一起来的投机解码框架。为了保持草稿质量，DSpark 使用半自回归架构，将并行 backbone 与轻量级顺序模块结合起来，引入块内依赖建模并缓解后缀衰减。为了优化系统效率，DSpark 使用置信度调度式验证，根据估计出的前缀存活概率和特定推理引擎的吞吐曲线，为每个请求动态调整验证长度。  
译注：这段几乎就是 DSpark 的完整方法概括：一半是模型结构，一半是系统调度。

**English**

> On offline benchmarks across diverse domains, DSpark substantially improves the accepted length over state-of-the-art autoregressive and parallel drafters. When deployed within the DeepSeek-V4 serving system under live user traffic, DSpark successfully mitigates verification waste. Compared to the established production baseline (MTP-1), DSpark accelerates per-user generation speeds by 60%–85% at matched throughput levels. More importantly, by preventing severe throughput degradation under strict interactivity constraints, it enables performance tiers that were previously unattainable, shifting the Pareto frontier of our serving system. To facilitate community progress, we open-source the DSpark checkpoints alongside DeepSpec, an algorithm-driven training repository for speculative decoding.

**中文翻译**

在覆盖多个领域的离线 benchmark 上，DSpark 相比当前先进的自回归草稿模型和并行草稿模型，都显著提升了 accepted length。在 DeepSeek-V4 serving 系统中承载真实用户流量时，DSpark 成功减少了验证浪费。相比已经建立的生产基线 MTP-1，在匹配吞吐水平下，DSpark 将单用户生成速度提升了 60% 到 85%。更重要的是，在严格交互性约束下，DSpark 能防止严重吞吐退化，从而实现过去不可达到的性能档位，推动 serving 系统的 Pareto frontier 外移。为了促进社区发展，作者开源了 DSpark checkpoints，同时也开源了 DeepSpec，这是一个面向 speculative decoding 的算法驱动训练仓库。  
译注：`matched throughput levels` 表示在总吞吐相近时比较单用户速度；`Pareto frontier` 表示“总吞吐”和“单用户速度”两者之间能达到的最好边界。

---

## 1. Introduction

**English**

> Large Language Models (LLMs) generate text autoregressively: each new token requires a full forward pass conditioned on all preceding tokens, making inference latency proportional to the output length. The resulting low GPU utilization and high user-perceived waiting time constitute a primary bottleneck in production LLM serving, particularly for latency-sensitive scenarios such as real-time conversational assistants and multi-turn agentic workflows.

**中文翻译**

大语言模型以自回归方式生成文本：每个新 token 都需要在所有先前 token 的条件下进行一次完整前向传播，因此推理延迟与输出长度成正比。由此带来的 GPU 利用率低和用户感知等待时间长，构成了生产级 LLM serving 的主要瓶颈，尤其是在实时对话助手、多轮 agent 工作流等对延迟敏感的场景中。  
译注：自回归生成的核心问题是“串行依赖”：下一个 token 必须等上一个 token 确定后才能生成。

**English**

> Speculative decoding offers a principled solution: a lightweight draft model proposes a block of candidate tokens, and the full-size target model verifies the entire block in a single forward pass via rejection sampling, accepting the longest prefix consistent with the target distribution and appending one bonus token. Because verification is parallel and the acceptance rule preserves the target distribution exactly, speculative decoding accelerates generation without any quality loss.

**中文翻译**

投机解码提供了一种原则性的解决方案：一个轻量级 draft model 先提出一块候选 token，完整尺寸的 target model 再通过一次前向传播和拒绝采样验证整个块，接受与目标分布一致的最长前缀，并追加一个 bonus token。由于验证过程是并行的，并且接受规则可以精确保持目标分布，投机解码能够在不损失质量的前提下加速生成。  
译注：这里的“无质量损失”不是经验上的近似，而是指采样分布理论上仍等于 target model。

**English**

> The design of the draft model governs the trade-off between drafting latency and acceptance rate. Early drafters are autoregressive, conditioning each position on previously sampled tokens. However, their drafting latency grows linearly with the block size, forcing these methods to use short blocks and shallow architectures.

**中文翻译**

draft model 的设计决定了草稿生成延迟与接受率之间的权衡。早期草稿模型是自回归式的，每个位置都以之前已经采样出的 token 为条件。然而，它们的草稿生成延迟会随着 block size 线性增长，因此这些方法不得不使用较短的草稿块和较浅的架构。  
译注：EAGLE、EAGLE3 这类强草稿模型虽然质量高，但长草稿会受到顺序生成成本限制。

**English**

> To break this sequential bottleneck, parallel drafters have emerged as a compelling alternative: all draft positions are produced in a single forward pass, making drafting latency nearly independent of block size. This structural advantage theoretically allows parallel drafters to efficiently generate substantially longer draft blocks.

**中文翻译**

为了打破这个顺序瓶颈，并行草稿模型成为一种有吸引力的替代方案：所有草稿位置都在一次前向传播中产生，使草稿生成延迟几乎不依赖 block size。这种结构优势理论上允许并行草稿模型高效生成明显更长的草稿块。  
译注：DFlash 就属于这种路线，优点是“一次 forward 猜一串”。

**English**

> However, fully unlocking the potential of large parallel draft blocks introduces two critical bottlenecks—one in generation quality, and the other in system efficiency. First, because parallel drafters predict each position independently, they cannot model inter-token dependencies within a block. This independence leads to multi-modal collisions and rapid acceptance decay at later positions.

**中文翻译**

然而，要完全释放大并行草稿块的潜力，会引入两个关键瓶颈：一个在生成质量上，另一个在系统效率上。首先，由于并行草稿模型独立预测每个位置，它们无法建模一个块内部 token 之间的依赖。这种独立性会导致多模态碰撞，并使靠后位置的接受率快速下降。  
译注：“多模态碰撞”可以理解为多个合理续写路径被混在一起，例如 “of course” 和 “no problem” 被混成 “of problem”。

**English**

> Second, determining the optimal verification length remains a challenge. While parallel generation easily produces long draft blocks, indiscriminately verifying all proposed tokens degrades system throughput, particularly under high-concurrency workloads. The ideal verification length varies along two axes. On the data side, structured requests like code naturally sustain higher acceptance rates than open-ended chat. On the system side, verifying extra tokens is nearly free under light loads. Under heavy loads, however, verifying tokens with a high rejection risk occupies critical batch capacity that could otherwise serve other active requests.

**中文翻译**

其次，如何确定最优验证长度仍然是一个挑战。虽然并行生成可以轻松产生长草稿块，但无差别地验证所有提出的 token 会降低系统吞吐，尤其是在高并发负载下。理想的验证长度沿两个维度变化：在数据侧，代码这类结构化请求天然比开放式聊天具有更高接受率；在系统侧，轻负载下额外验证 token 几乎没有代价，而在重负载下，验证高拒绝风险 token 会占用关键 batch 容量，这些容量本可以服务其他活跃请求。  
译注：这就是为什么固定 MTP-N 或固定草稿长度在线上可能不稳定。

**English**

> To address these bottlenecks, we introduce DSpark, a speculative decoding framework that unifies high-throughput parallel generation with adaptive, load-aware verification. At its core, DSpark is designed to resolve the inherent trade-offs in draft generation and verification through two complementary mechanisms.

**中文翻译**

为了解决这些瓶颈，作者提出 DSpark，一个将高吞吐并行生成与自适应、负载感知验证统一起来的投机解码框架。DSpark 的核心设计是通过两个互补机制解决草稿生成和验证中的内在权衡。  
译注：接下来两个 bullet 分别对应 DSpark 的模型侧创新和系统侧创新。

**English**

> First, to overcome the lack of inter-token dependencies, DSpark adopts a semi-autoregressive architecture. It keeps the computationally expensive draft backbone fully parallel, appending only a lightweight serial output head to inject local transition information. This design preserves the drafting speed of parallel models while significantly mitigating suffix decay.

**中文翻译**

第一，为了克服 token 间依赖缺失的问题，DSpark 采用半自回归架构。它保持计算昂贵的 draft backbone 完全并行，只追加一个轻量级串行输出 head，用来注入局部转移信息。这个设计保留了并行模型的草稿生成速度，同时显著缓解后缀衰减。  
译注：可以把它理解成“重计算并行做，轻量连贯性修正在采样时顺序做”。

**English**

> Second, to resolve the system-level bottleneck, DSpark employs confidence-scheduled verification. By coupling a confidence head—which estimates per-position prefix survival probabilities—with a hardware-aware scheduler, DSpark dynamically tailors the verification length for each request. This scheduler leverages real-time engine throughput profiles to route target verification budget only toward tokens with the highest expected return.

**中文翻译**

第二，为了解决系统级瓶颈，DSpark 使用置信度调度式验证。通过将 confidence head 与硬件感知 scheduler 结合起来，DSpark 为每个请求动态定制验证长度；其中 confidence head 用于估计每个位置的前缀存活概率。这个 scheduler 利用实时的引擎吞吐曲线，只把 target verification 预算分配给预期收益最高的 token。  
译注：这不是简单设置一个 confidence threshold，而是把 token 的预期收益和当前硬件负载放在一起优化。

**English**

> We extensively evaluate DSpark across both controlled offline benchmarks and production-scale online deployments. On controlled offline benchmarks—spanning mathematical reasoning, code generation, and daily chat—DSpark consistently outperforms strong baselines. Specifically, across the Qwen3-4B, 8B, and 14B target models, it improves the macro-average accepted length over the autoregressive Eagle3 by 30.9%, 26.7%, and 30.0%, and over the parallel DFlash by 16.3%, 18.4%, and 18.3%, respectively.

**中文翻译**

作者在受控离线 benchmark 和生产规模在线部署中都对 DSpark 进行了广泛评估。在覆盖数学推理、代码生成和日常聊天的离线 benchmark 上，DSpark 稳定超过强 baseline。具体来说，在 Qwen3-4B、8B 和 14B 目标模型上，相比自回归 Eagle3，DSpark 的 macro-average accepted length 分别提升 30.9%、26.7% 和 30.0%；相比并行 DFlash，分别提升 16.3%、18.4% 和 18.3%。  
译注：accepted length 越高，表示每轮验证平均能推进更多 token。

**English**

> Beyond top-line metrics, our fine-grained position-wise analysis reveals the distinct generation characteristics of different drafters, empirically demonstrating how DSpark successfully combines the high initial-token capacity of parallel models with the suffix coherence of autoregressive models.

**中文翻译**

除了总体指标之外，作者的细粒度逐位置分析揭示了不同草稿模型的生成特征，并从经验上说明 DSpark 如何成功结合并行模型的高首 token 能力与自回归模型的后缀连贯性。  
译注：这句话是论文实验分析的主线：并行模型不是全差，自回归模型也不是全优，关键是组合两者优势。

**English**

> Beyond offline evaluation, we deployed DSpark within the DeepSeek-V4 serving system to assess its performance under live user traffic. Compared to the prior MTP-1 production baseline, DSpark significantly broadens the system’s operational envelope. Specifically, it consistently accelerates per-user generation speeds by 60%–85% for V4-Flash and 57%–78% for V4-Pro at matched aggregate throughput capacities.

**中文翻译**

除了离线评估，作者还将 DSpark 部署到 DeepSeek-V4 serving 系统中，在真实用户流量下评估性能。相比此前的 MTP-1 生产基线，DSpark 显著扩展了系统的运行范围。具体而言，在匹配聚合吞吐能力时，DSpark 稳定地将 V4-Flash 的单用户生成速度提升 60% 到 85%，将 V4-Pro 的单用户生成速度提升 57% 到 78%。  
译注：线上指标比离线 accepted length 更能说明 DSpark 是否真的适合生产。

**English**

> Furthermore, under strict Service Level Agreements (SLAs) where the baseline’s capacity deteriorates severely—such as 120 TPS for Flash and 50 TPS for Pro—DSpark mitigates verification overhead to maintain robust throughput. By overcoming this performance cliff, DSpark unlocks strict interactivity tiers that were previously unattainable, effectively shifting the Pareto frontier of LLM serving.

**中文翻译**

此外，在严格的服务等级协议（SLA）下，当基线容量严重退化时，例如 Flash 的 120 TPS 和 Pro 的 50 TPS，DSpark 能减轻验证开销并维持稳健吞吐。通过克服这种性能悬崖，DSpark 解锁了过去无法达到的严格交互性档位，有效推动了 LLM serving 的 Pareto frontier。  
译注：这里的 TPS 指 token/s/user，即单用户看到 token 输出的速度。

**English**

> To foster collective advancement within the open-source community, we are making our artifacts publicly available. Specifically, we release the trained DSpark checkpoints for both the DeepSeek-V4-Flash (preview) and DeepSeek-V4-Pro (preview) models. Furthermore, we open-source DeepSpec, an algorithm-driven training repository, including Eagle3, DFlash and DSpark. These artifacts are intended to support further research on efficient LLM serving.

**中文翻译**

为了促进开源社区的共同进展，作者公开发布相关产物。具体而言，作者发布了 DeepSeek-V4-Flash preview 和 DeepSeek-V4-Pro preview 的 DSpark 训练 checkpoint。此外，作者还开源了 DeepSpec，这是一个算法驱动的训练仓库，包含 Eagle3、DFlash 和 DSpark。这些产物旨在支持高效 LLM serving 的进一步研究。  
译注：DeepSpec 可以看作研究 speculative decoding drafter 训练的配套工具箱。

---

## 2. Background

### 2.1 Speculative Decoding

**English**

> Autoregressive language models generate one token per forward pass, making inference latency proportional to output length. Speculative decoding accelerates the inference of a target model using a lightweight draft model. At each decoding cycle, the draft model proposes candidate tokens. The target model verifies all candidates in a single forward pass, accepting the longest prefix consistent with its own distribution.

**中文翻译**

自回归语言模型每次前向传播只生成一个 token，因此推理延迟与输出长度成正比。投机解码使用轻量级 draft model 来加速 target model 的推理。在每个解码周期中，draft model 提出若干候选 token；target model 在一次前向传播中验证所有候选，并接受与自身分布一致的最长前缀。  
译注：这里的“最长前缀”是理解投机解码的关键，后面的 token 只有当前面的 token 都被接受时才有意义。

**English**

> Concretely, at each draft position k, the target model computes its own distribution and compares it against the draft distribution. The token is accepted with probability min(1, p_t(x_k) / p_d(x_k)). Verification proceeds left to right: the first rejection at position k discards all subsequent tokens, regardless of their quality.

**中文翻译**

具体来说，在每个草稿位置 `k`，target model 计算自己的分布，并与 draft distribution 比较。token `x_k` 以 `min(1, p_t(x_k) / p_d(x_k))` 的概率被接受。验证从左到右进行：一旦在位置 `k` 第一次拒绝，后续所有 token 都会被丢弃，不论它们本身质量如何。  
译注：这就是为什么第一个 token 非常重要；第一个 token 错了，整个草稿块直接失效。

**English**

> Let tau denote the number of accepted tokens per cycle, and let T_draft and T_verify be the wall-clock times of the drafting and verification passes, respectively. The average latency per generated token is L = (T_draft + T_verify) / tau. Improving speedup therefore reduces to three levers: lowering T_draft, raising tau, or reducing the effective T_verify.

**中文翻译**

令 `tau` 表示每个周期接受的 token 数，`T_draft` 和 `T_verify` 分别表示草稿生成和验证过程的实际耗时。每个生成 token 的平均延迟为 `L = (T_draft + T_verify) / tau`。因此，要提升加速效果，本质上有三个杠杆：降低 `T_draft`，提高 `tau`，或降低有效的 `T_verify`。  
译注：DSpark 同时作用于这三个方向：并行 backbone 降低长草稿 draft 成本，顺序 head 提高接受数，scheduler 减少无效验证。

### 2.2 Drafter Architectures

**English**

> The design of the draft model determines how T_draft and tau trade off. Existing approaches fall into two categories.

**中文翻译**

draft model 的设计决定了 `T_draft` 与 `tau` 如何权衡。现有方法可以分为两类。  
译注：接下来就是自回归 drafter 与并行 drafter 的对比。

**English**

> Autoregressive drafters generate draft tokens sequentially, conditioning each position on previously sampled tokens. This explicit dependency gives strong modeling capacity, but the drafting cost grows linearly with block size, which forces autoregressive drafters to use small block sizes and shallow architectures to keep drafting latency low. To compensate for the short block, tree-based verification expands candidates into a tree and verifies multiple paths via tree attention, but the large number of verification tokens reduces overall serving throughput.

**中文翻译**

自回归草稿模型按顺序生成草稿 token，每个位置都依赖之前采样出的 token。这种显式依赖带来强建模能力，但草稿生成成本会随 block size 线性增长，因此自回归草稿模型必须使用较小的 block size 和较浅的架构，以保持较低的草稿延迟。为了弥补短 block，一些 tree-based verification 方法会把候选扩展成树，并通过 tree attention 验证多条路径，但大量验证 token 会降低整体 serving 吞吐。  
译注：自回归 drafter 的优势是连贯，弱点是长草稿太贵；tree 方法能扩候选，但验证负担也会变重。

**English**

> Parallel drafters produce all draft tokens in a single forward pass, making drafting latency nearly independent of block size. This allows substantially larger blocks without proportionally increasing latency.

**中文翻译**

并行草稿模型在一次前向传播中产生所有草稿 token，使草稿生成延迟几乎不依赖 block size。因此它们可以使用明显更大的草稿块，而不会让延迟按比例增加。  
译注：并行 drafter 的核心收益就是“长草稿的边际 draft 成本低”。

**English**

> Among them, DFlash is a state-of-the-art parallel drafter, which conditions its draft model on rich context features extracted from the target model through KV injection. During prefill, hidden states from a set of target layers are concatenated and projected into the draft hidden space. These context features are injected into every draft layer by concatenating them with the draft block representations along the sequence dimension of keys and values.

**中文翻译**

其中，DFlash 是一种先进的并行草稿模型，它通过 KV injection 让 draft model 以从 target model 抽取的丰富上下文特征为条件。在 prefill 阶段，来自若干 target layer 的 hidden states 被拼接起来，并投影到 draft hidden space 中。这些上下文特征会在每个 draft layer 中沿 key 和 value 的序列维度与 draft block 表示拼接，从而注入到模型中。  
译注：KV injection 可以理解为把 target model 的中间特征喂给 draft model，让 draft model 更接近 target model 的语义状态。

**English**

> All positions within a block attend bidirectionally to each other and to the injected target context. The draft model shares the target model’s embedding layer and language modeling head, both frozen. It takes as input the embedding of an anchor token followed by mask token embeddings, and produces logits for all mask positions in a single forward pass. Since drafting requires only a single forward pass regardless of block size, DFlash can afford deeper architectures and larger blocks than autoregressive drafters under the same latency budget.

**中文翻译**

block 内所有位置都可以双向关注彼此以及注入的 target context。draft model 共享 target model 的 embedding 层和语言建模 head，并冻结二者。它以一个 anchor token 的 embedding 加上一组 mask token embeddings 作为输入，并在一次前向传播中为所有 mask 位置产生 logits。由于无论 block size 多大，草稿生成都只需要一次 forward，在相同延迟预算下，DFlash 可以使用比自回归 drafter 更深的架构和更大的 block。  
译注：DFlash 强在并行和容量，但也正因为 block 内双向且非因果，采样后前缀依赖不足。

---

## 3. Architecture

**English**

> The overview of DSpark is shown in Figure 1. Recall that the per-token latency of speculative decoding is L = (T_draft + T_verify) / tau. Autoregressive drafters achieve high tau but pay drafting cost proportional to block size; parallel drafters collapse drafting to a single pass but sacrifice tau because each position is predicted independently. Meanwhile, fixed-length verification wastes verification time on low-confidence suffix tokens that are almost certain to be rejected.

**中文翻译**

DSpark 的整体架构如图 1 所示。回顾 speculative decoding 的单 token 延迟公式 `L = (T_draft + T_verify) / tau`：自回归草稿模型能获得较高的 `tau`，但要付出与 block size 成正比的草稿成本；并行草稿模型把草稿生成压缩到一次 forward，但由于每个位置独立预测，会牺牲 `tau`。与此同时，固定长度验证会把验证时间浪费在几乎必然被拒绝的低置信后缀 token 上。  
译注：这一段把 DSpark 的两个目标重新放回公式里：提高 `tau`，降低浪费的 `T_verify`。

**English**

> DSpark addresses these limitations with two complementary components: semi-autoregressive generation and confidence-scheduled verification. A parallel backbone handles the bulk of draft computation, keeping draft latency nearly independent of block size. A lightweight sequential block then injects dependency among draft tokens, improving accepted length at minimal additional latency. A confidence head estimates per-position acceptance probabilities, and a hardware-aware scheduler uses these estimates to prune low-confidence suffix tokens.

**中文翻译**

DSpark 通过两个互补组件解决这些限制：半自回归生成和置信度调度式验证。并行 backbone 处理大部分草稿计算，使草稿延迟几乎不依赖 block size；轻量级顺序 block 随后在草稿 token 之间注入依赖，在只增加很小延迟的情况下提升 accepted length。confidence head 估计每个位置的接受概率，硬件感知 scheduler 使用这些估计剪掉低置信后缀 token。  
译注：如果只记一句话，可以记“并行生成负责快，顺序 head 负责连贯，scheduler 负责别浪费验证”。

**English**

> Given prompt tokens ABC, the target model executes one step to generate the next token D, which serves as the anchor for the drafting phase. Using D as the input, DSpark employs a heavy parallel backbone and a lightweight sequential head to generate draft tokens EFGH along with their corresponding confidence scores. The hardware-aware prefix scheduler then evaluates these scores to retain the prefix EFG and drop the low-confidence token H. Finally, the target model verifies the scheduled prefix in parallel. E and F may be accepted while G is rejected, prompting the model to generate a corrected token G* to complete the current round.

**中文翻译**

给定 prompt token `A B C`，target model 先执行一步生成下一个 token `D`，它作为草稿阶段的 anchor。DSpark 以 `D` 为输入，使用较重的并行 backbone 和轻量级顺序 head 生成草稿 token `E F G H` 及其对应 confidence scores。硬件感知前缀 scheduler 随后评估这些分数，保留前缀 `E F G`，丢弃低置信 token `H`。最后，target model 并行验证调度后的前缀。`E` 和 `F` 可能被接受，而 `G` 被拒绝，于是模型生成修正 token `G*` 来完成当前轮。  
译注：这段是图 1 的文字版。注意 scheduler 只能选择连续前缀，不能跳过 `G` 直接验证 `H`。

### 3.1 Semi-Autoregressive Generation

**English**

> A parallel drafter produces all draft logits in one forward pass, so each prediction cannot condition on tokens sampled elsewhere in the block. When the context admits multiple plausible continuations, such as “of course” and “no problem”, a parallel drafter may produce incoherent combinations such as “of problem” or “no course”, because each position marginalizes over all possible predecessors rather than conditioning on the one actually sampled.

**中文翻译**

并行草稿模型在一次 forward 中产生所有 draft logits，因此每个预测位置都无法以同一个 block 中其他位置实际采样出的 token 为条件。当上下文允许多个合理续写时，例如 “of course” 和 “no problem”，并行草稿模型可能产生 “of problem” 或 “no course” 这样的不连贯组合，因为每个位置都在对所有可能前驱做边缘化，而不是以实际采样出的前驱为条件。  
译注：这就是 pure parallel drafter 的多模态碰撞问题。

**English**

> Acceptance rate thus decays rapidly along the block, wasting both draft and verification compute. We therefore adopt a semi-autoregressive structure that splits draft generation into two stages.

**中文翻译**

因此，接受率会沿着 block 快速衰减，浪费 draft 计算和 verification 计算。于是作者采用一种半自回归结构，将草稿生成拆成两个阶段。  
译注：第一阶段解决“快”，第二阶段解决“连贯”。

**English**

> In the parallel stage, a parallel backbone runs a single forward pass over the entire block, producing hidden states and base logits. In our instantiation, the backbone is DFlash. We make a minor modification: instead of feeding an anchor token plus gamma mask tokens and predicting only the mask positions, we treat the anchor itself as the first prediction position, so gamma input tokens yield gamma draft logits. This reduces draft computation while maintaining similar draft quality.

**中文翻译**

在并行阶段，一个并行 backbone 对整个 block 运行一次 forward，产生 hidden states 和 base logits。在作者的实现中，这个 backbone 是 DFlash。作者做了一个小修改：不再输入一个 anchor token 加 `gamma` 个 mask token 并只预测 mask 位置，而是把 anchor 本身也作为第一个预测位置，因此 `gamma` 个输入 token 就能产生 `gamma` 个 draft logits。这样可以减少 draft 计算，同时保持相近的草稿质量。  
译注：这个改动偏工程优化，目的是少放一个无用输入位置。

**English**

> In the sequential stage, the model supplements the base logits with a prefix-dependent transition bias, allowing each draft position to condition on previously sampled tokens within the block. Rather than defining a globally normalized energy model, the sequential stage induces a causal block distribution through an autoregressive factorization.

**中文翻译**

在顺序阶段，模型为 base logits 补充一个依赖前缀的 transition bias，使每个草稿位置能够以 block 内之前采样出的 token 为条件。顺序阶段并不定义一个全局归一化的能量模型，而是通过自回归分解诱导出一个 causal block distribution。  
译注：选择“局部 softmax + 自回归分解”很重要，因为 speculative decoding 需要每个 token 的精确 draft probability。

**English**

> At inference time, the sequential block samples left to right. Because this sampling process is inherently sequential, the block must be computationally lightweight so that the overall draft latency remains dominated by the parallel stage.

**中文翻译**

推理时，顺序 block 从左到右采样。由于这个采样过程本质上是顺序的，该 block 必须计算非常轻量，才能让整体 draft latency 仍然主要由并行阶段主导。  
译注：如果顺序 head 太重，DSpark 就会退化成普通自回归 drafter，失去意义。

**English**

> The simplest instantiation is the Markov head. It restricts the transition bias to depend only on the immediately preceding token, reducing it to a first-order transition. A full vocabulary-by-vocabulary transition matrix would be too expensive, so DSpark approximates it with a low-rank factorization. Given the preceding token, the transition bias is obtained by an embedding lookup followed by a logit projection.

**中文翻译**

最简单的实现是 Markov head。它限制 transition bias 只依赖紧邻前一个 token，从而把问题化为一阶转移。完整的词表乘词表转移矩阵成本太高，因此 DSpark 用低秩分解来近似它。给定前一个 token 后，模型先做 embedding lookup，再做 logit projection，得到当前词表上的转移 bias。  
译注：Markov head 只记“一步历史”，但很多短语搭配已经能靠一步历史显著改善。

**English**

> The low-rank factorization keeps both storage and per-step compute small, making the sequential loop efficient even for large vocabularies. Returning to the earlier example: once position 1 samples “of”, the Markov head boosts “course” and suppresses “problem” at position 2, which mitigates cross-mode collision.

**中文翻译**

低秩分解让存储和每步计算都保持较小，因此即使词表很大，顺序循环也很高效。回到前面的例子：一旦位置 1 采样出 “of”，Markov head 会在位置 2 提升 “course” 并压低 “problem”，从而缓解跨模式碰撞。  
译注：这就是“用很少顺序性换来很多连贯性”的直观例子。

**English**

> The RNN head relaxes the Markov head’s memoryless assumption by maintaining a recurrent state that accumulates the full prefix history within a block. At each step, it concatenates the current state, the previous token embedding, and the backbone hidden state, then applies a single gated update. The state is initialized to zero.

**中文翻译**

RNN head 放宽了 Markov head 的无记忆假设，通过维护一个 recurrent state，在 block 内累积完整前缀历史。每一步，它把当前状态、前一个 token embedding 和 backbone hidden state 拼接起来，然后进行一次门控更新。状态初始化为 0。  
译注：RNN head 表达能力更强，但实现和部署复杂度更高；论文默认选择 Markov head。

### 3.2 Confidence-Scheduled Verification

**English**

> The semi-autoregressive architecture enables DSpark to generate large draft blocks efficiently. However, producing more draft tokens does not automatically translate to higher end-to-end speedups. Indiscriminately verifying the full draft block can actually degrade overall system throughput, especially in high-concurrency scenarios.

**中文翻译**

半自回归架构使 DSpark 能够高效生成较大的草稿块。然而，生成更多 draft token 并不会自动转化为更高的端到端加速。无差别地验证完整草稿块实际上可能降低整体系统吞吐，尤其是在高并发场景下。  
译注：这是 DSpark 与普通“多 token 预测”思路最重要的区别之一：多猜不等于都该验证。

**English**

> This performance bottleneck stems from two interacting factors. First, on the data side, draft acceptance rates vary across domains: structured text like code naturally yields high acceptance, whereas open-ended chat has significantly lower acceptance. Second, on the system side, the actual cost of verifying an extra token depends on engine load.

**中文翻译**

这个性能瓶颈来自两个相互作用的因素。首先，在数据侧，不同领域的草稿接受率不同：代码这类结构化文本天然具有较高接受率，而开放式聊天的接受率明显更低。其次，在系统侧，额外验证一个 token 的实际成本取决于引擎负载。  
译注：同一个 confidence，在系统空闲和系统满载时，是否值得验证可能完全不同。

**English**

> Under light system load, an extra verification incurs minimal penalty even if rejected. Under high-concurrency deployments, every unnecessary verification occupies target model batch capacity that could otherwise serve other active requests. Therefore, fully unlocking large draft blocks requires routing target model compute only toward tokens with positive expected return.

**中文翻译**

在系统轻负载下，即使额外验证的 token 被拒绝，代价也很小。而在高并发部署中，每一次不必要验证都会占用 target model 的 batch capacity，这些容量本可以服务其他活跃请求。因此，要充分释放大草稿块的潜力，需要只把 target model 计算路由给具有正预期收益的 token。  
译注：这就是硬件感知 scheduler 的动机。

#### 3.2.1 Confidence Head

**English**

> The confidence head outputs a scalar estimate for each draft position. Crucially, this score models the conditional probability that the draft token at position k will survive target verification, given that all preceding tokens in the block have been accepted. The architecture is a lightweight linear projection followed by a sigmoid function.

**中文翻译**

confidence head 为每个草稿位置输出一个标量估计。关键在于，这个分数建模的是一个条件概率：在 block 中所有前序 token 都已经被接受的前提下，位置 `k` 的 draft token 能通过 target verification 的概率。其结构是轻量级线性投影后接 sigmoid 函数。  
译注：这里不是普通“单 token 置信度”，而是“前缀条件下的接受概率”。

**English**

> DSpark supervises the confidence score using the analytical acceptance rate per step. This rate is determined by the total variation distance between the draft distribution and the target distribution: the closer the two distributions are, the higher the expected acceptance probability.

**中文翻译**

DSpark 使用每一步的解析接受率来监督 confidence score。这个接受率由 draft distribution 与 target distribution 之间的 total variation distance 决定：两个分布越接近，预期接受概率越高。  
译注：这比只用“最终是否接受”的硬标签更平滑，也更贴近拒绝采样规则。

**English**

> Unlike threshold-based verification heuristics, which only require confidence scores to correctly rank draft token qualities, the hardware-aware scheduling approach requires the absolute magnitudes of cumulative acceptance probabilities to compute expected acceptance length. Because neural confidence estimates are often overconfident, using raw confidence scores directly would distort throughput estimation and lead to suboptimal scheduling.

**中文翻译**

不同于基于阈值的验证启发式方法，后者只要求 confidence score 能正确排序 draft token 的质量；硬件感知调度需要累计接受概率的绝对数值，用来计算期望接受长度。由于神经网络的置信度估计常常过度自信，如果直接使用原始 confidence score，会扭曲吞吐估计并导致次优调度。  
译注：调度器要做数值优化，因此 confidence 不仅要“谁高谁低”对，还要“数值大概是多少”准。

**English**

> To address this, DSpark introduces Sequential Temperature Scaling (STS). Since each confidence score models a conditional probability, the joint probability of a draft prefix being accepted factorizes into the cumulative product of the scores. Using a held-out validation set, STS calibrates this joint probability consecutively from left to right. Temperature scaling is order-preserving, so it corrects predicted probabilities without disrupting relative rankings.

**中文翻译**

为了解决这个问题，DSpark 引入 Sequential Temperature Scaling（STS）。由于每个 confidence score 建模的是条件概率，一个 draft prefix 被接受的联合概率可以分解为这些分数的累计乘积。STS 使用 held-out validation set，从左到右连续校准这个联合概率。温度缩放保持排序不变，因此它能校正预测概率，而不破坏相对排序。  
译注：STS 的作用是把“模型说 0.9”校准成“真实统计接近 0.9”。

#### 3.2.2 Hardware-Aware Prefix Scheduler

**English**

> Prior methods typically apply a static threshold to confidence scores to determine verification length. While effective under isolated single-request assumptions, static thresholds can be suboptimal in high-concurrency production systems, where the utility of verifying a draft token depends heavily on the current system load.

**中文翻译**

以往方法通常对 confidence score 应用静态阈值来确定验证长度。虽然这在孤立的单请求假设下有效，但在高并发生产系统中，静态阈值可能是次优的，因为验证一个 draft token 的效用高度依赖当前系统负载。  
译注：这也是本文标题里 `Scheduled` 的真正含义：不是固定规则，而是动态调度。

**English**

> DSpark formulates verification length selection as a global throughput maximization problem. For each request, the scheduler considers a scheduled verification length. Because speculative decoding accepts draft tokens only as a continuous prefix, the survival probability of a token at position j is the cumulative product of confidence scores up to j.

**中文翻译**

DSpark 将验证长度选择表述为一个全局吞吐最大化问题。对每个请求，scheduler 都考虑一个待调度的验证长度。由于 speculative decoding 只接受连续前缀，位置 `j` 的 token 存活概率就是从第 1 个位置到第 `j` 个位置 confidence score 的累计乘积。  
译注：这就是 prefix survival probability，后面 token 的价值会被前面所有 token 的通过概率折扣。

**English**

> In a single verification step, the total batch size sent to the target model is the sum over requests of one target token plus the scheduled draft tokens. The expected number of successfully accepted tokens is the sum over requests of one bonus token plus the survival probabilities of the scheduled draft positions. Let SPS(B) denote the engine throughput in steps per second for a given batch size B. The scheduler aims to maximize expected system-wide token throughput.

**中文翻译**

在一次验证 step 中，送入 target model 的总 batch size 是所有请求的“一枚 target token 加上被调度的 draft token”之和。预期成功接受的 token 数则是所有请求的“一枚 bonus token 加上被调度 draft 位置的存活概率”之和。令 `SPS(B)` 表示给定 batch size `B` 时引擎的 steps per second。scheduler 的目标是最大化系统范围内的预期 token 吞吐。  
译注：目标函数可以理解为 `有效 token/step * step/second = 有效 token/second`。

**English**

> Although finding the global maximum appears to be a combinatorial search, the objective structure allows an efficient greedy solution. Since prefix survival probabilities are monotonically non-increasing along each request, sorting candidate tokens globally by survival probability naturally respects intra-block prefix dependencies.

**中文翻译**

虽然寻找全局最大值看起来像组合搜索，但目标结构允许高效贪心求解。由于每个请求内的前缀存活概率沿位置单调不增，将所有候选 token 按存活概率全局排序，自然会尊重块内前缀依赖。  
译注：同一请求的第 2 个 token 不可能排在第 1 个 token 前面，因为第 2 个前缀概率一定不高于第 1 个。

**English**

> Lossless speculative decoding strictly requires the non-anticipating property: admission decisions must not depend on future candidate tokens. Because the confidence head relies on the Markov feature of the previously sampled token, computing the next survival probability requires the instantiated candidate. A retrospective global search would leak token information into earlier admission decisions and introduce selection bias.

**中文翻译**

无损 speculative decoding 严格要求 non-anticipating property：接纳决策不能依赖未来候选 token。由于 confidence head 依赖前一个已采样 token 的 Markov feature，计算下一位置的存活概率需要知道已经实例化的候选 token。回溯式全局搜索会把 token 信息泄漏到更早的接纳决策中，引入选择偏差。  
译注：这段是理论上最容易忽略的点。不是 target model 最终验证就一定无损，调度本身也不能偷看未来。

**English**

> To enforce strict causality, the scheduler employs an early-stopping mechanism. By breaking the greedy search immediately when throughput drops, the truncation decision relies only on the prefix processed up to that step. This isolates the admission event from future tokens and ensures exact target-distribution recovery. The stepwise early-stopping yields the global maximum if the throughput objective is unimodal; practical adaptations for non-smooth hardware curves are discussed later.

**中文翻译**

为了强制严格因果性，scheduler 使用 early-stopping 机制。当吞吐下降时立即停止贪心搜索，使截断决策只依赖到当前 step 为止已经处理的前缀。这样可以把接纳事件与未来 token 隔离，确保精确恢复 target distribution。若吞吐目标是单峰的，这种逐步 early-stopping 可以得到全局最大值；对于真实硬件曲线不平滑时的工程适配，论文后文会讨论。  
译注：理论版用 early stopping 保证无损，生产版用异步历史预测解决硬件曲线锯齿问题。

### 3.3 Training

**English**

> During training, DSpark randomly samples multiple anchor positions from each target sequence to form gamma-token blocks as training data. The target model is frozen throughout training; the draft model shares its embedding layer and language modeling head with the target model and keeps them frozen, updating only the backbone drafter, sequential block, and confidence head.

**中文翻译**

训练时，DSpark 从每个目标序列中随机采样多个 anchor position，构造 `gamma` 个 token 的 block 作为训练数据。target model 在整个训练过程中冻结；draft model 共享 target model 的 embedding 层和 language modeling head，并保持它们冻结，只更新 backbone drafter、sequential block 和 confidence head。  
译注：共享并冻结 embedding/LM head 有助于让 draft logits 与 target model 的词表空间对齐。

**English**

> The training objective consists of three terms: a cross-entropy loss, a distribution-matching loss, and a confidence loss. All three are position-weighted to emphasize earlier block positions, which contribute more to expected acceptance length under prefix-based verification.

**中文翻译**

训练目标由三项组成：cross-entropy loss、distribution-matching loss 和 confidence loss。三者都按位置加权，以强调 block 中较早的位置，因为在基于前缀的验证中，较早位置对期望 accepted length 的贡献更大。  
译注：第一个 token 错了后面全作废，所以训练时前面位置更重要。

**English**

> The cross-entropy loss trains the drafter to predict the correct next token. The distribution-matching loss penalizes the total variation distance between the draft and target distributions. Since this distance is a direct proxy for the acceptance rate, minimizing the distribution-matching loss directly maximizes the expected acceptance rate.

**中文翻译**

cross-entropy loss 训练 drafter 预测正确的 next token。distribution-matching loss 惩罚 draft distribution 与 target distribution 之间的 total variation distance。由于这个距离是接受率的直接代理，最小化 distribution-matching loss 就是在直接最大化期望接受率。  
译注：普通 CE 只看数据答案，TV loss 则让 draft 分布整体贴近 target 分布，更符合 speculative decoding 的目标。

**English**

> The confidence loss is a binary cross-entropy that trains the confidence head to predict the soft acceptance label derived from the analytical acceptance rate. The overall objective is a weighted combination of the three terms, with default weights 0.1 for cross entropy, 0.9 for distribution matching, and 1.0 for confidence.

**中文翻译**

confidence loss 是一个 binary cross-entropy，用来训练 confidence head 预测由解析接受率得到的软接受标签。总目标是三项损失的加权组合，默认权重为：cross entropy 0.1，distribution matching 0.9，confidence 1.0。  
译注：权重显示作者更重视与 target 分布对齐和 confidence 估计，而不是单纯拟合数据 token。

---

## 4. Experiments

### 4.1 Experimental Setup

**English**

> We validate the draft quality of DSpark using offline benchmarks and report the effectiveness of the confidence scheduler under online production traffic later. We evaluate DSpark on four target models spanning different scales and model families: Qwen3-4B, Qwen3-8B, Qwen3-14B, and Gemma4-12B.

**中文翻译**

作者使用离线 benchmark 验证 DSpark 的草稿质量，并在后文报告 confidence scheduler 在在线生产流量下的效果。DSpark 在四个覆盖不同规模和模型家族的 target model 上评估：Qwen3-4B、Qwen3-8B、Qwen3-14B 和 Gemma4-12B。  
译注：离线部分主要看 drafter 本身，在线部分主要看 scheduler 和系统收益。

**English**

> For draft models, DSpark is compared with two representative drafters: DFlash, a state-of-the-art parallel drafter, and Eagle3, an autoregressive drafter based on Training-Time Test. For fair comparison, all drafters are retrained in the same training framework and on the same data. Eagle3’s horizon is aligned with the block size used by DFlash and DSpark, and the same target-model feature layers are used for all drafters.

**中文翻译**

对于 draft model，DSpark 与两个代表性草稿模型比较：DFlash，一个先进的并行草稿模型；Eagle3，一个基于 Training-Time Test 的自回归草稿模型。为了公平比较，所有 drafter 都在同一训练框架和同一数据上重新训练。Eagle3 的 horizon 与 DFlash 和 DSpark 使用的 block size 对齐，并且所有 drafter 使用相同的 target-model feature layers。  
译注：这避免了“baseline 训练不充分”或“特征层不同”造成的不公平。

**English**

> The training data is Open-PerfectBlend, an open-source version of PerfectBlend consisting of 1.3 million samples. Only prompts are used, and responses are regenerated by each target model with recommended sampling parameters. Each drafter is trained for 10 epochs. Evaluation covers mathematical reasoning, code generation, and daily chat benchmarks. All benchmarks use standard speculative decoding with sampling temperature 1.0, and report accepted length per decoding round.

**中文翻译**

训练数据使用 Open-PerfectBlend，这是 PerfectBlend 的开源版本，包含 130 万样本。作者只使用 prompts，responses 则由每个 target model 按推荐采样参数重新生成。每个 drafter 训练 10 个 epoch。评估覆盖数学推理、代码生成和日常聊天 benchmark。所有 benchmark 都使用标准 speculative decoding，采样温度为 1.0，并报告每轮解码的 accepted length。  
译注：重新用 target model 生成 response，可以让训练数据更贴近目标模型自身分布。

### 4.2 Experimental Results

**English**

> To isolate raw draft quality from system-level scheduling policies, the offline evaluation disables the confidence scheduler and forces all drafters to propose a fixed block of tokens. The main results, measured by average accepted length per round, show that DSpark consistently outperforms both the autoregressive baseline Eagle3 and the parallel baseline DFlash across all target models and benchmark domains.

**中文翻译**

为了将原始草稿质量与系统级调度策略隔离开来，离线评估禁用了 confidence scheduler，并强制所有 drafter 提出固定长度的 token block。以每轮平均 accepted length 衡量的主结果显示，DSpark 在所有 target model 和 benchmark domain 上都稳定超过自回归 baseline Eagle3 和并行 baseline DFlash。  
译注：这里故意不启用 scheduler，是为了证明 DSpark 的 drafter 本身更强。

**English**

> Across Qwen3-4B, Qwen3-8B, and Qwen3-14B, DSpark improves macro-average accepted length over Eagle3 by 30.9%, 26.7%, and 30.0%, respectively. Compared to DFlash, DSpark yields relative improvements of 16.3%, 18.4%, and 18.3% across the three scales. The advantage also generalizes across model families, as shown by consistent gains on Gemma4-12B.

**中文翻译**

在 Qwen3-4B、Qwen3-8B 和 Qwen3-14B 上，DSpark 相比 Eagle3 的 macro-average accepted length 分别提升 30.9%、26.7% 和 30.0%。相比 DFlash，DSpark 在三个规模上分别带来 16.3%、18.4% 和 18.3% 的相对提升。这个优势也能跨模型家族泛化，因为 Gemma4-12B 上同样有稳定收益。  
译注：DSpark 同时超过强自回归和强并行 baseline，这是论文离线结果的核心。

**English**

> The results reveal a strong domain effect: accepted length is naturally higher on structured tasks such as math and code than on open-ended chat. This inherent variance in data predictability means a static verification length often wastes compute on trailing tokens that are highly likely to be rejected. This directly motivates confidence-scheduled verification.

**中文翻译**

结果揭示了强烈的领域效应：在数学和代码这类结构化任务上，accepted length 天然高于开放式聊天。这种数据可预测性的内在差异意味着，静态验证长度经常会把计算浪费在高度可能被拒绝的尾部 token 上。这直接支持了 confidence-scheduled verification 的动机。  
译注：代码补全通常约束强，聊天续写空间大，所以同样的草稿长度不应一视同仁。

### 4.3 Experimental Analysis

#### 4.3.1 Why Can Parallel Generation Outperform Autoregression?

**English**

> The main results present a counter-intuitive observation: the parallel drafter DFlash and the semi-autoregressive drafter DSpark often yield longer accepted lengths than the fully autoregressive drafter Eagle3. This contrasts with the standard expectation that step-by-step autoregression produces higher-quality sequences than parallel models.

**中文翻译**

主结果呈现了一个反直觉观察：并行草稿模型 DFlash 和半自回归草稿模型 DSpark 经常比完全自回归草稿模型 Eagle3 获得更长的 accepted length。这与通常预期相反，因为人们一般认为逐步自回归会比并行模型生成更高质量的序列。  
译注：论文接下来用位置级接受率解释这个现象。

**English**

> To analyze this behavior, the authors examine position-wise conditional acceptance during actual speculative decoding rollouts. For a given draft position k, the denominator counts only instances where all preceding draft tokens have been accepted, and the metric measures how often token k is also accepted. This isolates the predictive quality at each position without penalizing it for earlier prefix errors.

**中文翻译**

为了分析这种行为，作者在真实 speculative decoding rollout 中考察逐位置条件接受率。对于给定草稿位置 `k`，分母只统计所有前序草稿 token 都已经被接受的实例，然后计算位置 `k` 的 token 也被接受的比例。这样可以隔离每个位置自身的预测质量，而不因更早的前缀错误惩罚它。  
译注：这是一个很干净的诊断指标，可以看出不同位置到底谁更强。

**English**

> At the first draft position, both architectures predict the next token based solely on the target context. The performance divergence stems from architectural capacity: autoregressive models like Eagle3 are constrained to shallow networks due to O(gamma) latency, whereas O(1) parallel drafters can afford much deeper networks. This yields a substantial accuracy margin at position 1.

**中文翻译**

在第一个草稿位置，两类架构都只基于 target context 预测下一个 token。性能差异来自架构容量：Eagle3 这类自回归模型由于 `O(gamma)` 延迟限制，必须使用较浅网络；而 `O(1)` 的并行草稿模型可以承受更深网络。因此，并行模型在位置 1 上获得显著准确率优势。  
译注：第一个 token 权重极高，所以 DFlash 的首 token 容量优势能显著拉高整体 accepted length。

**English**

> Examining later positions exposes the limitation of independent parallel generation. As earlier tokens lock in a semantic path, subsequent tokens become more predictable. Autoregressive models leverage this conditional certainty, maintaining or increasing conditional acceptance deeper into the block. DFlash, however, suffers rapid acceptance decay because each parallel position marginalizes over all possible previous tokens rather than conditioning on the exact sampled prefix.

**中文翻译**

观察后续位置会暴露独立并行生成的限制。当前面的 token 锁定了一条语义路径后，后续 token 会变得更可预测。自回归模型能利用这种条件确定性，在 block 更深处保持甚至提高条件接受率。然而 DFlash 会遭遇快速接受率衰减，因为每个并行位置都对所有可能前序 token 做边缘化，而不是以实际采样出的精确前缀为条件。  
译注：这解释了为什么并行 drafter 首 token 强，但后缀弱。

**English**

> This analysis motivates DSpark’s semi-autoregressive design. DSpark inherits the high initial acceptance of a deep parallel drafter while its lightweight sequential head mitigates rapid suffix decay. By resolving this trade-off, DSpark maintains a high and stable conditional acceptance rate throughout the draft block.

**中文翻译**

这一分析直接推动了 DSpark 的半自回归设计。DSpark 继承了深并行草稿模型的高初始接受率，同时用轻量级顺序 head 缓解快速后缀衰减。通过解决这一权衡，DSpark 能在整个草稿块中保持较高且稳定的条件接受率。  
译注：这是 DSpark 模型结构的核心实验解释。

#### 4.3.2 A Little Autoregression Goes a Long Way

**English**

> The authors explore DSpark’s design space along two dimensions: drafter depth and proposal length. Increasing the number of transformer layers expands predictive capacity. With block size fixed, DSpark’s performance improves monotonically with depth, and the steepest marginal gain occurs from one to two layers. Notably, a two-layer DSpark outperforms a five-layer DFlash baseline across domains.

**中文翻译**

作者从两个维度探索 DSpark 的设计空间：drafter depth 和 proposal length。增加 Transformer 层数会扩大预测能力。在固定 block size 时，DSpark 的性能随深度单调提升，并且从 1 层增加到 2 层时边际收益最大。值得注意的是，2 层 DSpark 在各领域都超过了 5 层 DFlash baseline。  
译注：这说明轻量顺序依赖比单纯堆叠并行层更划算。

**English**

> With drafter depth fixed, the authors scale draft length across several values and evaluate both Markov and RNN heads. DSpark consistently outperforms DFlash at every proposal length, and the performance gap widens as the proposal length increases. Because pure parallel generation suffers suffix decay, its marginal utility diminishes for long blocks. DSpark mitigates this decay, so its relative gains grow for longer proposals.

**中文翻译**

在固定 drafter depth 后，作者将 draft length 扩展到多个取值，并评估 Markov head 与 RNN head。DSpark 在每个 proposal length 上都稳定超过 DFlash，并且随着 proposal length 增加，性能差距变大。由于纯并行生成存在后缀衰减，它在长 block 上的边际收益会下降；DSpark 缓解了这种衰减，因此在更长 proposal 上相对收益更大。  
译注：长草稿越长，DSpark 的“少量自回归”越值钱。

**English**

> The RNN head provides only marginal additional gains over the Markov head, mainly at longer proposal lengths. Given its higher implementation complexity and less favorable deployment properties, the Markov head is used as the default. The measured latency overhead of the sequential loop is negligible, adding only a small percentage to full-round latency while delivering substantial accepted-length improvements.

**中文翻译**

RNN head 相比 Markov head 只带来较小额外收益，主要出现在更长 proposal length 下。考虑到 RNN head 实现复杂度更高、部署属性不如 Markov head，论文默认使用 Markov head。实测顺序循环的延迟开销可以忽略，只给整轮延迟增加很小比例，却带来显著 accepted length 提升。  
译注：默认选 Markov head 是工程上很现实的选择。

#### 4.3.3 Verify Smarter, Not Longer

**English**

> While DSpark sustains high acceptance over long draft blocks, verifying the entire proposal remains inefficient. Because of domain variance, trailing tokens in open-ended chat still face high rejection risks, making blind verification wasteful. The authors first evaluate the confidence head in isolation through an offline threshold sweep.

**中文翻译**

虽然 DSpark 能在长草稿块上保持较高接受率，但验证完整 proposal 仍然低效。由于领域差异，开放式聊天中的尾部 token 仍然有很高拒绝风险，盲目验证会造成浪费。作者首先通过离线 threshold sweep 单独评估 confidence head。  
译注：这里先验证 confidence head 是否能区分“值得验证”和“不值得验证”的 token。

**English**

> As the confidence threshold increases, the overall acceptance rate steadily rises because the estimator filters out tokens that would ultimately be rejected. This pruning is most pronounced on chat workloads, where higher-entropy token distributions limit the efficiency of fixed-length verification. Structured tasks experience milder pruning and retain more draft tokens.

**中文翻译**

随着 confidence threshold 升高，整体接受率稳步上升，因为 estimator 过滤掉了最终会被拒绝的 token。这种剪枝在聊天负载上最明显，因为更高熵的 token 分布限制了固定长度验证的效率。结构化任务的剪枝更温和，会保留更多 draft token。  
译注：这再次说明“不同领域应该验证不同长度”。

**English**

> A static threshold is useful for diagnostics but suboptimal in dynamic serving environments because it ignores system load. Maximizing system-level throughput requires the confidence model to have both strong discrimination and precise calibration. Reliability diagrams show that raw confidence estimates discriminate well but are overconfident; post-hoc STS calibration aligns predicted prefix survival probabilities with empirical acceptance rates.

**中文翻译**

静态 threshold 对诊断有用，但在动态 serving 环境中是次优的，因为它忽略系统负载。最大化系统级吞吐要求 confidence model 既有强区分能力，又有精确校准。可靠性图显示，原始 confidence 估计区分能力很好，但存在过度自信；后验 STS 校准能让预测的前缀存活概率与经验接受率对齐。  
译注：confidence head 不只是分类器，还是 scheduler 的数值输入。

---

## 5. Real-World Deployment of DSpark

### 5.1 Scalable and Flexible Training

**English**

> DSpark draft models are co-deployed with preview versions of DeepSeek-V4-Flash and DeepSeek-V4-Pro. The parallel backbone comprises three MoE layers with mHC and sliding window attention. The maximum block size is set to gamma = 5, and the Markov head is used for sequential modeling. The confidence head is trained end-to-end alongside the draft model and then calibrated via STS.

**中文翻译**

DSpark draft models 与 DeepSeek-V4-Flash preview 和 DeepSeek-V4-Pro preview 共同部署。并行 backbone 由三层 MoE layer 组成，使用 mHC 和 sliding window attention。最大 block size 设置为 `gamma = 5`，顺序建模使用 Markov head。confidence head 与 draft model 端到端训练，随后通过 STS 校准。  
译注：生产配置比离线 Qwen 实验更贴近真实 DeepSeek-V4 架构。

**English**

> Training the draft model requires the target model’s output distributions for supervision. Evaluating both models over the full document context incurs substantial memory footprints and inter-worker communication overhead. To address these bottlenecks, the authors implement two system-level optimizations in their internal training framework.

**中文翻译**

训练 draft model 需要 target model 的输出分布作为监督。在完整文档上下文上同时评估两个模型，会带来巨大的显存占用和 worker 间通信开销。为了解决这些瓶颈，作者在内部训练框架中实现了两个系统级优化。  
译注：DSpark 训练贵的地方在于需要 target distribution，而不是只需要数据标签。

**English**

> First, instead of transferring full-vocabulary logits across parallel workers, the system temporarily caches target-model activations and communicates only the hidden states immediately preceding the language modeling head. The LM head projection is then executed locally on draft-model workers only for sampled target positions, reducing per-token communication complexity to O(d).

**中文翻译**

第一，系统不在并行 worker 之间传输全词表 logits，而是临时缓存 target model activation，并只通信 language modeling head 之前的 hidden states。随后，LM head projection 只在 draft-model worker 本地、针对采样的目标位置执行，从而将每 token 通信复杂度降为 `O(d)`。  
译注：全词表 logits 维度通常很大，传 hidden state 可以显著省带宽。

**English**

> Second, anchor-bounded sequence packing decouples the draft model’s computational cost from the target model’s context length. A fixed number of draft anchors is sampled from the training sequence, and isolated prediction blocks are packed into dense training batches. Token-level attention indices maintain exact causal masking while avoiding padding overhead.

**中文翻译**

第二，anchor-bounded sequence packing 将 draft model 的计算成本与 target model 的上下文长度解耦。系统从训练序列中采样固定数量的 draft anchors，并把这些孤立预测 block 打包成 dense training batches。token-level attention indices 用于保持精确 causal masking，同时避免 padding 开销。  
译注：这让训练 draft block 时不用为整段长上下文付出完整 padding 成本。

### 5.2 Hardware-Aware Prefix Scheduler in Practice

**English**

> Algorithm 1 provides a theoretically sound and lossless scheduling mechanism, but direct production deployment exposes two conflicts with real infrastructure. First, the algorithm assumes a smooth, unimodal capacity curve, whereas true hardware throughput is discrete and jagged. Second, dynamic per-step scheduling conflicts with continuous CUDA graph replay and zero-overhead scheduling.

**中文翻译**

算法 1 提供了理论上合理且无损的调度机制，但直接部署到生产环境会暴露两个与真实基础设施的冲突。第一，算法假设 capacity curve 平滑且单峰，而真实硬件吞吐是离散且锯齿状的。第二，每 step 动态调度与连续 CUDA graph replay 和 zero-overhead scheduling 存在冲突。  
译注：理论算法要落地到高性能推理系统，需要适配 CUDA graph、kernel 形状和调度流水线。

**English**

> DSpark adapts the scheduler to operate asynchronously. Because zero-overhead scheduling requires the next step’s batch size to be known before the current step completes, synchronous scheduling would stall the GPU pipeline. Instead, the system approximates upcoming verification capacity using confidence outputs from two steps prior. Current candidate tokens are still sorted by their up-to-date cumulative confidence scores; the historical prediction is used only to determine a dynamic truncation length.

**中文翻译**

DSpark 将 scheduler 改造成异步运行。因为 zero-overhead scheduling 要求下一步的 batch size 在当前 step 完成之前就已知，同步调度会让 GPU pipeline 停顿。因此，系统使用两步之前的 confidence 输出来近似即将到来的 verification capacity。当前候选 token 仍然按最新累计 confidence score 排序；历史预测只用于决定动态截断长度。  
译注：可以理解为“用历史信息定容量，用当前信息排优先级”。

**English**

> This asynchronous pipeline also resolves the hardware utilization bottleneck. To avoid being trapped by jagged SPS cliffs, the production scheduler removes the early-stopping break and enables unconstrained global search. Ordinarily this retrospective search would violate losslessness, but because it evaluates only historical predictions from two steps prior, the admission decision is isolated from the realization of the current token. Thus the asynchronous design forms a causal barrier while maximizing physical throughput.

**中文翻译**

这个异步流水线也解决了硬件利用率瓶颈。为了避免被锯齿状 SPS cliff 困在局部点，生产 scheduler 移除了 early-stopping break，并允许无约束全局搜索。通常这种回溯搜索会违反无损性，但由于它只评估两步之前的历史预测，接纳决策与当前 token 的实际取值隔离。因此，异步设计形成了一个因果屏障，同时最大化物理吞吐。  
译注：这是论文很精彩的系统设计：理论上 early stopping 保因果，工程上用异步历史预测保因果。

### 5.3 High-Throughput and Low-Latency Inference

**English**

> During decoding, production serving systems must simultaneously optimize per-request latency and aggregate throughput. Speculative decoding inherently navigates this trade-off, trading extra system compute for faster per-request generation.

**中文翻译**

在解码过程中，生产 serving 系统必须同时优化单请求延迟和聚合吞吐。投机解码本质上就在处理这一权衡：用额外的系统计算换取更快的单请求生成速度。  
译注：用户关心自己的 token/s，服务方还关心每张 GPU 总 token/s。

**English**

> In the deployment setting, the number of requests processed per step is often constrained by resource limits such as fixed KV-cache capacity per request and by available user traffic. As a result, effective batch size often remains below the GPU compute-saturation threshold. In this regime, maximizing per-GPU total token throughput and maximizing generation speed per user become highly correlated rather than competing objectives.

**中文翻译**

在作者的部署场景中，每 step 处理的请求数量常常受到资源限制约束，例如每个请求固定 KV-cache 容量，也受到可用用户流量池限制。因此，有效 batch size 经常低于 GPU 计算饱和阈值。在这种 regime 下，最大化每 GPU 总 token 吞吐和最大化单用户生成速度不再强烈竞争，而是高度相关。  
译注：如果 GPU 还没吃满，多验证高质量草稿可以同时提高用户速度和总吞吐。

**English**

> The asynchronous scheduler routes idle compute toward the most promising draft tokens. However, this dynamic routing requires the inference framework to efficiently support variable-length queries within a single batch. Standard decode kernels are optimized for fixed query lengths; naive padding would cause GPU under-utilization.

**中文翻译**

异步 scheduler 会把空闲计算路由给最有希望的 draft token。然而，这种动态路由要求推理框架在单个 batch 内高效支持 variable-length queries。标准 decode kernel 通常针对固定 query length 优化；朴素 padding 会导致 GPU 利用率下降。  
译注：调度算法选出不同请求的不同验证长度后，底层 kernel 也必须能高效执行。

**English**

> The system decouples physical execution from logical sequence tracking. In compute kernels, all tokens across requests are flattened and processed as independent elements. Intra-sequence dependencies are conveyed via a marker tensor integrated into sparse attention. On DeepSeek-V4, only the index-attention and compress kernels require modification to support variable-length routing.

**中文翻译**

系统将物理执行与逻辑序列追踪解耦。在计算 kernel 中，来自不同请求的所有 token 被 flatten，并作为独立元素处理。序列内部依赖则通过集成到 sparse attention 中的 marker tensor 传递。在 DeepSeek-V4 上，只需要修改 index-attention 和 compress kernels，就能支持 variable-length routing。  
译注：这段说明 DSpark 不是只在 Python 调度层工作，还需要底层 attention kernel 支持变长验证。

### 5.4 Performance under Live User Traffic

**English**

> DSpark-5, configured with maximum draft length gamma = 5, is evaluated against the MTP-1 production baseline within the serving engines of DeepSeek-V4-Flash and DeepSeek-V4-Pro. MTP-1 represents the former production setup and was maintained historically because static multi-token drafters such as MTP-3 or MTP-5 degrade aggregate throughput under high concurrency due to excessive verification overhead.

**中文翻译**

作者将 DSpark-5（最大 draft length `gamma = 5`）与 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro serving engine 中的 MTP-1 生产基线进行比较。MTP-1 代表此前的生产配置，历史上之所以维持这个单 token 设置，是因为静态多 token drafter（如 MTP-3 或 MTP-5）在高并发下会因过多验证开销降低聚合吞吐。  
译注：这明确说明 DSpark 的目标不是简单“比 MTP-1 多猜几个”，而是安全地释放多 token 草稿潜力。

**English**

> The serving Pareto frontier plots aggregate output token throughput against per-user generation speed. For V4-Flash, at a moderate 80 tok/s/user SLA, DSpark improves aggregate throughput by 51% over MTP-1. At a stricter 120 tok/s/user SLA, MTP-1 approaches its operational boundary and sustains only a very small concurrent batch, while DSpark achieves a much higher aggregate throughput. At matched practical throughput levels, DSpark accelerates per-user generation speeds by 60% to 85%.

**中文翻译**

serving Pareto frontier 展示聚合输出 token 吞吐与单用户生成速度之间的关系。对于 V4-Flash，在中等严格的 80 tok/s/user SLA 下，DSpark 相比 MTP-1 将聚合吞吐提升 51%。在更严格的 120 tok/s/user SLA 下，MTP-1 接近运行边界，只能维持很小的并发 batch，而 DSpark 达到明显更高的聚合吞吐。在匹配实际吞吐水平时，DSpark 将单用户生成速度提升 60% 到 85%。  
译注：极高 SLA 点的倍数提升很大，但作者也提醒它主要说明 DSpark 扩展了可行边界。

**English**

> The V4-Pro deployment shows the same pattern. At a moderate 35 tok/s/user SLA, DSpark improves aggregate throughput by 52%. At the stricter 50 tok/s/user SLA, MTP-1 again enters a low-concurrency regime, while DSpark sustains useful throughput. At matched system capacities, DSpark delivers 57% to 78% faster per-user generation.

**中文翻译**

V4-Pro 部署呈现相同模式。在中等严格的 35 tok/s/user SLA 下，DSpark 将聚合吞吐提升 52%。在更严格的 50 tok/s/user SLA 下，MTP-1 再次进入低并发 regime，而 DSpark 仍能维持有用吞吐。在匹配系统容量时，DSpark 带来 57% 到 78% 的单用户生成速度提升。  
译注：Flash 和 Pro 都有效，说明收益不是某个引擎的偶然现象。

**English**

> Throughput dynamics under load reveal the mechanism behind these gains. Under moderate concurrency, the hardware-aware scheduler uses available target compute capacity by allocating longer verification budgets, expanding from MTP-1’s static two tokens to roughly four to six tokens per request. As system concurrency increases and target capacity saturates, the scheduler dynamically restricts the budget, pruning low-confidence draft tokens before they consume critical batch capacity.

**中文翻译**

负载下的吞吐动态揭示了这些收益背后的机制。在中等并发下，硬件感知 scheduler 利用可用 target compute capacity，分配更长验证预算，从 MTP-1 的静态两个 token 扩展到每请求大约四到六个 token。随着系统并发增加、target capacity 饱和，scheduler 会动态限制预算，在低置信 draft token 消耗关键 batch capacity 之前剪掉它们。  
译注：这就是“轻载多验证，重载少验证”的生产行为。

**English**

> Although the prefix scheduler minimizes wasted target-model verification, DSpark still incurs a fixed draft-side cost to generate the initial gamma-token block via the parallel backbone. For complex queries with inherently low acceptance rates, this upfront drafting compute is unrecoverable. Future optimizations could introduce difficulty-aware early exiting within the draft model, enabling such requests to bypass full-block generation.

**中文翻译**

虽然 prefix scheduler 最小化了 target model 验证浪费，DSpark 仍然需要固定的 draft-side 成本，通过并行 backbone 生成初始 `gamma` token block。对于天然接受率低的复杂请求，这部分预先草稿计算无法回收。未来优化可以在 draft model 内引入 difficulty-aware early exiting，使这类请求绕过完整 block generation。  
译注：DSpark 解决了验证浪费，但还没有完全解决“低接受率请求是否值得生成长草稿”的问题。

---

## 6. Related Work

**English**

> Speculative decoding accelerates autoregressive generation by decoupling token proposal from verification. Modern approaches use rejection sampling to exactly preserve the target distribution. Because inference speedup depends directly on the drafter’s efficiency and accuracy, extensive research has focused on optimizing its architecture.

**中文翻译**

投机解码通过将 token proposal 与 verification 解耦，加速自回归生成。现代方法使用拒绝采样来精确保持 target distribution。由于推理加速直接取决于 drafter 的效率和准确性，大量研究都集中在优化 drafter 架构上。  
译注：DSpark 也属于这一脉络，但它还额外强调系统调度。

**English**

> Beyond standalone small language models, subsequent work integrates multi-token heads or feature extrapolators directly into the target model. Other strategies include self-speculation via early exits, dynamic vocabulary compression, prompt lookup, suffix automata, and retrieval. To remove the sequential bottleneck of drafting itself, recent methods propose parallel or blockwise generation, including P-EAGLE, PARD, DART, and DFlash.

**中文翻译**

除了使用独立小语言模型，后续工作还将 multi-token heads 或 feature extrapolators 直接集成到 target model 中。其他策略包括通过 early exit 做 self-speculation、动态词表压缩、prompt lookup、suffix automata 和 retrieval。为了移除草稿生成本身的顺序瓶颈，近期方法提出并行或块式生成，包括 P-EAGLE、PARD、DART 和 DFlash。  
译注：MTP、EAGLE、DFlash 分别代表了不同 drafter 设计方向。

**English**

> Beyond drafter architecture, another line of work focuses on determining the optimal number of speculative tokens to generate or verify in each round. Various approaches adapt draft lengths using confidence heuristics, learned acceptance predictors, or bandit-style policies. Recent system-level methods optimize goodput and latency by adjusting speculation budgets according to real-time system load and request priority.

**中文翻译**

除了 drafter 架构，另一条研究线关注每一轮应该生成或验证多少 speculative token。不同方法使用 confidence heuristics、learned acceptance predictors 或 bandit-style policies 来自适应调整 draft length。近期系统级方法则根据实时系统负载和请求优先级调整 speculation budget，以优化 goodput 和 latency。  
译注：DSpark 的 hardware-aware scheduler 就属于这条系统感知调度路线。

**English**

> Parallel generation models offer decoding latency nearly independent of output length, making them attractive alternatives to autoregressive decoding. Non-autoregressive Transformers pioneered this direction by predicting all positions independently in a single pass. However, this forces the model to average over plausible modes, often producing outputs that mix fragments from different valid sequences.

**中文翻译**

并行生成模型的解码延迟几乎不依赖输出长度，因此是自回归解码的有吸引力替代。Non-autoregressive Transformers 最早推动了这个方向，它们在一次 forward 中独立预测所有位置。然而，这会迫使模型对多个合理模式取平均，常常产生混合不同有效序列片段的输出。  
译注：这就是 DSpark 所说多模态碰撞的更广泛背景。

**English**

> Speculative decoding imposes an additional requirement: the drafter must provide exact per-token probabilities for the rejection sampling rule. Many parallel generation techniques cannot readily provide such probabilities due to iterative refinement, latent marginalization, or global normalization. DSpark circumvents these limitations by keeping the sequential correction local, so per-token probabilities remain exact softmax evaluations.

**中文翻译**

投机解码还施加了一个额外要求：drafter 必须为拒绝采样规则提供精确的 per-token probabilities。许多并行生成技术由于 iterative refinement、latent marginalization 或 global normalization，无法方便地提供这种概率。DSpark 通过保持顺序修正的局部性绕过了这些限制，因此每个 token 的概率仍然是精确 softmax 计算。  
译注：这解释了为什么 DSpark 不采用更复杂的全局结构层；无损拒绝采样需要精确 `p_d`。

---

## 7. Conclusion

**English**

> In this paper, we present DSpark, a speculative decoding framework designed to overcome the structural and system-level bottlenecks of large language model inference in high-concurrency production environments. Algorithmically, DSpark introduces a semi-autoregressive generation paradigm—coupling a computationally heavy parallel backbone with a lightweight sequential head—to mitigate the rapid suffix decay of independent parallel drafters.

**中文翻译**

本文提出 DSpark，一个投机解码框架，旨在克服高并发生产环境中大语言模型推理的结构性和系统级瓶颈。在算法上，DSpark 引入半自回归生成范式，将计算较重的并行 backbone 与轻量级顺序 head 结合起来，以缓解独立并行草稿模型的快速后缀衰减。  
译注：结论再次强调 DSpark 的第一条主线：更好的长草稿质量。

**English**

> At the system level, DSpark formulates verification length selection as a global throughput maximization problem, employing a hardware-aware prefix scheduler that dynamically tailors the target model’s verification budget based on calibrated survival probabilities and real-time engine load. Extensive offline evaluations demonstrate that DSpark substantially outperforms state-of-the-art autoregressive and parallel baselines across diverse domains.

**中文翻译**

在系统层面，DSpark 将验证长度选择表述为一个全局吞吐最大化问题，使用硬件感知前缀 scheduler，根据校准后的存活概率和实时引擎负载，动态定制 target model 的验证预算。广泛的离线评估表明，DSpark 在多个领域显著超过当前先进的自回归和并行 baseline。  
译注：这是 DSpark 的第二条主线：不是固定验证，而是按收益调度验证预算。

**English**

> Furthermore, real-world deployment within DeepSeek-V4 validates its practical value in production serving. By intelligently managing verification overhead, DSpark sustains robust concurrency under heavy load, consistently accelerates per-user generation speeds, and effectively shifts the Pareto frontier of LLM serving outward.

**中文翻译**

此外，DSpark 在 DeepSeek-V4 中的真实部署验证了它在生产 serving 中的实用价值。通过智能管理验证开销，DSpark 在重负载下保持稳健并发，持续提升单用户生成速度，并有效推动 LLM serving 的 Pareto frontier 向外移动。  
译注：这句总结了论文线上部署最重要的价值：既快，又不把高并发吞吐压垮。

---

## Appendix A. Counterexample: Selection Bias Without Early-Stopping

**English**

> We provide a simple counterexample to illustrate how an offline global search, operating without the break condition in Algorithm 1, violates the non-anticipating property required by lossless speculative decoding. Formally, the admission event for the k-th draft token must be determined by scheduler-visible information available before the token is sampled. It must not depend on the realization of the token itself.

**中文翻译**

作者给出一个简单反例，说明如果离线全局搜索不使用算法 1 中的 break condition，就会违反无损 speculative decoding 所需的 non-anticipating property。形式上，第 `k` 个 draft token 的接纳事件，必须由该 token 被采样之前 scheduler 可见的信息决定，不能依赖该 token 自身的实际取值。  
译注：这一点保证 scheduler 不会因为看到 token 内容后再决定是否接纳它，从而改变输出分布。

**English**

> Consider a scenario with a single request and maximum draft length two. Suppose the pre-token confidence for the first position is 0.8, and the profiled capacity curve gives SPS(1)=1.0, SPS(2)=0.5, and SPS(3)=0.45. The expected throughputs for verifying zero and one draft token are 1.0 and 0.9 respectively.

**中文翻译**

考虑一个只有单个请求、最大 draft length 为 2 的场景。假设第一个位置的 pre-token confidence 为 0.8，profile 出的 capacity curve 为 `SPS(1)=1.0`、`SPS(2)=0.5`、`SPS(3)=0.45`。验证 0 个和 1 个 draft token 的期望吞吐分别为 1.0 和 0.9。  
译注：此时如果使用 early stopping，因为验证 1 个比验证 0 个更差，就会直接停止。

**English**

> Without early stopping, the scheduler proceeds to evaluate the throughput for length two before committing any admission decisions. Because the Markov confidence head uses the previously sampled token, the next confidence score depends explicitly on the realization of the first token. If the first token yields high continuation confidence, length two may become the global maximum and the first token is admitted. If the first token yields low continuation confidence, length zero may remain best and the first token is not admitted.

**中文翻译**

如果没有 early stopping，scheduler 会在提交任何接纳决策之前继续评估长度为 2 的吞吐。由于 Markov confidence head 使用之前采样出的 token，下一个 confidence score 会显式依赖第一个 token 的实际取值。如果第一个 token 导致高后续置信度，长度 2 可能成为全局最大值，于是第一个 token 被接纳；如果第一个 token 导致低后续置信度，长度 0 可能仍然最好，于是第一个 token 不被接纳。  
译注：问题就在这里：第一个 token 是否被接纳，反过来依赖了第一个 token 自己是什么。

**English**

> Thus, the admission of the first draft token dynamically depends on the value of the first draft token itself. This retrospective dependence introduces selection bias: the scheduler favors tokens that lead to highly confident continuations, even though the admission decision for the first token should have been made before observing that token.

**中文翻译**

因此，第一个 draft token 是否被接纳，会动态依赖第一个 draft token 自身的值。这种回溯依赖引入了选择偏差：scheduler 会偏好那些导向高置信后续的 token，尽管第一个 token 的接纳决策本应在观察它之前就做出。  
译注：这说明“调度策略”也可能改变最终输出分布。

**English**

> To make the distributional bias explicit, let the vocabulary be {A, B}. Suppose the target distribution at the first position is p_t(A)=0.7 and p_t(B)=0.3, while the draft distribution is p_d(A)=0.5 and p_d(B)=0.5. The standard speculative acceptance probability at the first position is min(0.7,0.5)+min(0.3,0.5)=0.8.

**中文翻译**

为了明确展示分布偏差，令词表为 `{A, B}`。假设第一个位置的 target distribution 为 `p_t(A)=0.7`、`p_t(B)=0.3`，draft distribution 为 `p_d(A)=0.5`、`p_d(B)=0.5`。标准 speculative acceptance probability 为 `min(0.7,0.5)+min(0.3,0.5)=0.8`。  
译注：这个 0.8 与前面设定的第一个位置 pre-token confidence 对齐。

**English**

> Suppose the retrospective scheduler behaves as above: A yields high continuation confidence and therefore length two, while B yields low continuation confidence and therefore length zero. If the first draft token is A, it is admitted and accepted with probability one. If the first draft token is B, it is not admitted; the target model instead generates a fresh token from the target distribution.

**中文翻译**

假设回溯式 scheduler 的行为如前所述：`A` 导致高后续置信度，因此选择长度 2；`B` 导致低后续置信度，因此选择长度 0。如果第一个 draft token 是 `A`，它会被接纳，并以概率 1 被接受。如果第一个 draft token 是 `B`，它不会被接纳；target model 会改为从 target distribution 重新生成一个 fresh token。  
译注：这样 A 获得了额外机会，而 B 没有。

**English**

> Therefore, the output probability of A becomes 0.5 * 1 + 0.5 * 0.7 = 0.85, and the output probability of B becomes 0.15. This output distribution differs from the target distribution (0.7, 0.3), proving that the retrospective scheduler is not lossless.

**中文翻译**

因此，输出为 `A` 的概率变成 `0.5 * 1 + 0.5 * 0.7 = 0.85`，输出为 `B` 的概率变成 0.15。这个输出分布不同于 target distribution `(0.7, 0.3)`，证明回溯式 scheduler 不是无损的。  
译注：这就是选择偏差的数字证明。

**English**

> The early-stopping mechanism prevents this issue in the causal greedy scheduler. Since the throughput for length one is less than the throughput for length zero, the scheduler halts immediately and returns length zero before evaluating any continuation-dependent quantity. The admission decision for the first position therefore depends only on pre-token information and cannot be biased by the realization of the first token. This restores the non-anticipating property required by the standard losslessness argument.

**中文翻译**

early-stopping 机制可以在因果贪心 scheduler 中阻止这个问题。由于长度 1 的吞吐低于长度 0，scheduler 会立即停止并返回长度 0，而不会评估任何依赖后续的量。因此，第一个位置的接纳决策只依赖 pre-token information，不会被第一个 token 的实际取值偏置。这恢复了标准无损性论证所需的 non-anticipating property。  
译注：这也是为什么算法 1 里看似保守的 break 条件在理论上很重要。

---

## References

参考文献列表主要由论文、报告、代码仓库和网页引用构成，未在本英中对照版中逐条翻译。阅读 DSpark 主线时，最相关的几个参考入口如下：

- DSpark / DeepSpec 官方仓库：https://github.com/deepseek-ai/DeepSpec
- DeepSeek-V3 Technical Report（MTP 相关）：https://arxiv.org/abs/2412.19437
- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty：https://arxiv.org/abs/2401.15077
- EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test：https://arxiv.org/abs/2503.01840
- Better & Faster Large Language Models via Multi-token Prediction：https://proceedings.mlr.press/v235/gloeckle24a.html
