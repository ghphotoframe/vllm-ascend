# DSpark 论文中文详细解读版：从 Transformer 初学者视角读懂半自回归投机解码

> 这是面向初学者的详细版。当前目录中原有的简约译注版 `DSpark_paper_zh_annotated.md` 已保留不动。  
> 本文档不是逐字直译，而是“翻译 + 解释 + 例子 + 对比”。目标读者是假设你只知道一点 Transformer：知道模型会根据上下文预测下一个 token，大概听过 attention、logits、softmax、KV cache，但还不熟悉 speculative decoding、MTP、EAGLE、DFlash、serving 调度这些概念。

## 0. 先用一句话说清楚 DSpark

DSpark 想解决的问题是：大模型一个 token 一个 token 地生成太慢，于是我们让一个更便宜的草稿模块先猜一串 token，再让大模型一次性检查这些 token；但草稿不能乱猜，检查也不能无脑检查太长，否则高并发时反而更慢。

DSpark 的做法有两半：

1. **草稿怎么猜得又快又准**：用并行模型一次猜一串，再加一个很轻的小顺序模块，让这一串 token 内部更连贯。
2. **草稿该检查多长**：每个请求、每个位置都估计“这个前缀有多大概率能被大模型接受”，再结合 GPU 当前忙不忙，动态决定验证几个 token。

用生活化类比：

- **target model** 像正式老师，批改准确但慢。
- **draft model** 像助教，先快速写出草稿答案。
- **speculative decoding** 像让助教一次写几步，老师一次性检查。前面都对，就省下老师一步步写的时间；一旦中间错了，后面的草稿都作废。
- **DSpark** 的新意是：助教先并行写一串，再快速检查草稿之间是否连贯；同时系统会判断老师现在忙不忙，如果老师很忙，就只拿最可能对的前几步去批改。

## 1. 阅读路线图

这篇论文可以按三层理解：

| 层次 | 要回答的问题 | 对应论文部分 |
|---|---|---|
| 解码算法层 | 如何在不改变输出分布的前提下让 LLM 生成更快？ | 第 2 节背景 |
| 草稿模型层 | 如何让 draft model 又快又有块内依赖？ | 第 3.1 节半自回归生成 |
| serving 系统层 | 高并发时，到底该验证多少草稿 token？ | 第 3.2 节调度，第 5 节部署 |

如果你是第一次接触这类论文，建议先抓住三个词：

- **accepted length**：每一轮投机解码平均能被大模型接受多少 token。越大越好。
- **suffix decay**：草稿越往后，越容易被拒绝。尤其是纯并行草稿模型，后缀容易不连贯。
- **verification waste**：草稿 token 被送去 target model 验证，但很快被拒绝，白白占用 batch 容量。

## 1.5 前置知识：MTP、EAGLE、EAGLE3、DFlash 和投机推理路线图

在读 DSpark 之前，最好先把几个名字放到同一张地图里。它们都想解决同一个大问题：

```text
target model 一个 token 一个 token 生成太慢，
能不能先便宜地猜多个 token，
再让 target model 一次性验证？
```

但不同方法选择的“草稿怎么来”不一样。可以先按两个维度分类：

```text
维度 1：草稿 token 之间有没有依赖？
  有依赖：后一个 token 知道前面实际采样了什么，更连贯，但更串行。
  弱依赖/无显式采样依赖：一次生成多个位置，更快，但后缀容易不连贯。

维度 2：优化重点是模型结构，还是 serving 调度？
  模型结构：让 draft token 猜得更准。
  serving 调度：决定哪些 draft token 值得送去 target model 验证。
```

### 1.5.1 一条简化 roadmap

下面这条路线不是严格按论文年份排队，而是按思想演进来理解：

```text
普通自回归解码
  每次 target model 只生成 1 个 token，最稳但慢。

经典 speculative decoding
  小 draft model 先猜几个 token，target model 一次验证。
  关键问题：draft model 要便宜且接近 target。

MTP / 多 token 预测头
  在模型内部增加预测未来多个 token 的能力。
  常见直觉：不用另起一个小模型，而是在 target 模型旁边长出“未来 token 预测分支”。

EAGLE / EAGLE3
  利用 target model 的中间特征做更强的自回归草稿。
  关键直觉：不要只在 token id 上猜，借 target hidden states 帮忙。

DFlash / 并行 block drafter
  一次 forward 预测整个草稿 block，draft latency 对 block size 不敏感。
  关键问题：多个位置没有基于实际采样前缀条件化，后缀容易衰减。

DSpark
  继承 DFlash 式并行 backbone，再加轻量 Markov/RNN head 补一点块内依赖。
  同时增加 confidence scheduler，按负载动态决定验证多长。
```

更短一点说：

```text
MTP / EAGLE / EAGLE3：更偏“怎么生成高质量草稿”
DFlash：更偏“怎么一次快速生成长草稿”
DSpark：既要长草稿快，也要后缀更连贯，还要线上验证不浪费
```

### 1.5.2 MTP：让模型具备预测未来多个 token 的能力

MTP 是 Multi-Token Prediction，多 token 预测。它的基本想法是：普通语言模型通常只训练“预测下一个 token”，而 MTP 让模型同时或逐步学习预测更远的未来 token。

普通 next-token prediction 像这样：

```text
当前位置 hidden state -> 预测 x_{t+1}
```

MTP 想多看几步：

```text
当前位置 hidden state -> 预测 x_{t+1}
当前位置 hidden state / 后续 MTP 状态 -> 预测 x_{t+2}
当前位置 hidden state / 后续 MTP 状态 -> 预测 x_{t+3}
```

在 DeepSeek-V3 技术报告和很多工程实现语境中，MTP 模块会尽量保留完整因果链。也就是说，预测更后面的 token 时，会利用前面位置的信息，而不是完全独立地一次性猜所有位置。这样做的好处是：

- token 间依赖更强，草稿更连贯。
- MTP 层可以和主模型共享 embedding、LM head 等结构。
- 训练时多 token 预测也能给模型提供更密集的训练信号。

但它的问题也很现实：

- 如果推理时要预测多个 token，草稿路径仍然带有顺序成本或额外层成本。
- 如果生产系统固定验证 MTP-3、MTP-5 这类多 token 草稿，高并发下可能把 target verification batch 撑大。
- 因此很多生产场景会保守使用 MTP-1，也就是只多预测/验证一个 token。

所以你前面说“mtp 这种预测是串行的，后一个 token 预测依赖前一个 token，因此 token 有依赖关系，但是串行效率低”，作为 DeepSeek/VLLM 语境下的直觉基本是对的。更严谨一点说：广义 MTP 可以有不同实现，但 DSpark 论文里拿来做生产对比的 MTP-1，是偏保守、强因果链、低风险的基线。

### 1.5.3 EAGLE：在特征空间做自回归草稿

EAGLE 的出发点是：draft model 如果只在 token 层面猜，可能和 target model 的内部状态不够对齐。既然 target model 在生成时会产生 hidden states，那能不能让 draft model 利用这些 hidden states，预测 target model 下一步可能处在什么特征状态？

可以用一个简化流程理解 EAGLE：

```text
target model 已经算出当前上下文的 hidden states
EAGLE drafter 读取这些特征
EAGLE 预测下一步的 feature / token
采样出一个 draft token
再基于这个 draft token 继续预测下一个
```

它的核心优势是：

- 比普通小 draft model 更贴近 target model。
- 因为使用 target feature，草稿质量通常更强。
- 生成后续 token 时会依赖前面已经采样出的 token，因此块内依赖较好。

但它仍然属于自回归 drafter 路线。也就是说，要生成多个 draft token，仍然需要一步一步推进。草稿长度越长，draft latency 越容易增长。

注意：EAGLE 不是 DSpark 摘要里说的那类“所有位置一次 forward 独立预测”的纯并行 drafter。DSpark 论文把 Eagle3 明确作为 autoregressive baseline，而把 DFlash 作为 parallel baseline。

### 1.5.4 EAGLE3：更强的 EAGLE 系自回归草稿

EAGLE3 可以理解为 EAGLE 路线的强化版。相比早期 EAGLE，它更强调：

- 使用 target model 多层特征，而不是只依赖单一层特征。
- 减少或放弃“必须精确预测下一层 feature”的约束，转向更直接的 token 预测目标。
- 通过 Training-Time Test 等训练方式，让 drafter 在训练时更贴近推理时的使用形态。

对初学者来说，不必一开始纠结 EAGLE3 的所有训练细节，可以先抓住这个定位：

```text
EAGLE3 是强自回归 drafter。
它的强项是草稿质量和块内条件依赖。
它的弱项是长草稿仍然有顺序生成成本。
```

这也是 DSpark 论文为什么拿 Eagle3 做强 baseline：如果 DSpark 能超过 Eagle3，说明它不是只赢了弱 baseline，而是在“强自回归草稿”和“强并行草稿”之间找到了更好的折中。

### 1.5.5 DFlash：一次 forward 生成整段草稿 block

DFlash 代表另一条路线：不要一步一步生成草稿，而是一次 forward 直接生成一整个 block。

它的直觉可以写成：

```text
输入 anchor token + 若干 mask token
利用 target model 的上下文特征做 KV injection
一次 forward 输出多个 mask 位置的 logits
一次性得到多个 draft token
```

这样做的好处非常明显：

- draft latency 几乎不随草稿长度线性增长。
- 可以用更深的 draft backbone。
- 第一个 draft token 往往很强，因为并行模型容量更大。

但它也有一个核心问题：

```text
第 2、3、4 个位置在预测时，
并不知道第 1 个位置最后实际采样了什么。
```

例如上下文可能接：

```text
of course
no problem
```

并行模型可能同时觉得：

```text
第 1 位：of / no 都合理
第 2 位：course / problem 都合理
```

因为它不是先采样出 `of` 再去预测 `course`，所以可能混出：

```text
of problem
no course
```

这就是 DSpark 反复提到的 multi-modal collision 和 suffix decay。越靠后的 token 越依赖前面实际选了什么，纯并行模型越容易吃亏。

### 1.5.6 DSpark 站在这些方法的哪里

DSpark 的定位很清楚：

```text
它不是回到完全自回归 drafter；
也不是接受纯并行 drafter 的后缀衰减；
而是在 DFlash 式并行 backbone 后面，
加一个很便宜的 Markov/RNN head，
让后面的 token 知道前面实际采样了什么。
```

所以 DSpark 的草稿生成可以理解为：

```text
第一步：DFlash 式并行 backbone 给每个位置一个基础预测
第二步：Markov/RNN head 从左到右做轻量修正
第三步：confidence head 估计每个位置的条件接受概率
第四步：scheduler 根据系统负载决定验证几个 token
```

一张表总结：

| 方法 | 草稿生成方式 | token 间依赖 | 长草稿 draft 成本 | 主要问题 | DSpark 如何吸收/改进 |
|---|---|---|---|---|---|
| MTP | 模型内部预测未来 token，常保留因果链 | 较强 | 中到高，依实现而定 | 固定多 token 验证高并发下可能浪费 | 用动态 scheduler 避免固定验证长度 |
| EAGLE | 利用 target feature 自回归生成草稿 | 强 | 随草稿长度增长 | 长草稿顺序成本高 | DSpark 保留并行 backbone，降低长草稿成本 |
| EAGLE3 | 更强的 EAGLE 系自回归 drafter | 强 | 随草稿长度增长 | 仍受自回归 draft latency 限制 | DSpark 用半自回归折中质量与速度 |
| DFlash | 一次 forward 并行生成 block | 弱 | 低 | 后缀衰减、多模态碰撞 | DSpark 加 Markov/RNN head 补块内依赖 |
| DSpark | 并行 backbone + 轻量顺序 head + 动态验证调度 | 中到强 | 低到中 | draft side 对低接受率请求仍有固定成本 | 后续可做 difficulty-aware early exit |

从这张表再看 DSpark，就容易理解它为什么要同时做两件事：

```text
DFlash 解决 draft 快的问题，但后缀弱；
EAGLE/EAGLE3/MTP 解决依赖问题，但长草稿更串行；
DSpark 试图用很小的顺序成本，补上并行草稿的后缀依赖；
然后再用 scheduler，避免多出来的草稿在高并发下拖垮 target verification。
```

## 2. 必要背景：Transformer 生成为什么慢

### 2.1 自回归生成

大多数 LLM 都是自回归生成。意思是：

```text
给定前文 x_1, x_2, ..., x_t
模型预测下一个 token x_{t+1}
然后把 x_{t+1} 拼回上下文，再预测 x_{t+2}
```

例如模型要生成一句话：

```text
我 今天 想 去 图书馆
```

它不是一次生成整句，而是：

```text
我 -> 今天 -> 想 -> 去 -> 图书馆
```

每一步都要跑一次模型。即使有 KV cache，decode 阶段每生成一个 token 仍然要做一次 forward。模型越大，单步越贵；输出越长，总耗时越长。

### 2.2 logits 和 softmax 是什么

模型每一步不会直接说“下一个 token 是 A”，而是输出整个词表的分数，也就是 logits。再通过 softmax 变成概率分布。

例如词表只有四个 token：

```text
["苹果", "香蕉", "今天", "因为"]
```

模型输出 logits 后，softmax 可能得到：

```text
苹果 0.60
香蕉 0.20
今天 0.15
因为 0.05
```

采样时可能选择概率最高的，也可能按概率随机抽一个。这一点很重要，因为 speculative decoding 要保证加速后采样分布仍然等于 target model 的分布，不能只是“看起来差不多”。

### 2.3 为什么不是直接让模型一次生成多个 token

直觉上，你可能会问：既然一个 token 一个 token 慢，为什么不让模型一次输出 5 个 token？

难点在于，后面的 token 依赖前面的 token。比如上下文是：

```text
不用客气，祝你
```

可能续写：

```text
生活愉快
工作顺利
今天开心
```

第 2 个 token 该是什么，取决于第 1 个 token 选了什么。如果第 1 个 token 是“生活”，第 2 个更可能是“愉快”；如果第 1 个 token 是“工作”，第 2 个更可能是“顺利”。

纯并行模型如果每个位置独立猜，就可能混出：

```text
生活顺利
工作愉快
```

这不一定完全错，但会降低与 target model 的一致性。论文里把这种问题叫多模态碰撞，也就是多个合理续写模式被搅在一起。

## 3. 投机解码：让小模型先猜，大模型再验

### 3.1 基本流程

投机解码的基本流程可以写成：

```text
1. draft model 先生成 gamma 个候选 token
2. target model 一次 forward 验证这 gamma 个 token
3. 从左到右接受最长正确前缀
4. 一旦某个位置拒绝，后面的候选都丢弃
5. target model 额外生成一个 bonus token，进入下一轮
```

例子：

```text
上下文：我 今天 想
draft 猜：去 图书馆 看 书
target 验证：
  去：接受
  图书馆：接受
  看：拒绝
那么本轮接受：去 图书馆
target 生成修正 token：学习
```

最后输出变成：

```text
我 今天 想 去 图书馆 学习
```

后面的“书”虽然可能不错，但因为“看”被拒绝，它就不能继续用。投机解码只接受连续前缀。

### 3.2 为什么它是无损的

投机解码不是简单地“让小模型替大模型写”。它有拒绝采样规则，可以保证最终输出分布与 target model 一样。

在第 `k` 个 draft token `x_k` 上：

- draft model 给它的概率是 `p_d(x_k)`
- target model 给它的概率是 `p_t(x_k)`

接受概率为：

```text
min(1, p_t(x_k) / p_d(x_k))
```

直觉解释：

- 如果 target model 比 draft model 更喜欢这个 token，那就一定接受。
- 如果 draft model 过度自信，而 target model 没那么喜欢，就按比例接受。

这样做的效果是：被接受的 token 分布不会偏离 target model。这个“无损”很重要，尤其是生产系统不能为了速度随便改变模型行为。

### 3.3 速度取决于什么

论文用一个公式概括投机解码每生成一个 token 的平均延迟：

```text
L = (T_draft + T_verify) / tau
```

其中：

- `T_draft`：draft model 生成草稿的时间。
- `T_verify`：target model 验证草稿的时间。
- `tau`：本轮平均接受 token 数，也就是 accepted length。

想变快，有三条路：

1. **draft 更快**：降低 `T_draft`。
2. **draft 更准**：提高 `tau`。
3. **验证更聪明**：降低有效 `T_verify`，别验证明显会错的 token。

DSpark 三条都碰到了：

- 并行 backbone 让 draft 快。
- 半自回归 head 让 draft 准。
- confidence scheduler 让验证更聪明。

## 4. 现有 drafter 的两条路线

### 4.1 自回归 drafter：一步一步猜

自回归 drafter 和普通 LLM 一样，也是一步一步生成草稿：

```text
先猜第 1 个 token
再基于第 1 个 token 猜第 2 个
再基于前两个猜第 3 个
```

优点：

- token 之间依赖清楚。
- 后面的 token 能知道前面实际采样了什么。
- 生成出来通常更连贯。

缺点：

- 要猜 `gamma` 个 token，draft 也要跑 `gamma` 步。
- 草稿越长，draft latency 越高。
- 为了控制延迟，模型通常不能太深，草稿长度也不能太长。

EAGLE、EAGLE3 大体属于这一类强自回归 drafter。它们不是普通小模型，而是利用 target model 的特征来做更贴近 target 的草稿预测，但生成草稿仍然有明显顺序性。

这里很容易混淆：论文中说的“并行草稿模型缺少 token 之间依赖”，主要不是指 EAGLE/EAGLE3，而是指 DFlash/PARD/DART 这类一次 forward 同时预测多个草稿位置的模型。EAGLE/EAGLE3 反而更接近自回归 drafter，后一个 draft token 会依赖前面已经采样出的 draft token，所以块内关联更强，但草稿长度增加时顺序成本更明显。MTP 在很多工程实现中也更接近保留因果链/逐步预测的一侧，因此可以粗略理解为“依赖关系更强，但多 token 草稿或固定多 token 验证在高并发下成本更高”。不过在 DSpark 这篇论文里，`并行但后缀衰减` 的直接 baseline 是 DFlash，不是 EAGLE/EAGLE3。

### 4.2 并行 drafter：一次猜一串

并行 drafter 一次 forward 直接输出多个位置：

```text
位置 1：猜 token A
位置 2：猜 token B
位置 3：猜 token C
...
```

优点：

- draft latency 基本不随草稿长度线性增长。
- 可以用更深的 draft backbone。
- 第一个 token 往往预测得很强，因为模型容量更大。

缺点：

- 每个位置不知道前面实际采样了什么。
- 后缀容易越来越不可靠，也就是 suffix decay。

DFlash 就是论文采用的强并行 baseline。它会从 target model 的多个层抽取 hidden states，并注入 draft model 的 KV 中，让 draft model 有丰富上下文。

### 4.3 DSpark 为什么要折中

DSpark 观察到：

- 自回归 drafter 后缀连贯，但长草稿慢。
- 并行 drafter 首 token 强，长草稿便宜，但后缀衰减。

所以它采用折中：

```text
重计算：并行做
轻依赖：顺序补
```

也就是：

- 大部分计算仍然像 DFlash 一样并行跑。
- 只在输出层附近加一个很轻的顺序 head，给后面的 token 加一点“我知道前面选了什么”的信息。

## 5. DSpark 架构详解

### 5.1 整体解码流程

论文图 1 可以用下面的流程解释：

```text
已有上下文：A B C

1. target model 先生成一个 token D
   D 作为 anchor token

2. DSpark 用 D 作为起点生成草稿：
   E F G H

3. DSpark 同时给每个位置打置信度：
   c1 c2 c3 c4

4. scheduler 判断：
   E、F、G 值得验证
   H 置信度太低，先丢掉

5. target model 验证 E F G：
   E 接受
   F 接受
   G 拒绝

6. target model 生成修正 token G*
   进入下一轮
```

这比固定验证 `E F G H` 更省，因为 `H` 可能几乎肯定会被前面的拒绝连带丢掉，或者自己通过概率很低。

### 5.2 anchor token 是什么

anchor token 是上一轮 target model 真正生成的最后一个 token。它是可信的，因为它来自 target model。

为什么需要 anchor？

因为 draft model 需要一个可靠起点。比如上下文已经生成到：

```text
今天 天气 很
```

target model 生成 anchor：

```text
好
```

下一轮 draft model 就可以从“好”之后开始猜：

```text
， 我们 去
```

论文有时也把 anchor token 和 bonus token 混用，因为 bonus token 通常会成为下一轮的 anchor。

## 6. 半自回归生成：DSpark 的第一个核心

### 6.1 纯并行为什么会 suffix decay

设上下文是：

```text
没问题，我马上
```

可能的续写有：

```text
帮你处理
给你回复
开始检查
```

纯并行 drafter 同时预测 3 个位置。位置 1 可能觉得“帮”“给”“开始”都不错；位置 2 也在多个模式之间摇摆；位置 3 同样如此。它不是先选了“帮”，再基于“帮”去选“你”，而是每个位置都在混合所有可能路径。

于是可能出现：

```text
帮 你 检查
给 你 处理
开始 你 回复
```

越往后，这种不一致越容易累积。target model 从左到右验证时，后缀通过率就下降。

### 6.2 DSpark 的两阶段生成

DSpark 把草稿生成拆成：

```text
并行阶段：一次产生每个位置的基础 logits 和 hidden states
顺序阶段：从左到右采样，同时给 logits 加前缀相关 bias
```

可以把并行阶段看成“每个位置先给出初稿意见”，顺序阶段看成“根据前面已经选出来的词，给后面位置做微调”。

公式写法是：

```text
p_k(v | x_0, x_<k)
  = softmax(U_k(v) + B_k(x_0, x_<k, v))
```

其中：

- `U_k(v)` 是并行 backbone 对第 `k` 个位置给 token `v` 的基础分数。
- `B_k(...)` 是顺序 head 给 token `v` 加的修正分数。
- `x_0` 是 anchor。
- `x_<k` 是当前 block 中已经采样出来的前缀。

朴素理解：

```text
最终分数 = 并行模型原本觉得它好不好 + 前缀连贯性修正
```

### 6.3 Markov head：只看前一个 token 的轻量修正

Markov head 是默认版本。它只看上一个 token。

例如已经采样：

```text
of
```

那么下一个 token：

```text
course
```

应该被加分，而：

```text
problem
```

应该被减分。

如果前一个 token 是：

```text
no
```

那么：

```text
problem
```

应该被加分。

这就是一阶转移：只看 `x_{k-1}` 对 `x_k` 的影响。

理论上可以学一个巨大矩阵：

```text
前一个 token x 当前 token
```

如果词表有 10 万个 token，这个矩阵就是：

```text
100000 x 100000
```

太大了。DSpark 用低秩分解：

```text
B = W1 W2
```

其中：

- `W1[x_{k-1}]` 像查表，取出前一个 token 的小向量。
- `W2` 把这个小向量投影回整个词表的 bias。

默认中间维度 `r=256`，比完整词表矩阵小得多。

### 6.4 RNN head：记住更长的块内历史

Markov head 只看前一个 token。如果有些依赖跨度更长，就不够了。

例如：

```text
if 用户 已经 登录 ， 那么
```

后面生成什么可能依赖更早的“if”和“登录”，不只是前一个 token。

RNN head 用一个小状态 `s_k` 记录 block 内已经采样的历史。每一步把：

- 上一步状态 `s_{k-1}`
- 前一个 token embedding
- 并行 backbone 的 hidden state `h_k`

拼在一起，更新成新的状态，再输出 bias。

实验里 RNN head 在长草稿时略有提升，但更复杂，不如 Markov head 好部署，所以论文默认 Markov head。

### 6.5 为什么说“一点自回归就够了”

DSpark 的关键不是重新做一个完整自回归 draft model，而是只加一点点顺序性。

原因是：

- 并行 backbone 已经很强，尤其第一个 token。
- 后缀问题主要来自“完全不知道前面采样了什么”。
- 只要给后面 token 一些局部转移信息，就能显著减少不连贯后缀。

论文实验显示，2 层 DSpark 就能超过 5 层 DFlash，说明加轻量依赖比单纯堆更多并行层更有效。

## 7. 置信度调度式验证：DSpark 的第二个核心

### 7.1 为什么不能总是验证全部草稿

假设 draft model 每轮都生成 5 个 token。要不要都给 target model 验证？

低并发时：

```text
GPU 还有空闲
多验证几个 token 就算错了也不太亏
```

高并发时：

```text
很多请求排队
每个多验证的 token 都占用 batch 容量
如果这些 token 大概率会错，就很亏
```

所以固定验证长度并不理想。更好的策略是：

```text
当前请求的草稿很可靠 -> 多验证几个
当前请求的草稿不可靠 -> 少验证几个
系统负载低 -> 可以宽松一点
系统负载高 -> 只验证高收益 token
```

这就是 DSpark 的 confidence-scheduled verification。

### 7.2 confidence head 预测什么

confidence head 给每个位置输出一个数：

```text
c_k in (0, 1)
```

它不是简单表示“这个 token 看起来好不好”，而是条件概率：

```text
如果前面的 token 都已经被接受，
第 k 个 token 被接受的概率是多少？
```

例如 draft 是：

```text
去 图书馆 看 书
```

confidence 可能是：

```text
去       c1 = 0.95
图书馆   c2 = 0.90
看       c3 = 0.60
书       c4 = 0.50
```

那么前缀存活概率是累计乘积：

```text
第 1 个 token 存活：0.95
前 2 个都存活：0.95 * 0.90 = 0.855
前 3 个都存活：0.95 * 0.90 * 0.60 = 0.513
前 4 个都存活：0.95 * 0.90 * 0.60 * 0.50 = 0.2565
```

注意第 4 个 token 自己的 `c4=0.50` 看起来还行，但作为“前 4 个都通过”的概率只有 0.2565。因为它依赖前面全部通过。

### 7.3 confidence 的训练标签从哪里来

论文用 draft 分布和 target 分布的 total variation distance 构造软标签：

```text
c*_k = 1 - 1/2 * ||p_d_k - p_t_k||_1
```

直觉：

- 如果 draft 分布和 target 分布很像，说明 draft 很可靠，接受概率高。
- 如果两个分布差很多，说明 draft 容易被 target 拒绝，接受概率低。

这比只看采样出的 token 是否最终被接受更平滑，因为它利用了完整分布信息。

### 7.4 为什么要校准

神经网络常常过度自信。例如它说：

```text
我有 90% 把握
```

但真实统计下来可能只有 75%。

如果只是排序，过度自信问题没那么严重。比如我们只要知道 A 比 B 可靠就行。但是 DSpark scheduler 要计算：

```text
预期接受 token 数 * 当前 batch size 下的 steps per second
```

这需要概率数值本身可信。0.9 和 0.7 的差别会直接影响是否值得验证后缀。

所以论文使用 Sequential Temperature Scaling（STS）校准。可以理解为给每个位置的 confidence 做温度修正，让预测概率和真实接受率对齐，同时不改变排序。

## 8. 硬件感知前缀调度器

### 8.1 它要优化什么

一次 target verification 会把多个请求的 token 合在一个 batch 里跑。假设有 `R` 个请求，每个请求都至少要让 target model 生成或验证一个 token，因此基础 batch size 是 `R`。

如果给某个请求多验证一个 draft token，batch size 就多 1。验证更多 token 可能增加 accepted length，但也可能让 target forward 变慢。

调度器的目标是最大化：

```text
Theta = tau * SPS(B)
```

其中：

- `tau`：这一批请求预期总共能产出多少被接受 token。
- `B`：target model 这一步要处理的 token 数。
- `SPS(B)`：batch size 为 `B` 时，引擎每秒能跑多少 step。

朴素理解：

```text
总吞吐 = 每一步能产出多少有效 token * 每秒能跑多少步
```

### 8.2 一个小例子

假设当前有 2 个请求：

```text
请求 A 的前缀存活概率：
A1 = 0.90
A2 = 0.72
A3 = 0.30

请求 B 的前缀存活概率：
B1 = 0.85
B2 = 0.40
B3 = 0.10
```

如果 GPU 很空，scheduler 可能验证：

```text
A 验 2 个
B 验 2 个
```

因为额外 token 的机会成本低。

如果 GPU 很忙，scheduler 可能只验证：

```text
A 验 2 个
B 验 1 个
```

甚至：

```text
A 验 1 个
B 验 1 个
```

因为 `A3=0.30`、`B2=0.40`、`B3=0.10` 这些 token 预期收益不够高。

### 8.3 为什么按前缀存活概率排序是合理的

对同一个请求来说，越往后的前缀存活概率不会更高：

```text
a_{r,1} >= a_{r,2} >= a_{r,3} ...
```

因为每往后一步都要乘一个小于等于 1 的 confidence。

所以可以把所有请求的候选扩展放在一个池子里，按 `a_{r,j}` 从大到小排序。高的先拿去验证，低的后拿。

但是还要注意前缀约束：

```text
不能验证第 3 个 token，却不验证第 1、2 个 token
```

累计存活概率的单调性保证了全局排序不会破坏这个约束：同一请求中第 1 个一定排在第 2 个前，第 2 个一定排在第 3 个前。

### 8.4 算法 1 的朴素版本

论文算法可以解释为：

```text
1. 先假设每个请求都不验证 draft token，只让 target model 正常走一步
2. 计算当前吞吐
3. 把所有可能加入验证的 draft token 按收益排序
4. 一个一个尝试加入
5. 每加入一个，就更新：
   - 预期 accepted token 数
   - batch size
   - 查表得到 SPS(batch size)
   - 计算吞吐
6. 如果吞吐变好，就保留
7. 如果吞吐不变好，就停止
```

这个策略的本质是：

```text
把 target model 的验证预算分配给最值得验证的 token
```

### 8.5 为什么需要 early stopping 保证无损

这是论文中较难但很关键的一点。

无损 speculative decoding 要求：是否验证某个 token，不能依赖这个 token 采样出来之后的信息，更不能依赖未来 token。

举个简化例子：

```text
第 1 个 draft token 可能是 A 或 B
A 会让后面的 confidence 很高
B 会让后面的 confidence 很低
```

如果 scheduler 先偷看了第 1 个 token 是 A 还是 B，再决定要不要验证第 1 个 token，就会产生偏差：

- 看到 A：觉得后续收益高，于是验证 A。
- 看到 B：觉得后续收益低，于是不验证 B，让 target model 重新采样。

这样输出就会偏向 A，偏离 target model 原始分布。

论文附录用数字证明了这一点：

```text
target 分布：A 0.7, B 0.3
draft  分布：A 0.5, B 0.5
```

如果 scheduler 因为 A 后续置信高而接纳 A，因为 B 后续置信低而不接纳 B，最后输出 A 的概率会变成 0.85，而不是 target 的 0.7。

所以理论算法里，一旦吞吐不再提升就 early stop，避免继续看会依赖未来 token 的信息。

### 8.6 生产系统里怎么处理真实硬件曲线

理论算法假设 `SPS(B)` 比较平滑。但真实 GPU kernel 的吞吐曲线经常是锯齿状的。例如 batch size 从 127 到 128 可能很好，从 128 到 129 可能突然掉一下，因为底层矩阵计算、CUDA graph、kernel tile 都有离散边界。

如果严格 early stop，可能卡在局部点，错过后面更好的 batch size。

DeepSeek 的生产实现用了异步方案：

- 当前 step 的 token 仍按最新 confidence 排序。
- 但“这一步最多验证多少 token”的容量 `K` 用两步之前的信息预测。
- 这样调度不会偷看当前 token 的实现，仍保持因果。
- 同时可以做更自由的全局搜索，绕过锯齿状硬件曲线。

直觉上，这是把“容量规划”和“当前 token 选择”隔开：

```text
容量规划：用历史信息，避免偷看当前 token
token 选择：用当前 confidence 排序，优先验证最可靠 token
```

## 9. 训练目标详解

DSpark 训练时 target model 冻结。draft model 共享 target model 的 embedding 和 LM head，也冻结。训练更新：

- 并行 draft backbone
- Markov/RNN sequential head
- confidence head

### 9.1 为什么共享 embedding 和 LM head

embedding 层把 token id 变成向量。LM head 把 hidden state 映射回词表 logits。

共享它们有两个好处：

1. draft model 和 target model 使用同一套 token 表示空间，更容易对齐。
2. 冻结这些大矩阵可以减少训练参数和不稳定性。

### 9.2 位置加权

论文给每个位置一个权重：

```text
w_k = exp(-(k - 1) / gamma)
```

越靠前的 token 权重越大。

原因是 speculative decoding 只接受连续前缀。第一个 token 错了，后面全作废；第一个 token 对 accepted length 的影响最大。

用例子说：

```text
草稿：A B C D E
```

如果 A 错，接受长度可能只有 bonus token。  
如果 E 错，A B C D 仍可能都被接受。  
所以训练时更重视前面位置很合理。

### 9.3 三个损失

#### Cross-entropy loss

让 draft model 学会预测真实下一个 token：

```text
L_ce = - sum w_k log p_d(x*_k)
```

这就是语言模型常见训练目标。

#### Distribution-matching loss

让 draft 分布接近 target 分布：

```text
L_tv = sum w_k ||p_d - p_t||_1
```

为什么这比只预测标准答案更贴近 speculative decoding？

因为投机解码关心的是 draft 分布和 target 分布是否一致。即使 ground truth token 是某个词，target model 可能认为另一个词也很合理。让两个分布接近，能直接提高拒绝采样接受率。

#### Confidence loss

训练 confidence head 预测软接受概率：

```text
L_conf = BCE(c_k, c*_k)
```

这里标签 `c*_k` 不是硬 0/1，而是根据 draft-target 分布差异算出来的软概率。

总损失：

```text
L = 0.1 L_ce + 0.9 L_tv + 1.0 L_conf
```

可以看到论文更重视 `L_tv`，因为它直接对应接受率。

## 10. 实验结果怎么读

### 10.1 accepted length 是什么

论文主要离线指标是每轮 accepted length `tau`。它包含 bonus token。

如果 `tau = 5`，大概表示每轮 target verification 后平均能推进 5 个 token。越大，单位 target forward 能产出的 token 越多。

但注意：

```text
accepted length 高不一定线上吞吐就一定高
```

因为线上还要考虑：

- draft 成本
- target verification batch 变大后的成本
- 并发请求数
- GPU kernel 利用率
- KV cache 资源

这就是 DSpark 为什么还要做 scheduler。

### 10.2 主结果表说明

论文在 Qwen3-4B、8B、14B 和 Gemma4-12B 上比较：

- Eagle3：强自回归 drafter。
- DFlash：强并行 drafter。
- DSpark：半自回归 + 默认 Markov head。

结果显示 DSpark 在数学、代码、聊天三类任务上都更高。

例如 Qwen3-4B：

```text
Math:
Eagle3 约 4.56 到 5.14
DFlash 约 4.15 到 5.40
DSpark 约 4.89 到 6.11

Chat:
Eagle3 约 2.26 到 2.55
DFlash 约 2.83 到 3.07
DSpark 约 3.29 到 3.64
```

chat 的 accepted length 明显更低，因为开放式聊天可能续写很多样，draft 更难猜中 target 的具体采样路径。

### 10.3 为什么 DFlash 有时超过 Eagle3

这点看起来反直觉。自回归不是更连贯吗？

论文解释是：第一个 token 太重要了。

纯并行 DFlash 因为可以用更深 backbone，第一个 token 很强。第一个 token 一错，后面全没了；所以第一个 token 的优势会被放大。

而 Eagle3 虽然后面位置更稳定，但它为了控制 draft latency，模型较浅，首 token 可能不如 DFlash。

DSpark 同时拿到：

- DFlash 的首 token 容量优势。
- 自回归式的后缀依赖优势。

### 10.4 长草稿时 DSpark 优势更大

当 proposal length 从 4 增加到 16，DFlash 的后缀问题更明显，因为越往后越容易模式混合。

DSpark 的 Markov/RNN head 能缓解后缀衰减，所以 block 越长，相对 DFlash 的提升越大。论文报告：

```text
gamma=7 时：
math +16%
code +15%
chat +18%

gamma=15 时：
math +30%
code +26%
chat +22%
```

### 10.5 sequential head 的延迟为什么小

Markov head 每步只是：

```text
查一个小 embedding
做一个低秩投影
加到 logits 上
采样
```

相比 target model 的大规模 transformer forward，这点计算很小。论文在 batch size 128 下测到，draft length 从 4 到 16，相比 DFlash 整轮延迟只增加 0.2% 到 1.3%。

这就是 DSpark 能成立的关键：顺序性有收益，但开销不能太大。

## 11. 真实部署结果怎么理解

### 11.1 为什么线上比离线更复杂

离线 benchmark 常常只看一个请求或固定 batch，重点比较 accepted length。

线上 serving 要考虑：

- 同时有多少用户请求。
- 每个请求上下文长度不同。
- 每个请求草稿质量不同。
- GPU batch size 变化会影响 kernel 性能。
- KV cache 容量可能限制并发。
- 用户更关心 token/s/user，也就是自己看到字出来的速度。
- 服务方还关心 token/s/gpu，也就是每张 GPU 总产出。

DSpark 第 5 节就是把算法放到真实系统里看是否还划算。

### 11.2 MTP-1 baseline 是什么意义

论文拿 DSpark-5 和 MTP-1 比。

MTP-1 可以理解为生产中比较稳健的单 token 投机配置。为什么不是 MTP-3 或 MTP-5？

因为静态多 token MTP 在高并发时可能会验证太多低收益 token，导致 aggregate throughput 降低。生产系统宁愿用保守的 MTP-1，也不愿让多 token 验证把 batch 容量压垮。

DSpark 的目标正是安全释放更长草稿：

```text
轻载：多验证，提升单用户速度
重载：少验证，保护吞吐
```

### 11.3 Pareto frontier 是什么意思

这里的 Pareto frontier 指：

```text
单用户速度 token/s/user
和
总吞吐 token/s/gpu
之间能达到的最好边界
```

通常单用户速度更高，需要给单个请求更多计算资源，可能牺牲总吞吐。总吞吐更高，则可能每个用户看到的生成速度下降。

DSpark 的图 7 表示：在真实流量下，相比 MTP-1，DSpark 把这条边界往外推了。

论文报告：

- V4-Flash：匹配吞吐水平下，单用户速度提升 60% 到 85%。
- V4-Pro：匹配吞吐水平下，单用户速度提升 57% 到 78%。
- 在严格 SLA 下，MTP-1 会进入低并发、低吞吐区域，而 DSpark 还能保持有用吞吐。

### 11.4 verification budget 随负载变化

图 8 的核心意思：

```text
并发低：平均每请求验证 4 到 6 个 token
并发高：平均验证长度自动下降
```

这非常符合直觉：

- 空闲时多试几个草稿，反正 GPU 有余量。
- 忙的时候只验证最有把握的，别让低置信后缀占坑。

这也是 DSpark 与静态 MTP-N 的主要区别。

## 12. 和 MTP / EAGLE / EAGLE3 / DFlash 的详细对比

### 12.1 一张总表

| 方法 | 可以怎么理解 | 草稿生成 | 长草稿成本 | 块内依赖 | 验证长度调度 | DSpark 相比它的新意 |
|---|---|---|---|---|---|---|
| MTP / MTP-1 | 在模型中加未来 token 预测能力 | 多 token prediction head 或额外 MTP 层 | 常随预测深度增加 | 依具体实现，通常更接近顺序链 | 生产中常是固定或保守配置 | DSpark 把长草稿与动态验证预算结合，重载时自动收缩 |
| EAGLE | 利用 target feature 做自回归草稿 | 逐步特征外推和采样 | 随草稿步数增加 | 强 | 通常不是核心 | DSpark 用并行 backbone 保留长 block 低延迟 |
| EAGLE3 | 更强的 EAGLE 系自回归 drafter | 多层目标特征 + TTT 等 | 仍有顺序生成成本 | 强 | 不是核心 | DSpark 在线收益来自半自回归 + 硬件感知调度 |
| DFlash | 一次 forward 生成整个 block | 并行 block drafter | 低 | 弱，后缀衰减 | 无本文调度 | DSpark 在 DFlash 上加轻量顺序 head 和 confidence scheduler |

### 12.2 DSpark 与 MTP

MTP 的直觉是：让模型在训练或推理结构里预测未来多个 token。

例如普通 LM 只预测：

```text
当前位置 -> 下一个 token
```

MTP 还预测：

```text
当前位置 -> 下下个 token
当前位置 -> 下下下个 token
```

它能提供额外训练信号，也能用于 speculative decoding。

DSpark 与 MTP 的核心区别：

1. **目标不同**：MTP 更像模型结构中的未来 token 预测能力；DSpark 是完整 speculative decoding 框架，包括 drafter、confidence、scheduler、kernel 适配。
2. **验证策略不同**：MTP-N 如果静态验证 N 个 token，高并发下容易浪费 target batch；DSpark 每轮动态决定每个请求验证多长。
3. **生产定位不同**：论文说 MTP-1 是此前生产 baseline，因为静态 MTP-3/5 会在高并发下降低吞吐；DSpark 则是为了解锁多 token 草稿但避免吞吐崩掉。

### 12.3 DSpark 与 EAGLE

EAGLE 的直觉是：不要训练一个完全独立的小模型，而是利用 target model 的中间特征，让 draft model 外推下一步特征，再通过 LM head 得到 token。

它的优势：

- 和 target model 特征空间贴近。
- 自回归生成，块内依赖强。

它的限制：

- 草稿 token 逐步生成。
- 长草稿会增加 draft latency。

DSpark 的不同点：

- 重计算并行完成。
- 只用轻量顺序 head 建模块内依赖。
- 还把线上验证预算作为核心问题处理。

### 12.4 DSpark 与 EAGLE3

EAGLE3 是更强的 EAGLE 系 baseline。论文里把它作为 state-of-the-art autoregressive drafter 来比。

初学者可以把它理解为：

```text
一个更会利用 target model 特征、更强的自回归草稿器
```

DSpark 与 EAGLE3 的区别不是简单“谁预测更准”，而是取舍不同：

- EAGLE3 更偏 draft quality。
- DSpark 同时优化 draft quality 和 serving 中的 verification waste。
- DSpark 在 Qwen3 系列上 macro accepted length 比 Eagle3 高约 26.7% 到 30.9%。
- 在线部署时，DSpark 的硬件感知调度带来了 MTP-1 无法达到的高交互性能档位。

### 12.5 DSpark 与 DFlash

DFlash 是 DSpark 的重要基础。DSpark 可以看作：

```text
DFlash-style parallel drafter
+ 轻量 Markov/RNN sequential head
+ confidence head
+ 硬件感知 scheduler
+ 生产系统 variable-length verification 支持
```

DFlash 的问题是并行位置之间缺少已采样前缀依赖。DSpark 的 Markov head 直接补这个缺口。

## 13. 论文里容易卡住的几个概念

### 13.1 prefix survival probability

它不是“第 j 个 token 自己的通过率”，而是：

```text
前 j 个 token 全部通过的概率
```

因为 speculative decoding 只接受连续前缀。

### 13.2 accepted length 包含 bonus token

论文说明 accepted length 和 acceptance rate 默认包含 target-generated bonus token。这会让数值比“只统计 draft token 接受数”大 1 左右。读表时要注意。

### 13.3 static threshold 为什么不够

静态 threshold 只看 token confidence：

```text
confidence > 0.7 就验证
```

但线上系统还要看 GPU 是否忙。

同样一个 0.6 confidence 的 token：

- GPU 空闲时，验证一下可能值得。
- GPU 满载时，验证它可能挤掉其他请求更高收益的 token。

所以 DSpark 用 `tau * SPS(B)` 做系统级目标。

### 13.4 为什么调度也会影响无损性

很多人第一次读会觉得：只要 target model 最后验证，怎么调度都无损吧？

不完全是。验证哪些 token 这件事本身如果依赖 token 内容，就可能改变输出分布。

例如：

```text
看到某类 token 就验证
看到另一类 token 就不验证
```

这会让被验证并接受的 token 分布偏向某些 token。无损性要求调度决策不能偷看它不该看的未来信息。

### 13.5 为什么 confidence 要校准而不是只排序

如果只做：

```text
挑 top-k confidence token
```

排序可能够用。

但 DSpark 要计算预期吞吐：

```text
expected accepts = 1 + sum prefix_survival_prob
```

如果模型把真实 0.6 说成 0.9，scheduler 会过度验证后缀；如果把真实 0.9 说成 0.6，又会太保守。所以概率要校准。

## 14. 用一个完整小故事串起来

假设一个在线服务同时有 3 个用户：

```text
用户 A：写代码补全
用户 B：数学推理
用户 C：开放聊天
```

DSpark 每轮会：

1. target model 给每个请求生成一个 anchor。
2. DSpark 并行 backbone 为每个请求生成 5 个草稿位置的基础 logits。
3. Markov head 从左到右采样，让草稿内部更连贯。
4. confidence head 给每个位置预测条件接受概率。
5. 计算前缀存活概率。
6. scheduler 查看当前 GPU 的 `SPS(B)` 曲线。
7. 如果当前负载不高：
   - 代码请求 A 可能验证 5 个。
   - 数学请求 B 可能验证 4 个。
   - 聊天请求 C 可能验证 3 个。
8. 如果当前负载很高：
   - A 也许验证 3 个。
   - B 验证 2 个。
   - C 只验证 1 个。
9. target model 一次 batch forward 验证这些变长前缀。
10. 每个请求接受最长正确前缀，拒绝处由 target model 修正。

这个流程里，DSpark 没有让 draft model 直接替代 target model。target model 仍然掌握最终分布。DSpark 只是更聪明地组织“谁先猜、猜多长、验多长”。

## 15. 这篇论文的主要贡献，用初学者语言总结

1. **提出半自回归 drafter**：重 backbone 并行跑，轻 head 顺序补依赖，兼顾速度与连贯性。
2. **解释并行 drafter 为什么能超过自回归 drafter**：并行模型首 token 容量强，而首 token 对 accepted length 权重最大。
3. **用 confidence head 预测前缀存活概率**：不是只判断单个 token 好不好，而是估计连续前缀能活到哪里。
4. **把验证长度选择做成硬件感知调度问题**：目标是最大化 `tau * SPS(B)`，而不是固定验证 N 个 token。
5. **处理无损性中的选择偏差**：理论上用 early stopping，生产上用异步历史预测形成因果屏障。
6. **真实部署到 DeepSeek-V4**：相比 MTP-1，在 matched throughput 下明显提升单用户生成速度，并扩展严格 SLA 下的可行 serving 边界。

## 16. 读完后可以如何继续

如果你只懂一点 Transformer，建议按下面顺序补背景：

1. 先理解普通自回归 decode 和 KV cache。
2. 再看 speculative decoding 的拒绝采样规则。
3. 再看 EAGLE / MTP 这类 drafter 如何利用 target model 特征。
4. 再回到 DSpark，看它为什么从“草稿质量”进一步走到“系统调度”。

你在阅读相关代码时，可以重点找这些实现点：

- draft model 是否共享 target embedding / LM head。
- 草稿 token 是逐步生成还是一次 forward 生成。
- 每个 draft token 的概率 `p_d` 是否能被精确拿到。
- verification length 是固定的，还是每请求动态变化。
- 高并发下 batch size 如何计算，是否支持 variable-length verification。

## 17. 关键参考链接

- DSpark / DeepSpec 官方仓库：https://github.com/deepseek-ai/DeepSpec
- DeepSeek-V3 Technical Report（MTP 相关）：https://arxiv.org/abs/2412.19437
- Better & Faster Large Language Models via Multi-token Prediction：https://proceedings.mlr.press/v235/gloeckle24a.html
- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty：https://arxiv.org/abs/2401.15077
- EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test：https://arxiv.org/abs/2503.01840
- DFlash: Block Diffusion for Flash Speculative Decoding：https://arxiv.org/abs/2602.06036

## 18. 逐章细读：论文每一节到底在说什么

这一节把论文按原始结构再讲一遍。前面更像概念入门，这里更像“带着你读论文正文”。

### 18.1 Abstract：摘要在强调什么

摘要里有几句话特别关键：

1. 并行 drafter 可以一次提出长 token 序列。
2. 但并行 drafter 缺少 token 之间依赖，后面 token 接受率衰减很快。
3. 高并发时，无差别验证长草稿会浪费 batch capacity。
4. DSpark 用半自回归架构提升草稿质量。
5. DSpark 用置信度调度验证提升系统效率。

初学者容易只注意第 4 点，也就是“模型结构更强”。但论文真正完整的贡献是第 4 点加第 5 点。

如果只做更强 drafter，可能离线 accepted length 很好，但线上一堆请求同时来时，target model 验证的 token 数增加，batch 变大，系统吞吐可能下降。DSpark 要证明的是：它不只是让草稿更准，还让生产系统知道什么时候该多验、什么时候该少验。

### 18.2 Introduction：为什么大模型 serving 需要这个

论文开头说 LLM 生成是 autoregressive，也就是一个 token 一个 token 来。对用户来说，看到的速度通常是：

```text
每秒出现多少 token
```

对服务商来说，还要看：

```text
每张 GPU 每秒总共能服务多少 token
```

这两个指标不总是一致。为了让单个用户更快，可以给他更多计算资源；但这样可能减少同一张 GPU 上能服务的用户数量。投机解码本质上就是在这两个目标之间找平衡。

Introduction 中提到两个 bottleneck：

**第一个 bottleneck 是生成质量。**

并行 drafter 一次生成多个位置，但第 3 个位置不知道第 1、2 个位置实际采样了什么。这就像几个人同时写一句话的不同位置，但彼此不沟通，最后拼起来可能不自然。

**第二个 bottleneck 是系统效率。**

即使草稿模型能便宜地写出 16 个 token，也不代表 target model 应该验证 16 个。因为 target model 才是重计算部分。验证一个低概率通过的 token，可能挤掉另一个请求的高概率 token。

DSpark 的设计刚好一一对应：

```text
生成质量问题 -> 半自回归生成
系统效率问题 -> 置信度调度验证
```

### 18.3 Background：speculative decoding 的公式怎么读

论文写：

```text
L = (T_draft + T_verify) / tau
```

初看很简单，但它把 speculative decoding 的所有改进方向都框住了。

假设没有投机解码：

```text
target model 每 forward 一次 -> 生成 1 个 token
```

现在有投机解码：

```text
draft 花一点时间先猜
target 花一次时间验证
一次可能推进多个 token
```

如果 draft 很慢，`T_draft` 大，收益被吃掉。  
如果 draft 很不准，`tau` 小，也没收益。  
如果验证太多低质量 token，`T_verify` 的有效成本上升，线上吞吐下降。

所以不同方法的侧重点可以按这个公式归类：

| 方法方向 | 主要优化公式中的哪一项 |
|---|---|
| 更小 draft model | 降低 `T_draft` |
| 更强 draft model | 提高 `tau` |
| 并行 drafter | 降低长草稿下的 `T_draft` |
| confidence pruning | 降低浪费的 `T_verify` |
| hardware-aware scheduler | 在真实 `SPS(B)` 下优化总吞吐 |

DSpark 是一个组合拳，不是只动某一项。

### 18.4 Drafter Architectures：为什么要比较 Eagle3 和 DFlash

论文选择 Eagle3 和 DFlash 做 baseline 很有代表性：

- Eagle3 代表强自回归 drafter。
- DFlash 代表强并行 drafter。

这两个 baseline 分别站在两端：

```text
Eagle3：依赖强，但长草稿慢
DFlash：长草稿快，但依赖弱
```

DSpark 要证明自己不是简单地比某个弱 baseline 好，而是在这两条强路线中间找到更好的折中。

特别是论文观察到 DFlash 有时会超过 Eagle3，这个现象很值得注意。它说明 speculative decoding 的质量不是“整句生成质量”这么简单，而是强烈受到第一个 draft token 的影响。第一个 token 过不了，后面全作废；所以首 token 能力非常关键。

### 18.5 Architecture：图 1 可以怎么脑补

可以把图 1 想成流水线：

```text
target model:
  先吐出一个可靠 token D

DSpark parallel backbone:
  基于 D 和上下文，一次给出多个位置的基础预测

DSpark sequential head:
  从左到右让这些预测更连贯

confidence head:
  给每个位置一个通过概率

scheduler:
  决定 E F G H 里哪些值得送去 target model 验证

target model:
  一次性验证保留下来的前缀
```

这个流程中，target model 只做两类事：

- 生成 anchor / bonus token。
- 验证被 scheduler 选中的草稿 token。

draft model 做的事是：

- 提出候选。
- 提供候选概率。
- 提供 confidence。

scheduler 做的事是：

- 根据 confidence 和硬件负载分配验证预算。

把这三个角色分开，论文就好读很多。

### 18.6 Semi-Autoregressive Generation：半自回归不是折中得含糊，而是很精确

“半自回归”这个词容易让人误会，以为它只是模糊地介于自回归和非自回归之间。DSpark 里的含义更具体：

```text
hidden state 的主干计算：并行
token 采样时的局部修正：顺序
```

并行 backbone 输出 `U_k`，每个位置都有自己的基础 logits。然后 sequential head 输出 `B_k`，作为 bias 加到 logits 上。

为什么是加 bias，而不是重新跑一个 Transformer？

因为 DSpark 想要的只是块内依赖的补充，不是再做一次完整上下文建模。完整 Transformer 顺序跑会太慢，失去并行 drafter 的意义。

Markov head 是最便宜的版本：

```text
只看上一个 token
给当前词表所有 token 加一组转移 bias
```

这听起来很简单，但对很多局部搭配很有效。例如：

```text
San -> Francisco
New -> York
machine -> learning
of -> course
no -> problem
```

纯并行模型可能知道这些词都常见，但不知道当前路径到底选了哪个前缀。Markov head 直接把“前一个 token 是什么”注入进来。

### 18.7 Confidence Head：为什么它预测条件概率

confidence head 的定义很讲究。它不是预测：

```text
第 k 个 token 在任何情况下会不会被接受
```

而是预测：

```text
在前面 1..k-1 都已经被接受的条件下，第 k 个 token 会不会被接受
```

这符合 speculative decoding 的验证过程。因为如果前面已经失败，第 k 个 token 根本没有机会被验证为 accepted prefix 的一部分。

例子：

```text
c1 = 0.9
c2 = 0.8
c3 = 0.7
```

它们的含义是：

```text
c1：第 1 个 token 通过概率 0.9
c2：如果第 1 个通过，第 2 个通过概率 0.8
c3：如果第 1、2 个都通过，第 3 个通过概率 0.7
```

所以前 3 个都通过的概率是：

```text
0.9 * 0.8 * 0.7 = 0.504
```

这就是 scheduler 需要的 `a_{r,j}`。

### 18.8 Hardware-Aware Prefix Scheduler：为什么这是系统算法，不是模型算法

很多模型论文的改进只发生在神经网络结构里。但 DSpark 的 scheduler 更像系统算法。

它要知道：

- 当前 batch 里有多少请求。
- 每个请求不同位置的 prefix survival probability。
- target engine 在不同 batch size 下的实际 steps per second。

这些信息不是单个模型 forward 能决定的。它们属于 serving engine 的运行状态。

因此 DSpark 的框架跨越两层：

```text
模型层：生成草稿和 confidence
系统层：根据负载调度验证预算
```

这也是为什么论文会花一整节讲真实部署。只在离线 benchmark 上 accepted length 更高，还不足以说明它能在生产中带来收益。

### 18.9 Training：为什么训练里有 target 分布

普通语言模型训练只需要 ground truth token。但 DSpark 训练需要 target model 的输出分布，因为 speculative decoding 的接受率由 `p_d` 和 `p_t` 的接近程度决定。

如果 draft model 只是学数据答案，可能会和 target model 的偏好不一致。比如同一句话后面：

```text
ground truth: 好的
target model: 好的 0.4，可以 0.35，没问题 0.25
draft model: 好的 0.9，可以 0.05，没问题 0.05
```

draft model 对 ground truth 很自信，但和 target 分布差距很大。采样时一旦选到别的 token，容易被 target 拒绝。`L_tv` 就是让 draft 分布整体贴近 target 分布。

### 18.10 Experiments：为什么要分 Math、Code、Chat

这三个领域代表不同可预测性：

- Math：推理路径和答案形式更结构化。
- Code：语法和上下文约束强，也比较可预测。
- Chat：开放性高，同一上下文有很多合理续写。

如果一个方法只在 code 上好，可能只是利用了强结构。DSpark 在三个领域都有提升，说明它不是只适合某种高确定性任务。

同时，chat 上 accepted length 低也说明 scheduler 很必要。chat 低置信后缀更多，固定长验证更容易浪费。

### 18.11 Deployment：为什么论文强调 MTP-1 而不是 MTP-5

从直觉看，MTP-5 应该比 MTP-1 更快，因为一次多猜几个 token。但生产系统不是这么简单。

如果多猜出来的 token 接受率不够高，它们会：

```text
增加 target verification batch size
降低 SPS(B)
挤占其他请求
```

在高并发下，MTP-5 可能让总吞吐更差。所以生产中宁愿用 MTP-1 这种保守配置。

DSpark 的 scheduler 正是为了解决这个矛盾：

```text
不要固定 MTP-1，也不要固定 MTP-5
而是每轮、每请求动态选择接近最优的验证长度
```

### 18.12 Limitations：DSpark 还有什么没解决

论文提到一个重要局限：即使 scheduler 能剪掉 target verification 的浪费，draft side 的固定成本仍然存在。

也就是说，DSpark 每轮仍要用并行 backbone 生成初始 `gamma` 个 token。对于非常难、接受率天然很低的请求，这个 draft 成本可能收不回来。

未来可能的方向：

- 先判断请求难度，难请求少 draft 或不 draft。
- draft backbone 内部做 early exit。
- 对不同领域用不同 block size。
- 对不同用户 SLA 使用不同调度策略。

## 19. 数字算例：手算一次 scheduler 为什么会改变验证长度

这一节用一个完全虚构的小例子，让你感受 `tau * SPS(B)` 怎么影响决策。

### 19.1 场景设定

当前有 2 个请求：

```text
R = 2
```

每个请求最多有 3 个 draft token。prefix survival probability 是：

```text
请求 A:
a_A1 = 0.90
a_A2 = 0.70
a_A3 = 0.40

请求 B:
a_B1 = 0.80
a_B2 = 0.50
a_B3 = 0.20
```

假设 engine 的 SPS 曲线如下：

```text
B=2 -> SPS=100
B=3 -> SPS=90
B=4 -> SPS=80
B=5 -> SPS=65
B=6 -> SPS=50
B=7 -> SPS=40
B=8 -> SPS=32
```

基础 batch size 是 2，因为每个请求至少有一个 target step。

### 19.2 不验证任何 draft token

每个请求只靠 target model 走一步：

```text
B = 2
tau = 2
Theta = 2 * 100 = 200
```

### 19.3 加入最高收益 token A1

候选里最高的是 `A1=0.90`。

```text
B = 3
tau = 2 + 0.90 = 2.90
Theta = 2.90 * 90 = 261
```

吞吐提高，值得加。

### 19.4 加入 B1

下一个是 `B1=0.80`。

```text
B = 4
tau = 2 + 0.90 + 0.80 = 3.70
Theta = 3.70 * 80 = 296
```

继续提高，值得加。

### 19.5 加入 A2

下一个是 `A2=0.70`。

```text
B = 5
tau = 4.40
Theta = 4.40 * 65 = 286
```

吞吐从 296 降到 286。理论算法会停止，选择：

```text
请求 A 验证 1 个 token
请求 B 验证 1 个 token
```

### 19.6 如果系统更空闲会怎样

如果 GPU 更空闲，SPS 随 batch size 下降没那么快：

```text
B=2 -> 100
B=3 -> 98
B=4 -> 95
B=5 -> 92
B=6 -> 88
B=7 -> 83
B=8 -> 78
```

重新算：

```text
不验证：2 * 100 = 200
加 A1：2.9 * 98 = 284.2
加 B1：3.7 * 95 = 351.5
加 A2：4.4 * 92 = 404.8
加 B2：4.9 * 88 = 431.2
加 A3：5.3 * 83 = 439.9
加 B3：5.5 * 78 = 429
```

这时最优可能是验证：

```text
请求 A 验 3 个
请求 B 验 2 个
```

同样的 confidence，在不同系统负载下，最优验证长度不同。这就是 hardware-aware 的意义。

## 20. 从代码实现角度想 DSpark

如果你之后要看 vLLM 或类似推理框架里的代码，可以把 DSpark 拆成几个可能的模块。

### 20.1 Draft model forward

输入可能包括：

- 当前 token ids。
- anchor token。
- target model 提供的 hidden states 或 KV 注入特征。
- draft block 的 mask token 或位置表示。

输出可能包括：

- draft logits：每个草稿位置对词表的分数。
- draft hidden states：给 sequential head 和 confidence head 使用。
- draft probabilities：拒绝采样需要 `p_d`。

### 20.2 Sequential head

Markov head 需要：

- 上一个采样 token id。
- `W1` embedding lookup。
- `W2` projection。

每个位置做：

```text
bias = W1[prev_token] @ W2
final_logits = base_logits + bias
sample token
```

如果是 RNN head，还要维护 block 内 state：

```text
state = update(state, prev_token_embedding, h_k)
bias = projection(state and h_k)
```

### 20.3 Confidence head

confidence head 可能和 Markov head 共享 `W1[prev_token]`：

```text
confidence = sigmoid(linear([h_k, prev_token_embedding]))
```

推理时还要应用 STS 校准参数，把 raw confidence 变成 calibrated confidence。

### 20.4 Scheduler

scheduler 需要维护：

- 每个请求当前最多能验证的候选。
- 每个候选的 prefix survival probability。
- 当前 batch 的请求数 `R`。
- profile 得到的 `SPS(B)` 表。
- 每个请求最终选择的 `l_r`。

输出是：

```text
每个请求本轮验证几个 draft token
```

在 vLLM 这类系统里，这通常会影响：

- target model 输入 token 的展开方式。
- attention metadata。
- slot mapping。
- 输出 token 接受/拒绝后的状态更新。

### 20.5 Target verification

target model 会一次性处理变长前缀。难点是不同请求验证长度不一样：

```text
请求 A: 验 5 个
请求 B: 验 2 个
请求 C: 验 0 个
```

为了 GPU 友好，生产实现通常不希望简单 padding 成统一长度，因为 padding token 也会浪费计算。论文说 DeepSeek-V4 中通过 flatten token 和 marker tensor，让 sparse attention 知道逻辑序列关系。

## 21. 常见误解和纠正

### 21.1 误解：DSpark 让小模型替代了大模型

不是。DSpark 仍然让 target model 决定最终输出分布。draft model 只是提出候选，target model 通过拒绝采样验证。

### 21.2 误解：accepted length 越高线上一定越快

不一定。accepted length 高说明每轮可能推进更多 token，但如果 draft 成本很高，或者 target verification batch 太大导致 SPS 下降，线上可能不快。

DSpark 强调线上调度，就是因为离线 accepted length 不能完全代表 serving 性能。

### 21.3 误解：confidence threshold 固定成 0.8 就行

固定 threshold 忽略系统负载。

同样 confidence 0.6：

- 空闲时值得试。
- 满载时可能不值得。

所以 DSpark 用硬件曲线动态决定。

### 21.4 误解：并行 drafter 一定比自回归差

不一定。并行 drafter 可以用更深结构，首 token 能力很强。而 speculative decoding 中首 token 权重极大，所以并行 drafter 可能整体 accepted length 更好。

DSpark 的观点不是“并行完全好”，而是“并行 backbone 的容量优势很好，但要补上后缀依赖”。

### 21.5 误解：只要最后 target model 验证，就一定无损

还要看调度决策是否偷看了不该看的 token。如果 scheduler 根据当前 token 的取值决定这个 token 是否进入验证，就会改变输出分布。

论文附录 A 就是在证明这个问题。

## 22. 学习检查题

你可以用下面这些问题检查自己是否真正读懂：

1. 为什么 speculative decoding 只能接受连续前缀？
2. 为什么第一个 draft token 对 accepted length 影响最大？
3. DFlash 的 suffix decay 来自哪里？
4. DSpark 的 Markov head 为什么能缓解多模态碰撞？
5. confidence head 的 `c_k` 是普通接受概率，还是条件接受概率？
6. prefix survival probability 为什么是 confidence 的累计乘积？
7. 为什么 scheduler 要看 `SPS(B)`，而不是只看 confidence？
8. 为什么静态 MTP-5 在高并发下可能比 MTP-1 更差？
9. 为什么没有 early stopping 的全局搜索可能破坏无损性？
10. 生产系统中异步两步前预测为什么能形成因果屏障？

如果这些问题能讲清楚，DSpark 的主线基本就通了。
