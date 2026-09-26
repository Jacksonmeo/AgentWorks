# RAG 混合检索与存储重构方案

> 状态：设计方案，尚未实施
>
> 适用项目：AgentWorks / EchoMind
>
> 更新时间：2026-09-26

## 1. 目标

本次重构不追求论文式创新，而是建设一条成熟、简洁、容易验证，也适合在面试中完整讲清楚的 RAG 链路：

```text
父子切块
  → 中文稠密向量 + BM25 并行召回
  → RRF 融合
  → CrossEncoder 重排
  → Retrieval Sufficiency Gate
  → Top3 父块批量读取
  → 将精简上下文交给 LLM
```

同时保留条件查询改写，但只在指代不清、问题过短或首轮检索证据不足时触发；取消轻量 LLM 精排，避免无必要的延迟、成本和链路复杂度。

最终存储选型：

- Redis：短期会话状态和 Agent 工作记忆；
- Qdrant：知识库和长期语义记忆；
- 原始文件目录：保存可重建索引的原始知识文档；
- 不再保留 ChromaDB，也不单独维护一份本地 BM25 索引。

## 2. 选型结论

### 2.1 为什么选择 Qdrant

Qdrant 可以在同一个子块集合中保存命名稠密向量和 BM25 稀疏向量，并通过一次查询完成多路预取和 RRF 融合。父块使用同一 Qdrant 服务中的无向量集合按 ID 批量读取。这样既能避免 `ChromaDB + rank_bm25` 两套检索索引之间的双写问题，也不需要引入新的数据库。

Qdrant 当前官方能力包括：

- Dense Vector 与 Sparse Vector；
- 服务端 BM25；
- 面向中文等非空格分词语言的 `multilingual` tokenizer；
- 多路查询预取与 RRF 融合；
- Payload 元数据过滤；
- 本地部署。

参考资料：

- [Qdrant BM25 与全文检索](https://qdrant.tech/documentation/search/text-search/full-text-search/)
- [Qdrant Hybrid Query 与 RRF](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Qdrant Collection 与命名向量](https://qdrant.tech/documentation/manage-data/collections/)
- [Qdrant Inference 方式](https://qdrant.tech/documentation/inference/)

部署时应固定一个经过验证、支持多语言 BM25 的 Qdrant 版本，建议不低于 1.19，不直接使用浮动的 `latest` 标签。

### 2.2 为什么不继续使用 ChromaDB

ChromaDB 适合快速完成向量检索原型，但如果继续使用它，就需要在应用内另外维护 BM25 索引：

```text
写入文档
  ├── 写入 Chroma 向量索引
  └── 写入 BM25 关键词索引
```

这会带来以下问题：

- 两套索引可能写入成功状态不一致；
- 更新、删除和重建需要执行两遍；
- RRF 融合完全由应用维护；
- 部署和故障排查不够简洁。

使用 Qdrant 后，Dense、BM25、Payload 和 RRF 都集中在同一个检索服务中。

### 2.3 为什么暂不选择 OpenSearch / Elasticsearch

OpenSearch 能完整支持 BM25、向量检索和 RRF，但对于当前小规模知识库，运行和维护成本相对偏高。Elasticsearch 的原生 RRF 还需要考虑具体订阅等级。

当前项目的重点是展示 RAG 链路设计，而不是搜索集群运维，因此 Qdrant 更符合“能力够用、组件最少、面试容易解释”的目标。

## 3. 最终存储架构

```text
                         ┌──────────────────────────┐
                         │       FastAPI 应用        │
                         │                          │
用户问题 ───────────────▶│ 条件改写 / BGE Embedding │
                         │ CrossEncoder / 证据 Gate │
                         └───────────┬──────────────┘
                                     │
                         Dense + BM25│查询
                                     ▼
                         ┌──────────────────────────┐
                         │          Qdrant          │
                         │                          │
                         │ rag_children_v1          │
                         │ rag_parents_v1           │
                         │ memory_episodic_v1       │
                         │ memory_profile_v1        │
                         └──────────────────────────┘

Redis
└── 当前会话、工作记忆、Agent 临时状态

data/knowledge/raw/
└── 上传或爬取的原始文档，作为索引重建的数据源
```

这套架构中只有两个运行时存储组件：

| 组件 | 职责 |
| --- | --- |
| Redis | 高频、短生命周期、可过期的会话状态 |
| Qdrant | 需要持久化和检索的知识块、情景记忆、用户画像记忆 |

原始文件目录不参与在线检索，只用于审计、重新切块和重建索引。

## 4. Qdrant 集合设计

### 4.1 子块检索集合

```text
collection: rag_children_v1
point: 一个子块对应一个 Point
point_id: child_id
```

每个 Point 保存两类向量：

```yaml
vectors:
  dense: BGE 中文稠密向量
  bm25: Qdrant BM25 稀疏向量
```

建议 Payload：

```yaml
document_id: 文档稳定 ID
document_version: 文档版本
parent_id: 父块 ID
child_id: 子块 ID
title: 文档标题
heading_path: 标题层级路径
child_text: 用于召回和重排的子块文本
source_url: 来源地址
source_type: upload / crawl / builtin
content_hash: 内容摘要，用于去重和增量更新
active: 是否为当前有效版本
```

建议用于生成 Dense 与 BM25 表示的文本为：

```text
retrieval_text = title + heading_path + child_text
```

标题和章节路径通常包含较强的主题信息，加入检索文本后可以改善短子块语义不完整的问题。

### 4.2 父块内容集合与延迟读取

父块正文不再复制到每个子块中，而是单独保存在同一个 Qdrant 服务的无向量集合中：

```text
collection: rag_parents_v1
point: 一个父块对应一个 Point
point_id: parent_id
```

建议 Payload：

```yaml
document_id: 文档稳定 ID
document_version: 文档版本
parent_id: 父块 ID
title: 文档标题
heading_path: 标题层级路径
parent_text: 最终交给 LLM 的父块文本
source_url: 来源地址
source_type: upload / crawl / builtin
content_hash: 内容摘要
active: 是否为当前有效版本
```

在线检索和 CrossEncoder 重排阶段只返回子块必要字段，不携带 `parent_text`。通过 Sufficiency Gate 后，先按 `parent_id` 去重并确定 Top3，再调用一次批量 Retrieve 获取父块正文。

这不是引入第二个数据库，而是在同一个 Qdrant 服务内做职责清晰的两个逻辑集合。代价是增加一次批量读取，但避免了重复传输大段父块正文，也消除了每个子块重复保存 `parent_text` 的存储冗余。

### 4.3 长期记忆集合

最终可以将原 ChromaDB 中的长期记忆也迁移至 Qdrant：

```text
memory_episodic_v1
└── 历史事件、问题与解决过程，使用 Dense 检索

memory_profile_v1
└── 用户偏好和稳定画像，使用 Dense 检索及 Payload 过滤
```

知识库集合启用 Dense 和 BM25；记忆集合只使用 Dense，不需要为了统一形式强行启用 BM25。

## 5. 父子切块方案

### 5.1 切块规则

建议初始参数：

| 类型 | 建议大小 | 重叠 | 用途 |
| --- | ---: | ---: | --- |
| 父块 | 700～1000 tokens | 视章节边界决定 | 最终提供给 LLM |
| 子块 | 180～280 tokens | 30～50 tokens | BM25、Dense 召回与重排 |

参数只是初始工程值，应使用与 Embedding 模型一致的 tokenizer 计算，不使用字符数粗略代替 token 数。

切块优先级：

1. Markdown 标题或文档章节边界；
2. 完整段落边界；
3. 完整句子边界；
4. 达到最大长度后才进行硬切分。

表格、代码块和问答对尽量保持整体，不应从中间拆开。

### 5.2 检索与返回对象

```text
召回对象：child_text
重排对象：query + child_text
Gate 判断对象：重排后的 TopK child_text
去重对象：parent_id
批量读取对象：Top3 parent_id
生成上下文：对应的 parent_text
```

子块更聚焦，适合定位相关内容；父块保留上下文，适合交给大模型生成答案。父块只在 Gate 通过且 Top3 已确定后读取，既避免直接使用大块召回，也避免在候选阶段传输大量最终不会使用的父块正文。

## 6. 中文 Embedding 与 BM25

### 6.1 Dense Embedding

第一阶段使用显式配置的中文 Embedding 模型：

```text
BAAI/bge-small-zh-v1.5
```

模型由应用侧本地加载并生成向量，Qdrant 只负责保存和检索。这样可以：

- 明确知道生产环境实际使用的模型；
- 避免依赖框架默认模型；
- 方便后续更换模型和重建索引；
- 保持本地可部署和可复现。

参考：[BAAI/bge-small-zh-v1.5 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)

Embedding 模型名称、模型版本、向量维度和归一化方式必须记录在集合配置或索引清单中。更换模型时新建版本化集合，不直接向旧集合混写不同模型的向量。

### 6.2 中文 BM25

Qdrant BM25 建议采用语言中立配置：

```yaml
tokenizer: multilingual
stemmer:
  type: none
stopwords: {}
```

写入和查询时必须使用完全一致的 BM25 配置。

BM25 主要补充以下能力：

- 产品名、错误码和订单号等精确词匹配；
- Dense Embedding 不容易区分的专有名词；
- 新出现但未被模型充分学习的领域词汇。

Dense 则主要处理同义表达、口语表达和语义相关性。两路召回互补，而不是让其中一路完全替代另一路。

## 7. 在线查询链路

### 7.1 默认路径

```text
1. 规范化用户问题
2. 仅在指代不清或问题过短时，先改写为一个完整查询
3. 生成 BGE 查询向量
4. Qdrant 并行召回
   ├── Dense Top20
   └── BM25 Top20
5. 使用 Qdrant 默认 RRF 配置融合
6. 截取候选子块 Top12
7. CrossEncoder 批量重排
8. Retrieval Sufficiency Gate 判断证据是否充分
   ├── 不足且尚未改写：改写一次并返回步骤 3
   ├── 改写后仍不足：明确返回无有效证据
   └── 充分：继续执行
9. 按 parent_id 去重并选择 Top3
10. 从 rag_parents 批量读取父块正文
11. 拼装上下文并调用 LLM
```

以上 TopK 是初始值，不需要为求职项目设计大量组合实验。准备一组固定测试问题，结合质量与延迟做一次小范围调整即可。父块正文在步骤 10 之前不进入网络响应和应用内存。

### 7.2 RRF 融合

RRF 只依赖不同召回列表中的名次，不要求 Dense 分数和 BM25 分数具有相同量纲：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

第一版不在业务代码中指定 `k` 或召回通道权重，直接使用 Qdrant 当前固定版本提供的默认 RRF 配置：

```text
rrf: {}
```

这样可以先验证整条链路，不把未经测试的经验值写死。只有固定测试集显示默认配置存在稳定问题时，才调整 `k` 或权重，并记录调整前后的 Recall@K、MRR 与 P95 延迟。固定 Qdrant 镜像版本可以保证默认行为可复现。

不建议直接把不同来源的原始分数相加，因为 Dense 相似度和 BM25 分数没有统一尺度。

RRF 分数也不应用作跨查询的绝对置信度。它会受参与融合的列表数量、排名以及 `k` 影响，更适合排序而不是判断“答案是否可靠”。

### 7.3 CrossEncoder 重排

RRF 后只保留约 12 个候选子块，再由 CrossEncoder 批量计算：

```text
(query, child_text) → relevance_score
```

CrossEncoder 放在应用层，而不是与数据库强绑定，方便：

- 本地批量推理；
- 独立替换重排模型；
- 控制候选数量和延迟。

每轮检索只做一次 CrossEncoder 重排；只有 Gate 未通过并触发唯一一次改写时，才会执行第二轮检索和重排。不再增加轻量 LLM 精排。

### 7.4 Retrieval Sufficiency Gate

Gate 位于 CrossEncoder 重排之后、父块读取之前，回答的问题不是“哪个结果排第一”，而是“当前检索证据是否值得进入生成阶段”。

第一版不引入新的 LLM Judge 或分类模型，只复用 CrossEncoder 的归一化相关性分数：

```text
sufficient = 存在候选结果
             AND top1_rerank_score >= sufficiency_threshold
```

`sufficiency_threshold` 不凭经验写死，通过固定测试集校准后放入配置。更换 CrossEncoder 时必须重新校准。Gate 不使用 RRF 分数，也不叠加通道重合度、Top1/Top2 差值或实体覆盖率等额外规则。

Gate 的处理结果只有三种：

1. 证据充分：选择 Top3 `parent_id`，批量读取父块并生成答案；
2. 首轮证据不足且未使用改写：改写一次并重新完成检索、融合和重排；
3. 改写后仍不足：不调用生成链路，明确返回“当前知识库中没有足够证据回答”。

它是一个保守的“检索是否足够相关”判断，不等价于事实正确性验证。最终生成提示词仍应要求答案只能基于给定证据，无法支持时拒绝回答。

## 8. 条件查询改写

查询改写不是默认步骤，只在以下情况触发：

- 存在“它、这个、刚才那个”等上下文指代；
- 问题过短，结合当前会话仍缺少明确实体或意图；
- 首轮重排后的结果没有通过 Retrieval Sufficiency Gate。

前两类可以在检索前触发；第三类只在首轮检索和重排完成后触发。不再根据 Dense/BM25 重合度、分数差或实体覆盖率增加额外启发式规则。

限制规则：

- 每个请求最多改写一次，只生成一个完整查询；
- 原始问题保留用于日志和最终回答语义，不再与多个改写版本并行扇出检索；
- 改写后的查询重新执行同一套 Dense + BM25 + RRF + CrossEncoder 链路；
- 如果请求已在检索前改写，Gate 未通过时不再进行第二次改写；
- 改写后仍未通过 Gate，直接返回无答案。

推荐执行方式：

```text
收到用户问题
  │
  ├── 指代不清或过短：改写一次后检索
  │
  └── 信息明确：直接检索
              │
              ▼
      混合召回 + RRF + 重排
              │
              ▼
        Sufficiency Gate
         ├── 通过：读取 Top3 父块
         ├── 未通过且未改写：改写一次后重试
         └── 重试仍未通过：返回无答案
```

这部分是项目最值得讲的小型工程改进：改写由明确的输入缺陷或 Sufficiency Gate 驱动，而不是每次都付出额外模型调用和多路召回成本。

## 9. 数据写入、更新和删除

### 9.1 写入流程

```text
1. 保存原始文件
2. 生成 document_id、document_version 和 content_hash
3. 解析标题、章节、段落和正文
4. 生成父块和子块
5. 构造 retrieval_text
6. 应用侧批量生成 Dense Vector
7. 生成 BM25 Sparse Vector
8. 先批量 Upsert 父块到 rag_parents
9. 再批量 Upsert 子块及向量到 rag_children
10. 两侧写入完成后再将该版本标记为 active
```

### 9.2 增量更新

- `content_hash` 未变化：跳过重新索引；
- 内容发生变化：生成新的 `document_version`；
- 新版本的父块和子块均写入完成后再停用旧版本；
- 两个集合的查询或读取都默认过滤 `active = true`；
- 稳定后再异步删除旧版本。

不要先删除旧数据再写新数据，否则写入失败时会造成知识暂时不可用。

### 9.3 删除

根据 `document_id` 分别删除 `rag_children` 中的子块和 `rag_parents` 中的父块，同时移除或归档原始文件和索引清单。虽然需要清理两个逻辑集合，但它们位于同一个 Qdrant 服务中；Dense 与 BM25 仍随子块 Point 一起删除，不存在两套检索索引状态。

## 10. 索引版本与重建

集合名称应带版本：

```text
rag_children_v1
rag_parents_v1

rag_children_v2
rag_parents_v2
```

以下变化需要新建集合并重新索引：

- 更换 Embedding 模型；
- 向量维度或距离度量变化；
- 父子切块策略发生明显变化；
- BM25 tokenizer 或模型配置变化；
- Payload Schema 出现不兼容调整。

推荐使用稳定别名，例如：

```text
rag_children_active → rag_children_v2
rag_parents_active  → rag_parents_v2
```

同一版本的父、子集合构建并验证完成后，在同一次别名操作中切换两个别名，从而避免停机、父子版本错配以及新旧向量混写。

## 11. 迁移顺序

不做一次性的大爆炸替换，按以下顺序逐步迁移：

### 第一阶段：建设新的知识检索链路

1. 在 Docker Compose 中增加 Qdrant，并固定版本；
2. 建立 `rag_children_v1` 和 `rag_parents_v1`；
3. 实现父子切块和原始文件清单；
4. 接入显式中文 Embedding；
5. 接入 Qdrant BM25、Dense 与 RRF；
6. 接入 CrossEncoder；
7. 接入 Retrieval Sufficiency Gate；
8. 接入单次条件查询改写和无答案回退；
9. 使用固定问题集验证质量和延迟，再决定是否调整 RRF 默认参数。

### 第二阶段：切换线上 RAG

1. 保留旧实现作为临时回退；
2. 新旧链路对同一批问题执行对比；
3. 确认写入、查询、更新、删除均正常；
4. 默认流量切换至 Qdrant；
5. 删除旧知识库检索实现。

### 第三阶段：统一长期记忆存储

1. 将 episodic memory 迁移至 Qdrant；
2. 将 profile memory 迁移至 Qdrant；
3. 验证 Agent 长期记忆功能；
4. 删除 ChromaDB 服务、依赖、配置和遗留数据目录。

最终保留 Redis + Qdrant 两个运行时存储组件。

## 12. 最小验证方案

本项目不需要完整科研式消融实验，只保留一套用于证明选型有效的固定测试集。

测试问题建议覆盖：

1. 关键词、错误码、产品名等 BM25 优势问题；
2. 同义表达和自然语言描述等 Dense 优势问题；
3. 同时包含关键词与语义表达的问题；
4. 带上下文指代、需要改写的问题；
5. 不应触发改写的明确问题；
6. 知识库中没有答案的问题；
7. 首轮不足、改写后能够找到证据的问题；
8. 改写后仍应返回无答案的问题；
9. 多个子块来自同一个父块、只应批量读取一次父块的问题。

验收指标只保留工程上真正有用的几项：

| 指标 | 作用 |
| --- | --- |
| Recall@K | 正确证据是否进入候选集 |
| MRR / 首条正确排名 | 重排后正确证据是否靠前 |
| P50 / P95 延迟 | 链路是否满足交互要求 |
| 改写触发率 | 是否避免所有查询都改写 |
| Gate 误拒绝率 | 有答案的问题是否被错误拒绝 |
| 无答案误答率 | 没有证据时是否仍强行回答 |

只需对比当前基线和最终方案，不需要组合出大量消融版本。

## 13. 明确不做的内容

第一版不做：

- 轻量 LLM 精排；
- 每个问题默认查询改写；
- 复杂 Query Router；
- 多个向量数据库并存；
- Elasticsearch / OpenSearch 集群；
- 独立的父块数据库或存储服务；
- 为 Sufficiency Gate 单独引入 LLM Judge；
- 大规模自动参数搜索；
- 论文式全量消融实验；
- 为未来百万级数据提前设计分布式架构。

这些内容只有在实际数据和测试结果证明有必要时才引入。

## 14. 风险与取舍

### 14.1 中文 BM25 分词不是无限精细

Qdrant 的多语言 tokenizer 足够支撑当前混合检索，但不等同于为具体领域定制的中文词典。可以先通过标题、章节、实体标准化和 Dense 召回来弥补；只有实际出现明显问题时，再评估领域词典或更重的搜索引擎。

### 14.2 延迟读取父块会增加一次请求

Gate 通过后需要按照 Top3 `parent_id` 增加一次 Qdrant 批量读取。由于只请求一次、最多读取三个父块，它换来了更小的候选响应和更少的父块正文传输。应记录这次批量读取的 P95 延迟；只有实际成为瓶颈时才考虑短期缓存。

### 14.3 CrossEncoder 会增加延迟

通过以下方式控制：

- RRF 后只保留约 12 个候选；
- 批量推理；
- 设置超时；
- 不增加第二层 LLM 精排。

### 14.4 Sufficiency Gate 可能误拒绝

CrossEncoder 相关性分数并不等于完整的事实正确性，因此 Gate 只能作为保守的检索充分性判断。阈值应通过固定测试集校准并配置化；更换重排模型后必须重新校准，同时持续观察 Gate 误拒绝率与无答案误答率。

### 14.5 改写策略可能误触发

限制为每个请求最多改写一次，并记录触发原因、原始问题、改写结果和耗时。后续根据日志收紧“指代不清”和“问题过短”的判断即可，不再新增更多检索启发式规则。

## 15. 面试表达版本

### 30 秒版本

> 项目最初使用 Chroma 做向量检索，我将其重构为基于 Qdrant 的混合检索。一个知识子块同时保存中文 Dense Vector 和 BM25 Sparse Vector，查询时并行召回，通过 Qdrant 默认 RRF 配置融合，再使用 CrossEncoder 重排。重排后先经过 Retrieval Sufficiency Gate：证据不足时只改写重试一次，仍不足就明确拒答；证据充分时才批量读取 Top3 父块。这样兼顾召回质量、上下文完整性和传输开销。

### 选型依据

> 我没有叠加 Chroma、本地 BM25 和额外关系数据库，因为当前数据量不需要这么多组件。Qdrant 的子块集合统一保存 Dense 和 Sparse 表示并执行 RRF，父块则放在同一服务的无向量集合中。召回阶段只传输 child，确定最终 Top3 后才按 ID 批量读取 parent，避免候选阶段重复携带大段正文。

### 项目的差异点

> 算法本身采用行业常见方案，项目的差异点是工程组合：条件查询改写、Dense 与 BM25 混合召回、子块重排、证据充分性判断和父块延迟读取。重点不是发明新的检索算法，而是控制召回质量、上下文完整性、拒答能力和响应延迟之间的平衡。

## 16. 最终结论

当前项目最合适的最终方案是：

```text
Redis
  └── 短期状态与工作记忆

Qdrant
  ├── rag_children：Dense + BM25 + RRF
  ├── rag_parents：父块正文，按 parent_id 批量读取
  ├── Episodic Memory：Dense
  └── Profile Memory：Dense + Payload

应用层
  ├── 父子切块
  ├── BGE 中文 Embedding
  ├── 条件查询改写
  ├── CrossEncoder 重排
  ├── Retrieval Sufficiency Gate
  └── 父块去重、延迟读取与上下文拼装
```

该方案不追求组件数量或算法名词的堆叠，而是用一个统一检索服务完成当前项目真正需要的能力。它实现难度可控、链路完整、可渐进迁移，并且能够在面试中清楚说明每一项设计的原因、收益和边界。
