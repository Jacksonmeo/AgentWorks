"""Qdrant 混合检索知识库。

职责边界：
  1. 父子切块并写入 Qdrant；
  2. BGE 中文稠密向量与 BM25 并行召回；
  3. 使用 Qdrant 默认 RRF 融合；
  4. CrossEncoder 重排与 Retrieval Sufficiency Gate；
  5. Gate 通过后才批量读取最终父块。

查询改写由 MCPToolManager 负责，这里只执行一轮完整检索。
"""
import asyncio
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParentChunk:
    heading_path: str
    text: str


class KnowledgeBase:
    """基于 Qdrant 的中文混合检索知识库。"""

    CHILD_COLLECTION = "rag_children_v1"
    PARENT_COLLECTION = "rag_parents_v1"
    DENSE_VECTOR = "dense"
    SPARSE_VECTOR = "bm25"
    BM25_MODEL = "qdrant/bm25"
    DENSE_DIM = 512

    PARENT_MAX_TOKENS = 900
    CHILD_MAX_TOKENS = 240
    CHILD_OVERLAP_TOKENS = 40
    DENSE_RECALL_K = 20
    SPARSE_RECALL_K = 20
    RRF_CANDIDATE_K = 12
    MAX_PARENT_RESULTS = 3

    _TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
    _SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])|\n{2,}")
    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        dense_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        model_cache_dir: Optional[str] = None,
        sufficiency_threshold: float = 0.0,
        *,
        client: Optional[QdrantClient] = None,
        embedder: Optional[Any] = None,
        reranker: Optional[Any] = None,
        load_default_docs: bool = True,
    ):
        self._client = client or QdrantClient(url=qdrant_url, timeout=30)
        self._embedder = embedder or TextEmbedding(
            model_name=dense_model,
            cache_dir=model_cache_dir,
            lazy_load=True,
        )
        self._reranker = reranker or TextCrossEncoder(
            model_name=reranker_model,
            cache_dir=model_cache_dir,
            lazy_load=True,
        )
        self._dense_model = dense_model
        self._reranker_model = reranker_model
        self._sufficiency_threshold = float(sufficiency_threshold)
        # Document.options 在 qdrant-client 的本地 FastEmbed 路径中要求普通映射；
        # 使用 JSON 形式也能直接透传给远端 Qdrant 推理接口。
        self._bm25_options = models.Bm25Config(
            tokenizer=models.TokenizerType.MULTILINGUAL,
            stemmer=models.DisabledStemmerParams(type=models.NoStemmer.NONE),
            stopwords=models.StopwordsSet(),
        ).model_dump(exclude_none=True, mode="json")

        self._ensure_collections()
        if load_default_docs and self.doc_count == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """父块先写入，再写入携带 Dense 与 BM25 表示的子块。"""
        parent_points: List[models.PointStruct] = []
        child_records: List[Tuple[str, Dict[str, Any], str]] = []
        versions: List[Tuple[str, str]] = []

        for doc in documents:
            title = str(doc.get("title", "")).strip()
            content = str(doc.get("content", "")).strip()
            if not content:
                continue

            source_url = str(doc.get("source_url", "")).strip()
            source_type = str(doc.get("source_type", "upload")).strip() or "upload"
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            document_id = str(doc.get("document_id", "")).strip() or str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{source_type}:{source_url or title}")
            )
            document_version = content_hash[:16]
            versions.append((document_id, document_version))

            parents = self._build_parent_chunks(title, content)
            for parent_index, parent in enumerate(parents):
                parent_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{document_id}:{document_version}:parent:{parent_index}",
                ))
                common_payload = {
                    "document_id": document_id,
                    "document_version": document_version,
                    "parent_id": parent_id,
                    "title": title,
                    "heading_path": parent.heading_path,
                    "source_url": source_url,
                    "source_type": source_type,
                    "content_hash": content_hash,
                    "active": True,
                }
                parent_points.append(models.PointStruct(
                    id=parent_id,
                    vector={},
                    payload={**common_payload, "parent_text": parent.text},
                ))

                children = self._split_with_overlap(
                    parent.text,
                    self.CHILD_MAX_TOKENS,
                    self.CHILD_OVERLAP_TOKENS,
                )
                for child_index, child_text in enumerate(children):
                    child_id = str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{parent_id}:child:{child_index}",
                    ))
                    retrieval_text = "\n".join(
                        part for part in (title, parent.heading_path, child_text) if part
                    )
                    payload = {
                        **common_payload,
                        "child_id": child_id,
                        "child_index": child_index,
                        "child_text": child_text,
                    }
                    child_records.append((child_id, payload, retrieval_text))

        if not child_records:
            return 0

        dense_vectors = self._embed_passages([item[2] for item in child_records])
        child_points = [
            models.PointStruct(
                id=child_id,
                vector={
                    self.DENSE_VECTOR: dense_vector,
                    self.SPARSE_VECTOR: models.Document(
                        text=retrieval_text,
                        model=self.BM25_MODEL,
                        options=self._bm25_options,
                    ),
                },
                payload=payload,
            )
            for (child_id, payload, retrieval_text), dense_vector
            in zip(child_records, dense_vectors)
        ]

        # 先写父块，确保任何可见子块都有可读取的父块。
        self._client.upsert(self.PARENT_COLLECTION, parent_points, wait=True)
        self._client.upsert(self.CHILD_COLLECTION, child_points, wait=True)
        for document_id, document_version in versions:
            self._deactivate_old_versions(document_id, document_version)

        logger.info(
            "知识库导入完成: parents=%s children=%s",
            len(parent_points),
            len(child_points),
        )
        return len(child_points)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        return await asyncio.to_thread(self.add_documents, documents)

    # ── 检索 ──────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """执行混合召回、重排、充分性判断和父块延迟读取。"""
        query = str(query or "").strip()
        if not query:
            return self._empty_search_result(query, "query 不能为空")

        query_vector = self._embed_query(query)
        active_filter = self._active_filter()
        response = self._client.query_points(
            collection_name=self.CHILD_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=self.DENSE_VECTOR,
                    filter=active_filter,
                    limit=self.DENSE_RECALL_K,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.BM25_MODEL,
                        options=self._bm25_options,
                    ),
                    using=self.SPARSE_VECTOR,
                    filter=active_filter,
                    limit=self.SPARSE_RECALL_K,
                ),
            ],
            # 不指定 k 和权重，先使用固定 Qdrant 版本的默认 RRF 配置。
            query=models.RrfQuery(rrf=models.Rrf()),
            limit=self.RRF_CANDIDATE_K,
            with_payload=models.PayloadSelectorInclude(include=[
                "child_id",
                "parent_id",
                "title",
                "heading_path",
                "child_text",
                "source_url",
                "source_type",
            ]),
            with_vectors=False,
        )

        candidates = []
        for point in response.points:
            payload = point.payload or {}
            child_text = str(payload.get("child_text", "")).strip()
            parent_id = str(payload.get("parent_id", "")).strip()
            if not child_text or not parent_id:
                continue
            candidates.append({
                "child_id": str(payload.get("child_id") or point.id),
                "parent_id": parent_id,
                "title": str(payload.get("title", "")),
                "heading_path": str(payload.get("heading_path", "")),
                "child_text": child_text,
                "source_url": str(payload.get("source_url", "")),
                "source_type": str(payload.get("source_type", "")),
                "rrf_score": float(point.score),
            })

        if not candidates:
            return self._empty_search_result(query, "没有召回到候选证据")

        rerank_scores = [
            float(score)
            for score in self._reranker.rerank(
                query,
                [item["child_text"] for item in candidates],
            )
        ]
        for item, score in zip(candidates, rerank_scores):
            item["rerank_score"] = score
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)

        best_score = float(candidates[0]["rerank_score"])
        sufficient = best_score >= self._sufficiency_threshold
        if not sufficient:
            return {
                "query": query,
                "results": [],
                "sufficient": False,
                "reranked": True,
                "best_score": round(best_score, 6),
                "candidate_count": len(candidates),
                "reason": "重排后没有足够相关的证据",
            }

        requested_k = min(max(int(top_k or 1), 1), self.MAX_PARENT_RESULTS)
        # Gate 通过只说明“至少有一条证据可用”；最终上下文仍需剔除低于
        # 同一相关性边界的候选，避免为了凑满 Top3 注入明显无关父块。
        relevant_candidates = [
            item
            for item in candidates
            if float(item["rerank_score"]) >= self._sufficiency_threshold
        ]
        selected = self._select_unique_parents(relevant_candidates, requested_k)
        parent_records = self._client.retrieve(
            collection_name=self.PARENT_COLLECTION,
            ids=[item["parent_id"] for item in selected],
            with_payload=models.PayloadSelectorInclude(include=[
                "parent_id",
                "title",
                "heading_path",
                "parent_text",
                "source_url",
                "source_type",
            ]),
            with_vectors=False,
        )
        parents_by_id = {
            str(record.id): (record.payload or {})
            for record in parent_records
        }

        results = []
        for child in selected:
            parent = parents_by_id.get(child["parent_id"])
            if not parent:
                continue
            results.append({
                "title": str(parent.get("title", child["title"])),
                "content": str(parent.get("parent_text", "")),
                "heading_path": str(parent.get("heading_path", child["heading_path"])),
                "score": round(float(child["rerank_score"]), 6),
                "rrf_score": round(float(child["rrf_score"]), 6),
                "parent_id": child["parent_id"],
                "evidence_child": child["child_text"],
                "source_url": str(parent.get("source_url", child["source_url"])),
                "source_type": str(parent.get("source_type", child["source_type"])),
            })

        if not results:
            return self._empty_search_result(query, "候选父块读取失败", reranked=True)
        return {
            "query": query,
            "results": results,
            "sufficient": True,
            "reranked": True,
            "best_score": round(best_score, 6),
            "candidate_count": len(candidates),
            "reason": "证据充分",
        }

    async def search_async(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return int(self._client.count(
            self.CHILD_COLLECTION,
            count_filter=self._active_filter(),
            exact=True,
        ).count)

    async def doc_count_async(self) -> int:
        return await asyncio.to_thread(lambda: self.doc_count)

    async def search_handler(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        query = str(params.get("query", ""))
        top_k = int(params.get("top_k", self.MAX_PARENT_RESULTS) or self.MAX_PARENT_RESULTS)
        return await self.search_async(query, top_k=top_k)

    # ── Qdrant 初始化 ─────────────────────────────────────────────────────────

    def _ensure_collections(self) -> None:
        if not self._client.collection_exists(self.CHILD_COLLECTION):
            self._client.create_collection(
                collection_name=self.CHILD_COLLECTION,
                vectors_config={
                    self.DENSE_VECTOR: models.VectorParams(
                        size=self.DENSE_DIM,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )

        if not self._client.collection_exists(self.PARENT_COLLECTION):
            self._client.create_collection(
                collection_name=self.PARENT_COLLECTION,
                vectors_config={},
            )

        for collection in (self.CHILD_COLLECTION, self.PARENT_COLLECTION):
            for field_name, schema in (
                ("document_id", models.PayloadSchemaType.KEYWORD),
                ("document_version", models.PayloadSchemaType.KEYWORD),
                ("active", models.PayloadSchemaType.BOOL),
            ):
                try:
                    self._client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=schema,
                        wait=True,
                    )
                except Exception as ex:
                    # 已存在索引时不同 Qdrant 版本的返回行为略有差异。
                    logger.debug("Payload 索引无需重复创建: %s.%s (%s)", collection, field_name, ex)

    def _deactivate_old_versions(self, document_id: str, current_version: str) -> None:
        old_version_filter = models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                models.FieldCondition(key="active", match=models.MatchValue(value=True)),
            ],
            must_not=[
                models.FieldCondition(
                    key="document_version",
                    match=models.MatchValue(value=current_version),
                ),
            ],
        )
        for collection in (self.PARENT_COLLECTION, self.CHILD_COLLECTION):
            self._client.set_payload(
                collection_name=collection,
                payload={"active": False},
                points=old_version_filter,
                wait=True,
            )

    @staticmethod
    def _active_filter() -> models.Filter:
        return models.Filter(must=[
            models.FieldCondition(key="active", match=models.MatchValue(value=True)),
        ])

    # ── Embedding / Rerank ────────────────────────────────────────────────────

    def _embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._embedder.passage_embed(list(texts))
        return [self._vector_to_list(vector) for vector in vectors]

    def _embed_query(self, text: str) -> List[float]:
        vector = next(iter(self._embedder.query_embed(text)))
        return self._vector_to_list(vector)

    @staticmethod
    def _vector_to_list(vector: Any) -> List[float]:
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        return [float(value) for value in values]

    @staticmethod
    def _select_unique_parents(candidates: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        selected = []
        seen = set()
        for item in candidates:
            parent_id = item["parent_id"]
            if parent_id in seen:
                continue
            seen.add(parent_id)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _empty_search_result(query: str, reason: str, reranked: bool = False) -> Dict[str, Any]:
        return {
            "query": query,
            "results": [],
            "sufficient": False,
            "reranked": reranked,
            "best_score": None,
            "candidate_count": 0,
            "reason": reason,
        }

    # ── 父子切块 ──────────────────────────────────────────────────────────────

    @classmethod
    def _token_count(cls, text: str) -> int:
        """轻量 token 估算：中文按字、英文和数字按词，避免额外加载 tokenizer。"""
        return len(cls._TOKEN_RE.findall(text))

    @classmethod
    def _build_parent_chunks(cls, title: str, content: str) -> List[ParentChunk]:
        sections = cls._split_sections(title, content)
        parents: List[ParentChunk] = []
        for heading_path, section_text in sections:
            for chunk in cls._split_with_overlap(section_text, cls.PARENT_MAX_TOKENS, 0):
                if chunk.strip():
                    parents.append(ParentChunk(heading_path=heading_path, text=chunk.strip()))
        return parents

    @classmethod
    def _split_sections(cls, title: str, content: str) -> List[Tuple[str, str]]:
        heading_stack: List[str] = []
        current_lines: List[str] = []
        current_path = title
        sections: List[Tuple[str, str]] = []

        def flush() -> None:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append((current_path or title, text))
            current_lines.clear()

        for raw_line in content.replace("\r\n", "\n").split("\n"):
            match = cls._HEADING_RE.match(raw_line.strip())
            if not match:
                current_lines.append(raw_line)
                continue
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(heading)
            current_path = " > ".join(part for part in ([title] + heading_stack) if part)

        flush()
        return sections or [(title, content.strip())]

    @classmethod
    def _split_with_overlap(cls, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        units: List[str] = []
        for part in cls._SENTENCE_BOUNDARY_RE.split(text.strip()):
            part = part.strip()
            if not part:
                continue
            units.extend(cls._hard_split(part, max_tokens))

        chunks: List[str] = []
        current: List[str] = []
        for unit in units:
            candidate = "\n".join(current + [unit])
            if current and cls._token_count(candidate) > max_tokens:
                chunks.append("\n".join(current).strip())
                current = cls._overlap_tail(current, overlap_tokens)
                if current and cls._token_count("\n".join(current + [unit])) > max_tokens:
                    current = []
            current.append(unit)

        if current:
            final = "\n".join(current).strip()
            if not chunks or final != chunks[-1]:
                chunks.append(final)
        return chunks

    @classmethod
    def _hard_split(cls, text: str, max_tokens: int) -> List[str]:
        if cls._token_count(text) <= max_tokens:
            return [text]

        parts: List[str] = []
        remaining = text
        while remaining:
            low, high = 1, len(remaining)
            while low < high:
                mid = (low + high + 1) // 2
                if cls._token_count(remaining[:mid]) <= max_tokens:
                    low = mid
                else:
                    high = mid - 1
            cut = max(low, 1)
            parts.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        return [part for part in parts if part]

    @classmethod
    def _overlap_tail(cls, units: Sequence[str], token_budget: int) -> List[str]:
        if token_budget <= 0:
            return []
        tail: List[str] = []
        used = 0
        for unit in reversed(units):
            unit_tokens = cls._token_count(unit)
            if used + unit_tokens > token_budget:
                break
            tail.append(unit)
            used += unit_tokens
            if used >= token_budget:
                break
        return list(reversed(tail))

    # ── 默认知识 ──────────────────────────────────────────────────────────────

    def _load_default_docs(self) -> None:
        default_docs = [
            {
                "title": "退款政策",
                "source_type": "builtin",
                "content": (
                    "退款政策说明。用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "source_type": "builtin",
                "content": (
                    "订单查询指南。用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "source_type": "builtin",
                "content": (
                    "账户安全说明。建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "source_type": "builtin",
                "content": (
                    "常见技术问题排查。应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "source_type": "builtin",
                "content": (
                    "会员积分规则。每消费 1 元累积 1 积分。积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "source_type": "builtin",
                "content": (
                    "配送服务说明。标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        count = self.add_documents(default_docs)
        logger.info("已导入默认知识库: documents=%s children=%s", len(default_docs), count)
