from types import SimpleNamespace

from qdrant_client import models

from mcp.knowledge_base import KnowledgeBase


class FakeEmbedder:
    def passage_embed(self, texts):
        for index, _ in enumerate(texts):
            yield [float(index + 1)] * KnowledgeBase.DENSE_DIM

    def query_embed(self, text):
        yield [0.5] * KnowledgeBase.DENSE_DIM


class FakeReranker:
    def __init__(self, scores):
        self.scores = list(scores)

    def rerank(self, query, documents):
        assert query
        assert list(documents)
        return iter(self.scores)


class FakeQdrant:
    def __init__(self, points=None, parents=None):
        self.points = list(points or [])
        self.parents = dict(parents or {})
        self.query_calls = []
        self.retrieve_calls = []
        self.upsert_calls = []
        self.payload_updates = []

    def collection_exists(self, name):
        return True

    def create_payload_index(self, **kwargs):
        return None

    def count(self, collection_name, **kwargs):
        return SimpleNamespace(count=0)

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=self.points)

    def retrieve(self, collection_name, ids, **kwargs):
        self.retrieve_calls.append((collection_name, list(ids), kwargs))
        return [
            SimpleNamespace(id=parent_id, payload=self.parents[parent_id])
            for parent_id in ids
            if parent_id in self.parents
        ]

    def upsert(self, collection_name, points, wait=True):
        self.upsert_calls.append((collection_name, list(points), wait))

    def set_payload(self, **kwargs):
        self.payload_updates.append(kwargs)


def make_point(point_id, parent_id, child_text, score=0.1):
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={
            "child_id": point_id,
            "parent_id": parent_id,
            "title": "测试文档",
            "heading_path": "测试文档 > 章节",
            "child_text": child_text,
            "source_url": "",
            "source_type": "test",
        },
    )


def build_kb(client, scores, threshold=0.0):
    return KnowledgeBase(
        client=client,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(scores),
        sufficiency_threshold=threshold,
        load_default_docs=False,
    )


def test_parent_child_chunking_preserves_heading_and_limits_child_size():
    content = "# 登录问题\n" + "登录失败，请检查验证码。" * 100

    parents = KnowledgeBase._build_parent_chunks("帮助中心", content)
    children = [
        child
        for parent in parents
        for child in KnowledgeBase._split_with_overlap(
            parent.text,
            KnowledgeBase.CHILD_MAX_TOKENS,
            KnowledgeBase.CHILD_OVERLAP_TOKENS,
        )
    ]

    assert parents
    assert all(parent.heading_path == "帮助中心 > 登录问题" for parent in parents)
    assert len(children) > 1
    assert all(KnowledgeBase._token_count(child) <= KnowledgeBase.CHILD_MAX_TOKENS for child in children)


def test_add_documents_stores_parent_text_only_in_parent_collection():
    client = FakeQdrant()
    kb = build_kb(client, scores=[])

    added = kb.add_documents([{
        "title": "退款政策",
        "content": "购买后七天内可以申请退款。审核通过后原路退回。",
        "source_type": "test",
    }])

    assert added == 1
    assert [call[0] for call in client.upsert_calls] == [
        KnowledgeBase.PARENT_COLLECTION,
        KnowledgeBase.CHILD_COLLECTION,
    ]
    parent_point = client.upsert_calls[0][1][0]
    child_point = client.upsert_calls[1][1][0]
    assert "parent_text" in parent_point.payload
    assert "parent_text" not in child_point.payload
    assert set(child_point.vector) == {KnowledgeBase.DENSE_VECTOR, KnowledgeBase.SPARSE_VECTOR}
    sparse_document = child_point.vector[KnowledgeBase.SPARSE_VECTOR]
    assert isinstance(sparse_document, models.Document)
    assert isinstance(sparse_document.options, dict)


def test_search_uses_default_rrf_and_fetches_unique_parents_after_gate():
    client = FakeQdrant(
        points=[
            make_point("c1", "p1", "普通相关内容", 0.8),
            make_point("c2", "p1", "最相关内容", 0.7),
            make_point("c3", "p2", "第二相关内容", 0.6),
            make_point("c4", "p3", "明显不相关内容", 0.5),
        ],
        parents={
            "p1": {"title": "父块一", "parent_text": "父块一正文"},
            "p2": {"title": "父块二", "parent_text": "父块二正文"},
            "p3": {"title": "父块三", "parent_text": "不应进入上下文"},
        },
    )
    kb = build_kb(client, scores=[0.2, 2.0, 1.0, -1.0], threshold=0.0)

    result = kb.search("如何退款", top_k=3)

    assert result["sufficient"] is True
    assert [item["parent_id"] for item in result["results"]] == ["p1", "p2"]
    assert len(client.retrieve_calls) == 1
    assert client.retrieve_calls[0][1] == ["p1", "p2"]

    query_call = client.query_calls[0]
    assert isinstance(query_call["query"], models.RrfQuery)
    assert query_call["query"].rrf.k is None
    assert query_call["query"].rrf.weights is None
    assert {prefetch.using for prefetch in query_call["prefetch"]} == {
        KnowledgeBase.DENSE_VECTOR,
        KnowledgeBase.SPARSE_VECTOR,
    }
    assert "parent_text" not in query_call["with_payload"].include


def test_sufficiency_gate_blocks_parent_read_when_evidence_is_weak():
    client = FakeQdrant(
        points=[make_point("c1", "p1", "不相关内容")],
        parents={"p1": {"title": "父块一", "parent_text": "不应读取"}},
    )
    kb = build_kb(client, scores=[-2.0], threshold=0.0)

    result = kb.search("完全不同的问题")

    assert result["sufficient"] is False
    assert result["results"] == []
    assert client.retrieve_calls == []
