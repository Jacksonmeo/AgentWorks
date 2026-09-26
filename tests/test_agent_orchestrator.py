import asyncio

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    TechnicalAgent,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from mcp.tool_manager import MCPToolManager, ToolResult
from monitor.performance_monitor import PerformanceMonitor


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


class FakeStreamManager:
    def __init__(self, chunks, stop_reason="end_turn"):
        self._chunks = chunks
        self._final = type(
            "Response",
            (),
            {
                "content": [type("TextBlock", (), {"type": "text", "text": "".join(chunks)})()],
                "stop_reason": stop_reason,
            },
        )()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    @property
    def text_stream(self):
        async def iterate():
            for chunk in self._chunks:
                await asyncio.sleep(0)
                yield chunk

        return iterate()

    async def get_final_message(self):
        return self._final


class StreamingClient:
    def __init__(self, streams):
        self.streams = list(streams)
        self.stream_calls = []
        self.create_calls = []

        class Messages:
            def __init__(inner, owner):
                inner.owner = owner

            def stream(inner, **kwargs):
                inner.owner.stream_calls.append(kwargs)
                chunks, stop_reason = inner.owner.streams.pop(0)
                return FakeStreamManager(chunks, stop_reason)

            async def create(inner, **kwargs):
                inner.owner.create_calls.append(kwargs)
                raise AssertionError("streaming path should not use create")

        self.messages = Messages(self)


def make_request(**kwargs):
    values = {
        "message": "登录时报 401，同时这笔订单被重复扣款",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["401"], "amount": ["99 元"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != TechnicalAgent.profile.role
    assert TechnicalAgent.profile.workflow != BillingAgent.profile.workflow
    assert TechnicalAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_error_code" in TechnicalAgent.profile.tool_scope
    assert "check_billing_fields" in BillingAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    technical_packet = TechnicalAgent(FakeClient(), "test-model")._build_role_packet(req)
    billing_packet = BillingAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "diagnostic_fields" in technical_packet
    assert "verification_fields" in billing_packet
    assert general_packet != technical_packet != billing_packet


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "人工升级" in result.content
    assert client.calls == []


def test_composer_fallback_preserves_primary_and_supporting_results():
    client = FakeClient(error=RuntimeError("provider down"))
    composer = ResponseComposer(client, "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.TECHNICAL, "先排查 Token 是否过期。", True),
        AgentResponse(AgentType.BILLING, "请提供两笔扣款的时间和金额。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("先排查 Token 是否过期。")
    assert "补充说明" in content
    assert "两笔扣款" in content
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_specific_intent_fast_path_skips_llm():
    recognizer = IntentRecognizer(api_key="test-key", model="test-model")
    client = FakeClient(error=AssertionError("fast path should not call the LLM"))
    recognizer.client = client

    result = asyncio.run(recognizer.recognize("登录接口返回 401，请帮我排查"))

    assert result.intent == IntentCategory.TECHNICAL_LOGIN
    assert result.confidence >= 0.85
    assert result.source_scores["llm"] == 0.0
    assert client.calls == []


def test_rag_uses_direct_result_without_query_rewrite():
    manager = MCPToolManager(api_key="test-key", model="test-model")
    calls = []

    async def direct_call(name, params, context=None, **kwargs):
        calls.append((name, params, kwargs))
        return ToolResult(
            success=True,
            data=[
                {"title": "技术故障排查", "content": "401 表示认证失败", "score": 0.9},
                {"title": "技术故障排查", "content": "401 表示认证失败", "score": 0.8},
            ],
            tool_name=name,
        )

    async def unexpected_rewrite(query, n=3):
        raise AssertionError("direct result should skip query rewriting")

    manager.call = direct_call
    manager.rewrite_query = unexpected_rewrite

    result = asyncio.run(manager.search_with_rewrite("knowledge_search", "登录接口返回 401"))

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["score"] == 0.9
    assert len(calls) == 1


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    technical_tools = set(TechnicalAgent(FakeClient(), "test-model").get_tools())
    billing_tools = set(BillingAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields"}
    assert technical_tools == {"lookup_error_code", "build_diagnostic_plan"}
    assert billing_tools == {"check_billing_fields", "compare_amounts"}
    assert escalation_tools == {"create_handoff_summary"}
    assert not general_tools & technical_tools
    assert not technical_tools & billing_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "7 天内可退款"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, technical, billing, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_specific_agent_prefetches_rag_before_single_llm_call():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return ToolResult(
                success=True,
                data=[{"title": "技术故障排查", "content": "401 表示认证失败"}],
                tool_name=tool_name,
            )

    class TextBlock:
        type = "text"
        text = "请检查 Token 是否过期。"

    client = FakeClient(response=type("Response", (), {"content": [TextBlock()]})())
    agent = TechnicalAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(RagManager()))

    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["search_knowledge_base"]
    assert len(response.tool_traces) == 1
    assert len(client.calls) == 1
    assert "tools" not in client.calls[0]
    assert "知识库检索结果" in str(client.calls[0]["messages"])


def test_agent_streams_text_chunks_and_returns_the_same_complete_content():
    client = StreamingClient([(["你", "好，", "请问有什么可以帮你？"], "end_turn")])
    agent = GeneralAgent(client, "test-model")
    emitted = []

    async def run():
        async def on_token(text):
            emitted.append(text)

        return await agent.handle(make_request(
            message="你好",
            intent=IntentCategory.GREETING,
            intent_group="general",
            entities={},
        ), on_token=on_token)

    response = asyncio.run(run())

    assert response.success is True
    assert emitted == ["你", "好，", "请问有什么可以帮你？"]
    assert response.content == "".join(emitted)
    assert len(client.stream_calls) == 1
    assert client.create_calls == []


def test_agent_continues_once_when_stream_hits_max_tokens():
    client = StreamingClient([
        (["这是前半段，"], "max_tokens"),
        (["这是自动续写的后半段。"], "end_turn"),
    ])
    agent = GeneralAgent(client, "test-model")
    emitted = []

    async def run():
        async def on_token(text):
            emitted.append(text)

        return await agent.handle(make_request(
            message="请详细介绍服务",
            intent=IntentCategory.GREETING,
            intent_group="general",
            entities={},
        ), on_token=on_token)

    response = asyncio.run(run())

    assert response.content == "这是前半段，这是自动续写的后半段。"
    assert response.content == "".join(emitted)
    assert len(client.stream_calls) == 2
    assert "续写" in client.stream_calls[1]["messages"][-1]["content"]


def test_composer_streams_only_the_merged_final_answer():
    client = StreamingClient([(["技术结论。", "账单补充。"], "end_turn")])
    composer = ResponseComposer(client, "test-model")
    emitted = []

    async def run():
        async def on_token(text):
            emitted.append(text)

        return await composer.compose(make_request(), [
            AgentResponse(AgentType.TECHNICAL, "检查 Token。", True),
            AgentResponse(AgentType.BILLING, "核对扣款。", True),
        ], on_token=on_token)

    content = asyncio.run(run())

    assert emitted == ["技术结论。", "账单补充。"]
    assert content == "技术结论。账单补充。"
    assert len(client.stream_calls) == 1


def test_streaming_falls_back_before_first_chunk_when_provider_rejects_stream():
    class BrokenStream:
        async def __aenter__(self):
            raise RuntimeError("stream unsupported")

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class TextBlock:
        type = "text"
        text = "已使用兼容模式返回。"

    class FallbackClient:
        def __init__(self):
            self.create_calls = []

            class Messages:
                def __init__(inner, owner):
                    inner.owner = owner

                def stream(inner, **kwargs):
                    return BrokenStream()

                async def create(inner, **kwargs):
                    inner.owner.create_calls.append(kwargs)
                    return type("Response", (), {"content": [TextBlock()], "stop_reason": "end_turn"})()

            self.messages = Messages(self)

    client = FallbackClient()
    emitted = []

    async def run():
        async def on_token(text):
            emitted.append(text)

        return await GeneralAgent(client, "test-model").handle(make_request(
            message="你好",
            intent=IntentCategory.GREETING,
            intent_group="general",
            entities={},
        ), on_token=on_token)

    response = asyncio.run(run())

    assert response.content == "已使用兼容模式返回。"
    assert emitted == ["已使用兼容模式返回。"]
    assert len(client.create_calls) == 1


def test_tool_planning_stays_internal_and_final_answer_streams():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_stream"
        name = "lookup_error_code"
        input = {"error_code": "401"}

    class HybridClient:
        def __init__(self):
            self.create_calls = []
            self.stream_calls = []

            class Messages:
                def __init__(inner, owner):
                    inner.owner = owner

                async def create(inner, **kwargs):
                    inner.owner.create_calls.append(kwargs)
                    return type("Response", (), {"content": [ToolUseBlock()]})()

                def stream(inner, **kwargs):
                    inner.owner.stream_calls.append(kwargs)
                    return FakeStreamManager(["请检查 Token。"], "end_turn")

            self.messages = Messages(self)

    client = HybridClient()
    agent = TechnicalAgent(client, "test-model")
    emitted = []

    async def run():
        async def on_token(text):
            emitted.append(text)

        return await agent.handle(make_request(), on_token=on_token)

    response = asyncio.run(run())

    assert response.success is True
    assert response.content == "请检查 Token。"
    assert emitted == ["请检查 Token。"]
    assert len(client.create_calls) == 1
    assert len(client.stream_calls) == 1
    assert "tools" in client.create_calls[0]
    assert "tools" not in client.stream_calls[0]
    assert "tool_result" in str(client.stream_calls[0]["messages"])


def test_tool_input_validation_rejects_unknown_fields():
    agent = TechnicalAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_error_code"]

    try:
        agent._validate_tool_input(spec, {"error_code": "401", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_error_code"
        input = {"error_code": "401"}

    class TextBlock:
        type = "text"
        text = "已根据 401 错误码给出排查建议。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TechnicalAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["lookup_error_code"]
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_error_code",
        "build_diagnostic_plan",
    }
    assert "tools" not in client.calls[1]
    assert all(
        call["extra_body"] == {"thinking": {"type": "disabled"}}
        for call in client.calls
    )
    assert "tool_result" in str(client.calls[1]["messages"])


def test_monitor_deduplicates_and_resolves_threshold_alerts():
    monitor = PerformanceMonitor(orchestrator=None, tool_manager=None)

    monitor._check_threshold("agent_avg_ms", 8000, "technical_0")
    monitor._check_threshold("agent_avg_ms", 7000, "technical_0")

    assert len(monitor._alerts) == 1
    assert monitor._alerts[0].value == 7000
    assert monitor._alerts[0].resolved is False

    monitor._check_threshold("agent_avg_ms", 2000, "technical_0")
    assert monitor._alerts[0].resolved is True
