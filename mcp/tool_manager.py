"""
亮点：MCP 工具调用框架

核心问题：工具调用出错（检索不全、召回不好）怎么优化？

本模块的答案：
  1. 直接检索优先 —— 正常召回时立即返回，减少不必要的模型调用。
  2. 条件查询改写 —— 只在指代不清、问题过短或证据不足时改写一次。
  3. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩。
  4. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用。
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。
"""
import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # 是否经过重排
    sufficient:     Optional[bool] = None
    rewritten:      bool = False
    query_used:     Optional[str] = None


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN：放行一次探测

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    MCP 工具调用框架。

    RAG 检索链路：
      用户查询 → 一轮混合检索；证据不足时 → 改写一次 → 重试；仍不足则拒答。
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at)

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 缓存写入
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params)
            if cached is not None:
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached,
                    tool_name=name,
                    cached=True,
                    reranked=bool(cached.get("reranked")) if isinstance(cached, dict) else False,
                    sufficient=cached.get("sufficient") if isinstance(cached, dict) else None,
                    query_used=cached.get("query") if isinstance(cached, dict) else None,
                )

        # 熔断检查
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            # 参数校验（根据 JSON Schema 的 required 和 properties.type）
            self._validate_params(tool, params)

            data = await asyncio.wait_for(self._run_handler(tool, params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()

            reranked = bool(data.get("reranked")) if isinstance(data, dict) else False
            sufficient = data.get("sufficient") if isinstance(data, dict) else None
            query_used = data.get("query") if isinstance(data, dict) else None

            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl)

            return ToolResult(success=True, data=data, tool_name=name,
                              latency_ms=latency, reranked=reranked,
                              sufficient=sufficient, query_used=query_used)

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """
        执行工具 handler。

        优先支持 async handler；如果历史工具仍是同步函数，则放入线程池执行，
        避免阻塞事件循环。
        """
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params, context)
        result = await asyncio.to_thread(tool.handler, params, context)
        if inspect.isawaitable(result):
            return await result
        return result

    # ── 条件查询改写 ───────────────────────────────────────────────────────────

    async def rewrite_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """将问题改写为一个语义完整的检索查询，不生成多路变体。"""
        context_text = self._rewrite_context(context)
        prompt = f"""请把用户问题改写成一个可独立检索知识库的完整查询。
要求：
1. 只补全对话中已经出现的指代、实体和意图，不扩展问题范围；
2. 只输出一条改写后的查询，不要解释、编号或 JSON；
3. 如果信息不足以补全，原样返回用户问题。

最近对话：
{context_text or "（无）"}

用户问题：{query}"""
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=128,
                temperature=0.0,
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": self._clean_text(prompt)}],
            )
            rewritten = extract_text_content(resp.content).strip().strip('"“”')
            return rewritten or query
        except Exception as ex:
            logger.warning("查询改写失败，使用原始查询: %s", ex)
            return query

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """至多改写一次；第二轮仍不足时返回明确的无证据结果。"""
        original_query = str(query or "").strip()
        if not original_query:
            return ToolResult(
                success=False,
                data=[],
                tool_name=tool_name,
                error="query 不能为空",
                sufficient=False,
            )

        effective_query = original_query
        rewritten = False
        if self._should_rewrite_before_search(original_query, context):
            effective_query = await self.rewrite_query(original_query, context)
            rewritten = effective_query.strip() != original_query

        first = await self.call(
            tool_name,
            {"query": effective_query, "top_k": top_k},
            context,
            use_cache=True,
        )
        normalized = self._normalize_search_result(first, top_k, rewritten, effective_query)
        if not first.success or normalized.sufficient is not False:
            return normalized

        if rewritten:
            normalized.error = normalized.error or "改写后仍没有足够证据"
            return normalized

        retry_query = await self.rewrite_query(original_query, context)
        if retry_query.strip() == original_query:
            normalized.error = normalized.error or "没有足够证据，且查询无法进一步改写"
            return normalized

        logger.info("证据不足，查询改写后重试: %r → %r", original_query, retry_query)
        retry = await self.call(
            tool_name,
            {"query": retry_query, "top_k": top_k},
            context,
            use_cache=True,
        )
        result = self._normalize_search_result(retry, top_k, True, retry_query)
        if result.sufficient is False:
            result.error = result.error or "改写后仍没有足够证据"
        return result

    @classmethod
    def _normalize_search_result(
        cls,
        result: ToolResult,
        top_k: int,
        rewritten: bool,
        query_used: str,
    ) -> ToolResult:
        if not result.success:
            result.rewritten = rewritten
            result.query_used = query_used
            return result

        if isinstance(result.data, dict):
            payload = result.data
            items = payload.get("results", [])
            result.data = cls._deduplicate_results(items, top_k)
            result.reranked = bool(payload.get("reranked", result.reranked))
            result.sufficient = bool(payload.get("sufficient", False))
            result.error = None if result.sufficient else str(payload.get("reason") or "证据不足")
        elif isinstance(result.data, list):
            # 兼容非 RAG 检索工具和测试替身；非空列表视为已有可用结果。
            result.data = cls._deduplicate_results(result.data, top_k)
            result.sufficient = bool(result.data)
        else:
            result.data = []
            result.sufficient = False
            result.error = "检索工具返回了无法识别的数据结构"

        result.rewritten = rewritten
        result.query_used = query_used
        return result

    @staticmethod
    def _deduplicate_results(items: Any, top_k: int) -> List[Any]:
        if not isinstance(items, list):
            return []
        seen = set()
        unique = []
        for item in items:
            if isinstance(item, dict):
                key = item.get("parent_id") or (
                    str(item.get("title", "")).strip(),
                    str(item.get("content", "")).strip(),
                )
            else:
                key = str(item).strip()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
            if len(unique) >= top_k:
                break
        return unique

    @classmethod
    def _should_rewrite_before_search(
        cls,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> bool:
        if not cls._rewrite_context(context):
            return False
        compact = re.sub(r"[\s，。！？、,.!?;；:：]", "", query)
        context_dependent = bool(re.search(r"它|这个|那个|上述|前面|刚才|该问题|该订单", query))
        has_explicit_code = bool(re.search(r"[A-Za-z]*\d{3,}", query))
        too_short = len(compact) <= 6 and not has_explicit_code
        return context_dependent or too_short

    @staticmethod
    def _rewrite_context(context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return ""
        history = context.get("history") if isinstance(context, dict) else None
        if not isinstance(history, list):
            return ""
        lines = []
        for item in history[-4:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user"))
            content = str(item.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)[-1600:]

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict) -> str:
        payload = {"params": params}
        return f"{name}:{hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict) -> Optional[Any]:
        key = self._cache_key(name, params)
        if key in self._cache:
            data, expire_at = self._cache[key]
            if time.monotonic() < expire_at:
                return data
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
    ) -> None:
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params)] = (data, time.monotonic() + ttl)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
