"""针对运行中 AgentWorks `/search` 接口的最小 RAG 回归评测。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import httpx


DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases" / "rag_retrieval_cases.json"


@dataclass(frozen=True)
class RagRetrievalCase:
    case_id: str
    query: str
    expected_titles: Sequence[str]
    expect_sufficient: bool


def load_cases(path: Path) -> List[RagRetrievalCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("RAG 评测文件必须是 JSON 数组")

    cases = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("每个 RAG 评测用例必须是对象")
        case_id = str(raw.get("id", "")).strip()
        query = str(raw.get("query", "")).strip()
        expected_titles = raw.get("expected_titles", [])
        expect_sufficient = raw.get("expect_sufficient")
        if not case_id or not query:
            raise ValueError("RAG 评测用例必须包含非空 id 和 query")
        if not isinstance(expected_titles, list) or not all(
            isinstance(title, str) and title.strip() for title in expected_titles
        ):
            raise ValueError(f"用例 {case_id} 的 expected_titles 必须是字符串数组")
        if not isinstance(expect_sufficient, bool):
            raise ValueError(f"用例 {case_id} 的 expect_sufficient 必须是布尔值")
        if expect_sufficient and not expected_titles:
            raise ValueError(f"正例 {case_id} 至少需要一个 expected_titles")
        cases.append(RagRetrievalCase(
            case_id=case_id,
            query=query,
            expected_titles=tuple(title.strip() for title in expected_titles),
            expect_sufficient=expect_sufficient,
        ))
    return cases


def evaluate_case(case: RagRetrievalCase, payload: Dict[str, Any]) -> Dict[str, Any]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []

    raw_sufficient = payload.get("sufficient")
    sufficient = raw_sufficient if isinstance(raw_sufficient, bool) else None
    titles = [
        str(item.get("title", "")).strip()
        for item in results
        if isinstance(item, dict)
    ]
    parent_ids = [
        str(item.get("parent_id", "")).strip()
        for item in results
        if isinstance(item, dict) and item.get("parent_id")
    ]

    sufficiency_ok = sufficient is not None and sufficient == case.expect_sufficient
    title_hit = (
        any(expected in titles for expected in case.expected_titles)
        if case.expect_sufficient
        else not results
    )
    result_limit_ok = len(results) <= 3
    unique_parents_ok = len(parent_ids) == len(set(parent_ids))
    passed = sufficiency_ok and title_hit and result_limit_ok and unique_parents_ok

    return {
        "id": case.case_id,
        "query": case.query,
        "passed": passed,
        "expected_sufficient": case.expect_sufficient,
        "actual_sufficient": sufficient,
        "expected_titles": list(case.expected_titles),
        "actual_titles": titles,
        "query_used": payload.get("query_used"),
        "rewritten": bool(payload.get("rewritten")),
        "checks": {
            "sufficiency": sufficiency_ok,
            "title_hit_or_empty_rejection": title_hit,
            "max_three_results": result_limit_ok,
            "unique_parents": unique_parents_ok,
        },
        "message": payload.get("message"),
    }


def build_report(case_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    details = list(case_results)
    passed = sum(1 for item in details if item.get("passed") is True)
    total = len(details)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "details": details,
    }


def run_remote(base_url: str, cases: Sequence[RagRetrievalCase], timeout_s: float) -> Dict[str, Any]:
    case_results = []
    endpoint = f"{base_url.rstrip('/')}/search"
    with httpx.Client(timeout=timeout_s) as client:
        for case in cases:
            try:
                response = client.post(endpoint, params={"query": case.query, "top_k": 3})
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("/search 响应不是 JSON 对象")
                case_results.append(evaluate_case(case, payload))
            except Exception as ex:
                case_results.append({
                    "id": case.case_id,
                    "query": case.query,
                    "passed": False,
                    "error": str(ex),
                })
    return build_report(case_results)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 AgentWorks 固定 RAG 回归集")
    parser.add_argument("--base-url", default="http://localhost:8000", help="AgentWorks API 地址")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="评测集 JSON 路径")
    parser.add_argument("--timeout", type=float, default=120.0, help="单个查询超时秒数")
    parser.add_argument("--output", type=Path, help="可选：保存 JSON 报告")
    args = parser.parse_args()

    report = run_remote(args.base_url, load_cases(args.cases), args.timeout)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
