import json

import pytest

from evaluation.rag_retrieval_evaluator import (
    DEFAULT_CASES_PATH,
    RagRetrievalCase,
    build_report,
    evaluate_case,
    load_cases,
)


def test_fixed_rag_cases_are_valid_and_cover_answerable_and_unanswerable_queries():
    cases = load_cases(DEFAULT_CASES_PATH)

    assert len(cases) == 10
    assert any(case.expect_sufficient for case in cases)
    assert any(not case.expect_sufficient for case in cases)
    assert len({case.case_id for case in cases}) == len(cases)


def test_positive_case_requires_expected_title_and_unique_top_three_parents():
    case = RagRetrievalCase("refund", "退款多久到账", ("退款政策",), True)
    payload = {
        "sufficient": True,
        "results": [
            {"title": "退款政策", "parent_id": "p1"},
            {"title": "配送说明", "parent_id": "p2"},
        ],
        "query_used": "退款多久到账",
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_negative_case_requires_gate_rejection_and_empty_results():
    case = RagRetrievalCase("weather", "明天天气如何", (), False)

    accepted = evaluate_case(case, {"sufficient": False, "results": []})
    missing_gate_result = evaluate_case(case, {"results": []})
    leaked = evaluate_case(case, {
        "sufficient": False,
        "results": [{"title": "配送说明", "parent_id": "p1"}],
    })

    assert accepted["passed"] is True
    assert missing_gate_result["passed"] is False
    assert leaked["passed"] is False


def test_duplicate_parent_or_more_than_three_results_fails_constraints():
    case = RagRetrievalCase("refund", "退款", ("退款政策",), True)
    payload = {
        "sufficient": True,
        "results": [
            {"title": "退款政策", "parent_id": "p1"},
            {"title": "退款政策", "parent_id": "p1"},
            {"title": "订单查询", "parent_id": "p2"},
            {"title": "账户安全", "parent_id": "p3"},
        ],
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is False
    assert result["checks"]["max_three_results"] is False
    assert result["checks"]["unique_parents"] is False


def test_load_cases_rejects_positive_case_without_expected_title(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([{
        "id": "invalid",
        "query": "问题",
        "expected_titles": [],
        "expect_sufficient": True,
    }], ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="至少需要一个"):
        load_cases(path)


def test_report_has_simple_pass_rate_without_research_only_metrics():
    report = build_report([{"passed": True}, {"passed": False}])

    assert report["total"] == 2
    assert report["passed"] == 1
    assert report["failed"] == 1
    assert report["pass_rate"] == 0.5
