import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adk_research_planner as planner


# ---------------------------------------------------------
# Test _extract_json_decision (pure parsing helper)
# ---------------------------------------------------------

def test_extract_json_decision_valid():
    text = 'Some reasoning here.\n{"need_more_research": true, "reasoning": "thin coverage", "query_used": "crypt neuron wiring"}'
    res = planner._extract_json_decision(text)
    assert res == {
        "need_more_research": True,
        "reasoning": "thin coverage",
        "query_used": "crypt neuron wiring",
    }

def test_extract_json_decision_no_json_falls_back():
    text = "I could not decide anything useful."
    res = planner._extract_json_decision(text)
    assert res["need_more_research"] is False
    assert "I could not decide" in res["reasoning"]
    assert res["query_used"] is None

def test_extract_json_decision_malformed_json_falls_back():
    text = 'Reasoning: {"need_more_research": true, "reasoning": }'  # broken JSON
    res = planner._extract_json_decision(text)
    assert res["need_more_research"] is False
    assert res["query_used"] is None


# ---------------------------------------------------------
# Test _search_literature_tool (the ADK FunctionTool's underlying fn)
# ---------------------------------------------------------

def test_search_literature_tool_calls_search_and_formats():
    mock_papers = [{"title": "Paper X", "abstract": "Abstract X", "year": 2020, "citationCount": 3}]
    with patch("adk_research_planner.search_papers", return_value=mock_papers) as mock_search, \
         patch("adk_research_planner.format_papers_for_prompt", return_value="Formatted result") as mock_format:

        res = planner._search_literature_tool("olfactory wiring", limit=2)

        assert res == "Formatted result"
        mock_search.assert_called_once_with("olfactory wiring", limit=2)
        mock_format.assert_called_once_with(mock_papers)


# ---------------------------------------------------------
# Test build_research_planner_agent (real ADK LlmAgent construction)
# ---------------------------------------------------------

def test_build_research_planner_agent_structure():
    """Verify the ADK LlmAgent is wired up with the expected name, model,
    and exactly one FunctionTool wrapping our search function."""
    agent = planner.build_research_planner_agent()

    assert agent.name == "research_planner"
    assert agent.model == planner.PLANNER_MODEL
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "_search_literature_tool"
    assert "need_more_research" in agent.instruction


# ---------------------------------------------------------
# Test plan_research (mocked ADK Runner — fully offline)
# ---------------------------------------------------------

def _make_final_event(text: str):
    """Builds a lightweight stand-in for a google.adk.events.Event that
    looks like a final response containing `text`."""
    event = MagicMock()
    event.is_final_response.return_value = True
    part = MagicMock()
    part.text = text
    event.content.parts = [part]
    return event


def test_plan_research_returns_parsed_decision(monkeypatch):
    """Verify plan_research drives a (mocked) ADK Runner/session and
    correctly parses the final decision JSON, without any real network
    or API calls."""
    fake_session = MagicMock()
    fake_session.id = "session-123"

    mock_runner_instance = MagicMock()
    mock_runner_instance.session_service.create_session = AsyncMock(return_value=fake_session)

    final_event = _make_final_event(
        'Coverage looks thin.\n{"need_more_research": true, "reasoning": "only one paper per side", "query_used": "crypt neuron conserved targets"}'
    )

    async def fake_run_async(*args, **kwargs):
        for e in [final_event]:
            yield e

    mock_runner_instance.run_async = fake_run_async

    with patch("adk_research_planner.InMemoryRunner", return_value=mock_runner_instance) as mock_runner_cls, \
         patch("adk_research_planner.build_research_planner_agent", return_value=MagicMock()) as mock_build:

        decision = planner.plan_research(
            question="Do crypt neurons project to a conserved target?",
            hyp_a="Conserved target hypothesis",
            hyp_b="Variable target hypothesis",
            coverage_summary="1 paper per advocate",
        )

        assert decision == {
            "need_more_research": True,
            "reasoning": "only one paper per side",
            "query_used": "crypt neuron conserved targets",
        }
        mock_build.assert_called_once()
        mock_runner_cls.assert_called_once()
        mock_runner_instance.session_service.create_session.assert_called_once()


def test_plan_research_sufficient_coverage(monkeypatch):
    """Verify a 'no more research needed' decision round-trips correctly."""
    fake_session = MagicMock()
    fake_session.id = "session-456"

    mock_runner_instance = MagicMock()
    mock_runner_instance.session_service.create_session = AsyncMock(return_value=fake_session)

    final_event = _make_final_event(
        '{"need_more_research": false, "reasoning": "coverage is strong on both sides", "query_used": null}'
    )

    async def fake_run_async(*args, **kwargs):
        yield final_event

    mock_runner_instance.run_async = fake_run_async

    with patch("adk_research_planner.InMemoryRunner", return_value=mock_runner_instance), \
         patch("adk_research_planner.build_research_planner_agent", return_value=MagicMock()):

        decision = planner.plan_research(
            question="Q", hyp_a="A", hyp_b="B", coverage_summary="plenty of papers"
        )

        assert decision["need_more_research"] is False
        assert decision["query_used"] is None
