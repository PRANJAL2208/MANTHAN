import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import parse_json_safely, AdvocateAgent, JudgeAgent

def test_parse_json_safely_valid():
    """Verify parse_json_safely parses clean JSON correctly."""
    text = '{"hypothesis": "Target A", "rationale": "Reason A"}'
    res = parse_json_safely(text)
    assert res == {"hypothesis": "Target A", "rationale": "Reason A"}

def test_parse_json_safely_invalid():
    """Verify parse_json_safely returns empty dict on complete garbage."""
    text = 'This is not json at all!'
    res = parse_json_safely(text)
    assert res == {}

def test_parse_json_safely_surrounded_by_text():
    """Verify parse_json_safely can extract JSON when wrapped in code blocks or extra text."""
    # Text surrounding JSON
    text = 'Some prefix text {"hypothesis": "Target A", "rationale": "Reason A"} some suffix text'
    res = parse_json_safely(text)
    assert res == {"hypothesis": "Target A", "rationale": "Reason A"}
    
    # Markdown codeblock
    text_md = '```json\n{\n  "key": "val"\n}\n```'
    res_md = parse_json_safely(text_md)
    assert res_md == {"key": "val"}

def test_advocate_agent_propose_hypothesis():
    """Verify that propose_hypothesis calls search and call_llm, updates self.hypothesis, and returns results."""
    agent = AdvocateAgent("Advocate A")
    mock_papers = [{"title": "Paper A", "abstract": "Abstract A", "year": 2021, "citationCount": 5}]
    
    with patch("agents.search_papers", return_value=mock_papers) as mock_search, \
         patch("agents.call_llm", return_value='{"hypothesis": "Test Hypothesis A", "rationale": "Test Rationale A"}') as mock_call_llm:
         
        res = agent.propose_hypothesis("Test Topic")
        
        assert res == {"hypothesis": "Test Hypothesis A", "rationale": "Test Rationale A"}
        assert agent.hypothesis == "Test Hypothesis A"
        assert agent.last_papers == mock_papers
        
        mock_search.assert_called_once_with("Test Topic", limit=5)
        mock_call_llm.assert_called_once()
        # Verify prompt details
        sys_prompt, user_prompt = mock_call_llm.call_args[0][:2]
        assert "Advocate A" in sys_prompt
        assert "Test Topic" in user_prompt

def test_advocate_agent_oppose_hypothesis():
    """Verify that oppose_hypothesis generates a competing hypothesis successfully."""
    agent = AdvocateAgent("Advocate B")
    mock_papers = [{"title": "Paper B", "abstract": "Abstract B", "year": 2022, "citationCount": 1}]
    
    with patch("agents.search_papers", return_value=mock_papers) as mock_search, \
         patch("agents.call_llm", return_value='{"hypothesis": "Test Hypothesis B", "rationale": "Test Rationale B"}') as mock_call_llm:
         
        res = agent.oppose_hypothesis("Test Topic", "Test Hypothesis A")
        
        assert res == {"hypothesis": "Test Hypothesis B", "rationale": "Test Rationale B"}
        assert agent.hypothesis == "Test Hypothesis B"
        assert agent.last_papers == mock_papers
        
        mock_search.assert_called_once_with("Test Topic", limit=5)
        # Verify instructions explicitly mention Advocate B opposing A
        sys_prompt, user_prompt = mock_call_llm.call_args[0][:2]
        assert "Advocate B" in sys_prompt
        assert "Test Hypothesis A" in user_prompt

def test_advocate_agent_open_argument():
    """Verify that open_argument calls call_llm and runs the check_grounding guardrail check."""
    agent = AdvocateAgent("Advocate A", "Conserved projections")
    agent.last_papers = [{"title": "Paper A", "abstract": "This abstract maps olfactory bulb targets.", "year": 2020, "citationCount": 3}]
    
    argument_text = "As shown in [1], we mapped targets."
    mock_grounding_res = {"grounded": True, "matched_papers": ["Paper A"], "warning": None}
    
    with patch("agents.call_llm", return_value=argument_text) as mock_call_llm, \
         patch("agents.check_grounding", return_value=mock_grounding_res) as mock_check:
         
        res = agent.open_argument("What is the olfactory target?")
        
        assert res["speaker"] == "Advocate A"
        assert res["text"] == argument_text
        assert res["grounding"] == mock_grounding_res
        
        mock_call_llm.assert_called_once()
        mock_check.assert_called_once_with(argument_text, agent.last_papers)

def test_advocate_agent_rebuttal():
    """Verify that rebuttal calls call_llm comparing own papers with opponent's arguments and papers."""
    agent = AdvocateAgent("Advocate A", "Conserved projections")
    agent.last_papers = [{"title": "Paper A", "abstract": "Abstract A", "year": 2020, "citationCount": 3}]
    opponent_papers = [{"title": "Paper B", "abstract": "Abstract B", "year": 2022, "citationCount": 10}]
    
    rebuttal_text = "Opponent's citation of [Opponent-1] is flawed."
    mock_grounding_res = {"grounded": True, "matched_papers": [], "warning": None}
    
    with patch("agents.call_llm", return_value=rebuttal_text) as mock_call_llm, \
         patch("agents.check_grounding", return_value=mock_grounding_res) as mock_check:
         
        res = agent.rebuttal("Opponent claims variable wiring [1]", opponent_papers)
        
        assert res["speaker"] == "Advocate A"
        assert res["text"] == rebuttal_text
        assert res["grounding"] == mock_grounding_res
        
        mock_call_llm.assert_called_once()
        # Verify instructions compared opponent's argument and papers
        user_prompt = mock_call_llm.call_args[0][1]
        assert "Opponent claims variable wiring" in user_prompt
        assert "[Opponent-1]" in user_prompt
        mock_check.assert_called_once_with(rebuttal_text, agent.last_papers)

def test_judge_agent_evaluate_debate_success():
    """Verify that JudgeAgent evaluates the transcript and returns parsed JSON results."""
    judge = JudgeAgent()
    transcript = [
        {"speaker": "Advocate A", "text": "Argument A", "grounding": {"grounded": True, "warning": None}},
        {"speaker": "Advocate B", "text": "Argument B", "grounding": {"grounded": False, "warning": "No citation"}}
    ]
    
    judge_json_res = (
        '{"should_stop": true, '
        '"verdict_summary": "Summary of debate", '
        '"winner": "Advocate A", '
        '"deduction_rationale": "Deduction rationale info"}'
    )
    
    with patch("agents.call_llm", return_value=judge_json_res) as mock_call_llm:
        eval_res = judge.evaluate_debate("What is the olfactory target?", transcript)
        
        assert eval_res["should_stop"] is True
        assert eval_res["verdict_summary"] == "Summary of debate"
        assert eval_res["winner"] == "Advocate A"
        assert eval_res["deduction_rationale"] == "Deduction rationale info"
        
        mock_call_llm.assert_called_once()
        sys_prompt, user_prompt = mock_call_llm.call_args[0][:2]
        assert "decisive, neutral scientific judge" in sys_prompt
        assert "Advocate A (Grounded)" in user_prompt
        assert "Advocate B (Ungrounded (Warning: No citation))" in user_prompt

def test_judge_agent_evaluate_debate_fallback():
    """Verify that JudgeAgent falls back gracefully when judge LLM response is malformed."""
    judge = JudgeAgent()
    transcript = []
    
    # Completely invalid JSON response
    malformed_response = "I judge Advocate A as winner but refuse to return JSON!"
    
    with patch("agents.call_llm", return_value=malformed_response):
        eval_res = judge.evaluate_debate("What is the olfactory target?", transcript)
        
        # Verify fallback structure
        assert eval_res["should_stop"] is True
        assert eval_res["winner"] == "Both - Context Specific"
        assert "Failed to parse" in eval_res["verdict_summary"]
        assert eval_res["deduction_rationale"] == malformed_response

def test_judge_agent_summarize():
    """Verify JudgeAgent.summarize returns the formatted verdict markdown block."""
    judge = JudgeAgent()
    eval_mock = {
        "should_stop": True,
        "winner": "Advocate B",
        "verdict_summary": "Advocate B has stronger proof",
        "deduction_rationale": "Rationale details here"
    }
    
    with patch.object(judge, "evaluate_debate", return_value=eval_mock):
        summary = judge.summarize("Topic A", [])
        
        assert "**Verdict:** Advocate B" in summary
        assert "**Summary:** Advocate B has stronger proof" in summary
        assert "**Rationale:** Rationale details here" in summary
