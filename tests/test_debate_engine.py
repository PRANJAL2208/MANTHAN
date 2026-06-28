import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debate_engine import DebateState, execute_agent_turn_generator, run_debate_stream, run_debate
from agents import AdvocateAgent

MOCK_PAPERS = [
    {
        "paperId": "p1",
        "title": "Crypt neuron projection patterns",
        "abstract": "We mapped projection targets of olfactory cells and found a conserved glomerular target region.",
        "year": 2021,
        "citationCount": 5
    }
]

# ---------------------------------------------------------
# Test DebateState
# ---------------------------------------------------------

def test_debate_state_properties():
    """Verify that DebateState compiles metrics and formats output correctly."""
    state = DebateState(
        question="What is the olfactory target?",
        hypothesis_a="Conserved target",
        hypothesis_b="Variable target"
    )
    
    # Add one grounded turn
    state.add_turn({
        "speaker": "Advocate A",
        "text": "Conserved target [1]",
        "grounding": {"grounded": True, "warning": None}
    })
    
    # Add one unverified turn
    state.add_turn({
        "speaker": "Advocate B",
        "text": "Variable target [2]",
        "grounding": {"grounded": False, "warning": "Low overlap"}
    })
    
    assert state.grounded_count == 1
    assert state.unverified_count == 1
    
    flat_dict = state.to_dict()
    assert flat_dict["question"] == "What is the olfactory target?"
    assert flat_dict["hypothesis_a"] == "Conserved target"
    assert flat_dict["hypothesis_b"] == "Variable target"
    assert len(flat_dict["transcript"]) == 2
    assert flat_dict["metrics"]["total_turns"] == 2
    assert flat_dict["metrics"]["grounded_turns"] == 1
    assert flat_dict["metrics"]["unverified_turns"] == 1


# ---------------------------------------------------------
# Test execute_agent_turn_generator (Self-Correction Loop)
# ---------------------------------------------------------

def test_execute_agent_turn_generator_grounded_immediately():
    """Verify that if the first attempt is grounded, we yield the turn immediately with no retries."""
    agent = AdvocateAgent("Advocate A", "Conserved target")
    agent.last_papers = MOCK_PAPERS
    
    turn_res = {
        "speaker": "Advocate A",
        "text": "This has high overlap and is grounded [1].",
        "grounding": {"grounded": True, "warning": None}
    }
    
    with patch.object(agent, "open_argument", return_value=turn_res) as mock_open:
        events = list(execute_agent_turn_generator(agent, "open", "Olfactory topic"))
        
        assert len(events) == 1
        assert events[0]["type"] == "turn"
        assert events[0]["data"]["speaker"] == "Advocate A"
        assert events[0]["data"]["text"] == "This has high overlap and is grounded [1]."
        assert events[0]["data"]["grounding"]["grounded"] is True
        assert events[0]["data"]["retries"] == 0
        mock_open.assert_called_once_with("Olfactory topic")

def test_execute_agent_turn_generator_success_after_one_retry():
    """Verify that if the first attempt is ungrounded, it yields a retry warning and tries again."""
    agent = AdvocateAgent("Advocate A", "Conserved target")
    agent.last_papers = MOCK_PAPERS
    
    first_response = {
        "speaker": "Advocate A",
        "text": "This is ungrounded claim [1].",
        "grounding": {"grounded": False, "warning": "Low overlap"}
    }
    
    second_response_text = "This corrected text has high overlap and is grounded [1]."
    second_grounding_res = {"grounded": True, "warning": None}
    
    with patch.object(agent, "open_argument", return_value=first_response) as mock_open, \
         patch("debate_engine.call_llm", return_value=second_response_text) as mock_call_llm, \
         patch("debate_engine.check_grounding", return_value=second_grounding_res) as mock_check:
         
        events = list(execute_agent_turn_generator(agent, "open", "Olfactory topic", max_retries=2))
        
        # Should yield 1 grounding_retry event and then 1 turn event
        assert len(events) == 2
        assert events[0]["type"] == "grounding_retry"
        assert events[0]["speaker"] == "Advocate A"
        assert events[0]["retry_num"] == 1
        assert events[0]["warning"] == "Low overlap"
        
        assert events[1]["type"] == "turn"
        assert events[1]["data"]["text"] == second_response_text
        assert events[1]["data"]["grounding"]["grounded"] is True
        assert events[1]["data"]["retries"] == 1
        
        mock_open.assert_called_once_with("Olfactory topic")
        mock_call_llm.assert_called_once()
        # Verify feedback instructions prompt contains the warning
        feedback_prompt = mock_call_llm.call_args[0][1]
        assert "Low overlap" in feedback_prompt
        mock_check.assert_called_once_with(second_response_text, MOCK_PAPERS)

def test_execute_agent_turn_generator_max_retries_exceeded():
    """Verify that if the attempts fail grounding repeatedly, it yields retries and then yields ungrounded turn."""
    agent = AdvocateAgent("Advocate A", "Conserved target")
    agent.last_papers = MOCK_PAPERS
    
    first_response = {
        "speaker": "Advocate A",
        "text": "This is ungrounded claim [1].",
        "grounding": {"grounded": False, "warning": "Low overlap"}
    }
    
    retry_response_text = "Still ungrounded claim [1]."
    retry_grounding_res = {"grounded": False, "warning": "Still low overlap"}
    
    with patch.object(agent, "open_argument", return_value=first_response) as mock_open, \
         patch("debate_engine.call_llm", return_value=retry_response_text) as mock_call_llm, \
         patch("debate_engine.check_grounding", return_value=retry_grounding_res) as mock_check:
         
        events = list(execute_agent_turn_generator(agent, "open", "Olfactory topic", max_retries=2))
        
        # 1 first attempt failure + 2 retry attempts = 3 total calls to generator check
        # But max_retries is 2. So it yields 2 retry warning events, then the final turn.
        assert len(events) == 3
        assert events[0]["type"] == "grounding_retry"
        assert events[0]["retry_num"] == 1
        assert events[1]["type"] == "grounding_retry"
        assert events[1]["retry_num"] == 2
        
        assert events[2]["type"] == "turn"
        assert events[2]["data"]["text"] == retry_response_text
        assert events[2]["data"]["grounding"]["grounded"] is False
        assert events[2]["data"]["retries"] == 2
        
        assert mock_call_llm.call_count == 2
        assert mock_check.call_count == 2


# ---------------------------------------------------------
# Test run_debate_stream (Coroutine pausing & resumption)
# ---------------------------------------------------------

@patch("debate_engine.JudgeAgent")
@patch("debate_engine.AdvocateAgent")
@patch("debate_engine.search_papers", return_value=MOCK_PAPERS)
def test_run_debate_stream_flow_with_intermission(mock_search, mock_advocate_cls, mock_judge_cls):
    """Verify that run_debate_stream yields correct sequence, pauses for user input, and resumes successfully."""
    # Setup mocks
    mock_adv_a = MagicMock()
    mock_adv_b = MagicMock()
    mock_advocate_cls.side_effect = lambda name, *args: mock_adv_a if name.endswith("A") else mock_adv_b
    
    mock_adv_a.name = "Advocate A"
    mock_adv_a.last_papers = MOCK_PAPERS
    mock_adv_a.propose_hypothesis.return_value = {"hypothesis": "Hypothesis A", "rationale": "Rationale A"}
    mock_adv_a.open_argument.return_value = {"speaker": "Advocate A", "text": "Opening argument A", "grounding": {"grounded": True, "warning": None}}
    mock_adv_a.rebuttal.return_value = {"speaker": "Advocate A", "text": "Rebuttal A", "grounding": {"grounded": True, "warning": None}}
    
    mock_adv_b.name = "Advocate B"
    mock_adv_b.last_papers = MOCK_PAPERS
    mock_adv_b.oppose_hypothesis.return_value = {"hypothesis": "Hypothesis B", "rationale": "Rationale B"}
    mock_adv_b.open_argument.return_value = {"speaker": "Advocate B", "text": "Opening argument B", "grounding": {"grounded": True, "warning": None}}
    mock_adv_b.rebuttal.return_value = {"speaker": "Advocate B", "text": "Rebuttal B", "grounding": {"grounded": True, "warning": None}}
    
    mock_judge = MagicMock()
    mock_judge_cls.return_value = mock_judge
    mock_judge.evaluate_debate.return_value = {
        "should_stop": True,
        "winner": "Advocate A",
        "verdict_summary": "Judge summary",
        "deduction_rationale": "Deduction details"
    }
    
    # Initialize the generator stream
    stream = run_debate_stream("Topic question?", rounds=2, use_mock=False)
    
    # Step 1: Initial Stance & Hypotheses
    event = next(stream)
    assert event["type"] == "status"
    
    event = next(stream)
    assert event["type"] == "status"
    
    event = next(stream)
    assert event["type"] == "hypothesis"
    assert event["speaker"] == "Advocate A"
    
    event = next(stream)
    assert event["type"] == "status"
    
    event = next(stream)
    assert event["type"] == "hypothesis"
    assert event["speaker"] == "Advocate B"
    
    # Step 2: Opening arguments
    event = next(stream)
    assert event["type"] == "status"
    event = next(stream)
    assert event["type"] == "turn"
    assert event["data"]["speaker"] == "Advocate A"
    
    event = next(stream)
    assert event["type"] == "status"
    event = next(stream)
    assert event["type"] == "turn"
    assert event["data"]["speaker"] == "Advocate B"
    
    # Step 3: Intermission Pause
    event = next(stream)
    assert event["type"] == "pause"
    
    # Resume by sending user challenge!
    event = stream.send("User challenge about translation limits")
    
    # Assert advocates digest user challenge
    assert event["type"] == "status"
    assert "User challenge about translation limits" in event["message"]
    
    # Step 4: Rebuttals addressing the challenge
    event = next(stream)
    assert event["type"] == "status"
    event = next(stream)
    assert event["type"] == "turn"
    assert event["data"]["speaker"] == "Advocate A"
    # Verify that the rebuttal was generated with the user challenge directive injected
    mock_adv_a.rebuttal.assert_called_once()
    rebuttal_input = mock_adv_a.rebuttal.call_args[0][0]
    assert "User Cross-Examination Directive: 'User challenge about translation limits'" in rebuttal_input
    
    event = next(stream)
    assert event["type"] == "status"
    event = next(stream)
    assert event["type"] == "turn"
    assert event["data"]["speaker"] == "Advocate B"
    
    # Step 5: Judge verdict evaluation
    event = next(stream)
    assert event["type"] == "status"  # "Judge evaluating latest round..."
    
    event = next(stream)
    assert event["type"] == "status"  # "Judge performing final review and drawing deduction..."
    
    event = next(stream)
    assert event["type"] == "verdict"
    assert "**Verdict:** Advocate A" in event["val"]
    
    # Assert stream completes
    with pytest.raises(StopIteration):
        next(stream)


# ---------------------------------------------------------
# Test run_debate and Mock Run wrappers
# ---------------------------------------------------------

def test_run_debate_synchronous_wrapper():
    """Verify that the synchronous run_debate wrapper consumes the generator and outputs expected output format."""
    mock_stream_events = [
        {"type": "status", "message": "Starting"},
        {"type": "hypothesis", "speaker": "Advocate A", "hypothesis": "Hyp A", "rationale": "Rat A", "papers": []},
        {"type": "hypothesis", "speaker": "Advocate B", "hypothesis": "Hyp B", "rationale": "Rat B", "papers": []},
        {"type": "turn", "data": {"speaker": "Advocate A", "text": "Turn 1 text", "grounding": {"grounded": True}}},
        {"type": "pause"}, # Generator send(None) should auto-resume this
        {"type": "turn", "data": {"speaker": "Advocate B", "text": "Turn 2 text", "grounding": {"grounded": False}}},
        {"type": "verdict", "val": "**Verdict:** Advocate A\n\n**Summary:** summary\n\n**Rationale:** rationale"}
    ]
    
    def mock_stream_gen(*args, **kwargs):
        for event in mock_stream_events:
            if event["type"] == "pause":
                # Simulated pause coroutine yield
                yield event
            else:
                yield event

    with patch("debate_engine.run_debate_stream", side_effect=mock_stream_gen):
        res = run_debate("Test Question?", rounds=2)
        
        assert res["question"] == "Test Question?"
        assert res["hypothesis_a"] == "Hyp A"
        assert res["hypothesis_b"] == "Hyp B"
        assert len(res["transcript"]) == 2
        assert res["transcript"][0]["speaker"] == "Advocate A"
        assert res["transcript"][1]["speaker"] == "Advocate B"
        assert "**Verdict:** Advocate A" in res["verdict"]
        assert res["metrics"]["total_turns"] == 2
        assert res["metrics"]["grounded_turns"] == 1
        assert res["metrics"]["unverified_turns"] == 1

def test_run_mock_debate_simulation_runs_to_conclusion():
    """Verify that run_mock_debate (use_mock=True) yields all simulated events successfully without exceptions."""
    stream = run_debate_stream("Topic question?", rounds=2, use_mock=True)
    events = []
    try:
        event = next(stream)
        while True:
            events.append(event)
            if event["type"] == "pause":
                event = stream.send("Zebrafish translation limits")
            else:
                event = next(stream)
    except StopIteration:
        pass
        
    assert len(events) > 5
    # Verify we had a pause and a verdict event
    event_types = [e["type"] for e in events]
    assert "pause" in event_types
    assert "verdict" in event_types
    assert "hypothesis" in event_types
    assert "turn" in event_types