"""
debate_engine.py
----------------
This is the ORCHESTRATOR. It manages the sequence: who speaks when,
how many rounds happen, and exposes a generator-based streaming API 
supporting self-correction loops and interactive user interventions.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from agents import AdvocateAgent, JudgeAgent
from literature_search import search_papers
from llm_client import call_llm
from guardrails import check_grounding


class DebateState:
    """
    Structured state tracking the full lifecycle of one debate.
    """

    def __init__(self, question: str, hypothesis_a: str, hypothesis_b: str):
        self.question = question
        self.hypothesis_a = hypothesis_a
        self.hypothesis_b = hypothesis_b
        self.turns: list[dict] = []
        self.verdict: str | None = None
        self.papers_a: list[dict] = []
        self.papers_b: list[dict] = []
        self.hypothesis_a_rationale: str = ""
        self.hypothesis_b_rationale: str = ""

    def add_turn(self, turn: dict):
        self.turns.append(turn)

    @property
    def grounded_count(self) -> int:
        return sum(1 for t in self.turns if t["grounding"]["grounded"])

    @property
    def unverified_count(self) -> int:
        return sum(1 for t in self.turns if not t["grounding"]["grounded"])

    def to_dict(self) -> dict:
        """Flat dict shape, kept for backward compatibility with app.py."""
        return {
            "question": self.question,
            "hypothesis_a": self.hypothesis_a,
            "hypothesis_b": self.hypothesis_b,
            "hypothesis_a_rationale": self.hypothesis_a_rationale,
            "hypothesis_b_rationale": self.hypothesis_b_rationale,
            "transcript": self.turns,
            "verdict": self.verdict,
            "papers_a": self.papers_a,
            "papers_b": self.papers_b,
            "metrics": {
                "total_turns": len(self.turns),
                "grounded_turns": self.grounded_count,
                "unverified_turns": self.unverified_count,
            },
        }


def execute_agent_turn_generator(agent: AdvocateAgent, turn_type: str, prompt_input: str, opponent_papers: list[dict] = None, max_retries: int = 2):
    """
    Generator version of turn execution. Yields warnings for grounding failures
    before yielding the final turn data.
    """
    if turn_type == "open":
        turn_data = agent.open_argument(prompt_input)
    else:
        turn_data = agent.rebuttal(prompt_input, opponent_papers)

    turn_data["retries"] = 0

    while not turn_data["grounding"]["grounded"] and turn_data["retries"] < max_retries:
        turn_data["retries"] += 1
        warning = turn_data["grounding"]["warning"]
        
        yield {
            "type": "grounding_retry",
            "speaker": agent.name,
            "retry_num": turn_data["retries"],
            "warning": warning
        }

        feedback_prompt = (
            f"Your previous response failed the grounding guardrail check:\n"
            f"\"{warning}\"\n\n"
            f"Please rewrite your argument, strictly adhering to the literature. "
            f"Only cite papers using correct [n] indices when the claim matches the abstract, "
            f"and avoid friendly filler phrases."
        )

        text = call_llm(agent._system_prompt(), feedback_prompt)
        grounding = check_grounding(text, agent.last_papers)
        
        turn_data["text"] = text
        turn_data["grounding"] = grounding

    yield {"type": "turn", "data": turn_data}


def run_debate_stream(question: str, hypothesis_a: str = None, hypothesis_b: str = None, rounds: int = 2, use_mock: bool = False, use_adk_planner: bool = False):
    """
    Exposes the debate flow as a generator (coroutine). 
    Yields intermediate status, stances, turns, pauses, and the final verdict.
    Supports user intervention injections during pause events via .send().

    use_adk_planner: opt-in flag (default False). When True, invokes the
    Google ADK-built Research Planner agent (see adk_research_planner.py)
    after the opening round to decide whether evidence coverage is thin
    enough to warrant another targeted literature search before the
    rebuttal rounds begin. Yields an extra {"type": "adk_planner", ...}
    event when enabled. Left False by default so the existing, tested
    event sequence (see test_debate_engine.py) is completely unaffected.
    """
    if use_mock:
        yield from run_mock_debate(question, rounds)
        return

    judge = JudgeAgent()

    # 1. Hypotheses Generation & Research
    if hypothesis_a is None or hypothesis_b is None:
        yield {"type": "status", "message": f"Querying literature databases on topic: '{question}'..."}
        advocate_a = AdvocateAgent("Advocate A")
        advocate_b = AdvocateAgent("Advocate B")

        # Pre-fetch literature for both advocates in parallel (cuts search time ~50%)
        # Both search the same topic; the second thread hits cache after the first completes.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(search_papers, question, 5)
            future_b = executor.submit(search_papers, question, 5)
            advocate_a.last_papers = future_a.result()
            advocate_b.last_papers = future_b.result()

        # Advocate A proposes hypothesis (uses pre-loaded papers — no extra API call)
        yield {"type": "status", "message": "Advocate A formulating Hypothesis A from literature..."}
        prop_a = advocate_a.propose_hypothesis(question)
        hyp_a = prop_a.get("hypothesis", f"The primary evidence supports a positive effect on: {question}.")
        rat_a = prop_a.get("rationale", "")

        yield {
            "type": "hypothesis",
            "speaker": "Advocate A",
            "hypothesis": hyp_a,
            "rationale": rat_a,
            "papers": advocate_a.last_papers
        }

        # Advocate B proposes opposing hypothesis (uses pre-loaded papers — no extra API call)
        yield {"type": "status", "message": "Advocate B formulating opposing Hypothesis B from literature..."}
        prop_b = advocate_b.oppose_hypothesis(question, hyp_a)
        hyp_b = prop_b.get("hypothesis", f"The evidence supports a nuanced or skeptical interpretation of: {question}.")
        rat_b = prop_b.get("rationale", "")

        yield {
            "type": "hypothesis",
            "speaker": "Advocate B",
            "hypothesis": hyp_b,
            "rationale": rat_b,
            "papers": advocate_b.last_papers
        }
    else:
        # Backward compatibility mode
        hyp_a = hypothesis_a
        hyp_b = hypothesis_b
        rat_a = ""
        rat_b = ""
        advocate_a = AdvocateAgent("Advocate A", hypothesis_a)
        advocate_b = AdvocateAgent("Advocate B", hypothesis_b)
        
        yield {"type": "status", "message": "Querying literature databases on hypotheses..."}
        advocate_a.last_papers = search_papers(f"{question} {hypothesis_a}", limit=3)
        advocate_b.last_papers = search_papers(f"{question} {hypothesis_b}", limit=3)
        
        yield {
            "type": "hypothesis",
            "speaker": "Advocate A",
            "hypothesis": hypothesis_a,
            "rationale": rat_a,
            "papers": advocate_a.last_papers
        }
        yield {
            "type": "hypothesis",
            "speaker": "Advocate B",
            "hypothesis": hypothesis_b,
            "rationale": rat_b,
            "papers": advocate_b.last_papers
        }

    state = DebateState(question, hyp_a, hyp_b)
    state.papers_a = advocate_a.last_papers
    state.papers_b = advocate_b.last_papers
    state.hypothesis_a_rationale = rat_a
    state.hypothesis_b_rationale = rat_b

    # 2. Opening Round
    yield {"type": "status", "message": "Advocate A formulating opening argument..."}
    turn_a = None
    for event in execute_agent_turn_generator(advocate_a, "open", question):
        if event["type"] == "turn":
            turn_a = event["data"]
        yield event
    state.add_turn(turn_a)

    yield {"type": "status", "message": "Advocate B formulating opening argument..."}
    turn_b = None
    for event in execute_agent_turn_generator(advocate_b, "open", question):
        if event["type"] == "turn":
            turn_b = event["data"]
        yield event
    state.add_turn(turn_b)

    # 2b. Optional: Google ADK Research Planner assesses evidence coverage
    if use_adk_planner:
        yield {"type": "status", "message": "ADK Research Planner assessing evidence coverage..."}
        coverage_summary = (
            f"Advocate A: {len(state.papers_a)} papers retrieved, opening argument "
            f"{'grounded' if turn_a['grounding']['grounded'] else 'NOT grounded'}. "
            f"Advocate B: {len(state.papers_b)} papers retrieved, opening argument "
            f"{'grounded' if turn_b['grounding']['grounded'] else 'NOT grounded'}."
        )
        try:
            from adk_research_planner import plan_research
            planner_decision = plan_research(question, hyp_a, hyp_b, coverage_summary)
        except Exception as e:
            planner_decision = {
                "need_more_research": False,
                "reasoning": f"ADK Research Planner unavailable ({e}); proceeding without it.",
                "query_used": None,
            }
        yield {"type": "adk_planner", "data": planner_decision}

    # 3. INTERMISSION: Pause for user cross-examination challenge
    user_challenge = yield {"type": "pause"}
    
    if user_challenge:
        yield {"type": "status", "message": f"Advocates digesting cross-examination challenge: '{user_challenge}'..."}

    # 4. Rebuttal Rounds
    max_rounds = rounds if rounds > 0 else 3
    for r in range(max_rounds - 1):
        # A rebuts B
        yield {"type": "status", "message": "Advocate A formulating rebuttal..."}
        rebut_input_a = state.turns[-1]["text"]
        if user_challenge:
            rebut_input_a = f"User Cross-Examination Directive: '{user_challenge}'\n\nOpponent argument:\n{rebut_input_a}"
            
        turn_a = None
        for event in execute_agent_turn_generator(advocate_a, "rebuttal", rebut_input_a, opponent_papers=advocate_b.last_papers):
            if event["type"] == "turn":
                turn_a = event["data"]
            yield event
        state.add_turn(turn_a)

        # B rebuts A
        yield {"type": "status", "message": "Advocate B formulating rebuttal..."}
        rebut_input_b = state.turns[-1]["text"]
        if user_challenge:
            rebut_input_b = f"User Cross-Examination Directive: '{user_challenge}'\n\nOpponent argument:\n{rebut_input_b}"
            
        turn_b = None
        for event in execute_agent_turn_generator(advocate_b, "rebuttal", rebut_input_b, opponent_papers=advocate_a.last_papers):
            if event["type"] == "turn":
                turn_b = event["data"]
            yield event
        state.add_turn(turn_b)

        # Check if the Judge decides to stop early
        yield {"type": "status", "message": "Judge evaluating latest round..."}
        eval_res = judge.evaluate_debate(question, state.turns)
        if eval_res.get("should_stop", False) and r < (max_rounds - 2):
            yield {"type": "status", "message": "Judge stopped debate early..."}
            state.verdict = (
                f"**Verdict:** {eval_res.get('winner', 'Both - Context Specific')}\n\n"
                f"**Summary:** {eval_res.get('verdict_summary', '')}\n\n"
                f"**Rationale:** {eval_res.get('deduction_rationale', '')}"
            )
            yield {"type": "verdict", "val": state.verdict}
            return

    # Final Verdict Summary
    yield {"type": "status", "message": "Judge performing final review and drawing deduction..."}
    eval_res = judge.evaluate_debate(question, state.turns)
    state.verdict = (
        f"**Verdict:** {eval_res.get('winner', 'Both - Context Specific')}\n\n"
        f"**Summary:** {eval_res.get('verdict_summary', '')}\n\n"
        f"**Rationale:** {eval_res.get('deduction_rationale', '')}"
    )
    yield {"type": "verdict", "val": state.verdict}


def run_debate(question: str, hypothesis_a: str = None, hypothesis_b: str = None, rounds: int = 2, use_mock: bool = False, use_adk_planner: bool = False):
    """
    Synchronous wrapper around run_debate_stream.
    Consumes the generator to the end, skipping pauses, and reconstructs 
    the final result dictionary for 100% backward-compatibility with tests.
    """
    stream = run_debate_stream(question, hypothesis_a, hypothesis_b, rounds, use_mock, use_adk_planner)
    
    transcript = []
    hyp_a, hyp_b, rat_a, rat_b = "", "", "", ""
    papers_a, papers_b = [], []
    verdict = ""
    adk_planner_decision = None
    
    try:
        event = next(stream)
        while True:
            if event["type"] == "hypothesis":
                if event["speaker"] == "Advocate A":
                    hyp_a = event["hypothesis"]
                    rat_a = event["rationale"]
                    papers_a = event["papers"]
                else:
                    hyp_b = event["hypothesis"]
                    rat_b = event["rationale"]
                    papers_b = event["papers"]
            elif event["type"] == "turn":
                transcript.append(event["data"])
            elif event["type"] == "verdict":
                verdict = event["val"]
            elif event["type"] == "adk_planner":
                adk_planner_decision = event["data"]
            elif event["type"] == "pause":
                event = stream.send(None)  # Auto-resume without challenge
                continue
            
            event = next(stream)
    except StopIteration:
        pass
        
    grounded_count = sum(1 for t in transcript if t["grounding"]["grounded"])
    unverified_count = sum(1 for t in transcript if not t["grounding"]["grounded"])
    
    return {
        "question": question,
        "hypothesis_a": hyp_a,
        "hypothesis_b": hyp_b,
        "hypothesis_a_rationale": rat_a,
        "hypothesis_b_rationale": rat_b,
        "transcript": transcript,
        "verdict": verdict,
        "papers_a": papers_a,
        "papers_b": papers_b,
        "adk_planner_decision": adk_planner_decision,
        "metrics": {
            "total_turns": len(transcript),
            "grounded_turns": grounded_count,
            "unverified_turns": unverified_count,
        },
    }


def run_mock_debate(question: str, rounds: int = 2):
    """
    Simulates a debate using high-quality predefined responses, yielding
    progress events to show typewriter typing and self-correction loops.
    """
    yield {"type": "status", "message": "Querying literature database on topic..."}
    time.sleep(0.5)
    
    hyp_a = "Crypt neurons project to a conserved region of the olfactory bulb shared across species."
    rat_a = "Genetic tracing in mammalian species shows target glomeruli are located in a homologous, conserved dorsomedial domain."
    hyp_b = "Crypt neurons lack a single conserved target and instead show variable, individual-specific projections."
    rat_b = "Functional mapping and individual tracing reveal significant glomerular variability and species-specific divergence."

    # Advocate A Stance
    yield {"type": "status", "message": "Advocate A researching topic & formulating Hypothesis A..."}
    time.sleep(0.5)
    yield {
        "type": "hypothesis", 
        "speaker": "Advocate A", 
        "hypothesis": hyp_a, 
        "rationale": rat_a,
        "papers": [
            {"title": "Crypt neuron projection patterns in mammalian olfactory bulb", "year": 2021, "citationCount": 142, "url": "https://api.semanticscholar.org/paper/p1", "abstract": "We mapped projection targets of crypt neurons across multiple mammalian species using genetic tracing and found a conserved glomerular target region shared across individuals."},
            {"title": "Conserved olfactory domains in teleost fish", "year": 2018, "citationCount": 89, "url": "https://api.semanticscholar.org/paper/p2", "abstract": "Analysis of crypt-like sensory cells indicates conserved axonal projection to the dorsomedial bulb region, suggesting ancient developmental hardwiring."}
        ]
    }

    # Advocate B Stance
    yield {"type": "status", "message": "Advocate B researching topic & formulating Hypothesis B..."}
    time.sleep(0.5)
    yield {
        "type": "hypothesis", 
        "speaker": "Advocate B", 
        "hypothesis": hyp_b, 
        "rationale": rat_b,
        "papers": [
            {"title": "Axonal pathfinding variability in olfactory crypt sensory cells", "year": 2023, "citationCount": 24, "url": "https://api.semanticscholar.org/paper/p3", "abstract": "Using single-cell electroporation, we tracked crypt cell axons and observed substantial variance in target glomeruli across individual mice, challenging the rigid conserved target model."},
            {"title": "Stochastic wiring of atypical olfactory sensory neurons", "year": 2022, "citationCount": 31, "url": "https://api.semanticscholar.org/paper/p4", "abstract": "Crypt-like neurons exhibit high transcriptional heterogeneity, leading to stochastic glomerular mapping rather than fixed topographic target domains."}
        ]
    }

    # Turn 1
    yield {"type": "status", "message": "Advocate A formulating opening argument..."}
    time.sleep(0.5)
    turn_1 = {
        "speaker": "Advocate A",
        "text": "Based on genetic tracing studies in mammals [1], crypt neurons project to a conserved dorsomedial glomerular target region shared across individuals. This suggests that the mapping of crypt sensory neurons is developmental and evolutionarily hardwired, as similarly observed in teleost fish [2] where projections target a homologous dorsomedial domain.",
        "grounding": {
            "grounded": True,
            "matched_papers": ["Crypt neuron projection patterns in mammalian olfactory bulb", "Conserved olfactory domains in teleost fish"],
            "warning": None
        },
        "retries": 0
    }
    yield {"type": "turn", "data": turn_1}

    # Turn 2 (Simulated self-correction warning)
    yield {"type": "status", "message": "Advocate B formulating opening argument..."}
    time.sleep(0.5)
    yield {
        "type": "grounding_retry",
        "speaker": "Advocate B",
        "retry_num": 1,
        "warning": "Partially or fully ungrounded: [1] cited but claim does not overlap sufficiently with abstract. Overlap: ['axonal']. Need 3 non-stop words."
    }
    time.sleep(1.0)
    
    turn_2 = {
        "speaker": "Advocate B",
        "text": "While my opponent asserts rigid hardwiring, single-cell axonal tracking [1] has revealed substantial variance in target glomeruli across individual mice. This variable mapping is supported by transcriptional heterogeneity data [2] demonstrating stochastic, individual-specific mapping rather than a single conserved developmental target.",
        "grounding": {
            "grounded": True,
            "matched_papers": ["Axonal pathfinding variability in olfactory crypt sensory cells", "Stochastic wiring of atypical olfactory sensory neurons"],
            "warning": None
        },
        "retries": 1
    }
    yield {"type": "turn", "data": turn_2}
    
    # Optional ADK planner simulation in mock debate
    yield {"type": "status", "message": "ADK Research Planner assessing evidence coverage..."}
    time.sleep(0.5)
    yield {
        "type": "adk_planner",
        "data": {
            "need_more_research": False,
            "reasoning": "Evidence coverage is sufficient as both advocates' opening arguments on crypt neuron projection targets are fully grounded in the retrieved literature.",
            "query_used": None
        }
    }

    # PAUSE FOR INTERMISSION
    user_challenge = yield {"type": "pause"}
    
    if user_challenge:
        yield {"type": "status", "message": f"Advocates digesting cross-examination challenge: '{user_challenge}'..."}
        time.sleep(1.0)
    else:
        user_challenge = "Zebrafish translation limits"

    # Turn 3
    yield {"type": "status", "message": "Advocate A formulating rebuttal..."}
    time.sleep(0.5)
    turn_3 = {
        "speaker": "Advocate A",
        "text": f"Addressing the user's focus on '{user_challenge}': Advocate B's reliance on single-cell tracking in mouse models [1] ignores that the target domain itself remains regionally conserved. Even if individual glomerular coordinates vary slightly due to local pathfinding noise, the projections remain confined to the homologous dorsomedial bulb sector as defined in mammalian tracing studies [1]. Stochastic local mapping does not contradict regional conservation.",
        "grounding": {
            "grounded": True,
            "matched_papers": ["Crypt neuron projection patterns in mammalian olfactory bulb", "Axonal pathfinding variability in olfactory crypt sensory cells"],
            "warning": None
        },
        "retries": 0
    }
    yield {"type": "turn", "data": turn_3}

    # Turn 4
    yield {"type": "status", "message": "Advocate B formulating rebuttal..."}
    time.sleep(0.5)
    turn_4 = {
        "speaker": "Advocate B",
        "text": "Advocate A is diluting the definition of 'conserved' to sidestep the user's concern. If mapping is stochastic and glomeruli vary across individuals [1, 2], calling it a 'conserved target region' is an oversimplification. The high transcriptional heterogeneity of crypt cells means they lack the guidance receptor consistency required for homologous targeting, resulting in individual-specific pattern layouts.",
        "grounding": {
            "grounded": True,
            "matched_papers": ["Axonal pathfinding variability in olfactory crypt sensory cells", "Stochastic wiring of atypical olfactory sensory neurons"],
            "warning": None
        },
        "retries": 0
    }
    yield {"type": "turn", "data": turn_4}

    # Judge Verdict
    yield {"type": "status", "message": "Judge weighing arguments and preparing final deduction..."}
    time.sleep(0.8)
    verdict = (
        "**Verdict:** Both - Context Specific\n\n"
        f"**Summary:** In response to user challenge '{user_challenge}', the debate confirms that crypt neurons project to a regionally homologous dorsomedial domain across species (supporting Hypothesis A), but show substantial stochastic wiring and glomerular variability at the individual level (supporting Hypothesis B).\n\n"
        "**Rationale:** Advocate A established strong evidence for evolutionary regional homology (conservation of the dorsomedial sector). However, Advocate B successfully demonstrated that individual targeting is highly variable and stochastic due to cell-to-cell transcriptional variance. The most suitable deduction is that the target region is conserved at a macro (regional) level but variable and stochastic at a micro (individual glomerulus) level. Both advocates presented fully grounded arguments backed by literature."
    )
    yield {"type": "verdict", "val": verdict}