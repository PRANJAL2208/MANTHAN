"""
agents.py
---------
Defines the three "characters" in the debate:

  - AdvocateAgent: argues FOR one specific hypothesis, using real
    retrieved literature, and rebuts the other advocate's weak points.
  - JudgeAgent: reads the full transcript and produces a structured,
    neutral summary of where the real agreement/disagreement lies.

This is the MULTI-AGENT part of the project: each agent
has a distinct role, distinct instructions, and they interact with each
other's output.
"""

import json
import re
from llm_client import call_llm
from literature_search import search_papers, format_papers_for_prompt
from guardrails import check_grounding


def parse_json_safely(text: str) -> dict:
    """Helper to extract and parse JSON from LLM output robustly."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try using regex to find JSON structures
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


class AdvocateAgent:
    def __init__(self, name: str, hypothesis: str = ""):
        self.name = name
        self.hypothesis = hypothesis
        self.last_papers = []  # remembers what evidence IT retrieved

    def propose_hypothesis(self, topic: str) -> dict:
        """Researches the topic and proposes a hypothesis with rationale."""
        papers = search_papers(topic, limit=5)
        self.last_papers = papers
        evidence_block = format_papers_for_prompt(papers)

        system_prompt = (
            f"You are {self.name}, a rigorous, brutally honest, and purely fact-based scientific researcher. "
            "Your goal is to analyze the literature and propose a strong, clear scientific hypothesis on the given topic.\n\n"
            "Rules you must follow:\n"
            "1. Avoid friendly, polite, or vague filler. Be direct and analytical.\n"
            "2. Propose a specific, testable hypothesis backed by the provided literature."
        )

        user_prompt = (
            f"The topic of study is: \"{topic}\"\n\n"
            f"Here is the literature retrieved for this topic:\n"
            f"{evidence_block}\n\n"
            "Propose a clear, scientific hypothesis and a brief, fact-based rationale (100 words max) supporting it. "
            "Your output must be in JSON format with two keys: 'hypothesis' (a single concise sentence) and 'rationale' (the explanation)."
        )

        text = call_llm(system_prompt, user_prompt, max_tokens=1000)
        result = parse_json_safely(text)
        self.hypothesis = result.get("hypothesis", f"Conserved target regions are supported on topic {topic}.")
        return result

    def oppose_hypothesis(self, topic: str, opponent_hypothesis: str) -> dict:
        """Researches the topic and proposes an opposing/competing hypothesis to the opponent's."""
        papers = search_papers(topic, limit=5)
        self.last_papers = papers
        evidence_block = format_papers_for_prompt(papers)

        system_prompt = (
            f"You are {self.name}, an adversarial, brutally honest, and purely fact-based scientific investigator. "
            "Your opponent has proposed a hypothesis on the topic. Your goal is to propose an alternative, opposing, "
            "or competing hypothesis that is distinct and conflicts with their stance, backed by literature.\n\n"
            "Rules you must follow:\n"
            "1. Do NOT agree with the opponent. Your hypothesis must represent a competing model, mechanism, or interpretation.\n"
            "2. Be direct, dense, and fact-based."
        )

        user_prompt = (
            f"The topic of study is: \"{topic}\"\n\n"
            f"Your opponent proposed Hypothesis A:\n\"{opponent_hypothesis}\"\n\n"
            f"Here is the literature retrieved for this topic:\n"
            f"{evidence_block}\n\n"
            "Propose an opposing or competing Hypothesis B and a brief, fact-based rationale (100 words max) supporting it. "
            "Make sure Hypothesis B represents a contrasting view to Hypothesis A.\n"
            "Your output must be in JSON format with two keys: 'hypothesis' (a single concise sentence) and 'rationale' (the explanation)."
        )

        text = call_llm(system_prompt, user_prompt, max_tokens=1000)
        result = parse_json_safely(text)
        self.hypothesis = result.get("hypothesis", f"Variable and context-specific projections are supported on topic {topic}.")
        return result

    def _system_prompt(self) -> str:
        return (
            f"You are {self.name}, a rigorous scientific advocate. "
            f"You argue specifically for this hypothesis:\n"
            f"\"{self.hypothesis}\"\n\n"
            "Rules you must follow:\n"
            "1. Only use the evidence provided to you in this turn. Do not "
            "invent studies, statistics, or quotes. Be brutally honest and fact-based.\n"
            "2. CRITICAL: whenever you state a claim that comes from a "
            "retrieved paper, you MUST cite it using its bracket index "
            "exactly as given to you, e.g. \"[1]\" or \"[2]\" — and the "
            "claim must actually match what that paper's abstract says.\n"
            "3. If asked to rebut another advocate, you must identify a "
            "methodological or logical weak point in their argument and rebut "
            "it directly with evidence. Challenge their sample sizes, species models, "
            "or citation accuracy.\n"
            "4. NEVER use polite filler phrases such as 'you are correct', 'that is a good point', "
            "'I agree with', or 'as my opponent rightly pointed out'. Be adversarial and direct.\n"
            "5. Be concise and dense: 150-250 words per turn."
        )

    def open_argument(self, question: str) -> dict:
        """First turn: retrieve evidence, make the opening case."""
        # If we don't have papers yet, fetch them (backward compatibility/fallback)
        if not self.last_papers:
            self.last_papers = search_papers(f"{question} {self.hypothesis}", limit=3)
            
        evidence_block = format_papers_for_prompt(self.last_papers)

        user_prompt = (
            f"The open scientific question/topic is: \"{question}\"\n\n"
            f"Here is the literature retrieved for your position:\n"
            f"{evidence_block}\n\n"
            "Write your opening argument for your hypothesis, using this "
            "evidence directly (refer to paper titles/findings and cite using bracket indices like [1])."
        )
        text = call_llm(self._system_prompt(), user_prompt)
        grounding = check_grounding(text, self.last_papers)
        return {"speaker": self.name, "text": text, "grounding": grounding}

    def rebuttal(self, opponent_text: str, opponent_papers: list[dict] = None) -> dict:
        """Later turns: attack a specific weak point in the opponent's last argument."""
        own_evidence_block = format_papers_for_prompt(self.last_papers)
        
        opponent_evidence_block = ""
        if opponent_papers:
            opponent_evidence_block = "\nHere is the literature cited/retrieved by your opponent (for comparison):\n"
            for i, p in enumerate(opponent_papers, 1):
                opponent_evidence_block += (
                    f"[Opponent-{i}] {p['title']} ({p['year']})\n"
                    f"    Abstract: {p['abstract'][:1500]}\n"
                )
        else:
            opponent_evidence_block = "\nNo literature was retrieved by your opponent."

        user_prompt = (
            f"Your opponent just argued:\n\"\"\"\n{opponent_text}\n\"\"\"\n\n"
            f"Here is your evidence again, with the SAME citation numbers as before:\n"
            f"{own_evidence_block}\n"
            f"{opponent_evidence_block}\n\n"
            "Identify ONE specific weak point or citation mismatch in their argument and rebut it directly. "
            "Compare their cited abstract details with your evidence. Do not agree with them, and do not use friendly filler. "
            "You may cite your papers using their index numbers (e.g. [1])."
        )
        text = call_llm(self._system_prompt(), user_prompt)
        grounding = check_grounding(text, self.last_papers)
        return {"speaker": self.name, "text": text, "grounding": grounding}


class JudgeAgent:
    def evaluate_debate(self, question: str, transcript: list[dict]) -> dict:
        """
        Evaluates the current debate transcript, decides whether it is ready to stop,
        and provides a conclusion or intermediate verdict.
        """
        # Format the transcript with grounding information
        formatted_transcript = []
        for turn in transcript:
            grounded_status = "Grounded" if turn["grounding"]["grounded"] else "Ungrounded"
            warning_info = f" (Warning: {turn['grounding']['warning']})" if turn["grounding"]["warning"] else ""
            formatted_transcript.append(
                f"{turn['speaker']} ({grounded_status}{warning_info}):\n{turn['text']}"
            )
        transcript_text = "\n\n".join(formatted_transcript)

        system_prompt = (
            "You are a decisive, neutral scientific judge. Your job is to analyze the debate transcript "
            "between two advocates and decide whether the debate has reached a point where a conclusion can be made, "
            "or whether it needs to continue.\n\n"
            "Rules for stopping:\n"
            "1. If the arguments are exhausted or a clear evidentiary winner has emerged, set 'should_stop' to true.\n"
            "2. If new arguments or evidence are still actively being contested, set 'should_stop' to false.\n\n"
            "Rules for deduction (when 'should_stop' is true):\n"
            "1. You must avoid a lazy 'both are correct' or 'we need more research' cop-out.\n"
            "2. Evaluate the grounding status of each agent. If one agent frequently made ungrounded claims, penalize their credibility.\n"
            "3. Declare a definitive winner ('winner': 'Advocate A' or 'Advocate B') and provide a clear, evidence-based deduction explaining why their hypothesis is best supported.\n"
            "4. A joint or split conclusion ('winner': 'Both - Context Specific') is ONLY allowed if the evidence rigorously and explicitly supports both in different conditions (e.g. different species or sensory organs), backed by a detailed explanation.\n\n"
            "You MUST output your response in JSON format matching this schema:\n"
            "{\n"
            "  \"should_stop\": true,\n"
            "  \"verdict_summary\": \"brief summary of the debate outcome\",\n"
            "  \"winner\": \"Advocate A\" / \"Advocate B\" / \"Both - Context Specific\",\n"
            "  \"deduction_rationale\": \"detailed, fact-based scientific analysis explaining the decision\"\n"
            "}"
        )

        user_prompt = (
            f"Original scientific question/topic: \"{question}\"\n\n"
            f"Current debate transcript (including grounding audits):\n"
            f"{transcript_text}\n\n"
            "Generate your evaluation JSON now. Output raw JSON only."
        )

        text = call_llm(system_prompt, user_prompt, max_tokens=1000)
        
        # Parse response JSON
        eval_result = parse_json_safely(text)
        if not eval_result:
            # Fallback if parsing fails
            eval_result = {
                "should_stop": True,
                "verdict_summary": "Failed to parse Judge JSON.",
                "winner": "Both - Context Specific",
                "deduction_rationale": text
            }
        return eval_result

    def summarize(self, question: str, transcript: list[dict]) -> str:
        """For backward compatibility with existing tests."""
        eval_res = self.evaluate_debate(question, transcript)
        return (
            f"**Verdict:** {eval_res.get('winner', 'Both - Context Specific')}\n\n"
            f"**Summary:** {eval_res.get('verdict_summary', 'Evaluation completed.')}\n\n"
            f"**Rationale:** {eval_res.get('deduction_rationale', '')}"
        )