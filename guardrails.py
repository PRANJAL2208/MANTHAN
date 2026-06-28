"""
guardrails.py
-------------
This is the SECURITY / QUALITY feature for the project (Course Day 4).

The risk this protects against:
An LLM agent can easily write something like "studies show X is true"
without ever actually retrieving a real study. That's a hallucinated
citation — confident-sounding, unverifiable, and exactly the kind of
thing that makes AI agents untrustworthy.

v2 upgrade — citation-index alignment (stronger than v1):
The first version only checked whether an agent's argument mentioned
words from a paper's TITLE. That's gameable: an agent could cite
"[1] Crypt neuron projection patterns..." by name and then assert a
completely made-up finding that paper never actually contains.

This version requires agents to cite using explicit index tokens
(e.g. "[1]", "[2]" — exactly how format_papers_for_prompt() numbers
them). We then check that any sentence containing a citation token
ALSO overlaps meaningfully with that SPECIFIC paper's abstract content
— not just its title. This is a much harder check to fake, while still
being a fast, explainable, non-LLM check (no extra API call needed).
"""

import re


def check_grounding(argument_text: str, retrieved_papers: list[dict]) -> dict:
    """
    Checks whether an agent's argument cites specific retrieved papers
    by index (e.g. "[1]") AND whether the surrounding claim actually
    overlaps with THAT paper's abstract content — not just its title.

    Now stricter: if any cited index is invalid, or if any cited sentence
    fails to show meaningful overlap with the cited paper's abstract,
    the entire turn is marked as grounded=False.
    """
    if not retrieved_papers:
        return {
            "grounded": False,
            "matched_papers": [],
            "warning": "No literature was retrieved at all — argument is "
                        "unverifiable by this guardrail and should be treated "
                        "as low-confidence.",
        }

    # Define common scientific stop words to filter out
    SCIENTIFIC_STOP_WORDS = {
        "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
        "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
        "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
        "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
        "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
        "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
        "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
        "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
        "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
        "you've", "your", "yours", "yourself", "yourselves",
        # Common scientific meta-words that don't add semantic specificity
        "study", "studies", "paper", "papers", "research", "results", "result", "finding", "findings",
        "suggest", "suggests", "show", "shows", "shown", "demonstrate", "demonstrates", "demonstrated",
        "evidence", "analyse", "analysis", "significant", "significantly", "however", "therefore",
        "report", "reports", "reported", "observe", "observed", "observes", "data", "method", "methods",
        "conclude", "concluded", "conclusion", "conclusions", "author", "authors", "associated", "association"
    }

    def clean_word(w: str) -> str:
        return w.lower().strip(".,;:()[]{}'\"-+=_/*&^%$#@!~?<>|\\")

    sentences = re.split(r'(?<=[.!?])\s+', argument_text)
    matched_papers = []
    cited_but_unsupported = []
    has_citations = False

    for sentence in sentences:
        citation_refs = re.findall(r'\[(\d+)\]', sentence)
        if not citation_refs:
            continue

        has_citations = True
        for ref in citation_refs:
            idx = int(ref) - 1  # citations are 1-indexed
            if idx < 0 or idx >= len(retrieved_papers):
                cited_but_unsupported.append(f"[{ref}] — citation index out of bounds")
                continue

            paper = retrieved_papers[idx]
            
            # Clean abstract words
            abstract_words = {clean_word(w) for w in paper["abstract"].split()}
            abstract_words = {w for w in abstract_words if len(w) > 4 and w not in SCIENTIFIC_STOP_WORDS}
            
            # Clean sentence words
            sentence_words = {clean_word(w) for w in sentence.split()}
            sentence_words = {w for w in sentence_words if len(w) > 4 and w not in SCIENTIFIC_STOP_WORDS}
            
            overlap = abstract_words & sentence_words

            # Require meaningful overlap to count as supported
            if len(overlap) >= 3:
                if paper["title"] not in matched_papers:
                    matched_papers.append(paper["title"])
            else:
                cited_but_unsupported.append(
                    f"[{ref}] cited but claim does not overlap sufficiently with abstract. "
                    f"Overlap: {list(overlap)}. Need 3 non-stop words."
                )

    if not has_citations:
        return {
            "grounded": False,
            "matched_papers": [],
            "warning": "Argument contains no [n] citation markers referencing retrieved papers.",
        }

    if cited_but_unsupported:
        return {
            "grounded": False,
            "matched_papers": matched_papers,
            "warning": "Partially or fully ungrounded: " + "; ".join(cited_but_unsupported),
        }

    return {"grounded": True, "matched_papers": matched_papers, "warning": None}