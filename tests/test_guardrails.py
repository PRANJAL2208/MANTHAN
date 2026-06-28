import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrails import check_grounding

MOCK_PAPERS = [
    {
        "paperId": "p1",
        "title": "Crypt neuron projection patterns",
        "abstract": "We mapped projection targets of olfactory cells in mice and found a conserved glomerular target region.",
        "year": 2021,
        "citationCount": 5
    },
    {
        "paperId": "p2",
        "title": "Variable olfactory targeting",
        "abstract": "Single-cell tracking shows stochastic targeting of sensory neurons leading to high individual variability in glomeruli.",
        "year": 2022,
        "citationCount": 10
    }
]

def test_check_grounding_empty_retrieved_papers():
    """Verify that checking grounding returns False and a warning when no retrieved papers exist."""
    res = check_grounding("This claims something [1]", [])
    assert res["grounded"] is False
    assert "No literature was retrieved" in res["warning"]
    assert res["matched_papers"] == []

def test_check_grounding_no_citations():
    """Verify that checking grounding returns False and a warning when argument contains no citation bracket indices."""
    res = check_grounding("This is a bold scientific assertion without any references.", MOCK_PAPERS)
    assert res["grounded"] is False
    assert "contains no [n] citation markers" in res["warning"]
    assert res["matched_papers"] == []

def test_check_grounding_invalid_index():
    """Verify that checking grounding returns False and flags out-of-bounds citation indices."""
    res = check_grounding("This is based on [3] which does not exist.", MOCK_PAPERS)
    assert res["grounded"] is False
    assert "[3] — citation index out of bounds" in res["warning"]
    assert res["matched_papers"] == []

def test_check_grounding_successful_overlap():
    """Verify that checking grounding returns True when sentence has sufficient overlap (>= 3 words) with the cited abstract."""
    # Sentences with overlap: "mapped", "projection", "targets", "olfactory", "conserved", "glomerular"
    argument = "According to [1], we mapped the projection targets of olfactory cells, revealing a conserved glomerular region."
    res = check_grounding(argument, MOCK_PAPERS)
    assert res["grounded"] is True
    assert res["warning"] is None
    assert "Crypt neuron projection patterns" in res["matched_papers"]

def test_check_grounding_insufficient_overlap():
    """Verify that checking grounding returns False when cited claims lack sufficient overlap (fewer than 3 non-stop words)."""
    # Only "cells" overlaps
    argument = "According to [1], we are testing cells in the brain."
    res = check_grounding(argument, MOCK_PAPERS)
    assert res["grounded"] is False
    assert "overlap sufficiently" in res["warning"]
    assert "Overlap" in res["warning"]
    assert res["matched_papers"] == []

def test_check_grounding_partially_unverified_multiple_citations():
    """Verify that when multiple papers are cited, if one fails the overlap check, the entire argument is marked ungrounded."""
    # [1] has high overlap, but [2] does not overlap with the cited sentence
    argument = (
        "We mapped projection targets showing a conserved glomerular region [1]. "
        "Also, we think that teleost fish have olfactory cells [2]."
    )
    res = check_grounding(argument, MOCK_PAPERS)
    assert res["grounded"] is False
    assert "[2] cited but claim does not overlap sufficiently" in res["warning"]
    assert "Crypt neuron projection patterns" in res["matched_papers"]  # still matched paper 1

def test_check_grounding_scientific_stop_words_filtering():
    """Verify that stop words (common and scientific meta-words) are not counted towards semantic overlap."""
    # Citing [1], but sentence only shares stop words: "study", "shows", "significant", "associated"
    # Even if they match, they are filtered out.
    argument = "A study [1] shows significant association with observed methods."
    res = check_grounding(argument, MOCK_PAPERS)
    assert res["grounded"] is False
    assert "Overlap: []" in res["warning"]  # zero overlap since they are stop words or too short

def test_check_grounding_word_cleaning():
    """Verify that punctuation and casing are cleaned properly before checking overlap."""
    argument = "Based on [2], we found stochastic TARGETING and individual VARIABILITY!!"
    # abstract words: "stochastic", "targeting", "individual", "variability"
    # should match despite punctuation and casing
    res = check_grounding(argument, MOCK_PAPERS)
    assert res["grounded"] is True
    assert res["warning"] is None
    assert "Variable olfactory targeting" in res["matched_papers"]
