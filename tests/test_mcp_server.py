import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import search_literature, get_paper_details, verify_claim_grounding


# ──────────────────────────────────────────────
# search_literature
# ──────────────────────────────────────────────

class TestSearchLiterature:

    def test_returns_formatted_papers(self):
        """Happy path: valid query returns formatted paper summaries."""
        mock_papers = [{"title": "Test Paper", "year": 2023, "abstract": "Abstract content", "citationCount": 5}]

        with patch("mcp_server.search_papers", return_value=mock_papers) as mock_search, \
             patch("mcp_server.format_papers_for_prompt", return_value="Formatted papers") as mock_format:

            result = search_literature("olfactory receptors", limit=5)

            assert result == "Formatted papers"
            mock_search.assert_called_once_with("olfactory receptors", limit=5)
            mock_format.assert_called_once_with(mock_papers)

    def test_empty_query_raises_value_error(self):
        """Empty query string should raise ValueError, not silently pass to API."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            search_literature("")

    def test_whitespace_only_query_raises_value_error(self):
        """Whitespace-only query should be treated as empty."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            search_literature("    ")

    def test_limit_zero_raises_value_error(self):
        """limit=0 is invalid and should raise ValueError."""
        with pytest.raises(ValueError, match="limit must be between 1 and 10"):
            search_literature("neuroscience", limit=0)

    def test_limit_too_high_raises_value_error(self):
        """limit=99 would hammer the API — should be rejected."""
        with pytest.raises(ValueError, match="limit must be between 1 and 10"):
            search_literature("neuroscience", limit=99)

    def test_no_papers_found_returns_helpful_message(self):
        """When search returns empty list, return a helpful message instead of blank."""
        with patch("mcp_server.search_papers", return_value=[]), \
             patch("mcp_server.format_papers_for_prompt", return_value=""):

            result = search_literature("xyzzy gibberish topic")
            assert "No papers found" in result
            assert "xyzzy gibberish topic" in result

    def test_api_failure_raises_runtime_error(self):
        """When search_papers raises an exception, it should be wrapped in RuntimeError."""
        with patch("mcp_server.search_papers", side_effect=ConnectionError("API down")):
            with pytest.raises(RuntimeError, match="Literature search failed"):
                search_literature("mitochondria")


# ──────────────────────────────────────────────
# get_paper_details
# ──────────────────────────────────────────────

class TestGetPaperDetails:

    def test_returns_formatted_metadata(self):
        """Happy path: valid paper_id returns full formatted metadata string."""
        mock_details = {
            "title": "Olfactory projections",
            "venue": "J. Neurosci.",
            "year": 2021,
            "authors": ["Author A", "Author B"],
            "citationCount": 15,
            "tldr": "TL;DR sentence.",
            "abstract": "Abstract description."
        }

        with patch("mcp_server.fetch_paper_details", return_value=mock_details):
            result = get_paper_details("paper_123")

            assert "Title: Olfactory projections" in result
            assert "Venue: J. Neurosci. (2021)" in result
            assert "Authors: Author A, Author B" in result
            assert "Citations: 15" in result
            assert "TL;DR: TL;DR sentence." in result
            assert "Abstract: Abstract description." in result

    def test_tldr_none_shows_not_available(self):
        """When tldr is None, output should say 'Not available'."""
        mock_details = {
            "title": "T", "venue": "V", "year": 2020,
            "authors": ["A"], "citationCount": 1, "tldr": None,
            "abstract": "Abstract."
        }
        with patch("mcp_server.fetch_paper_details", return_value=mock_details):
            result = get_paper_details("paper_456")
            assert "TL;DR: Not available" in result

    def test_empty_paper_id_raises_value_error(self):
        """Empty paper_id should raise ValueError before making any API call."""
        with pytest.raises(ValueError, match="paper_id cannot be empty"):
            get_paper_details("")

    def test_whitespace_paper_id_raises_value_error(self):
        """Whitespace-only paper_id should be treated as empty."""
        with pytest.raises(ValueError, match="paper_id cannot be empty"):
            get_paper_details("   ")

    def test_paper_not_found_raises_value_error(self):
        """When fetch returns None (not found), raise ValueError with helpful message."""
        with patch("mcp_server.fetch_paper_details", return_value=None):
            with pytest.raises(ValueError, match="No paper found for ID"):
                get_paper_details("invalid_id_xyz")

    def test_api_failure_raises_runtime_error(self):
        """When fetch_paper_details raises, it should be wrapped in RuntimeError."""
        with patch("mcp_server.fetch_paper_details", side_effect=TimeoutError("timeout")):
            with pytest.raises(RuntimeError, match="Failed to fetch paper details"):
                get_paper_details("paper_789")


# ──────────────────────────────────────────────
# verify_claim_grounding
# ──────────────────────────────────────────────

class TestVerifyClaimGrounding:

    def test_grounded_claim_returns_success_message(self):
        """When grounding check passes, result should confirm grounded status."""
        mock_result = {
            "grounded": True,
            "matched_papers": ["Paper 1"],
            "warning": None
        }
        with patch("mcp_server._check_grounding", return_value=mock_result):
            result = verify_claim_grounding(
                "As shown in [1], mitochondria drive apoptosis.",
                ["Mitochondria regulate apoptosis via cytochrome c release."]
            )

            assert "✅ GROUNDED" in result
            assert "Paper 1" in result

    def test_ungrounded_claim_returns_warning(self):
        """When grounding check fails, result should include warning details."""
        mock_result = {
            "grounded": False,
            "matched_papers": [],
            "warning": "[1] cited but claim does not overlap sufficiently with abstract."
        }
        with patch("mcp_server._check_grounding", return_value=mock_result):
            result = verify_claim_grounding(
                "As shown in [1], neurons teleport.",
                ["Mitochondria regulate calcium signaling in cells."]
            )

            assert "⚠️ NOT GROUNDED" in result
            assert "overlap" in result
            assert "Revise" in result

    def test_empty_claim_raises_value_error(self):
        """Empty claim should raise ValueError immediately."""
        with pytest.raises(ValueError, match="claim cannot be empty"):
            verify_claim_grounding("", ["Some abstract text."])

    def test_empty_abstracts_list_raises_value_error(self):
        """Empty paper_abstracts list should raise ValueError."""
        with pytest.raises(ValueError, match="paper_abstracts must contain at least one"):
            verify_claim_grounding("As shown in [1], X happens.", [])

    def test_all_blank_abstracts_raises_value_error(self):
        """List of only whitespace strings should be treated as empty."""
        with pytest.raises(ValueError, match="All provided paper_abstracts were empty"):
            verify_claim_grounding("As shown in [1], X happens.", ["   ", ""])

    def test_passes_correctly_ordered_papers_to_guardrail(self):
        """Abstracts must be passed to check_grounding in index order (abstract[0] = paper [1])."""
        mock_result = {"grounded": True, "matched_papers": ["Paper 1"], "warning": None}

        with patch("mcp_server._check_grounding", return_value=mock_result) as mock_guard:
            verify_claim_grounding(
                "As shown in [1], X is true.",
                ["First abstract.", "Second abstract."]
            )

            call_args = mock_guard.call_args
            papers_passed = call_args[0][1]  # second positional arg
            assert papers_passed[0]["abstract"] == "First abstract."
            assert papers_passed[0]["title"] == "Paper 1"
            assert papers_passed[1]["abstract"] == "Second abstract."
            assert papers_passed[1]["title"] == "Paper 2"
