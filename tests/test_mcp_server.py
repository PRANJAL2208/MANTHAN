import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import search_literature, get_paper_details

def test_search_literature():
    """Verify that search_literature calls search_papers and format_papers_for_prompt."""
    mock_papers = [{"title": "Test Paper", "year": 2023, "abstract": "Abstract content", "citationCount": 5}]
    
    with patch("mcp_server.search_papers", return_value=mock_papers) as mock_search, \
         patch("mcp_server.format_papers_for_prompt", return_value="Formatted papers") as mock_format:
         
        res = search_literature("olfactory receptors", limit=5)
        
        assert res == "Formatted papers"
        mock_search.assert_called_once_with("olfactory receptors", limit=5)
        mock_format.assert_called_once_with(mock_papers)

def test_get_paper_details_success():
    """Verify that get_paper_details returns correctly formatted string on successful fetch."""
    mock_details = {
        "title": "Olfactory projections",
        "venue": "J. Neurosci.",
        "year": 2021,
        "authors": ["Author A", "Author B"],
        "citationCount": 15,
        "tldr": "TL;DR sentence.",
        "abstract": "Abstract description."
    }
    
    with patch("mcp_server.fetch_paper_details", return_value=mock_details) as mock_fetch:
        res = get_paper_details("paper_123")
        
        assert "Title: Olfactory projections" in res
        assert "Venue: J. Neurosci. (2021)" in res
        assert "Authors: Author A, Author B" in res
        assert "Citations: 15" in res
        assert "TL;DR: TL;DR sentence." in res
        assert "Abstract: Abstract description." in res
        mock_fetch.assert_called_once_with("paper_123")

def test_get_paper_details_failure():
    """Verify that get_paper_details handles failure fetching gracefully."""
    with patch("mcp_server.fetch_paper_details", return_value=None) as mock_fetch:
        res = get_paper_details("invalid_id")
        
        assert "Could not retrieve details for paper ID: invalid_id" in res
        mock_fetch.assert_called_once_with("invalid_id")
