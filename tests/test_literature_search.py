import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import sqlite3
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import literature_search

@pytest.fixture(autouse=True)
def mock_sleep():
    """Mock time.sleep to avoid wait time during retry tests."""
    with patch("time.sleep", return_value=None) as mock:
        yield mock

@pytest.fixture
def temp_db():
    """Create a temporary SQLite DB to test the caching logic in isolation."""
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    # Save the original CACHE_DB_PATH
    orig_path = literature_search.CACHE_DB_PATH
    literature_search.CACHE_DB_PATH = temp_db_path
    
    # Initialize the temp DB structure
    literature_search.init_cache_db()
    
    yield temp_db_path
    
    # Restore and cleanup
    literature_search.CACHE_DB_PATH = orig_path
    if os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except Exception:
            pass

def test_cache_db_init_and_table_exists(temp_db):
    """Verifies that the cache database creates the api_cache table correctly."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_cache';")
    row = cursor.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "api_cache"

def test_cache_get_and_set(temp_db):
    """Verifies that we can insert and retrieve objects from the caching DB."""
    key = "test_key"
    value = {"data": "some value"}
    
    # Ensure cache miss first
    assert literature_search.get_cached_response(key) is None
    
    # Set cached value
    literature_search.set_cached_response(key, value)
    
    # Verify cache hit
    cached = literature_search.get_cached_response(key)
    assert cached == value

def test_search_papers_cache_hit(temp_db):
    """Verifies that search_papers returns cached data and bypasses external API requests entirely."""
    query = "crypt neurons"
    cached_papers = [
        {"paperId": "1", "title": "Paper A", "abstract": "Abstract A", "year": 2020, "citationCount": 10, "url": "urlA"}
    ]
    literature_search.set_cached_response(f"search:{query}:3", cached_papers)
    
    with patch("requests.get") as mock_get:
        results = literature_search.search_papers(query, limit=3)
        assert results == cached_papers
        mock_get.assert_not_called()

def test_search_papers_cache_miss_success(temp_db):
    """Verifies search_papers queries the API on a cache miss, parses correctly, and updates cache."""
    query = "olfactory bulb"
    api_response_mock = {
        "data": [
            {
                "paperId": "abc",
                "title": "Title A",
                "abstract": "Abstract A",
                "year": 2021,
                "citationCount": 50,
                "url": "https://example.com/abc"
            }
        ]
    }
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = api_response_mock
    mock_resp.raise_for_status = MagicMock()
    
    with patch("requests.get", return_value=mock_resp) as mock_get:
        results = literature_search.search_papers(query, limit=3)
        
        # Verify result format
        assert len(results) == 1
        assert results[0]["paperId"] == "abc"
        assert results[0]["title"] == "Title A"
        assert results[0]["abstract"] == "Abstract A"
        assert results[0]["year"] == 2021
        assert results[0]["citationCount"] == 50
        assert results[0]["url"] == "https://example.com/abc"
        
        # Verify requests call
        mock_get.assert_called_once_with(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 3, "fields": "paperId,title,abstract,year,citationCount,url"},
            headers={},
            timeout=10
        )
        
        # Verify cached locally
        cached = literature_search.get_cached_response(f"search:{query}:3")
        assert cached == results

def test_search_papers_rate_limit_retry(temp_db):
    """Verifies that search_papers retries with backoff on a 429 rate limit response."""
    query = "projection mapping"
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {"data": []}
    mock_resp_200.raise_for_status = MagicMock()
    
    # 429 then 200
    with patch("requests.get", side_effect=[mock_resp_429, mock_resp_200]) as mock_get:
        results = literature_search.search_papers(query, limit=3)
        assert results == []
        assert mock_get.call_count == 2
        
        # Verify sleep was called for backoff
        import time
        time.sleep.assert_called_once_with(1)  # 2 ** 0

def test_search_papers_fails_gracefully(temp_db):
    """Verifies that search_papers returns an empty list and doesn't crash on persistent network errors."""
    query = "failing query"
    
    with patch("requests.get", side_effect=requests.RequestException("Connection timed out")) as mock_get:
        results = literature_search.search_papers(query, limit=3)
        assert results == []
        assert mock_get.call_count == 3  # Should retry up to max_retries (3)

def test_fetch_paper_details_success(temp_db):
    """Verifies fetching specific paper details by ID works, caches, and parses correctly."""
    paper_id = "paper_id_123"
    api_response_mock = {
        "title": "Specific Paper Title",
        "abstract": "Detailed abstract content",
        "year": 2022,
        "citationCount": 99,
        "venue": "Nature Neuroscience",
        "authors": [{"name": "Author 1"}, {"name": "Author 2"}],
        "tldr": {"text": "Simple TLDR summary"},
        "url": "https://example.com/paper"
    }
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = api_response_mock
    mock_resp.raise_for_status = MagicMock()
    
    with patch("requests.get", return_value=mock_resp) as mock_get:
        details = literature_search.fetch_paper_details(paper_id)
        
        assert details is not None
        assert details["title"] == "Specific Paper Title"
        assert details["abstract"] == "Detailed abstract content"
        assert details["year"] == 2022
        assert details["citationCount"] == 99
        assert details["venue"] == "Nature Neuroscience"
        assert details["authors"] == ["Author 1", "Author 2"]
        assert details["tldr"] == "Simple TLDR summary"
        assert details["url"] == "https://example.com/paper"
        
        # Verify cache hit on second call
        with patch("requests.get") as mock_get_sub:
            cached_details = literature_search.fetch_paper_details(paper_id)
            assert cached_details == details
            mock_get_sub.assert_not_called()

def test_fetch_paper_details_failure(temp_db):
    """Verifies fetch_paper_details returns None when all request retries fail."""
    paper_id = "invalid_id"
    with patch("requests.get", side_effect=requests.RequestException("Not Found")) as mock_get:
        details = literature_search.fetch_paper_details(paper_id)
        assert details is None
        assert mock_get.call_count == 3

def test_format_papers_for_prompt():
    """Verifies that literature output formatting is correct."""
    # Test empty list
    empty_res = literature_search.format_papers_for_prompt([])
    assert "No supporting literature" in empty_res
    
    # Test populated list
    papers = [
        {"title": "Title One", "year": 2019, "citationCount": 5, "abstract": "This is abstract one."},
        {"title": "Title Two", "year": 2020, "citationCount": 10, "abstract": "This is abstract two."}
    ]
    formatted = literature_search.format_papers_for_prompt(papers)
    assert "[1] Title One (2019, 5 citations)" in formatted
    assert "Abstract: This is abstract one." in formatted
    assert "[2] Title Two (2020, 10 citations)" in formatted
    assert "Abstract: This is abstract two." in formatted
