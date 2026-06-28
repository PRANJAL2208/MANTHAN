import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import call_llm

@pytest.fixture(autouse=True)
def mock_env_and_sleep():
    """Mock time.sleep to speed up retry tests, and reset environment variables."""
    with patch("time.sleep", return_value=None) as mock_sleep, \
         patch.dict(os.environ, {"GEMINI_API_KEY": "fake_gemini_key", "ANTHROPIC_API_KEY": "fake_anthropic_key"}):
        yield mock_sleep

def test_call_llm_gemini_success():
    """Tests a successful call to the Gemini provider."""
    # We patch _call_gemini directly to simplify provider switching tests
    with patch("llm_client._call_gemini", return_value="Gemini response") as mock_gemini, \
         patch("llm_client.LLM_PROVIDER", "gemini"):
        
        response = call_llm("sys prompt", "user prompt")
        assert response == "Gemini response"
        mock_gemini.assert_called_once_with("sys prompt", "user prompt", 3000)

def test_call_llm_anthropic_success():
    """Tests a successful call to the Anthropic provider."""
    with patch("llm_client._call_anthropic", return_value="Anthropic response") as mock_anthropic, \
         patch("llm_client.LLM_PROVIDER", "anthropic"):
        
        response = call_llm("sys prompt", "user prompt")
        assert response == "Anthropic response"
        mock_anthropic.assert_called_once_with("sys prompt", "user prompt", 3000)

def test_call_llm_retry_on_rate_limit():
    """Tests that the client retries on rate limit (429) errors."""
    call_count = 0

    def fail_then_succeed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("API Error: 429 Rate Limit Exceeded")
        return "Success on attempt 3"

    with patch("llm_client._call_gemini", side_effect=fail_then_succeed) as mock_gemini, \
         patch("llm_client.LLM_PROVIDER", "gemini"):
        
        response = call_llm("sys prompt", "user prompt")
        assert response == "Success on attempt 3"
        assert call_count == 3
        # Assert that our mock_env_and_sleep fixture's mock_sleep was called twice
        import time
        assert time.sleep.call_count == 2

def test_call_llm_raises_error_after_max_retries():
    """Tests that call_llm raises an exception when the error persists after 5 attempts."""
    with patch("llm_client._call_gemini", side_effect=Exception("Error code 503 Service Unavailable")) as mock_gemini, \
         patch("llm_client.LLM_PROVIDER", "gemini"):
        
        with pytest.raises(Exception) as exc_info:
            call_llm("sys prompt", "user prompt")
        
        assert "503" in str(exc_info.value)
        # Should call 5 times (attempts 0, 1, 2, 3, 4)
        assert mock_gemini.call_count == 5

def test_call_llm_non_retryable_error_raised_immediately():
    """Tests that non-retryable errors (e.g. invalid arguments) are raised immediately without retries."""
    with patch("llm_client._call_gemini", side_effect=ValueError("Incorrect arguments passed")) as mock_gemini, \
         patch("llm_client.LLM_PROVIDER", "gemini"):
        
        with pytest.raises(ValueError) as exc_info:
            call_llm("sys prompt", "user prompt")
            
        assert "Incorrect arguments" in str(exc_info.value)
        assert mock_gemini.call_count == 1  # No retries!

def test_call_llm_unknown_provider():
    """Tests that an unknown provider raises ValueError immediately."""
    with patch("llm_client.LLM_PROVIDER", "unknown_provider"):
        with pytest.raises(ValueError) as exc_info:
            call_llm("sys prompt", "user prompt")
        assert "Unknown LLM_PROVIDER" in str(exc_info.value)

def test_call_gemini_client_interaction():
    """Tests the interaction with the google.genai Client."""
    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = " Gemini generated content "
    mock_client_instance.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client_instance) as mock_client_cls, \
         patch("llm_client.LLM_PROVIDER", "gemini"):
        
        from llm_client import _call_gemini
        text = _call_gemini("sys instruct", "user message", 1000)
        
        assert text == "Gemini generated content"
        mock_client_cls.assert_called_once_with(api_key="fake_gemini_key")
        mock_client_instance.models.generate_content.assert_called_once()

def test_call_anthropic_client_interaction():
    """Tests the interaction with the anthropic Client."""
    mock_client_instance = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=" Anthropic generated content ")]
    mock_client_instance.messages.create.return_value = mock_message

    with patch("anthropic.Anthropic", return_value=mock_client_instance) as mock_client_cls, \
         patch("llm_client.LLM_PROVIDER", "anthropic"):
        
        from llm_client import _call_anthropic
        text = _call_anthropic("sys instruct", "user message", 2000)
        
        assert text == "Anthropic generated content"
        mock_client_cls.assert_called_once_with(api_key="fake_anthropic_key")
        mock_client_instance.messages.create.assert_called_once()
