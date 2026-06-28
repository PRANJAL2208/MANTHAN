"""
llm_client.py
-------------
A single, simple function call_llm() that every agent uses to "think."

Why this file exists:
Instead of hardcoding one AI provider everywhere in the code, every agent
calls this one function. If you want to switch from Gemini to Claude to
OpenAI, you change ONE place, not five.

This matters for the competition rubric too: judges want to see clean,
swappable model usage, not a provider hardcoded into every agent's logic.

Default provider is Gemini (since this is a Google course), with Anthropic
Claude as a fallback/alternative — set LLM_PROVIDER env var to switch.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the project folder and loads it into os.environ

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini" or "anthropic"


import time

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 3000) -> str:
    """
    Sends a system instruction + a user message to whichever LLM provider
    is configured, and returns the plain text response.
    Retries on rate-limit, spikes in demand, or temporary unavailable errors (429/503).
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if LLM_PROVIDER == "gemini":
                return _call_gemini(system_prompt, user_prompt, max_tokens)
            elif LLM_PROVIDER == "anthropic":
                return _call_anthropic(system_prompt, user_prompt, max_tokens)
            else:
                raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(msg in err_str for msg in ["503", "429", "unavailable", "rate limit", "demand", "resourceexhausted"])
            if is_retryable and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 2s, 4s, 6s, 8s backoff
                print(f"[llm_client] LLM Call failed: {e}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("LLM Call failed after max retries.")


def _call_gemini(system_prompt, user_prompt, max_tokens):
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")

    api_key = api_key.strip().strip('"').strip("'")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text.strip()


def _call_anthropic(system_prompt, user_prompt, max_tokens):
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your .env file.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()