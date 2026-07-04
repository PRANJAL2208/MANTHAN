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

load_dotenv(override=True)  # reads the .env file in the project folder and loads it into os.environ

def get_secret(key_name, default=None):
    """Try to get a secret from Streamlit secrets first, then fallback to os.environ."""
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.environ.get(key_name, default)

def get_available_provider():
    # If explicitly set, respect it
    explicit_provider = get_secret("LLM_PROVIDER")
    if explicit_provider in ["groq", "gemini", "anthropic"]:
        return explicit_provider
        
    # Auto-detect based on available keys
    if get_secret("GROQ_API_KEY"):
        return "groq"
    if get_secret("GEMINI_API_KEY"):
        return "gemini"
    if get_secret("ANTHROPIC_API_KEY"):
        return "anthropic"
        
    # Default to groq so the error messages point them in the right direction
    return "groq"

# Determine the provider dynamically at runtime
LLM_PROVIDER = get_available_provider()


import time

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 3000) -> str:
    """
    Sends a system instruction + a user message to whichever LLM provider
    is configured, and returns the plain text response.

    Retry policy:
    - Per-minute rate limit (429 RPM): retries up to 5 times with backoff
    - Daily quota exhausted (RESOURCE_EXHAUSTED + limit:0): fails fast with
      a clear message. No point waiting 7x60s for a daily limit to reset.
    - Server errors (503): retries up to 3 times
    """
    import re
    
    current_provider = get_available_provider()
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if current_provider == "gemini":
                return _call_gemini(system_prompt, user_prompt, max_tokens)
            elif current_provider == "anthropic":
                return _call_anthropic(system_prompt, user_prompt, max_tokens)
            elif current_provider == "groq":
                return _call_groq(system_prompt, user_prompt, max_tokens)
            else:
                raise ValueError(f"Unknown LLM provider: {current_provider}. Choose: gemini, anthropic, groq")
        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()

            # Detect daily quota exhaustion: only fail fast when the per-day quota
            # is the SOLE violation and per-minute limits are not also listed.
            # (Google often shows per-day alongside per-minute on RPM hits, which
            # resets in <60s and should be retried normally.)
            is_daily_exhausted = (
                "resource_exhausted" in err_lower
                and "PerDay" in err_str
                and "PerMinute" not in err_str  # if per-minute also listed, it's just RPM throttle
            )
            if is_daily_exhausted:
                raise RuntimeError(
                    "DAILY QUOTA EXHAUSTED: Your Gemini API key has reached its daily "
                    "free-tier limit. Options:\n"
                    "  1. Enable billing at https://console.cloud.google.com/billing\n"
                    "  2. Use a fresh API key from a different Google account\n"
                    "  3. Wait until tomorrow (quota resets ~midnight Pacific time)"
                ) from e

            is_retryable = any(msg in err_lower for msg in [
                "503", "429", "unavailable", "rate limit", "demand",
                "resource_exhausted", "resourceexhausted"
            ])
            if is_retryable and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                delay_match = re.search(r"Please retry in (\d+\.?\d*)s", err_str, re.IGNORECASE)
                if delay_match:
                    wait_time = float(delay_match.group(1)) + 1.0
                else:
                    delay_dict_match = re.search(
                        r"['\"]retryDelay['\"]\s*:\s*['\"](\d+)s['\"]", err_str, re.IGNORECASE
                    )
                    if delay_dict_match:
                        wait_time = float(delay_dict_match.group(1)) + 1.0
                # Cap wait at 15s — if it's a per-minute limit it resets in 60s max
                wait_time = min(wait_time, 15.0)
                print(f"[llm_client] Retrying in {wait_time:.1f}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("LLM call failed after max retries.")


def _call_groq(system_prompt, user_prompt, max_tokens):
    """
    Groq: ultra-fast LPU inference. Free tier = 14,400 req/day, no credit card needed.
    Model: llama-3.3-70b-versatile — comparable quality to Gemini 2.0 Flash.
    Switch back to Gemini anytime: set LLM_PROVIDER=gemini in .env
    """
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "groq package not installed. Run: venv\\Scripts\\pip install groq"
        )

    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file or Streamlit Secrets.")

    api_key = api_key.strip().strip('"').strip("'")
    model_name = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

    client = Groq(api_key=api_key)
    
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] [Groq] Calling {model_name}...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    duration = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] [Groq] Responded in {duration:.1f}s")
    
    return response.choices[0].message.content.strip()


def _call_gemini(system_prompt, user_prompt, max_tokens):
    from google import genai
    from google.genai import types

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file or Streamlit Secrets.")

    api_key = api_key.strip().strip('"').strip("'")
    model_name = get_secret("GEMINI_MODEL", "gemini-2.5-flash")

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

    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your .env file or Streamlit Secrets.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()