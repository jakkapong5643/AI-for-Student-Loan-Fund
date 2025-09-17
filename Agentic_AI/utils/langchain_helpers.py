import os
import time
import random
import math
import re
from typing import Optional, List, Dict

try:
    from groq import Groq
    from groq._exceptions import APIStatusError
except Exception:
    Groq = None
    APIStatusError = Exception


_MODEL_PROMPT_BUDGET: Dict[str, int] = {
    "llama-3.3-70b-versatile": 9000,
    "openai/gpt-oss-20b": 9000,
    "qwen/qwen3-32b": 9000,
    "moonshotai/kimi-k2-instruct": 9000,
}

_PRIMARY_MODEL = "llama-3.3-70b-versatile"

_FALLBACK_MODELS: List[str] = [
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
]


def _rough_token_count(s: str) -> int:
    """Estimate token count (~4 chars per token)."""
    if not s:
        return 0
    return max(1, math.ceil(len(s) / 4))


def _normalize_ws(text: str) -> str:
    """Normalize whitespace for cleaner prompts."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_head_tail_by_tokens(text: str, token_budget: int, head_ratio: float = 0.7) -> str:
    """Truncate text to fit within token budget, keeping head and tail."""
    if _rough_token_count(text) <= token_budget:
        return text

    char_budget = max(100, int(token_budget * 4))
    if len(text) <= char_budget:
        return text

    head_chars = int(char_budget * head_ratio)
    tail_chars = char_budget - head_chars
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip() if tail_chars > 0 else ""
    return head + "\n...\n" + tail


def _pack_prompt(user_prompt: str) -> str:
    return f"{user_prompt}".strip()


def _get_groq_client() -> Groq:
    if Groq is None:
        raise RuntimeError("error")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY")
    return Groq(api_key=api_key)


def _chat_once(
    client: Groq,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_new_tokens,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


def call_llm(
    prompt: str,
    max_new_tokens: int = 2048,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model: Optional[str] = None,
    retries: int = 3,
    sleep_after_success: float = 0.25,
) -> str:

    client = _get_groq_client()
    models: List[str] = [model] if model else [_PRIMARY_MODEL, *_FALLBACK_MODELS]

    last_err: Optional[Exception] = None
    raw_user_prompt = _normalize_ws(prompt)

    for m in models:
        budget = _MODEL_PROMPT_BUDGET.get(m, 9000)
        safety_pad = max(512, int(max_new_tokens * 1.25))
        prompt_budget_tokens = max(1000, budget - safety_pad)

        working_prompt = _pack_prompt(
            _truncate_head_tail_by_tokens(raw_user_prompt, prompt_budget_tokens)
        )

        local_budget = prompt_budget_tokens

        for attempt in range(retries):
            try:
                out = _chat_once(client, m, working_prompt, temperature, top_p, max_new_tokens)
                if sleep_after_success:
                    time.sleep(sleep_after_success)
                return out

            except APIStatusError as e:
                last_err = e
                code = getattr(e, "status_code", None)
                msg = str(getattr(e, "response", "")) or str(e)

                if code == 413 or "Request too large" in msg or "tokens per minute" in msg:
                    new_budget = max(1000, int(local_budget * 0.5))
                    if new_budget >= local_budget:
                        break
                    local_budget = new_budget
                    working_prompt = _pack_prompt(
                        _truncate_head_tail_by_tokens(raw_user_prompt, local_budget)
                    )
                    time.sleep(0.2)
                    continue

                if code == 429 or "Too Many Requests" in msg or "rate_limit" in msg:
                    time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.4))
                    continue

                time.sleep(0.4 + random.uniform(0, 0.3))
                continue

            except Exception as e:
                last_err = e
                time.sleep(0.6 + random.uniform(0, 0.4))
                continue

        continue

    raise RuntimeError(f"call failed: {last_err}")
