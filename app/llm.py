"""Thin Groq wrapper: retries with backoff on rate limits, JSON mode, streaming."""
import json
import random
import re
import time

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from . import config

_client = None


def client():
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        _client = Groq(api_key=config.GROQ_API_KEY, max_retries=0, timeout=90)
    return _client


def _extra(model):
    # gpt-oss models spend tokens on reasoning; keep it low for speed and TPM
    return {"extra_body": {"reasoning_effort": "low"}} if model.startswith("openai/gpt-oss") else {}


def _call(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except RateLimitError as e:
            ra = e.response.headers.get("retry-after") if e.response is not None else None
            wait = float(ra) + 1 if ra else min(2 ** i * 2 + random.random(), 40)
        except APIConnectionError:
            wait = 2 ** i
        except APIStatusError as e:
            if e.status_code < 500 or i == tries - 1:
                raise
            wait = 2 ** i
        time.sleep(wait)
    raise RuntimeError("Groq is rate limiting requests. Wait a minute and try again.")


def chat(messages, model, max_tokens=1500, temperature=0.0, json_mode=False):
    kw = dict(model=model, messages=messages, temperature=temperature,
              max_tokens=max_tokens, **_extra(model))
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    r = _call(lambda: client().chat.completions.create(**kw))
    return r.choices[0].message.content or ""


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def chat_json(messages, model, max_tokens=1500):
    return parse_json(chat(messages, model, max_tokens=max_tokens, json_mode=True))


def stream(messages, model, max_tokens=2000, temperature=0.1):
    resp = _call(lambda: client().chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True, **_extra(model)))
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
