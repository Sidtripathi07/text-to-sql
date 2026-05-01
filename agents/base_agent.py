"""
agents/base_agent.py  — v2 (FIXED)

Root-cause fixes applied:
  FIX-1: TPM-aware throttling.  Old code only checked RPM.  The eval showed 0-token
          rows from sample ~115 onwards — that is a TPM exhaustion.  Now we track
          a rolling 60-second token bucket per model and sleep BEFORE the call if
          the estimated prompt+completion would exceed the limit.

  FIX-2: Exponential back-off on 429/5xx.  Old code broke on first API error.
          Now we retry up to 4 times with 5 / 10 / 20 / 40 s waits.

  FIX-3: estimated_tokens passed into enforce so the budget check is realistic.
"""

import time
import re
import json
from abc import ABC, abstractmethod
from typing import Optional
from groq import Groq
from config import SMALL_MODEL, LARGE_MODEL, LIMITS


class AgentTrace:
    def __init__(self, agent_name: str, model: str):
        self.agent_name    = agent_name
        self.model         = model
        self.prompt        = ""
        self.raw_response  = ""
        self.parsed_output: dict | str = {}
        self.tokens_in     = 0
        self.tokens_out    = 0
        self.latency_ms    = 0
        self.error         = None

    def to_dict(self) -> dict:
        return {
            "agent"        : self.agent_name,
            "model"        : self.model,
            "tokens_in"    : self.tokens_in,
            "tokens_out"   : self.tokens_out,
            "latency_ms"   : self.latency_ms,
            "error"        : self.error,
            "prompt"       : self.prompt,
            "raw_response" : self.raw_response,
            "parsed"       : self.parsed_output,
        }


class BaseAgent(ABC):
    _groq_client : Optional[Groq] = None
    _api_key_used: str = ""

    # Sliding-window call timestamps and (timestamp, token_count) tuples — class-level
    _call_times  : dict[str, list] = {SMALL_MODEL: [], LARGE_MODEL: []}
    _token_window: dict[str, list] = {SMALL_MODEL: [], LARGE_MODEL: []}

    def __init__(self, name: str, model: str, api_key: str,
                 temperature: float = 0.0, max_tokens: int = 900):
        self.name        = name
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.api_key     = api_key
        if BaseAgent._api_key_used != api_key or BaseAgent._groq_client is None:
            BaseAgent._groq_client  = Groq(api_key=api_key)
            BaseAgent._api_key_used = api_key

    # ── Window management ────────────────────────────────────────────────────

    def _prune(self):
        now = time.time()
        BaseAgent._call_times[self.model] = [
            t for t in BaseAgent._call_times[self.model] if now - t < 60]
        BaseAgent._token_window[self.model] = [
            (t, k) for t, k in BaseAgent._token_window[self.model] if now - t < 60]

    def _tpm_used(self) -> int:
        return sum(k for _, k in BaseAgent._token_window[self.model])

    # ── FIX-1: TPM + RPM enforcement ─────────────────────────────────────────

    def _enforce_rate_limit(self, estimated_tokens: int):
        rpm_limit = LIMITS[self.model]["rpm"]
        tpm_limit = LIMITS[self.model]["tpm"]

        while True:
            self._prune()
            rpm_ok = len(BaseAgent._call_times[self.model]) < rpm_limit
            tpm_ok = (self._tpm_used() + estimated_tokens) <= tpm_limit

            if rpm_ok and tpm_ok:
                break

            # Sleep until the oldest entry leaves the 60-second window
            candidates = (
                BaseAgent._call_times[self.model] +
                [t for t, _ in BaseAgent._token_window[self.model]]
            )
            if candidates:
                oldest    = min(candidates)
                sleep_for = max(60.0 - (time.time() - oldest) + 1.0, 1.0)
            else:
                sleep_for = 2.0
            time.sleep(sleep_for)

        # Record upcoming call
        BaseAgent._call_times[self.model].append(time.time())

    def _record_tokens(self, n: int):
        BaseAgent._token_window[self.model].append((time.time(), n))

    # ── FIX-2: API call with exponential back-off ─────────────────────────────

    def call(self, prompt: str, system: str = "",
             estimated_tokens: int = 600) -> AgentTrace:
        trace        = AgentTrace(self.name, self.model)
        trace.prompt = prompt

        self._enforce_rate_limit(estimated_tokens)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        max_retries = 4
        for attempt in range(max_retries):
            start = time.time()
            try:
                resp = BaseAgent._groq_client.chat.completions.create(
                    model       = self.model,
                    messages    = messages,
                    temperature = self.temperature,
                    max_tokens  = self.max_tokens,
                )
                trace.raw_response = resp.choices[0].message.content or ""
                trace.tokens_in    = resp.usage.prompt_tokens
                trace.tokens_out   = resp.usage.completion_tokens
                trace.latency_ms   = int((time.time() - start) * 1000)
                trace.error        = None
                self._record_tokens(trace.tokens_in + trace.tokens_out)
                break
            except Exception as e:
                err = str(e)
                trace.error      = err
                trace.latency_ms = int((time.time() - start) * 1000)
                if any(x in err for x in ["429", "rate_limit", "503", "502",
                                           "500", "overloaded"]):
                    wait = (2 ** attempt) * 5
                    time.sleep(wait)
                    self._enforce_rate_limit(estimated_tokens)
                else:
                    break  # non-retriable

        return trace

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def extract_json(text: str) -> Optional[dict]:
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None

    @staticmethod
    def extract_sql(text: str) -> str:
        text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "").strip()
        m = re.search(
            r"((?:WITH|SELECT|INSERT|UPDATE|DELETE)[\s\S]+?)(?:;?\s*$)",
            text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().rstrip(";")
        return text.strip().rstrip(";")

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build_prompt(self, context: dict) -> str: ...

    @abstractmethod
    def parse_response(self, raw: str) -> dict | str: ...

    def run(self, context: dict, system: str = "") -> AgentTrace:
        prompt    = self.build_prompt(context)
        estimated = len(prompt) // 4 + self.max_tokens   # ~1 tok per 4 chars
        trace     = self.call(prompt, system=system, estimated_tokens=estimated)
        if not trace.error:
            trace.parsed_output = self.parse_response(trace.raw_response)
        return trace