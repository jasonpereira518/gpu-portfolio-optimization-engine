"""Stretch: plain-English rebalance explanations from a NIM-served model.

Scope is deliberately small. This is an explanation layer over a real
optimization engine, not the point of the project. It takes structured facts
that the optimizer already computed — which constraints bound, which positions
moved, where risk concentrated — and asks a local model to phrase them.

Design constraint that matters: **the model is never asked to compute
anything.** Every number in the prompt is produced by the optimizer and passed
in as text. An LLM asked to derive risk contributions would produce fluent
arithmetic errors, and the resulting explanation would be worse than none.

Run a NIM locally (single GPU is enough for Nemotron Nano):

    docker run --gpus all -p 8000:8000 \\
        -e NGC_API_KEY=$NGC_API_KEY \\
        nvcr.io/nim/nvidia/nemotron-3-nano-instruct:latest

The client below speaks the OpenAI-compatible API that NIM exposes, so it also
works against any other OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_ENDPOINT = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-instruct"

SYSTEM_PROMPT = """You explain portfolio rebalances to an investment committee.

You will be given structured facts computed by a mean-variance optimizer. Use
only those facts. Do not compute, estimate, or infer any number that is not
given to you. Do not give investment advice or make forecasts.

Write 3-5 sentences covering: what changed and why, which constraints were
binding, and where the portfolio's risk is concentrated. Plain language, no
jargon beyond what an investment committee uses."""


@dataclass
class RebalanceFacts:
    """Everything the explanation is allowed to reference."""

    date: str
    n_assets: int
    turnover: float
    transaction_cost: float
    top_buys: list[tuple[str, float]] = field(default_factory=list)
    top_sells: list[tuple[str, float]] = field(default_factory=list)
    binding_constraints: list[str] = field(default_factory=list)
    expected_return: float = 0.0
    expected_vol: float = 0.0
    top_risk_contributors: list[tuple[str, float]] = field(default_factory=list)

    def to_prompt(self) -> str:
        def pct_pairs(pairs):
            return ", ".join(f"{name} {value:+.2%}" for name, value in pairs) or "none"

        return json.dumps(
            {
                "rebalance_date": self.date,
                "universe_size": self.n_assets,
                "turnover": f"{self.turnover:.2%}",
                "transaction_cost": f"{self.transaction_cost:.4%} of portfolio value",
                "expected_annual_return": f"{self.expected_return:.2%}",
                "expected_annual_volatility": f"{self.expected_vol:.2%}",
                "largest_increases": pct_pairs(self.top_buys),
                "largest_decreases": pct_pairs(self.top_sells),
                "binding_constraints": self.binding_constraints or ["none"],
                "largest_risk_contributors": ", ".join(
                    f"{name} {share:.1%} of portfolio variance"
                    for name, share in self.top_risk_contributors
                )
                or "none",
            },
            indent=2,
        )


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each position's share of total portfolio variance.

    Marginal contribution to risk: RC_i = w_i * (Sigma w)_i / (w' Sigma w).
    These sum to 1 by construction, which is what makes "where is the risk"
    answerable rather than hand-waved.
    """
    portfolio_var = float(weights @ cov @ weights)
    if portfolio_var <= 0:
        return np.zeros_like(weights)
    return weights * (cov @ weights) / portfolio_var


def build_facts(
    date: str,
    new_weights: np.ndarray,
    old_weights: np.ndarray,
    tickers: list[str],
    cov: np.ndarray,
    exp_returns: np.ndarray,
    spec,
    transaction_cost: float = 0.0,
    top_k: int = 5,
) -> RebalanceFacts:
    """Extract the explainable facts from a solved rebalance."""
    delta = new_weights - old_weights
    order = np.argsort(delta)

    binding = []
    if np.any(new_weights >= spec.max_weight - 1e-6):
        n_capped = int((new_weights >= spec.max_weight - 1e-6).sum())
        binding.append(f"position cap of {spec.max_weight:.1%} binding on {n_capped} names")
    if np.any(new_weights <= spec.min_weight + 1e-9):
        n_floor = int((new_weights <= spec.min_weight + 1e-9).sum())
        binding.append(f"long-only floor binding on {n_floor} names (excluded)")
    if spec.turnover_budget is not None:
        used = float(np.abs(delta).sum())
        if used >= spec.turnover_budget - 1e-6:
            binding.append(f"turnover budget of {spec.turnover_budget:.1%} fully used")

    contributions = risk_contributions(new_weights, cov)
    risk_order = np.argsort(-contributions)[:top_k]

    return RebalanceFacts(
        date=date,
        n_assets=len(tickers),
        turnover=float(np.abs(delta).sum()),
        transaction_cost=transaction_cost,
        top_buys=[(tickers[i], float(delta[i])) for i in order[::-1][:top_k] if delta[i] > 1e-6],
        top_sells=[(tickers[i], float(delta[i])) for i in order[:top_k] if delta[i] < -1e-6],
        binding_constraints=binding,
        expected_return=float(exp_returns @ new_weights),
        expected_vol=float(np.sqrt(max(new_weights @ cov @ new_weights, 0.0))),
        top_risk_contributors=[(tickers[i], float(contributions[i])) for i in risk_order],
    )


def explain(
    facts: RebalanceFacts,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> tuple[str, dict]:
    """Send facts to a NIM endpoint. Returns (explanation, latency/usage metrics).

    Latency and token counts come back alongside the text so the explainer can
    be entered in the same benchmark table as every other stage — the project
    measures this component the way it measures the rest.
    """
    import requests

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": facts.to_prompt()},
        ],
        "temperature": temperature,
        "max_tokens": 400,
    }

    t0 = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=timeout)
    latency = time.perf_counter() - t0
    response.raise_for_status()
    body = response.json()

    usage = body.get("usage", {})
    metrics = {
        "latency_s": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "tokens_per_second": usage.get("completion_tokens", 0) / latency if latency else 0.0,
        "model": model,
    }
    return body["choices"][0]["message"]["content"].strip(), metrics


def explain_offline(facts: RebalanceFacts) -> tuple[str, dict]:
    """Template-based fallback when no NIM endpoint is reachable.

    Kept so the dashboard degrades to something truthful instead of an error,
    and so the value of the LLM layer can be judged against the trivial
    alternative rather than assumed.
    """
    buys = ", ".join(f"{t} ({d:+.1%})" for t, d in facts.top_buys[:3]) or "no material increases"
    sells = ", ".join(f"{t} ({d:+.1%})" for t, d in facts.top_sells[:3]) or "no material decreases"
    constraints = "; ".join(facts.binding_constraints) or "no constraints were binding"
    risk = ", ".join(f"{t} ({s:.1%})" for t, s in facts.top_risk_contributors[:3])

    text = (
        f"On {facts.date} the portfolio was re-optimized across {facts.n_assets} assets, "
        f"turning over {facts.turnover:.1%} at a cost of {facts.transaction_cost:.3%}. "
        f"Largest increases: {buys}. Largest decreases: {sells}. "
        f"Constraints: {constraints}. "
        f"Expected annual return is {facts.expected_return:.1%} at "
        f"{facts.expected_vol:.1%} volatility, with risk concentrated in {risk}."
    )
    return text, {"latency_s": 0.0, "model": "template (offline fallback)"}
