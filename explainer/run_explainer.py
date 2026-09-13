"""Run the NIM explainer against a real endpoint and record what it did.

    NVIDIA_API_KEY=<your key> python -m explainer.run_explainer --out benchmarks/results/nim-hosted

Builds one real rebalance — a turnover-capped mean-variance solve moving a
40-name synthetic book on from the previous quarter's optimum — asks the
model to explain it ``--runs`` times, and records every reply with its
latency, token counts and any number the model produced that the facts do
not support (``unsupported_numbers``). The template fallback's text is kept
alongside for comparison.

The default endpoint is NVIDIA's hosted API, which needs a key: it is read
from NVIDIA_API_KEY and never printed or written. A local NIM container
needs none: pass ``--endpoint http://localhost:8000/v1/chat/completions``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from data.universe import synthetic_prices
from explainer.nim_explainer import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    build_facts,
    explain,
    explain_offline,
    unsupported_numbers,
)
from optimizer.mean_variance_cpu import solve_mean_variance_cpu
from optimizer.spec import PortfolioSpec
from pipeline.cpu_baseline import build_risk_model


def rebalance_facts(n_assets: int = 40, seed: int = 3, cost_bps: float = 10.0):
    """The facts for one quarterly rebalance, computed by the optimizer."""
    prices = synthetic_prices(n_assets, n_days=1260, seed=seed).prices
    before = solve_mean_variance_cpu(build_risk_model(prices.iloc[:-63]),
                                     PortfolioSpec(risk_aversion=2.0, max_weight=0.15)).weights
    spec = PortfolioSpec(risk_aversion=2.0, max_weight=0.15, turnover_budget=0.25, w_prev=before)
    model = build_risk_model(prices)
    after = solve_mean_variance_cpu(model, spec).weights
    cost = cost_bps / 1e4 * float(np.abs(after - before).sum())
    return build_facts(str(prices.index[-1].date()), after, before, model.tickers,
                       model.nearest_psd(), model.exp_returns, spec, transaction_cost=cost)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    api_key = os.environ.get("NVIDIA_API_KEY")
    if args.endpoint == DEFAULT_ENDPOINT and not api_key:
        print("NVIDIA's hosted endpoint needs a key: set NVIDIA_API_KEY in this shell "
              "(from build.nvidia.com), or pass --endpoint for a local NIM.", file=sys.stderr)
        return 2

    facts = rebalance_facts()
    runs = []
    for _ in range(args.runs):
        try:
            text, metrics = explain(facts, endpoint=args.endpoint, model=args.model, api_key=api_key)
        except Exception as exc:  # recorded, not hidden: a failed call is a result too
            error = f"{type(exc).__name__}: {exc}"
            runs.append({"error": error.replace(api_key, "[key redacted]") if api_key else error})
            continue
        runs.append({**metrics, "explanation": text, "unsupported_numbers": unsupported_numbers(text, facts)})

    record = {
        "endpoint": args.endpoint,
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "client": f"{platform.platform()}, Python {platform.python_version()}",
        "facts": facts.to_prompt(),
        "offline_template": explain_offline(facts)[0],
        "runs": runs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "explanations.json").write_text(json.dumps(record, indent=2))

    answered = [run for run in runs if "explanation" in run]
    print(f"{len(answered)}/{len(runs)} calls answered by {args.model} at {args.endpoint}")
    if answered:
        print(f"median latency {statistics.median(r['latency_s'] for r in answered):.2f} s, "
              f"median {statistics.median(r['tokens_per_second'] for r in answered):.0f} tokens/s, "
              f"{sum(1 for r in answered if r['unsupported_numbers'])} replies with unsupported numbers")
        print("\nfirst reply:\n" + answered[0]["explanation"])
    for run in runs:
        if "error" in run:
            print("error:", run["error"], file=sys.stderr)
    print(f"\nwrote {args.out}/explanations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
