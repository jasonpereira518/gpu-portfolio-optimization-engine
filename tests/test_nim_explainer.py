"""The NIM explainer's contract, against a local stand-in for the endpoint.

The stand-in speaks the OpenAI-compatible chat API that NIM exposes, on
127.0.0.1, so nothing leaves the machine and no credential is involved. What
is checked is the client's side of the contract: what it sends, when it
authenticates, what it keeps from the reply, and how it fails.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from explainer.nim_explainer import SYSTEM_PROMPT, RebalanceFacts, explain, unsupported_numbers

FACTS = RebalanceFacts(
    date="2026-06-30", n_assets=40, turnover=0.1234, transaction_cost=0.00012,
    top_buys=[("SYN00001", 0.0321)], top_sells=[("SYN00007", -0.0254)],
    binding_constraints=["position cap of 15.0% binding on 3 names"],
    expected_return=0.0815, expected_vol=0.1432,
    top_risk_contributors=[("SYN00001", 0.221)],
)


class _Endpoint:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.status = 200
        self.reply: dict = {
            "choices": [{"message": {"role": "assistant", "content": "  The book rotated.  "}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 42},
        }


@pytest.fixture
def endpoint():
    state = _Endpoint()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — http.server's naming
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            payload = json.dumps(state.reply).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    yield state
    server.shutdown()


def test_explain_sends_the_optimizer_facts_and_returns_the_reply_with_its_cost(endpoint):
    text, metrics = explain(FACTS, endpoint=endpoint.url, model="some/model")

    sent = endpoint.requests[0]["body"]
    assert sent["model"] == "some/model"
    assert sent["messages"] == [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": FACTS.to_prompt()}]
    assert text == "The book rotated."
    assert (metrics["prompt_tokens"], metrics["completion_tokens"]) == (120, 42)
    assert metrics["latency_s"] > 0


def test_explain_asks_for_one_complete_answer_without_thinking(endpoint):
    """Nemotron 3 models think by default, and thinking counts against
    max_tokens — 400 tokens can be spent before any answer; the hosted API
    also streams unless told not to."""
    explain(FACTS, endpoint=endpoint.url)

    sent = endpoint.requests[0]["body"]
    assert sent["stream"] is False
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_reply_cut_off_at_max_tokens_is_an_error(endpoint):
    endpoint.reply["choices"][0]["message"]["content"] = "On 2026-06-30 the portfolio was re-optimized and"
    endpoint.reply["choices"][0]["finish_reason"] = "length"

    with pytest.raises(RuntimeError, match="cut off"):
        explain(FACTS, endpoint=endpoint.url)


def test_explain_authenticates_only_when_given_a_key(endpoint):
    explain(FACTS, endpoint=endpoint.url, api_key="not-a-real-key")
    explain(FACTS, endpoint=endpoint.url)

    with_key, without_key = (r["headers"] for r in endpoint.requests)
    assert with_key["Authorization"] == "Bearer not-a-real-key"
    assert "Authorization" not in without_key


def test_explain_raises_with_the_endpoints_own_reason_when_refused(endpoint):
    """A bare "401 Unauthorized" cannot say whether the key is wrong, expired
    or scoped to the wrong service; the endpoint's reply usually can."""
    endpoint.status = 401
    endpoint.reply = {"status": 401, "title": "Unauthorized", "detail": "Authentication failed"}

    with pytest.raises(requests.HTTPError, match="Authentication failed"):
        explain(FACTS, endpoint=endpoint.url)


def test_a_reply_with_no_text_is_an_error_not_an_empty_explanation(endpoint):
    """Reasoning models can spend the whole token budget thinking and return
    no content; an empty string must not pass for an explanation."""
    endpoint.reply["choices"][0]["message"]["content"] = None

    with pytest.raises(RuntimeError, match="no text"):
        explain(FACTS, endpoint=endpoint.url)


def test_a_reasoning_block_is_not_part_of_the_explanation(endpoint):
    endpoint.reply["choices"][0]["message"]["content"] = "<think>turnover is 12%...</think>\nThe book rotated."

    text, _ = explain(FACTS, endpoint=endpoint.url)

    assert text == "The book rotated."


# ---------------------------------------------------------------------------
# The model phrases numbers; it must not produce them
# ---------------------------------------------------------------------------

def test_numbers_taken_from_the_facts_are_supported():
    text = ("Turnover was 12.34% at a cost of 0.0120% of the book; SYN00001 rose 3.21% and "
            "SYN00007 fell 2.54%, and SYN00001 carries 22.1% of the risk.")

    assert unsupported_numbers(text, FACTS) == []


def test_a_fact_rounded_to_the_precision_written_is_supported():
    assert unsupported_numbers("Turnover was about 12% and volatility near 14.3%.", FACTS) == []


def test_a_number_no_fact_gives_is_flagged():
    """8.15% expected return and 14.32% volatility are facts; a Sharpe ratio
    of 0.57 is arithmetic the model was told not to do."""
    assert unsupported_numbers("Expected Sharpe is 0.57 on 8.15% return.", FACTS) == ["0.57"]


# ---------------------------------------------------------------------------
# The runner that records a real endpoint's behaviour
# ---------------------------------------------------------------------------

def test_the_runner_records_every_reply_and_never_writes_the_key(endpoint, tmp_path, monkeypatch):
    from explainer import run_explainer

    monkeypatch.setenv("NVIDIA_API_KEY", "not-a-real-key")
    endpoint.reply["choices"][0]["message"]["content"] = "Turnover was 99.9% this quarter."
    out = tmp_path / "nim"

    assert run_explainer.main(["--endpoint", endpoint.url, "--model", "some/model",
                               "--runs", "2", "--out", str(out)]) == 0

    record = json.loads((out / "explanations.json").read_text())
    assert [run["explanation"] for run in record["runs"]] == ["Turnover was 99.9% this quarter."] * 2
    assert all(run["unsupported_numbers"] == ["99.9"] for run in record["runs"])
    assert record["model"] == "some/model" and record["facts"]
    assert all(r["headers"]["Authorization"] == "Bearer not-a-real-key" for r in endpoint.requests)
    assert "not-a-real-key" not in (out / "explanations.json").read_text()


def test_the_runner_redacts_the_key_from_a_recorded_error(endpoint, tmp_path, monkeypatch):
    """Failed calls are recorded with the endpoint's reason, so an endpoint
    that echoed the credential back must not get it into the results file."""
    from explainer import run_explainer

    monkeypatch.setenv("NVIDIA_API_KEY", "not-a-real-key")
    endpoint.status = 401
    endpoint.reply = {"detail": "key not-a-real-key is not authorized"}
    out = tmp_path / "nim"

    run_explainer.main(["--endpoint", endpoint.url, "--runs", "1", "--out", str(out)])

    written = (out / "explanations.json").read_text()
    assert "not authorized" in written
    assert "not-a-real-key" not in written


def test_the_runner_refuses_the_hosted_endpoint_without_a_key(tmp_path, monkeypatch):
    from explainer import run_explainer

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    assert run_explainer.main(["--out", str(tmp_path / "nim")]) == 2
    assert not (tmp_path / "nim").exists()
