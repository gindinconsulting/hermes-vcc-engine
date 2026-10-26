"""Post-compaction prefix priming: the operator's next message should not pay for cold prefill."""

import pytest

pytest.importorskip("agent.conversation_compression")

import priming  # noqa: E402  (conftest puts vcc/ on sys.path)


class _Agent:
    """The attributes priming touches: live history, model id, cached system prompt."""

    def __init__(self, messages, model="Qwen-Flash:coding"):
        self.messages = list(messages)
        self.model = model
        self.compacts = False
        self.client: object = None  # Hermes agents carry the client they answer with
        self._cached_system_prompt = "session system prompt"


def _opts(**over):
    import engine

    vcc = {"warm_after_compaction": True, "warm_after_compaction_delay_s": 0.0}
    vcc.update(over)
    return engine._engine_options({"context": {"vcc": vcc}})


def _msgs(pairs=4):
    out = []
    for i in range(pairs):
        out += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    return out


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(priming, "_INSTALLED", {"wrapper": None, "original": None})
    for key in ("sent", "skipped", "errors", "installs"):
        monkeypatch.setitem(priming.PRIMES, key, 0)
    priming.PRIMES["last"] = ""


def test_the_wrapper_primes_a_commit_and_only_a_commit(monkeypatch):
    import agent.conversation_compression as cc

    def fake_compress(agent, messages, system_message, *args, **kwargs):
        return (messages[:-2], system_message) if agent.compacts else (messages, system_message)

    monkeypatch.setattr(cc, "compress_context", fake_compress)
    primed = []
    monkeypatch.setattr(priming, "prime", lambda agent, messages, system, *, opts: primed.append(len(messages)))
    priming.install(_opts())
    priming.install(_opts())  # second engine construction must not double-wrap
    assert priming.PRIMES["installs"] == 1

    shrinking = _Agent(_msgs(8))
    shrinking.compacts = True                 # a real commit: the history shrank
    steady = _Agent(_msgs(4))                 # an aborted pass: input returned unchanged

    assert len(cc.compress_context(shrinking, _msgs(8), "sys")[0]) == 14
    assert len(cc.compress_context(steady, _msgs(4), "sys")[0]) == 8
    assert primed == [14], "an aborted or size-unchanged compression must not prime"


def test_prime_sends_the_wire_copy_with_one_token(monkeypatch):
    import agent.auxiliary_client as aux

    def fake_build(agent, messages, **kwargs):
        assert kwargs["current_turn_user_idx"] is None, "there is no user turn to place yet"
        return [{"role": "system", "content": "S"}] + list(messages), "S"

    monkeypatch.setattr(priming, "agent_build_api_messages", fake_build)
    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (object(), "Qwen-Flash"))
    sent = {}

    priming.prime(_Agent(_msgs(4)), _msgs(4), "S", opts=_opts(),
                  sender=lambda opts, wire, model, **kw: sent.update(
                      wire=wire, model=model, max_tokens=opts["warm_after_compaction_max_tokens"]),
                  block=True)

    assert sent["max_tokens"] == 1, "a prime is a prefill, not an answer"
    assert sent["model"] == "Qwen-Flash:coding", "prime the instance the next turn will use"
    assert sent["wire"][0]["role"] == "system" and len(sent["wire"]) == 9
    assert priming.PRIMES["sent"] == 1


def test_prime_skips_when_a_turn_already_started(monkeypatch):
    """Racing a live request with a prefill makes the operator's message slower, not faster."""
    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (object(), "Qwen-Flash"))
    sent = []
    agent = _Agent(_msgs(6))                      # live history: 12 rows
    priming.prime(agent, _msgs(3), "S", opts=_opts(), sender=lambda *a: sent.append(a), block=True)
    assert sent == [] and priming.PRIMES["skipped"] == 1
    assert "a turn started first" in priming.PRIMES["last"]


def test_prime_skips_when_the_aux_route_is_a_different_model(monkeypatch):
    """The KV cache belongs to a model instance; warming a cheaper summariser warms nothing."""
    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (object(), "SomeCheapSummariser"))
    sent = []
    msgs = _msgs(3)
    priming.prime(_Agent(msgs), msgs, "S", opts=_opts(), sender=lambda *a: sent.append(a), block=True)
    assert sent == [] and "is not the session model" in priming.PRIMES["last"]


def test_the_agents_own_client_beats_a_different_summariser(monkeypatch):
    """The config on this box used to be: the summariser profile is NOT the session profile. The
    prime has to follow the instance that answers the next turn, which the agent object carries —
    but only as a fallback now, because that client is per-request (see the two tests below)."""
    import agent.auxiliary_client as aux

    sent: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            sent.append(kwargs)

    class _Client:
        chat = type("_Chat", (), {"completions": _Completions()})

    def wrong_route(task=""):                     # a different instance entirely
        return _Client(), "Qwen3-Summariser"

    monkeypatch.setattr(aux, "get_text_auxiliary_client", wrong_route)
    monkeypatch.setattr(aux, "get_auxiliary_extra_body", lambda *a, **k: {})
    monkeypatch.setattr(priming, "agent_build_api_messages",
                        lambda agent, messages, **kw: (list(messages), "S"))

    msgs = _msgs(3)
    agent = _Agent(msgs)
    agent.client = _Client()                      # the client Hermes will answer with
    priming.prime(agent, msgs, "S", opts=_opts(), block=True)

    assert sent and sent[0]["model"] == "Qwen-Flash:coding", "prime the session instance"
    assert sent[0]["max_tokens"] == 1 and sent[0]["stream"] is False
    assert "agent.client" in priming.PRIMES["last"], priming.PRIMES["last"]


def _live_client(sent, *, raise_with=None):
    """A stand-in OpenAI client: the shape `send_prime` actually calls is `client.chat.completions.create`."""
    class _Completions:
        def create(self, **kw):
            sent.append(kw)
            if raise_with is not None:
                raise raise_with

    return type("_Client", (), {"chat": type("_Chat", (), {"completions": _Completions()})})()


def test_a_matching_aux_route_outranks_the_per_turn_client(monkeypatch, caplog):
    """`agent.client` lives only for the duration of one request (`run_agent.py` nulls it,
    `chat_completion_helpers` closes it at turn end — measured: it tore a prime off mid-flight 17 s
    after sending it). Once the compression task points at the session model, the module-cached
    auxiliary client answers the same instance with no turn lifecycle, so it wins."""
    import agent.auxiliary_client as aux

    sent: list[dict] = []
    per_turn = _live_client(sent)
    long_lived = _live_client(sent)

    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (long_lived, "Qwen-Flash"))
    monkeypatch.setattr(aux, "get_auxiliary_extra_body", lambda *a, **k: {})
    monkeypatch.setattr(priming, "agent_build_api_messages",
                        lambda agent, messages, **kw: (list(messages), "S"))

    msgs = _msgs(3)
    agent = _Agent(msgs)
    agent.client = per_turn                        # live right now, gone at turn end
    priming.prime(agent, msgs, "S", opts=_opts(), block=True)

    assert len(sent) == 1
    assert "aux:compression" in priming.PRIMES["last"], priming.PRIMES["last"]
    assert "agent.client" not in priming.PRIMES["last"]


def test_a_failed_prime_is_visible_at_info_level_and_harmless(monkeypatch, caplog):
    """Every prime outcome has to be readable in the operator's log. A DEBUG-only failure hid a
    torn-off request for four minutes of debugging on 2026-10-26."""
    import logging

    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (
        _live_client([], raise_with=RuntimeError("client closed by turn end")), "Qwen-Flash"))
    monkeypatch.setattr(priming, "agent_build_api_messages",
                        lambda agent, messages, **kw: (list(messages), "S"))

    msgs = _msgs(3)
    with caplog.at_level(logging.INFO):
        note = priming.prime(_Agent(msgs), msgs, "S", opts=_opts(), block=True)

    assert priming.PRIMES["errors"] == 1
    assert "error: RuntimeError" in note, note
    assert any("prefix prime failed" in r.getMessage() and r.levelno >= logging.WARNING
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_a_broken_prime_never_breaks_compaction(monkeypatch):
    import agent.conversation_compression as cc

    def exploding(agent, messages, system_message, *args, **kwargs):
        raise RuntimeError("summary LLM exploded")

    monkeypatch.setattr(cc, "compress_context", exploding)
    monkeypatch.setattr(priming, "prime", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("prime broke")))
    priming.install(_opts())

    agent = _Agent(_msgs(4))
    agent.compacts = True
    with pytest.raises(RuntimeError, match="summary LLM exploded"):
        cc.compress_context(agent, _msgs(4), "sys")   # Hermes's own failure path, intact


def test_idle_probe_reads_the_engine_queue_counts():
    """`""` means the engine is idle and the prime may go; a count means stay out of the way."""
    import io
    import json as _json

    def _payload(active):
        body = _json.dumps({"active_models": {"total_active_requests": active,
                                              "total_waiting_requests": 0}}).encode()
        return lambda request, timeout=0: io.BytesIO(body)

    opts = _opts(warm_after_compaction_idle_probe="http://127.0.0.1:1/stats")
    assert priming._engine_busy(opts, opener=_payload(0)) == ""
    got = priming._engine_busy(opts, opener=_payload(2))
    assert got == "engine busy: 2 request(s) active or waiting", got


def test_an_unreachable_probe_counts_as_busy():
    """Skipping a warm-up costs the pre-priming status quo; a collided prefill costs the turn."""
    import urllib.error

    def boom(request, timeout=0):
        raise urllib.error.URLError("connection refused")

    got = priming._engine_busy(_opts(warm_after_compaction_idle_probe="http://127.0.0.1:1/stats"), opener=boom)
    assert "unreachable" in got, got


def test_prime_skips_when_the_engine_already_has_a_request(monkeypatch):
    """The 2026-10-26 09:50:39 collision: both requests carried 25 messages, so length staleness
    was blind to it and the prime prefill raced the operator's turn (1 token in 59.49 s)."""
    sent = []
    monkeypatch.setattr(priming, "_engine_busy",
                        lambda opts: "engine busy: 1 request(s) active or waiting")

    msgs = _msgs(3)
    note = priming.prime(_Agent(msgs), msgs, "S", opts=_opts(),
                         sender=lambda *a, **k: sent.append(a), block=True)

    assert sent == [], "a prime went out on top of a live turn"
    assert priming.PRIMES["skipped"] == 1
    assert "engine busy" in note, note


def test_send_prime_sends_the_tools_the_live_turn_sends():
    """Tool schemas render into the prompt, so a tools-less prime warms a prefix the next turn
    cannot reuse. Measured 2026-10-26: prime 31,050 tok, next turn 38,983 tok, `cached 6,592`,
    TTFT 13.9s — versus 0.9s for an ordinary warm turn-to-turn call."""
    seen = {}

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    tools = [{"type": "function", "function": {"name": "terminal"}}]
    priming.send_prime(_opts(), [{"role": "user", "content": "x"}], "Qwen-Flash",
                       client=_Client(), tools=tools)
    assert seen["tools"] == tools, seen
    assert seen["messages"] == [{"role": "user", "content": "x"}]

    only = {}

    class _Completions2:
        def create(self, **kwargs):
            only.update(kwargs)

    class _Chat2:
        completions = _Completions2()

    class _Client2:
        chat = _Chat2()

    priming.send_prime(_opts(), [{"role": "user", "content": "x"}], "Qwen-Flash", client=_Client2())
    assert "tools" not in only, only


def test_prime_passes_the_agent_tools_to_the_sender(monkeypatch):
    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (object(), "Qwen-Flash"))
    calls = []
    spy = lambda *a, **k: calls.append(k)

    agent = _Agent(_msgs(3))
    schemas = [{"type": "function", "function": {"name": "js"}}]
    setattr(agent, "tools", schemas)
    priming.prime(agent, agent.messages, "S", opts=_opts(), sender=spy, block=True)

    assert len(calls) == 1, calls
    assert calls[0]["tools"] == schemas, calls[0]


def test_prime_fires_when_the_probe_reports_the_engine_idle(monkeypatch):
    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (object(), "Qwen-Flash"))
    monkeypatch.setattr(priming, "_engine_busy", lambda opts: "")
    sent = []

    msgs = _msgs(3)
    priming.prime(_Agent(msgs), msgs, "S", opts=_opts(),
                  sender=lambda *a, **k: sent.append(a), block=True)

    assert len(sent) == 1, sent
    assert priming.PRIMES["skipped"] == 0



def _fake_wire(monkeypatch):
    import agent.auxiliary_client as aux

    monkeypatch.setattr(priming, "agent_build_api_messages",
                        lambda agent, messages, **kw: ([{"role": "system", "content": "S"}] + list(messages), "S"))
    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (object(), "Qwen-Flash"))


def test_dry_run_measures_without_paying_for_the_prefill(monkeypatch, caplog):
    """2026-10-26: the warm-up landed on 1 of 7 compactions and cost 12-21 s of prefill each time,
    so the config may collect the payload fingerprint and the cache-counter window while sending
    nothing. A dry run that still calls the sender would be paying for the thing it measures."""
    import logging

    _fake_wire(monkeypatch)
    sent: list = []
    msgs = _msgs(4)
    with caplog.at_level(logging.INFO, logger="hermes_plugins.hermes_vcc_engine.priming"):
        priming.prime(_Agent(msgs), msgs, "S",
                      opts=_opts(warm_after_compaction_dry_run=True, warm_after_compaction_delta_wait_s=5.0),
                      sender=lambda *a, **kw: sent.append(a), block=True)
    assert sent == [], "a dry run must not put a prefill on the engine"
    assert priming.PRIMES["sent"] == 0 and priming.PRIMES["skipped"] == 1
    assert priming.PRIMES["last"].startswith("dry-run")
    lines = [r.getMessage() for r in caplog.records if "dry-run" in r.getMessage()]
    assert lines and "head_sha=" in lines[0], "the payload fingerprint is the diagnostic being collected"


def test_the_live_prime_logs_the_reuse_count_it_earned(monkeypatch, caplog):
    """`cached_tokens` from the warm response, not a timing inference, is what settles whether a
    prime was reusable at all — so the log line has to carry it."""
    import logging

    _fake_wire(monkeypatch)
    with caplog.at_level(logging.INFO, logger="hermes_plugins.hermes_vcc_engine.priming"):
        priming.prime(_Agent(_msgs(4)), _msgs(4), "S", opts=_opts(),
                      sender=lambda opts, wire, model, **kw: {"prompt": 40228, "cached": 0}, block=True)
    line = next(r.getMessage() for r in caplog.records if "prefix primed" in r.getMessage())
    assert "prompt=40228 cached=0 tokens" in line and "head_sha=" in line


def test_usage_survives_provider_shape_drift():
    """OpenAI-style dicts, attribute objects, and no usage block at all: none may raise."""
    class _Details:
        cached_tokens = 7777

    class _Usage:
        prompt_tokens = 40228
        prompt_tokens_details = _Details()

    class _Resp:
        usage = _Usage()

    assert priming._usage_of(_Resp()) == {"prompt": 40228, "cached": 7777}
    assert priming._usage_of({"usage": {"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3}}}) \
        == {"prompt": 5, "cached": 3}
    assert priming._usage_of({"usage": {"prompt_tokens": 5, "cache_read_input_tokens": 2}}) == {"prompt": 5, "cached": 2}
    assert priming._usage_of(object()) == {}


def test_fingerprint_localises_where_two_prompts_diverge():
    """Same length, different head: the hash changes. Same head, different tail: it does not."""
    head = [{"role": "system", "content": "S" * 2000}, {"role": "user", "content": "old"}]
    tail_changed = [{"role": "system", "content": "S" * 2000}, {"role": "user", "content": "newer"}]
    head_changed = [{"role": "system", "content": "T" * 2000}, {"role": "user", "content": "old"}]
    base = priming._fingerprint(head, [1, 2])[0]
    assert priming._fingerprint(tail_changed, [1, 2])[0] == base
    assert priming._fingerprint(head_changed, [1, 2])[0] != base
    assert priming._fingerprint(head, [1, 2, 3])[2] == 3


def test_measurement_survives_a_busy_engine(monkeypatch, caplog):
    """2026-10-26 15:22: the busy gate ran before the dry run, so a compaction that could not warm
    also collected nothing. Counters and fingerprint cost the engine nothing, so they must be taken
    whatever the gate decides."""
    import logging

    _fake_wire(monkeypatch)
    monkeypatch.setattr(priming, "_engine_busy", lambda opts: "engine busy: 1 request(s) active")
    msgs = _msgs(4)
    with caplog.at_level(logging.INFO, logger="hermes_plugins.hermes_vcc_engine.priming"):
        priming.prime(_Agent(msgs), msgs, "S",
                      opts=_opts(warm_after_compaction_dry_run=True, warm_after_compaction_delta_wait_s=5.0),
                      sender=lambda *a, **kw: pytest.fail("dry run must not send"), block=True)
    line = next(r.getMessage() for r in caplog.records if "dry-run" in r.getMessage())
    assert "counters" in line and "head_sha=" in line


def test_the_prime_waits_for_the_idle_gap(monkeypatch):
    """The compacted prefix is not sent until the next turn, so a busy engine means wait, not
    give up — but each probe must be spaced by the configured delay, and the warm must be sent
    exactly once, the moment the engine looks free."""
    _fake_wire(monkeypatch)
    probes = iter(["engine busy: 1 request(s) active", "engine busy: 1 request(s) active", ""])
    monkeypatch.setattr(priming, "_engine_busy", lambda opts: next(probes))
    sent: list = []
    slept: list = []
    priming.prime(_Agent(_msgs(4)), _msgs(4), "S",
                  opts=_opts(warm_after_compaction_idle_attempts=5, warm_after_compaction_retry_delay_s=10.0),
                  sender=lambda opts, wire, model, **kw: sent.append(model),
                  sleep=lambda s: slept.append(s), block=True)
    assert len(sent) == 1, "exactly one warm, once the engine went idle"
    assert slept == [10.0, 10.0], f"probes spaced by the configured delay, got {slept}"


def test_a_turn_starting_during_the_wait_abandons_the_warm(monkeypatch):
    """A prefix that is no longer the next call's prefix is not worth warming: 2026-10-26's
    09:50:39 splash was exactly that race."""
    import agent.auxiliary_client as aux

    _fake_wire(monkeypatch)
    monkeypatch.setattr(priming, "_engine_busy", lambda opts: "engine busy: 1 request(s) active")
    msgs = _msgs(4)
    agent = _Agent(msgs)

    def _moving_sleep(_seconds: float) -> None:
        # _Agent copies the list it is handed; priming watches the agent's live history.
        agent.messages.append({"role": "user", "content": "a turn started"})

    priming.prime(agent, msgs, "S",
                  opts=_opts(warm_after_compaction_idle_attempts=5, warm_after_compaction_retry_delay_s=10.0),
                  sender=lambda *a, **kw: pytest.fail("a stale prefix must not be warmed"),
                  sleep=_moving_sleep, block=True)
    assert priming.PRIMES["sent"] == 0 and "history changed" in priming.PRIMES["last"]


def test_a_prime_borrows_the_turn_clock_it_was_never_given(monkeypatch):
    """agent/turn_context.py:1273 reads agent._current_turn_timestamp unconditionally and fails a
    caller that skipped the turn prologue - which is every prime, hence the AttributeError that put
    every payload on the raw-history route up to 2026-10-26 15:35. The clock must be borrowed, not
    kept, and must never overwrite a live turn's frozen value."""
    seen: list = []

    def _builder(agent, messages, **kw):
        seen.append(getattr(agent, "_current_turn_timestamp", None))
        return [{"role": "user", "content": "x"}], "S"

    monkeypatch.setattr(priming, "agent_build_api_messages", _builder)
    agent = _Agent(_msgs(2))

    _, route = priming.wire_payload(agent, agent.messages, "S")
    assert route == "build_api_messages", "the real wire copy, not the raw-history fallback"
    assert isinstance(seen[0], float), "the builder got a clock"
    assert not hasattr(agent, "_current_turn_timestamp"), "a borrowed clock must be handed back"

    agent._current_turn_timestamp = 12345.0
    priming.wire_payload(agent, agent.messages, "S")
    assert seen[1] == 12345.0 and agent._current_turn_timestamp == 12345.0, "a live turn owns its clock"


def test_a_broken_builder_still_degrades_to_raw_history(monkeypatch):
    """The borrowed clock must not swallow real builder failures: they still degrade to the
    documented fallback, and the slot is still returned."""
    def _boom(agent, messages, **kw):
        raise RuntimeError("plugin manager exploded")

    monkeypatch.setattr(priming, "agent_build_api_messages", _boom)
    agent = _Agent([{"role": "user", "content": "q"}])
    wire, route = priming.wire_payload(agent, agent.messages, "S")
    assert route.startswith("fallback raw history (RuntimeError") and wire[0]["role"] == "system"
    assert not hasattr(agent, "_current_turn_timestamp")


def test_one_prime_opens_exactly_one_cache_window_reporter(monkeypatch):
    """_delta_watch used to be started twice on the live path - once before the dry-run gate (free
    measurements go first) and once after the warm - so every prime logged its `cache window +180s`
    line twice with identical counters (16:38:00 and 16:38:11). One prime, one window, anchored on
    the pre-warm baseline and the session model, so it already covers the warm and every turn in it."""
    import agent.auxiliary_client as aux
    import threading as real_threading

    started: list = []

    class _Thread:
        def __init__(self, target=None, name="", daemon=None, args=(), **kw):
            self.args, self.name = args, name

        def start(self):
            if self.name == "vcc-cache-window":
                started.append(self.args)

    class _ThreadNs:
        Thread = _Thread

        def __getattr__(self, item):
            return getattr(real_threading, item)

    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task="": (object(), "Qwen-Flash"))
    monkeypatch.setattr(priming, "_engine_busy", lambda opts: "")
    monkeypatch.setattr(priming, "threading", _ThreadNs())

    msgs = _msgs(3)
    agent = _Agent(msgs)
    priming.prime(agent, msgs, "S", opts=_opts(), sender=lambda *a, **k: {"prompt": 10}, block=True)
    assert len(started) == 1, f"live prime must open one window, got {len(started)}"

    started.clear()
    opts = dict(_opts(), warm_after_compaction_dry_run=True)
    priming.prime(agent, msgs, "S", opts=opts, sender=lambda *a, **k: {"prompt": 10}, block=True)
    assert len(started) == 1, f"a dry-run prime measures too, exactly one window, got {len(started)}"
