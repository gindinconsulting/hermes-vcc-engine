"""Post-compaction prefix priming — spend the reading window, not the next keystroke.

A compaction rewrites the whole history, so the next model call has no cached prefix to
reuse. Measured on a live session (oMLX, 2026-10-26): the first call after a manual
`/compress` took **12.16 s to first token on a 38,108-token prompt**, against **0.35-2.38 s
for every other call that hour, including one on 74,292 tokens**. The model had not cooled —
idle eviction is off and it had decoded 1,845 tokens 31 seconds earlier — the *prefix* was
new. 38,108 / 12.16 s is ~3.1k tok/s of cold prefill, and an operator reading the handoff
banner pays all of it later, on their next message.

So fire one tiny request over the payload the next turn will send, the moment a compaction
commits. Providers that cache by prefix (oMLX: `cache.enabled` plus an SSD tier, and
`preserve_mid_system_cache`) then hold those KV blocks and the real turn prefills only its
delta. If the operator already sent that turn, skip: a prefill queued *ahead* of a live
request would make it slower, not faster.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, Callable, Optional

logger = logging.getLogger("hermes_plugins.hermes_vcc_engine.priming")

# Counters for tests and for a future telemetry line; `last` is the last decision, verbatim.
PRIMES: dict[str, Any] = {"installs": 0, "sent": 0, "skipped": 0, "errors": 0, "last": ""}
_INSTALLED: dict[str, Any] = {"wrapper": None, "original": None}


def _note(text: str) -> str:
    PRIMES["last"] = text
    return text


def wire_payload(agent: Any, messages: list, system_prompt: Optional[str]) -> tuple[list, str]:
    """The wire copy the NEXT turn will send, minus the user message it has not received yet.

    Hermes builds the request copy per turn (`turn_context.build_api_messages`) and replays
    historical rows from their `api_content` sidecars precisely so the prefix stays
    byte-stable — which is what makes a prime reusable at all. Outside a turn some of the
    prologue inputs are absent, so this degrades to `system + history` rather than failing:
    a slightly shorter prime still warms the part it covers.
    """
    try:
        api_messages, effective_system = agent_build_api_messages(
            agent, list(messages), current_turn_user_idx=None, ext_prefetch_cache=None,
            plugin_user_context=None, moa_config=None, active_system_prompt=system_prompt)
        if effective_system and not (api_messages and api_messages[0].get("role") == "system"):
            api_messages = [{"role": "system", "content": effective_system}] + list(api_messages)
        return list(api_messages), "build_api_messages"
    except Exception as exc:  # noqa: BLE001 - a degraded prime beats none
        system = system_prompt or getattr(agent, "_cached_system_prompt", "") or ""
        wire = ([{"role": "system", "content": system}] if system else []) + list(messages)
        return wire, f"fallback raw history ({type(exc).__name__}: {exc})"


def agent_build_api_messages(*args: Any, **kwargs: Any):
    from agent.turn_context import build_api_messages

    return build_api_messages(*args, **kwargs)


def _base_model(model: Any) -> str:
    return str(model or "").split(":")[0].strip().lower()


def _admin_key(explicit: Any) -> str:
    """Bearer for the engine admin API: config value, then env, then oMLX's own settings file."""
    for value in (explicit, os.environ.get("OMLX_ADMIN_KEY")):
        if value:
            return str(value).strip()
    try:
        with open(os.path.expanduser("~/.omlx/settings.json"), "r", encoding="utf-8") as fh:
            return str((json.load(fh).get("auth") or {}).get("api_key") or "").strip()
    except Exception:  # noqa: BLE001 - a missing key just means an unauthenticated probe
        return ""


def _cache_counters(opts: dict[str, Any], opener: Callable[..., Any] = urllib.request.urlopen,
                    model: str = "") -> Optional[dict[str, Any]]:
    """Cumulative prefix-cache counters, read from the same admin endpoint as the idle probe.

    `server.log` never reports reuse — I grepped a full day of it — but `/admin/api/stats` does:
    `total_cached_tokens` server-wide, plus per-model `cache_rates.cumulative.{prefix_hits,
    prefix_misses,evictions}`. Sampling this before and after the operator's first post-compaction
    call turns "I think it missed" into a number. Returns ``None`` when unreachable, never raises.
    """
    url = str(opts.get("warm_after_compaction_idle_probe") or "").strip()
    if not url:
        return None
    key = _admin_key(opts.get("warm_after_compaction_idle_probe_key"))
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with opener(request, timeout=float(opts.get("warm_after_compaction_probe_timeout_s", 0.5))) as resp:
            payload = json.load(resp)
    except Exception:  # an unreachable engine just means no counters this time
        return None
    engines = (payload.get("runtime_cache") or {}).get("models") or []
    chosen = next((m for m in engines
                   if _base_model(m.get("id") or "") == _base_model(model)), engines[0] if engines else {})
    rates = (chosen.get("cache_rates") or {}).get("cumulative") or {}
    return {
        "cached": payload.get("total_cached_tokens"),
        "prompt": payload.get("total_prompt_tokens"),
        "hits": rates.get("prefix_hits"),
        "misses": rates.get("prefix_misses"),
        "evictions": rates.get("evictions"),
    }


def _delta_watch(opts: dict[str, Any], before: Optional[dict[str, Any]], wait_s: float,
                 model: str = "", sleep: Callable[[float], None] = time.sleep) -> None:
    """Log what the cache actually served between the prime window and the operator's next turn."""
    sleep(wait_s)
    after = _cache_counters(opts, model=model)
    if not before or not after or before.get("cached") is None or after.get("cached") is None:
        logger.info("vcc: cache window unusable (counters unavailable before or after)")
        return
    delta = {k: (after[k] - before[k]) if isinstance(after.get(k), int) and isinstance(before.get(k), int)
             else "?" for k in ("cached", "prompt", "hits", "misses", "evictions")}
    logger.info("vcc: cache window %+.0fs after compaction: cached %+s tokens served, prompt %+s, "
                "prefix hits %+s, misses %+s, evictions %+s",
                wait_s, delta["cached"], delta["prompt"], delta["hits"], delta["misses"],
                delta["evictions"])
    _note("window " + " ".join(f"{k}={v}" for k, v in delta.items()))


def _fingerprint(wire: list, tools: Any) -> tuple[str, int, int]:
    """A hash of the payload HEAD plus its size — enough to say where two prompts diverge.

    A prime only helps if it is byte-identical to the request that follows, and "identical
    length" is not identical: on 2026-10-26 a 38,011-token prime preceded a 38,022-token turn
    that still paid a 10.51 s cold prefill. The head hash separates `the system/tools region
    moved` (head differs) from `only the tail grew` (head matches, total differs).
    """
    text = "".join(str(m.get("content") or "") for m in wire)
    return hashlib.sha256(text[:1024].encode("utf-8", "replace")).hexdigest()[:12], len(text), len(tools or [])


def _queued_requests(payload: Any) -> int:
    """How many requests the engine says it is running or holding; -1 if the shape is unknown."""
    active = payload.get("active_models") if isinstance(payload, dict) else None
    if not isinstance(active, dict):
        return -1
    total = sum(active[k] for k in ("total_active_requests", "total_waiting_requests")
                if isinstance(active.get(k), int))
    if total == 0:  # some builds report per-engine instead of totals
        for engine in active.values():
            if isinstance(engine, dict):
                total += sum(engine[k] for k in ("active_requests", "waiting_requests", "queued_requests")
                             if isinstance(engine.get(k), int))
    return total


def _engine_busy(opts: dict[str, Any], opener: Callable[..., Any] = urllib.request.urlopen) -> str:
    """``""`` when the serving engine looks idle; a reason string when a request is queued.

    The queue we must not join is the *engine's*, and message-count staleness cannot see it. At
    09:50:39 on 2026-10-26 the prime fired 1.1 s after a compaction committed while the operator's
    turn had been in flight for 38 ms: both requests carried the same 25 messages, so the length
    check passed and two prefills of one 38k-token prefix fought each other — the prime took
    1 token in 59.49 s and the turn's first token came at 21.04 s (32.7 tok/s, against 100.3 tok/s
    on the call before it). An unreachable probe counts as busy: skipping a warm-up costs us the
    pre-priming status quo, joining a live prefill costs the operator their turn.
    """
    url = str(opts.get("warm_after_compaction_idle_probe") or "").strip()
    if not url:
        return ""  # no probe configured: unknown backend, behave as before priming existed
    key = _admin_key(opts.get("warm_after_compaction_idle_probe_key"))
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with opener(request, timeout=float(opts.get("warm_after_compaction_probe_timeout_s", 0.5))) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - unreachable engine is a skip, not a crash
        return f"engine idle probe unreachable ({type(exc).__name__})"
    queued = _queued_requests(payload)
    if queued < 0:
        return "engine idle probe returned an unrecognised shape"
    return f"engine busy: {queued} request(s) active or waiting" if queued else ""


def _looks_stale(agent: Any, baseline: int) -> str:
    """Best-effort: has a turn already been admitted on top of what we primed for?"""
    if getattr(agent, "_active_compression_commit_fence", None):
        return "another compaction is in flight"
    live = getattr(agent, "messages", None)
    if isinstance(live, list) and len(live) != baseline:
        return f"history changed {baseline} -> {len(live)} (a turn started first)"
    return ""


def _session_route(agent: Any) -> tuple[Any, str, str]:
    """The client the next turn answers with — a last resort, not the first choice.

    A KV cache belongs to one model instance, so the prime has to land on the instance that
    answers, and the compression task is routinely configured to a different profile than the
    session model. But the client the agent object carries is a PER-REQUEST one
    (`run_agent.py` sets `self.client = None` and `chat_completion_helpers` swaps a live client
    in only for the duration of a request, closing it at turn end), so a prime sent on it can be
    torn off mid-flight by `agent_close` — measured on this box at 09:09:51, 17 s after the prime
    went out, with the response lost and the failure visible only at DEBUG. The auxiliary route
    is a module-cached client with no turn lifecycle, so it is preferred whenever it resolves to
    the same model; this one is kept for a Hermes build with no compression task configured.
    """
    model = str(getattr(agent, "model", "") or getattr(agent, "_model", "") or "")
    for attr in ("client", "_client", "_chat_client"):
        candidate = getattr(agent, attr, None)
        completions = getattr(getattr(candidate, "chat", None), "completions", None)
        if model and callable(getattr(completions, "create", None)):
            return candidate, model, f"agent.{attr}"
    return None, model, ""


def _aux_route(session_model: str) -> tuple[Any, str, str]:
    """The configured compression client — first choice when it *is* the session model."""
    try:
        from agent.auxiliary_client import get_text_auxiliary_client

        client, aux_model = get_text_auxiliary_client("compression")
    except Exception:  # noqa: BLE001 - no aux route is a skip, never a failure
        return None, "", ""
    if client is None:
        return None, "", ""
    if _base_model(aux_model) != _base_model(session_model):
        return None, "", (f"aux model {aux_model} is not the session model {session_model}")
    return client, session_model, "aux:compression"


def send_prime(opts: dict[str, Any], wire: list, model: str, client: Any = None,
               tools: Optional[list] = None) -> dict[str, Any]:
    """One non-streaming, single-token completion over `wire` — a prefill, not an answer.

    `client` is the instance that will serve the next turn; with none given this falls back to
    the compression auxiliary route, which is only correct when it resolves to the same model.
    """
    if client is None:
        from agent.auxiliary_client import get_text_auxiliary_client

        client, aux_model = get_text_auxiliary_client("compression")
        if client is None:
            raise RuntimeError("no auxiliary client resolved for the compression task")
        model = model or aux_model or ""
    kwargs: dict[str, Any] = {
        "model": model, "messages": wire,
        "max_tokens": int(opts["warm_after_compaction_max_tokens"]), "stream": False,
    }
    # Tool schemas are rendered INTO the prompt, so a prime without them prefills a token
    # sequence the following turn does not share. Measured 2026-10-26 10:01:38 → 10:03:05: the
    # prime was 31,050 tokens, the operator's turn 38,983 with `cached 6,592` — only the system
    # prompt overlapped, TTFT 13.9s where a warm turn-to-turn call gets 0.9s (splash log). The
    # live call sends `tools=agent.tools` (agent/chat_completion_helpers.py:1528), so must this.
    if tools:
        kwargs["tools"] = tools
    try:
        from agent.auxiliary_client import get_auxiliary_extra_body

        extra = get_auxiliary_extra_body()
        if extra:
            kwargs["extra_body"] = extra
    except Exception:  # noqa: BLE001 - extra_body is provider decoration, never the point
        pass
    response = client.chat.completions.create(**kwargs)
    return _usage_of(response)


def _usage_of(response: Any) -> dict[str, Any]:
    """``{prompt, cached}`` from a completion's usage block, tolerant of provider shape drift.

    oMLX answers a non-streaming call with `usage.prompt_tokens_details.cached_tokens` — measured
    on 2026-10-26: a cold 46,060-token prime reported `cached_tokens: 0`. That field, not timing,
    is the authoritative reuse count.
    """
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {}

    def _get(obj: Any, key: str) -> Any:
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    details = _get(usage, "prompt_tokens_details") or {}
    cached = _get(details, "cached_tokens")
    if cached is None:
        cached = _get(usage, "cache_read_input_tokens")
    return {"prompt": _get(usage, "prompt_tokens"), "cached": cached}


def prime(agent: Any, messages: list, system_prompt: Optional[str], *, opts: dict[str, Any],
          sender: Optional[Callable[..., dict[str, Any]]] = None,
          sleep: Callable[[float], None] = time.sleep, block: bool = False) -> str:
    """Warm the next turn's prefix on a daemon thread. Returns the decision, never raises.

    ``block=True`` runs the sequence inline instead of on a thread, for tests and for a
    caller that wants the decision rather than the fire-and-forget.
    """
    sender = sender or send_prime
    send: Callable[..., dict[str, Any]] = sender
    baseline = len(messages)
    started = time.perf_counter()

    def _run() -> None:
        try:
            delay = float(opts["warm_after_compaction_delay_s"])
            if delay > 0:
                sleep(delay)
            stale = _looks_stale(agent, baseline)
            if stale:
                PRIMES["skipped"] += 1
                logger.info("vcc: prefix prime skipped (%s)", _note(stale))
                return
            session_model = getattr(agent, "model", "") or getattr(agent, "_model", "")
            wire, payload_route = wire_payload(agent, messages, system_prompt)
            tools = getattr(agent, "tools", None) or None
            head_sha, payload_chars, tool_count = _fingerprint(wire, tools)
            # Free measurements go first, before any gate: a run that cannot warm the cache must
            # still report what the payload was and what the cache did with it afterwards.
            counters = _cache_counters(opts, model=str(session_model))
            threading.Thread(
                target=_delta_watch, name="vcc-cache-window", daemon=True,
                args=(opts, counters, float(opts.get("warm_after_compaction_delta_wait_s", 90)),
                      str(session_model))).start()
            if opts.get("warm_after_compaction_dry_run", False):
                # Measure, don't pay: the warm-up costs 12-21 s of prefill and landed on 1 of 7
                # compactions on 2026-10-26, so nothing reaches the engine until the numbers say so.
                PRIMES["skipped"] += 1
                logger.info("vcc: prime dry-run (no warm request sent): %d wire messages / %d chars, "
                            "head_sha=%s, tools=%d, payload from %s, session model %s; counters %s",
                            len(wire), payload_chars, head_sha, tool_count, payload_route,
                            session_model, counters)
                _note(f"dry-run: {len(wire)} messages / {payload_chars} chars head_sha={head_sha}")
                return
            busy = _engine_busy(opts)
            attempts = 1
            max_attempts = max(1, int(opts.get("warm_after_compaction_idle_attempts", 1)))
            while busy and attempts < max_attempts and not _looks_stale(agent, baseline):
                # Waiting for the idle gap is the point, not a courtesy: measured 2026-10-26 at
                # 15:22, three calls at 128-131k tokens went out on the *old* context after
                # "compression done" and the first call carrying the 42-message handoff arrived
                # 70 s later. Warming during the old turn both wastes the prefill and splashes.
                sleep(float(opts.get("warm_after_compaction_retry_delay_s", 10.0)))
                moved_on = _looks_stale(agent, baseline)
                if moved_on:
                    # A turn started during the wait: whatever we would warm is no longer the
                    # prefix the next call will send, and the engine is busy again anyway.
                    PRIMES["skipped"] += 1
                    logger.info("vcc: prefix prime skipped (%s, while waiting for idle)", _note(moved_on))
                    return
                busy = _engine_busy(opts)
                attempts += 1
            if busy:
                PRIMES["skipped"] += 1
                logger.info("vcc: prefix prime skipped (%s, after %d idle probe(s))",
                            _note(busy), attempts)
                return
            client, model, route = _aux_route(str(session_model))
            aux_why = "" if client is not None else route  # route slot carries the refusal reason
            if client is None:
                # No configured compression task on this build: the agent's own client is the
                # only route left, lifecycle risk and all (see _session_route).
                client, model, route = _session_route(agent)
            if client is None:
                PRIMES["skipped"] += 1
                why = aux_why or "no client reachable"
                logger.info("vcc: prefix prime skipped (%s)", why)
                _note(f"skipped: {why}")
                return
            usage = send(opts, wire, model, client=client, tools=tools) or {}
            PRIMES["sent"] += 1
            elapsed = (time.perf_counter() - started) * 1000
            logger.info("vcc: prefix primed: %d wire messages / %d chars (head_sha=%s, tools=%d) via "
                        "%s on %s in %.0f ms; warm request itself: prompt=%s cached=%s tokens",
                        len(wire), payload_chars, head_sha, tool_count,
                        f"{route}+{payload_route}", model, elapsed,
                        usage.get("prompt"), usage.get("cached"))
            _note(f"primed {len(wire)} messages / {payload_chars} chars via {route}+{payload_route}")
            threading.Thread(target=_delta_watch, name="vcc-cache-window", daemon=True,
                             args=(opts, counters,
                                   float(opts.get("warm_after_compaction_delta_wait_s", 90)),
                                   str(model))).start()
            # A prefill only helps if the operator's next message arrives after it. Re-checking
            # afterwards catches the case that cost latency on this box at 09:09:57: the prime was
            # still prefilling 41k tokens when the turn landed, so the turn queued behind it
            # (20.95 s time-to-first-token, against 15.06 s with no prime at all).
            overlapped = _looks_stale(agent, baseline)
            if overlapped:
                logger.warning("vcc: prefix prime overlapped a live turn (%s after %0.f ms of prefill); "
                               "the next message queued behind it rather than ahead of it",
                               _note(overlapped), elapsed)
        except Exception as exc:  # noqa: BLE001 - priming is a bonus, never a dependency
            PRIMES["errors"] += 1
            logger.warning("vcc: prefix prime failed (%s: %s); the next turn prefills cold",
                           type(exc).__name__, exc)
            _note(f"error: {type(exc).__name__}: {exc}")

    if block:
        _run()
        return PRIMES["last"]
    thread = threading.Thread(target=_run, name="vcc-prefix-prime", daemon=True)
    thread.start()
    return _note(f"scheduled {baseline} messages")


def install(opts: dict[str, Any]) -> None:
    """Wrap `conversation_compression.compress_context` once per process. Fail-open.

    The load-bearing detail: `compression_facade.py` imports `compress_context` *inside* the
    calling function, so replacing the module attribute is picked up on the very next call.
    Any exception inside the wrapper leaves compaction exactly as Hermes wrote it.
    """
    if _INSTALLED["wrapper"] is not None or not opts.get("warm_after_compaction", True):
        return
    import agent.conversation_compression as cc

    original = getattr(cc, "compress_context", None)
    if not callable(original):
        logger.info("vcc: prefix priming unavailable (no compress_context in this Hermes build)")
        _note("unavailable: no compress_context")
        return
    if getattr(original, "__vcc_prime_wrap__", False):
        _INSTALLED["wrapper"] = original
        return

    def wrapper(agent: Any, messages: list, system_message: str, *args: Any, **kwargs: Any):
        result = original(agent, messages, system_message, *args, **kwargs)
        try:
            if isinstance(result, tuple) and len(result) == 2:
                new_messages, new_system = result
                if isinstance(new_messages, list) and len(new_messages) < len(messages):
                    prime(agent, new_messages, new_system, opts=opts)
        except Exception as exc:  # noqa: BLE001
            PRIMES["errors"] += 1
            logger.warning("vcc: prime scheduling skipped (%s: %s)", type(exc).__name__, exc)
        return result

    wrapper.__name__ = getattr(original, "__name__", "compress_context")
    wrapper.__doc__ = getattr(original, "__doc__", "")
    setattr(wrapper, "__vcc_prime_wrap__", True)
    cc.compress_context = wrapper
    _INSTALLED.update({"wrapper": wrapper, "original": original})
    PRIMES["installs"] += 1
    logger.info("vcc: post-compaction prefix priming installed (delay %ss, max_tokens %s, dry_run %s)",
                opts["warm_after_compaction_delay_s"], opts["warm_after_compaction_max_tokens"],
                opts.get("warm_after_compaction_dry_run", False))


def uninstall() -> None:
    """Restore the wrapped function (tests, and a config flip to false)."""
    if _INSTALLED["original"] is not None:
        import agent.conversation_compression as cc

        cc.compress_context = _INSTALLED["original"]
    _INSTALLED.update({"wrapper": None, "original": None})
