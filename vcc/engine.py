"""``vcc`` context engine: Hermes's own compressor with the summary LLM call swapped out.

Everything that is hard to get right stays Hermes's code: when to compact, which turns to
keep, tool-call/result pairing, the session-DB archive commit, cancellation and secret
redaction. The one thing replaced is ``_generate_summary``, the step that sends the
compacted turns to a model and waits for prose. Here it is a deterministic projection
that takes milliseconds (see ``projector.py``).

If this Hermes build doesn't look like the one the engine was written against, or the
projection fails for any reason, it hands the call back to the stock implementation.
Worst case you get the normal LLM summary, never a broken session.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

from agent.auxiliary_client import (
    aux_interrupt_protection,
    auxiliary_max_tokens_param,
    call_llm,
)
from agent.context_compressor import ContextCompressor

try:  # loaded as a package (Hermes plugin loader)
    from . import anchors, projector
except ImportError:  # loaded as a top-level module (tests, tooling)
    import anchors  # type: ignore[no-redef]
    import projector  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

ENGINE_NAME = "vcc"

# Base-class pieces the override reuses. Checked once at construction; any missing one
# switches the engine to passthrough (stock LLM summaries) with a single warning.
_REQUIRED_METHODS = (
    "_generate_summary", "_compression_cancelled", "_ground_historical_task_snapshot",
    "_augment_summary_lean", "_validate_summary_user_provenance", "_with_summary_prefix",
    "_transcript_has_real_user_turn", "_is_synthetic_compression_user_turn",
    "_clear_compression_failure_cooldown", "_clear_terminal_summary_failures",
    "_reset_consecutive_overload_aborts", "_compute_summary_budget", "update_model",
)
_REQUIRED_FUNCS = (
    "_redact_compaction_text", "_reinject_pruned_skill_markers", "_collect_ghosted_skill_names",
    "_extract_pruned_skill_names", "_MAX_PRUNED_SKILL_MARKERS",
)

# Option defaults (``context.vcc`` in config.yaml).
_DEFAULTS: dict[str, Any] = {
    "budget_ratio": 0.06,        # summary budget as a share of the context window
    "min_budget_tokens": 1500,
    "max_budget_tokens": 8000,
    "fallback_to_llm": True,     # on an unexpected projection error, use the stock summary
    "fast_anchors": True,        # build Hermes's anchor index with anchors.py (same output, faster)
    "record_trigger": True,      # say in the handoff what asked for it (manual /compress vs automatic)
    "preserve_conclusions": False,  # hybrid leg: ONE bounded LLM call that keeps the agent's conclusions
    "conclusions_input_tokens": 2500,   # cap on the assistant prose handed to that call
    "conclusions_row_tokens": 1000,     # per-reply cap inside that window; prose-sized, not excerpt-sized
    "conclusions_output_tokens": 900,   # cap on the digest it may write back
    "conclusions_wire_tokens": 1600,    # max_tokens for the leg's OWN call; 0 = uncapped
    "conclusions_min_prose_rows": 3,    # below this the leg writes reconstruction, not digestion
    "conclusions_min_input_chars": 1200,  # and below this in characters, the same
    "conclusions_profile": "coding",  # profile forced on that call; "" = inherit the session's
    "warm_after_compaction": True,      # prime the new prefix so the next call is not cold
    "warm_after_compaction_delay_s": 1.0,  # grace period: a user turn already in flight wins
    "warm_after_compaction_max_tokens": 1,  # a prefill, not an answer
    "warm_after_compaction_idle_probe": "",  # engine admin URL to ask whether a turn is already queued
    "warm_after_compaction_idle_probe_key": "",  # bearer for it; "" = env OMLX_ADMIN_KEY or ~/.omlx/settings.json
    "warm_after_compaction_probe_timeout_s": 0.5,
    "projection_char_caps": {},
}
_CHARS_PER_TOKEN = 4  # same rough ratio Hermes's own estimator uses
_CONCLUSIONS_MIN_OUT_TOKENS = 400  # floor so a legitimate small window still gets a real section

# anchors.build() must reproduce Hermes's anchor index byte for byte. It is checked against
# Hermes's builder once per process on an edge-case fixture and again on the first real
# compaction; any disagreement or error turns it off for the rest of the process.
_FAST_ANCHORS: dict[str, Any] = {"fixture_ok": None, "live_ok": False, "disabled": ""}


def _disable_fast_anchors(reason: str) -> None:
    if not _FAST_ANCHORS["disabled"]:
        _FAST_ANCHORS["disabled"] = reason
        logger.warning("vcc: fast anchor index off (%s); Hermes builds it as usual", reason)


def _load_hermes_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        return cfg if isinstance(cfg, Mapping) else {}
    except Exception as exc:  # config is optional; defaults are fine
        logger.debug("vcc: could not read Hermes config (%s); using defaults", exc)
        return {}


def _section(cfg: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    node: Any = cfg
    for key in path:
        node = node.get(key) if isinstance(node, Mapping) else None
    return node if isinstance(node, Mapping) else {}


def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(hi, max(lo, number))


def _compressor_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Map ``compression:`` to ContextCompressor kwargs.

    Hermes doesn't pass its compression settings to plugin engines, so read the same keys
    it reads for the built-in compressor and default them the same way.
    """
    comp = _section(cfg, "compression")
    cap = comp.get("threshold_tokens")
    try:
        cap = int(cap) if cap is not None and int(cap) > 0 else None
    except (TypeError, ValueError):
        cap = None
    thresholds = {}
    for key, value in _section(comp, "model_thresholds").items():
        try:
            thresholds[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return {
        "threshold_percent": _num(comp.get("threshold", 0.75), 0.75, 0.05, 0.98),
        "protect_first_n": int(_num(comp.get("protect_first_n", 3), 3, 0, 50)),
        "protect_last_n": int(_num(comp.get("protect_last_n", 20), 20, 1, 500)),
        "summary_target_ratio": _num(comp.get("target_ratio", 0.20), 0.20, 0.10, 0.80),
        "abort_on_summary_failure": bool(comp.get("abort_on_summary_failure", False)),
        "model_thresholds": thresholds,
        "threshold_tokens_cap": cap,
        "min_tail_user_messages": int(_num(comp.get("min_tail_user_messages", 1), 1, 1, 20)),
        "tail_mode": str(comp.get("tail_mode", "lean")).strip().lower() or "lean",
    }


def _engine_options(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _section(cfg, "context", "vcc")
    return {
        "budget_ratio": _num(raw.get("budget_ratio", _DEFAULTS["budget_ratio"]), _DEFAULTS["budget_ratio"], 0.01, 0.30),
        "min_budget_tokens": int(_num(raw.get("min_budget_tokens", _DEFAULTS["min_budget_tokens"]),
                                      _DEFAULTS["min_budget_tokens"], 300, 50_000)),
        "max_budget_tokens": int(_num(raw.get("max_budget_tokens", _DEFAULTS["max_budget_tokens"]),
                                      _DEFAULTS["max_budget_tokens"], 500, 100_000)),
        "fallback_to_llm": bool(raw.get("fallback_to_llm", _DEFAULTS["fallback_to_llm"])),
        "fast_anchors": bool(raw.get("fast_anchors", _DEFAULTS["fast_anchors"])),
        "record_trigger": bool(raw.get("record_trigger", _DEFAULTS["record_trigger"])),
        "preserve_conclusions": bool(raw.get("preserve_conclusions", _DEFAULTS["preserve_conclusions"])),
        "conclusions_input_tokens": int(_num(raw.get("conclusions_input_tokens", _DEFAULTS["conclusions_input_tokens"]),
                                             _DEFAULTS["conclusions_input_tokens"], 500, 60_000)),
        "conclusions_row_tokens": int(_num(raw.get("conclusions_row_tokens", _DEFAULTS["conclusions_row_tokens"]),
                                           _DEFAULTS["conclusions_row_tokens"], 100, 20_000)),
        "conclusions_output_tokens": int(_num(raw.get("conclusions_output_tokens", _DEFAULTS["conclusions_output_tokens"]),
                                              _DEFAULTS["conclusions_output_tokens"], 100, 4000)),
        "conclusions_wire_tokens": int(_num(raw.get("conclusions_wire_tokens", _DEFAULTS["conclusions_wire_tokens"]),
                                            _DEFAULTS["conclusions_wire_tokens"], 0, 8000)),
        "conclusions_min_prose_rows": int(_num(raw.get("conclusions_min_prose_rows", _DEFAULTS["conclusions_min_prose_rows"]),
                                               _DEFAULTS["conclusions_min_prose_rows"], 1, 50)),
        "conclusions_min_input_chars": int(_num(raw.get("conclusions_min_input_chars", _DEFAULTS["conclusions_min_input_chars"]),
                                                _DEFAULTS["conclusions_min_input_chars"], 0, 100_000)),
        "projection_char_caps": (dict(raw["projection_char_caps"])
                                 if isinstance(raw.get("projection_char_caps"), dict) else {}),
        "conclusions_profile": str(raw.get("conclusions_profile", _DEFAULTS["conclusions_profile"]) or "").strip(),
        "warm_after_compaction": bool(raw.get("warm_after_compaction", _DEFAULTS["warm_after_compaction"])),
        "warm_after_compaction_delay_s": _num(raw.get("warm_after_compaction_delay_s",
                                                     _DEFAULTS["warm_after_compaction_delay_s"]),
                                              _DEFAULTS["warm_after_compaction_delay_s"], 0.0, 30.0),
        "warm_after_compaction_max_tokens": int(_num(raw.get("warm_after_compaction_max_tokens",
                                                            _DEFAULTS["warm_after_compaction_max_tokens"]),
                                                     _DEFAULTS["warm_after_compaction_max_tokens"], 1, 16)),
        "warm_after_compaction_idle_probe": str(
            raw.get("warm_after_compaction_idle_probe", _DEFAULTS["warm_after_compaction_idle_probe"]) or "").strip(),
        "warm_after_compaction_idle_probe_key": str(
            raw.get("warm_after_compaction_idle_probe_key",
                    _DEFAULTS["warm_after_compaction_idle_probe_key"]) or "").strip(),
        "warm_after_compaction_probe_timeout_s": _num(
            raw.get("warm_after_compaction_probe_timeout_s", _DEFAULTS["warm_after_compaction_probe_timeout_s"]),
            _DEFAULTS["warm_after_compaction_probe_timeout_s"], 0.05, 10.0),
    }


def _missing_seams() -> list[str]:
    import agent.context_compressor as cc
    missing = [m for m in _REQUIRED_METHODS if not callable(getattr(ContextCompressor, m, None))]
    missing += [f for f in _REQUIRED_FUNCS if not hasattr(cc, f)]
    try:
        from agent.conversation_compression import _raise_if_stale_attempt  # noqa: F401
    except ImportError:
        missing.append("conversation_compression._raise_if_stale_attempt")
    return missing


def _conclusions_prompt(material: list[dict[str, Any]], keep_tokens: int) -> str:
    """The leg's prompt: bullets, most-important-first, no section skeleton.

    Hermes' summariser prompt is built for the one summary that IS the handoff and always asks
    for its full skeleton, so a thin window still comes back long. This leg writes ONE section
    of a handoff that already carries the record, so it asks for exactly that.
    """
    turns = "\n\n".join(f"[{i}] {str(row.get('content') or '').strip()}"
                        for i, row in enumerate(material, 1))
    return (
        "Write the body of the 'What the Agent Concluded' section of a context handoff that "
        "replaces a compacted conversation. The input is the assistant's own prose from the "
        "folded window, oldest first.\n\n"
        f"Digest it as terse markdown bullets, most important FIRST, in at most {keep_tokens} "
        "tokens. Hard rules:\n"
        "- Conclusions, decisions and rationale — not a timeline of what was read.\n"
        "- Keep exact values verbatim: numbers, IDs, flags, config keys, paths, model names, timings.\n"
        "- Keep every user decision, constraint and correction, in their words or tighter.\n"
        "- Keep the last uncompleted step, and anything still blocked.\n"
        "- Bullets only: no heading, no preamble, no closing remarks, no meta-commentary.\n"
        "- Never invent: if it is not in the input, leave it out.\n\n"
        f"<turns>\n{turns}\n</turns>\n"
    )


def _digest_text(response: Any) -> tuple[str, Optional[str]]:
    """``(content, finish_reason)`` from a chat completion, dict- or object-shaped."""
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    try:
        choice = (_get(response, "choices") or [None])[0]
        message = _get(choice, "message") or {}
        text = _get(message, "content") or ""
        if isinstance(text, list):  # routes that return content parts
            text = "".join(str(p.get("text") or "") if isinstance(p, dict) else str(p) for p in text)
        text = str(text or "")
        if not text:
            logger.warning("vcc: conclusions response carried no text (shape=%s, finish=%s)",
                           type(response).__name__, _get(choice, "finish_reason"))
        return text, _get(choice, "finish_reason")
    except Exception as exc:
        logger.warning("vcc: could not read the conclusions response (%s: %s); shape=%s",
                       type(exc).__name__, exc, type(response).__name__)
        return "", None


class VccEngine(ContextCompressor):
    """ContextCompressor whose summaries are deterministic projections instead of LLM output."""

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None, **overrides: Any) -> None:
        cfg = _load_hermes_config() if config is None else config
        kwargs = _compressor_kwargs(cfg)
        kwargs.update(overrides)
        kwargs.setdefault("quiet_mode", True)
        super().__init__(model=kwargs.pop("model", ""), **kwargs)
        self._vcc_config = cfg
        self._vcc_options = _engine_options(cfg)
        self._vcc_overrides = dict(overrides)
        try:  # priming is a bonus path: a failure here must never take the engine down
            try:
                from . import priming
            except ImportError:  # flat sys.path load (tests, one-off tooling): same module, absolute name
                import priming  # type: ignore[no-redef]
            priming.install(self._vcc_options)
        except Exception as exc:  # noqa: BLE001
            logger.debug("vcc: prefix priming not installed (%s: %s)", type(exc).__name__, exc)
        self._vcc_last_ms: Optional[float] = None
        self._vcc_runs = 0
        self._vcc_fallbacks = 0
        # What asked for the compaction now in flight (set by compress(), read by _generate_summary).
        self._vcc_trigger: Optional[dict[str, Any]] = None
        missing = _missing_seams()
        self._vcc_passthrough = bool(missing)
        if missing:
            logger.warning(
                "vcc: this Hermes build lacks %s; using stock LLM summaries until the plugin is updated",
                ", ".join(missing),
            )

    @property
    def name(self) -> str:
        return ENGINE_NAME

    # The summariser model never runs on this engine's normal path, so its context window
    # must not cap our trigger (the session-start feasibility probe installs it).
    @property
    def _aux_context_ceiling(self) -> Optional[int]:
        return None

    @_aux_context_ceiling.setter
    def _aux_context_ceiling(self, _value: Optional[int]) -> None:
        pass

    def clone_for_agent(self) -> "VccEngine":
        # A fresh instance per agent: no shared mutable state, nothing deepcopy could trip on.
        return type(self)(config=self._vcc_config, **self._vcc_overrides)

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status.update({
            "vcc_passthrough": self._vcc_passthrough,
            "vcc_runs": self._vcc_runs,
            "vcc_fallbacks": self._vcc_fallbacks,
            "vcc_last_ms": self._vcc_last_ms,
            "vcc_fast_anchors": (
                "off (config)" if not self._vcc_options["fast_anchors"]
                else f"off ({_FAST_ANCHORS['disabled']})" if _FAST_ANCHORS["disabled"]
                else "on" if _FAST_ANCHORS["live_ok"] else "pending first check"
            ),
        })
        return status

    def _vcc_fast_anchor_section(self, cc: Any, summary: str, turns: list[dict[str, Any]]) -> Optional[str]:
        """Hermes's anchor section for ``turns``, built faster. ``None`` leaves it to Hermes."""
        state = _FAST_ANCHORS
        if not self._vcc_options["fast_anchors"] or state["disabled"]:
            return None
        if getattr(self, "tail_mode", "lean") != "lean":
            return None  # Hermes adds no lean sections in legacy mode
        try:
            if state["fixture_ok"] is None:
                state["fixture_ok"] = anchors.verify(cc)
                if not state["fixture_ok"]:
                    _disable_fast_anchors("self-check against Hermes's builder failed")
                    return None
            if cc._LEAN_ANCHOR_HEADING in summary:
                return None  # Hermes would skip the section, so must we
            fast = anchors.build(turns, cc)
            if not state["live_ok"]:
                if fast != cc._build_anchor_index(turns):
                    _disable_fast_anchors("differed from Hermes's builder on a real compaction")
                    return None
                state["live_ok"] = True
            return fast or None
        except Exception as exc:  # the optimization must never be the reason a compaction fails
            _disable_fast_anchors(f"{type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------ hybrid additions

    def compress(
        self, messages: list[dict[str, Any]], current_tokens: Optional[int] = None, focus_topic: Optional[str] = None,
        force: bool = False, memory_context: str = "", bypass_cooldown: bool = False,
    ) -> list[dict[str, Any]]:
        """Note what asked for this compaction, then defer to Hermes.

        Manual ``/compress`` reaches an engine as ``force=True`` (``compress_now`` →
        ``agent._compress_context``); an automatic pass reaches it as ``force=False``.
        ``focus_topic`` cannot tell them apart — core fills it with an auto-derived topic when
        the user gave none. The signature mirrors ``ContextCompressor.compress`` exactly,
        because core filters engine keywords by signature.
        """
        self._vcc_trigger = {
            "manual": bool(force),
            "focus": (focus_topic or "").strip() or None,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            return super().compress(
                messages, current_tokens=current_tokens, focus_topic=focus_topic, force=force,
                memory_context=memory_context, bypass_cooldown=bypass_cooldown)
        finally:
            self._vcc_trigger = None

    def _vcc_trigger_section(self, used_llm: bool) -> str:
        """Two lines naming the trigger — something the transcript itself cannot show.

        Deliberately no elapsed-time figure here: a handoff must be byte-reproducible for a
        given compaction (Hermes's own tests compare projections for equality), and the
        duration is already in the log line and in ``get_status()['vcc_last_ms']``.
        """
        trigger = self._vcc_trigger or {}
        when = trigger.get("at") or "unknown time"
        if trigger.get("manual"):
            focus = f', focused on "{trigger["focus"]}"' if trigger.get("focus") else ""
            first = (f"The user ran `/compress` at {when}{focus}. That command is not part of the "
                     "transcript above: this handoff replaced the turns it folded away.")
        else:
            first = f"Automatic: the context crossed the compaction threshold at {when}."
        how = " with one bounded summary-model call" if used_llm else " with no LLM call"
        return f"{projector.TRIGGER_HEADING}\n{first}\nBuilt by the {ENGINE_NAME} engine{how}."

    def _vcc_summary_route(self, profile: str) -> Optional[dict[str, Any]]:
        """Route the conclusions leg at ``profile`` of the session's OWN model; None means inherit.

        oMLX profiles carry per-mode sampling: ``:fiction`` runs a creative temperature and template
        that reads a context digest like a novelist wrote it. The leg inherits the session model, so
        an authoring session would otherwise summarise its own working context in fiction voice. Only
        the PROFILE SUFFIX is rewritten here — same weights, same server, same key, no second model
        loaded. A profile the server does not know fails the leg, and the deterministic handoff ships
        without the section, which is the same degradation as any other leg failure.
        """
        model = str(getattr(self, "model", "") or "")
        # A profile only exists as a suffix (oMLX's convention, see /v1/models: `id:fiction`,
        # `id:coding`). A bare provider model id has no per-mode sampling to protect against, and
        # inventing `gpt-5.1-codex:coding` would fail the call for nothing — so no suffix, no pin.
        if not model or not profile or ":" not in model:
            return None
        base = model.rsplit(":", 1)[0]
        target = f"{base}:{profile}"
        if target == model:
            return None  # already there; let core's normal route resolution stand
        route: dict[str, Any] = {"label": f"vcc conclusions ({profile} profile)", "model": target}
        for field in ("provider", "base_url", "api_key", "api_mode"):
            value = getattr(self, field, None)
            if value:
                route[field] = value
        return route

    def _vcc_digest_call(self, material: list[dict[str, Any]], opts: dict[str, Any]) -> str:
        """The leg's own single call: its own prompt, its own ``max_tokens``.

        Deliberately NOT ``_generate_summary``. That path writes the ONE summary that IS the
        handoff, so it asks for a fixed section skeleton — which is why its output tracks the
        template rather than the material (measured 10:30:27: 5,739 chars of prose in, **4,434
        tokens out**, 32.8 s of a 39.66 s leg decoding, and only 27% of it survived the keep-cap;
        the run before wrote 2,837 from *more* input). Core also refuses a wire cap there —
        "NO max_tokens: … a hard cap truncates summaries" (``_call_summary_llm``) — because a
        length-stop in a whole-handoff summary really is a partial memory.

        This section earns its cap: the result is trimmed to ``conclusions_output_tokens`` the
        moment it lands, and the prompt puts the most important content first, so stopping at the
        cap can only cut the tail. ``conclusions_wire_tokens`` sits ~1.3x above the keep-cap so
        the last bullet still finishes; ``0`` restores the uncapped behaviour.
        """
        route = self._vcc_summary_route(opts["conclusions_profile"]) or {}
        keep = int(opts["conclusions_output_tokens"])
        cap = int(opts["conclusions_wire_tokens"] or 0)
        kwargs: dict[str, Any] = {
            "task": "compression",
            "main_runtime": {
                "model": route.get("model") or self.model,
                "provider": route.get("provider") or self.provider,
                "base_url": route.get("base_url") or self.base_url,
                "api_key": route.get("api_key") or self.api_key,
                "api_mode": route.get("api_mode") or self.api_mode,
            },
            "messages": [{"role": "user", "content": _conclusions_prompt(material, keep)}],
            "route_info": {},
        }
        if cap > 0:
            # `max_tokens=` alone does NOT reach the wire on a local custom route: the aux client
            # forwards an explicit cap only on the routes it trusts it on (Anthropic/NIM/Gemini/
            # OpenRouter/managed llama-server) and drops it elsewhere — measured, the 1,600 I
            # passed to a 127.0.0.1:8000 call arrived as `request_max_tokens=None` and the model
            # wrote 9,543 tokens. extra_body is merged into the request body verbatim, and
            # auxiliary_max_tokens_param picks the key this wire accepts.
            kwargs["max_tokens"] = cap
            body = auxiliary_max_tokens_param(cap, model=kwargs["main_runtime"]["model"])
            # A cap alone is not enough for a thinking model: measured 10:47, it spent all 1,600
            # tokens on reasoning and returned `finish_reason=length` with an EMPTY message.content
            # — a fast leg that ships no section. Same trap core warns about ("thinking models burn
            # it on reasoning"). Turn thinking off for this call: `reasoning` is the key Hermes'
            # own compression fast lane uses on these routes, and `chat_template_kwargs` is where
            # oMLX-style Qwen servers actually read it (a top-level reasoning_effort is ignored).
            body["reasoning"] = {"effort": "none"}
            body["chat_template_kwargs"] = {"reasoning_effort": "none"}
            kwargs["extra_body"] = body
        with aux_interrupt_protection():  # compaction is atomic: an incoming message must not abort the leg (#23975)
            response = call_llm(**kwargs)
        text, finish = _digest_text(response)
        if finish == "length":
            logger.info("vcc: conclusions digest stopped at its %d-token wire cap (keep-cap %d) — the "
                        "trim keeps the head, which is what the prompt puts first", cap, keep)
        return text

    def _vcc_conclusions(
        self, turns_to_summarize: list[dict[str, Any]], focus_topic: Optional[str],
        memory_context: str, bypass_cooldown: bool,
    ) -> Optional[str]:
        """The hybrid leg: ONE bounded model call over the agent's own prose.

        The projection keeps what *happened*; only a model can keep what the agent *concluded*.
        Tool-result bodies are excluded outright and the input is capped, so this reads a
        fraction of what a full-context summary reads. Any failure, or an empty answer, simply
        means the handoff ships without the section: the deterministic core never depends on it.
        """
        opts = self._vcc_options
        if not opts["preserve_conclusions"] or self._vcc_passthrough:
            return None
        material = projector.conclusions_material(
            turns_to_summarize, budget_chars=opts["conclusions_input_tokens"] * _CHARS_PER_TOKEN,
            row_limit=opts["conclusions_row_tokens"] * _CHARS_PER_TOKEN)
        if not material:
            return None
        input_chars = sum(len(row["content"]) for row in material)
        # The stock summariser prompt always asks for its full ten-section skeleton, so a thin
        # window does not buy a thin digest: two truncated excerpts still come back as ~9k chars
        # of confident reconstruction. Below this much real prose the section would be invented
        # rather than preserved, so decline the call and ship the deterministic handoff.
        if (len(material) < opts["conclusions_min_prose_rows"]
                or input_chars < opts["conclusions_min_input_chars"]):
            logger.info("vcc: conclusions leg skipped (%d prose rows / %d chars in the compacted "
                        "window; needs %d rows / %d chars)", len(material), input_chars,
                        opts["conclusions_min_prose_rows"], opts["conclusions_min_input_chars"])
            return None
        # Hermes records a summary-model failure here; the structural fallback it inserts in the
        # model's place is recorded the same way (see _generate_summary's stall path).
        self._last_summary_error = None
        started = time.perf_counter()
        try:
            digest = self._vcc_digest_call(material, opts)
        except Exception as exc:
            logger.warning("vcc: conclusions leg failed (%s: %s); shipping the deterministic handoff",
                           type(exc).__name__, exc)
            return None
        if not digest or getattr(self, "_summary_model_fallen_back", False):
            return None
        # A refused or stalled summariser (an oMLX 507 memory ceiling, a timeout) makes Hermes
        # insert a STRUCTURAL fallback summary. That restates the transcript we already projected
        # deterministically, so shipping it as a "conclusion" buys nothing and charges the whole
        # leg's latency for it. Refuse it, and let the trigger line say "no LLM call" honestly.
        failure = str(getattr(self, "_last_summary_error", "") or "")
        if "deterministic fallback summary inserted" in failure or "summary model" in failure.lower():
            logger.info("vcc: conclusions leg produced no model output (%s); skipping the section",
                        failure[:160])
            return None
        import agent.context_compressor as cc
        # Digestion compresses; it must not inflate. Tying the allowance to the prose actually
        # read (with a floor for a legitimately small window) is what stops a thin input from
        # buying a long section written to fill the stock prompt's fixed skeleton.
        out_chars = min(opts["conclusions_output_tokens"] * _CHARS_PER_TOKEN,
                        max(_CONCLUSIONS_MIN_OUT_TOKENS * _CHARS_PER_TOKEN, input_chars))
        # The stock summariser returns a WHOLE handoff, handoff banner included; ours is one
        # section of a handoff, so the banner has to go or the result pays ~700 chars to say
        # the same thing twice.
        body = projector.clean_multiline(cc.ContextCompressor._strip_summary_prefix(str(digest).strip()),
                               out_chars)
        if not body.strip():
            return None
        logger.info("vcc: conclusions digest: %d prose turns / %d chars in -> %d chars out (%.1fx) in %.0f ms",
                    len(material), input_chars, len(body),
                    (len(body) / input_chars) if input_chars else 0.0,
                    (time.perf_counter() - started) * 1000)
        return f"{projector.CONCLUSIONS_HEADING}\n{body}"

    # ------------------------------------------------------------------ summary override

    def _vcc_budget_chars(self, turns: list[dict[str, Any]]) -> int:
        opts = self._vcc_options
        window = int(getattr(self, "context_length", 0) or 0)
        by_window = int(window * opts["budget_ratio"]) if window > 0 else opts["max_budget_tokens"]
        by_content = self._compute_summary_budget(turns)  # Hermes's own content-scaled budget
        tokens = min(by_window, by_content, opts["max_budget_tokens"])
        return max(opts["min_budget_tokens"], tokens) * _CHARS_PER_TOKEN

    def _vcc_is_real_user(self, message: Mapping[str, Any]) -> bool:
        try:
            return not self._is_synthetic_compression_user_turn(message)
        except Exception:
            return True

    def _generate_summary(
        self,
        turns_to_summarize: list[dict[str, Any]],
        focus_topic: Optional[str] = None,
        memory_context: str = "",
        bypass_cooldown: bool = False,
    ) -> Optional[str]:
        if self._vcc_passthrough:
            return super()._generate_summary(turns_to_summarize, focus_topic, memory_context, bypass_cooldown)

        import agent.context_compressor as cc
        from agent.auxiliary_client import AuxiliaryExplicitCancellation
        from agent.conversation_compression import _raise_if_stale_attempt

        if self._compression_cancelled():
            raise AuxiliaryExplicitCancellation()
        started = time.perf_counter()
        try:
            previous = self._previous_summary or ""
            pruned_skills = list(dict.fromkeys(
                cc._collect_ghosted_skill_names(turns_to_summarize)
                + cc._extract_pruned_skill_names(previous)
            ))[: cc._MAX_PRUNED_SKILL_MARKERS]
            has_user_turn = getattr(self, "_summary_has_user_turn", None)
            if has_user_turn is None:
                has_user_turn = self._transcript_has_real_user_turn(turns_to_summarize)

            body = projector.project(
                turns_to_summarize,
                budget_chars=self._vcc_budget_chars(turns_to_summarize),
                session_id=getattr(self, "_session_id", "") or "",
                previous=cc._redact_compaction_text(previous) if previous else None,
                is_real_user=self._vcc_is_real_user,
                focus=focus_topic,
                limits=projector.limits_from_opts(self._vcc_options),
            )
            if not has_user_turn:
                body = _force_no_user_task(body)

            # Same post-processing, same order, as ContextCompressor._generate_summary.
            summary = cc._redact_compaction_text(body)
            summary = cc._reinject_pruned_skill_markers(summary, pruned_skills)
            summary = self._ground_historical_task_snapshot(summary, turns_to_summarize)
            anchor_section = self._vcc_fast_anchor_section(cc, summary, turns_to_summarize)
            if anchor_section:
                # _augment_summary_lean skips any section whose heading is already present, so
                # this replaces only Hermes's slow anchor scan; the rest is still its own code.
                summary += cc._redact_compaction_text(anchor_section)
            summary = self._augment_summary_lean(summary, turns_to_summarize)
            self._validate_summary_user_provenance(summary, has_user_turn)
            conclusions = self._vcc_conclusions(turns_to_summarize, focus_topic, memory_context, bypass_cooldown)
            if conclusions:
                summary += "\n\n" + conclusions
        except AuxiliaryExplicitCancellation:
            raise
        except Exception as exc:
            self._vcc_fallbacks += 1
            if not self._vcc_options["fallback_to_llm"]:
                logger.warning("vcc: projection failed (%s); fallback disabled, no summary this pass", exc)
                return None
            logger.warning("vcc: projection failed (%s: %s); using the stock LLM summary for this pass",
                           type(exc).__name__, exc)
            return super()._generate_summary(turns_to_summarize, focus_topic, memory_context, bypass_cooldown)

        # A detached stale attempt must not publish onto shared compressor state.
        _raise_if_stale_attempt(self)
        self._previous_summary = summary
        self._clear_compression_failure_cooldown()
        self._summary_model_fallen_back = False
        self._last_summary_error = None
        self._clear_terminal_summary_failures()
        self._reset_consecutive_overload_aborts()

        self._vcc_runs += 1
        self._vcc_last_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("vcc: projected %d turns into %d chars in %.1f ms%s",
                    len(turns_to_summarize), len(summary), self._vcc_last_ms,
                    " (+ bounded conclusions leg)" if conclusions else " (no LLM call)")
        if self._vcc_options["record_trigger"]:
            summary += "\n\n" + self._vcc_trigger_section(bool(conclusions))
        return self._with_summary_prefix(summary)


def _force_no_user_task(body: str) -> str:
    """With no user-authored turn anywhere, the task section must be Hermes's exact sentinel."""
    preamble, sections = projector.split_sections(body)
    sections[projector.TASK_HEADING] = projector.NO_USER_TASK
    parts = [preamble] if preamble else []
    parts += [f"{heading}\n{text}".rstrip() for heading, text in sections.items()]
    return "\n\n".join(parts)
