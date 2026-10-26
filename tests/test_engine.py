from __future__ import annotations

import json
import os
from typing import Any

import pytest

from conftest import ROOT, requires_hermes
from transcripts import assistant, call, fix_the_build, long_session, result, terminal_turn, user

pytestmark = requires_hermes


@pytest.fixture()
def cc():
    import agent.context_compressor as module

    return module


@pytest.fixture()
def no_llm(monkeypatch, cc):
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("vcc must not call the summary LLM")

    monkeypatch.setattr(cc, "call_llm", forbidden)
    return calls


def make_engine(**overrides):
    import engine

    kwargs = dict(config={}, config_context_length=100_000, protect_first_n=1, protect_last_n=2)
    kwargs.update(overrides)
    eng = engine.VccEngine(**kwargs)
    eng.tail_token_budget = 50  # small verbatim tail so the fixtures actually compact
    return eng


def session(*parts):
    msgs = [{"role": "system", "content": "You are Hermes."}]
    for part in parts:
        msgs.extend(part)
    return msgs


def summary_of(eng, messages):
    found = [m for m in messages if eng._is_context_summary_message(m)]
    assert len(found) == 1, f"expected one summary message, got {len(found)}"
    content = found[0]["content"]
    return content if isinstance(content, str) else json.dumps(content)


def assert_tool_pairs_intact(messages):
    open_ids = set()
    for m in messages:
        if m.get("role") == "assistant":
            open_ids |= {tc["id"] for tc in m.get("tool_calls") or ()}
        elif m.get("role") == "tool":
            assert m["tool_call_id"] in open_ids, f"orphan tool result {m['tool_call_id']}"


# ----------------------------------------------------------------------------- basics

def test_hermes_api_has_every_seam_we_use():
    import engine

    assert engine._missing_seams() == []


def test_name_and_mode():
    eng = make_engine()
    assert eng.name == "vcc"
    assert eng._vcc_passthrough is False


def test_compress_makes_no_llm_call_and_keeps_history_addressable(no_llm):
    eng = make_engine()
    messages = session(fix_the_build(), long_session(40, output_chars=1500), [user("what's next?")])
    out = eng.compress(messages, force=True)

    assert no_llm == []
    text = summary_of(eng, out)
    assert "vcc projection" in text
    # the first user turn sits in Hermes's protected head; the goal is the first one compacted
    assert "## Goal\nAudit the repo and clean up the logging module." in text
    assert "→ fixed at [#" in text and "[#" in text
    # Hermes's own lean sections are still appended after ours
    assert "## User Messages (verbatim, newest first)" in text
    assert_tool_pairs_intact(out)
    assert sum(len(json.dumps(m)) for m in out) < sum(len(json.dumps(m)) for m in messages) / 3
    assert eng._vcc_runs == 1 and eng._vcc_last_ms is not None


def test_iterative_compaction_keeps_the_original_goal(no_llm):
    eng = make_engine()
    first = eng.compress(session(fix_the_build(), long_session(30, output_chars=800)), force=True)
    second = eng.compress(first + long_session(30, output_chars=800, seed=9) + [user("status?")], force=True)
    text = summary_of(eng, second)
    assert "## Goal\nAudit the repo and clean up the logging module." in text
    assert text.count("## Detailed Session Log") == 1
    assert no_llm == []


def test_session_without_user_turns_uses_hermes_sentinel(no_llm, cc):
    eng = make_engine()
    turns = [{"role": "system", "content": "cron"}]
    for i in range(30):
        turns += terminal_turn(f"c{i}", f"df -h /mnt/{i}", "ok " * 200)
    out = eng.compress(turns, force=True)
    text = summary_of(eng, out)
    assert cc._NO_USER_TASK_SENTINEL in text
    assert no_llm == []


# ----------------------------------------------------------------------------- fallbacks

def test_projection_error_falls_back_to_stock_llm_summary(monkeypatch, cc):
    import engine

    def boom(*a, **k):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(engine.projector, "project", boom)
    requests = []

    def fake_llm(**kwargs):
        requests.append(kwargs)
        return {"choices": [{"message": {"content": "## Goal\nfrom the llm"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(cc, "call_llm", fake_llm)
    eng = make_engine()
    out = eng.compress(session(fix_the_build(), long_session(30, output_chars=800)), force=True)
    assert requests, "fallback should reach the stock summary path"
    assert "from the llm" in summary_of(eng, out)
    assert eng._vcc_fallbacks == 1


def test_fallback_can_be_disabled(monkeypatch, no_llm):
    import engine

    monkeypatch.setattr(engine.projector, "project", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    eng = make_engine(config={"context": {"vcc": {"fallback_to_llm": False}}})
    eng.compress(session(fix_the_build(), long_session(30, output_chars=800)), force=True)
    assert no_llm == []
    assert eng._vcc_fallbacks == 1


def test_missing_seam_switches_to_passthrough(monkeypatch, cc):
    import engine

    monkeypatch.setattr(engine, "_missing_seams", lambda: ["_something_renamed"])
    requests = []
    monkeypatch.setattr(cc, "call_llm", lambda **k: requests.append(k) or
                        {"choices": [{"message": {"content": "## Goal\nstock"}, "finish_reason": "stop"}]})
    eng = make_engine()
    assert eng._vcc_passthrough is True
    eng.compress(session(fix_the_build(), long_session(30, output_chars=800)), force=True)
    assert requests


# ----------------------------------------------------------------------------- config & trigger

def test_trigger_defaults_to_75_percent_of_the_window():
    eng = make_engine()
    eng.update_model(model="local/qwen", context_length=98_304)
    assert eng.threshold_tokens == int(98_304 * 0.75)


def test_trigger_follows_context_changes():
    eng = make_engine()
    eng.update_model(model="local/qwen", context_length=98_304)
    assert eng.threshold_tokens == 73_728
    eng.update_model(model="local/qwen", context_length=131_072)
    assert eng.threshold_tokens == 98_304
    # Hermes itself raises the floor to 85% on windows near its 64K minimum; we inherit that
    eng.update_model(model="local/qwen", context_length=65_536)
    assert eng.threshold_tokens == int(65_536 * 0.85)


def test_explicit_threshold_tokens_cap_is_respected():
    eng = make_engine(config={"compression": {"threshold_tokens": 45_000}})
    eng.update_model(model="local/qwen", context_length=98_304)
    assert eng.threshold_tokens == 45_000


def test_summariser_window_never_caps_the_trigger():
    eng = make_engine()
    eng._aux_context_ceiling = 8_000  # what the feasibility probe would install
    eng.update_model(model="local/qwen", context_length=98_304)
    assert eng.threshold_tokens == int(98_304 * 0.75)


@pytest.mark.parametrize("junk", [None, "abc", -5, float("nan"), {"x": 1}, []])
def test_config_junk_falls_back_to_defaults(junk):
    import engine

    kw = engine._compressor_kwargs({"compression": {"threshold": junk, "protect_last_n": junk,
                                                    "threshold_tokens": junk, "model_thresholds": junk}})
    assert 0.05 <= kw["threshold_percent"] <= 0.98
    assert kw["threshold_tokens_cap"] is None
    opts = engine._engine_options({"context": {"vcc": {"budget_ratio": junk, "max_budget_tokens": junk}}})
    assert 0.01 <= opts["budget_ratio"] <= 0.30


def test_summary_budget_is_small_relative_to_the_window():
    eng = make_engine()
    eng.update_model(model="local/qwen", context_length=98_304)
    budget = eng._vcc_budget_chars(long_session(300, output_chars=3000))
    assert budget <= int(98_304 * 0.06) * 4


def test_clone_is_independent():
    eng = make_engine()
    twin = eng.clone_for_agent()
    assert twin is not eng and type(twin) is type(eng) and twin.name == "vcc"
    twin.update_model(model="other", context_length=50_000)
    assert eng.model != "other"


# ----------------------------------------------------------------------------- loading through Hermes

def test_loads_through_hermes_user_plugin_dir(sandbox_home):
    plugins = sandbox_home / "plugins"
    plugins.mkdir(exist_ok=True)
    link = plugins / "vcc"
    if not link.exists():
        os.symlink(ROOT, link, target_is_directory=True)
    from plugins.context_engine import load_context_engine

    eng = load_context_engine("vcc")
    assert eng is not None and eng.name == "vcc"
    assert type(eng).__mro__[1].__name__ == "ContextCompressor"


# ----------------------------------------------------------------------------- v0.2 fast anchors

@pytest.fixture(autouse=True)
def _fresh_fast_anchor_state():
    import anchors
    import engine

    def reset():
        engine._FAST_ANCHORS.update({"fixture_ok": None, "live_ok": False, "disabled": ""})
        anchors._FOOTER = None

    reset()
    yield
    reset()


_T_ONE = session(fix_the_build(), long_session(60, output_chars=900))
_T_TWO_A = long_session(40, output_chars=700, seed=3)
_T_TWO_B = long_session(40, output_chars=700, seed=4) + [user("and now?")]


def _summaries(fast: bool):
    """Same transcripts every call (ids come from a global counter in transcripts.py)."""
    import engine

    engine._FAST_ANCHORS.update({"fixture_ok": None, "live_ok": False, "disabled": ""})
    cfg = {"context": {"vcc": {"fast_anchors": fast}}}
    first = make_engine(config=cfg)._generate_summary(list(_T_ONE))
    two = make_engine(config=cfg)
    a = two._generate_summary(list(_T_TWO_A))
    b = two._generate_summary(list(_T_TWO_B))
    return first, a, b


def test_fast_anchors_change_no_output(no_llm):
    assert _summaries(fast=True) == _summaries(fast=False)
    assert no_llm == []


def test_fast_anchor_path_is_actually_used():
    import engine

    eng = make_engine()
    eng._generate_summary(session(fix_the_build(), long_session(30, output_chars=800)))
    assert engine._FAST_ANCHORS["live_ok"] is True
    assert eng.get_status()["vcc_fast_anchors"] == "on"


def test_shadow_mismatch_turns_the_fast_path_off(monkeypatch, no_llm):
    import anchors
    import engine

    expected = _summaries(fast=False)
    real_build = anchors.build
    monkeypatch.setattr(anchors, "build", lambda turns, cc=None: real_build(turns, cc) + "tampered")
    engine._FAST_ANCHORS.update({"fixture_ok": True, "live_ok": False, "disabled": ""})
    eng = make_engine()
    out = eng._generate_summary(list(_T_ONE))
    assert out == expected[0]
    assert "differed" in engine._FAST_ANCHORS["disabled"]
    assert eng.get_status()["vcc_fast_anchors"].startswith("off")
    assert no_llm == []


def test_fast_anchor_errors_never_reach_the_llm_fallback(monkeypatch, no_llm):
    import anchors
    import engine

    expected = _summaries(fast=False)
    monkeypatch.setattr(anchors, "build", lambda *a, **k: (_ for _ in ()).throw(MemoryError("boom")))
    engine._FAST_ANCHORS.update({"fixture_ok": True, "live_ok": True, "disabled": ""})
    eng = make_engine()
    out = eng._generate_summary(list(_T_ONE))
    assert out == expected[0]
    assert eng._vcc_fallbacks == 0
    assert no_llm == []


def test_failed_self_check_keeps_hermes_builder(monkeypatch, no_llm):
    import anchors
    import engine

    expected = _summaries(fast=False)
    monkeypatch.setattr(anchors, "verify", lambda cc=None: False)
    engine._FAST_ANCHORS.update({"fixture_ok": None, "live_ok": False, "disabled": ""})
    eng = make_engine()
    assert eng._generate_summary(list(_T_ONE)) == expected[0]
    assert "self-check" in engine._FAST_ANCHORS["disabled"]


def test_legacy_tail_mode_never_adds_an_anchor_section(cc):
    eng = make_engine(config={"compression": {"tail_mode": "legacy"}})
    out = eng._generate_summary(session(fix_the_build(), long_session(30, output_chars=800)))
    assert cc._LEAN_ANCHOR_HEADING not in out


# ------------------------------------------------------------- hybrid: trigger recording + conclusions


def _prose_session(rounds: int = 12) -> list[dict[str, Any]]:
    """A session with real assistant judgement, which the shared fixtures do not carry.

    Conclusions only exist in the text the agent writes around its tool calls, so the hybrid
    tests need turns that contain it; ``long_session`` is deliberately all tool output.
    """
    msgs = [{"role": "system", "content": "You are Hermes."}, user("did the tests pass, and what now?")]
    for i in range(rounds):
        msgs.append(assistant(
            f"Verdict {i}: the failures are the suite's real-home I/O guard, not a regression in "
            f"the patch, so the plan stands — keep it and move to the next gate."))
        msgs += terminal_turn(f"p{i}", f"pytest tests/gate_{i}.py", "46 failed, 21 passed")
    return msgs


def test_manual_compress_records_the_trigger(no_llm):
    eng = make_engine()
    out = eng.compress(session(fix_the_build(), long_session(30, output_chars=800)), force=True)
    text = summary_of(eng, out)
    assert "## Compaction Trigger" in text
    assert "The user ran `/compress`" in text
    assert "no LLM call" in text
    assert no_llm == []  # recording the trigger costs no model call


def test_the_handoff_is_still_byte_reproducible(no_llm):
    """The same turns must still project to identical bytes (no clock, no duration in the text).

    Two engines, one turn list: a second pass on the SAME engine legitimately differs, because
    iterative compaction folds the previous summary into ``## Earlier Summary``.
    """
    turns = _prose_session()
    assert make_engine()._generate_summary(list(turns)) == make_engine()._generate_summary(list(turns))
    assert no_llm == []


def test_automatic_pass_says_automatic(no_llm):
    eng = make_engine()
    eng._vcc_trigger = {"manual": False, "focus": None, "at": "2026-01-01 00:00:00"}
    text = eng._generate_summary(_prose_session())
    assert "Automatic: the context crossed the compaction threshold at 2026-01-01" in text
    assert no_llm == []


def test_record_trigger_off_restores_the_old_handoff(no_llm):
    eng = make_engine(config={"context": {"vcc": {"record_trigger": False}}})
    text = eng._generate_summary(_prose_session())
    assert "## Compaction Trigger" not in text


def test_conclusions_material_is_prose_only_and_bounded():
    import projector

    turns = _prose_session(rounds=30)
    prose = projector.assistant_prose(turns)
    assert len(prose) == 30 and all(m["role"] == "assistant" for m in prose)
    material = projector.conclusions_material(turns, budget_chars=4000)
    assert material, "expected some prose inside the budget"
    assert sum(len(m["content"]) + 1 for m in material) <= 4000
    assert len(material) < len(prose)  # the cap actually cut the selection
    assert material[-1]["content"] == prose[-1]["content"]  # newest prose is always kept
    assert all("tool_calls" not in m for m in material)     # tool traffic never reaches the leg


def test_conclusions_leg_defaults_off_and_costs_no_llm(no_llm):
    eng = make_engine()
    assert eng._vcc_options["preserve_conclusions"] is False
    out = eng.compress(session(fix_the_build(), long_session(30, output_chars=800)), force=True)
    assert "## What the Agent Concluded" not in summary_of(eng, out)
    assert no_llm == []


def test_conclusions_leg_appends_the_digest(monkeypatch, no_llm):
    import engine

    seen: dict[str, int] = {}

    def fake_llm_summary(self, turns, focus_topic=None, memory_context="", bypass_cooldown=False):
        seen["turns"] = len(turns)
        return "Decided the failing tests were an environment guard, not a regression."

    monkeypatch.setattr(engine.ContextCompressor, "_generate_summary", fake_llm_summary)
    eng = make_engine(config={"context": {"vcc": {"preserve_conclusions": True}}})
    eng._vcc_trigger = {"manual": True, "focus": None, "at": "2026-01-01 00:00:00"}
    text = eng._generate_summary(_prose_session())
    assert "## What the Agent Concluded" in text
    assert "environment guard" in text
    assert seen["turns"] > 0
    # the trigger line stops claiming "no LLM call" once the hybrid leg ran
    assert "one bounded summary-model call" in text


def test_conclusions_leg_strips_the_stock_handoff_banner(monkeypatch, no_llm):
    """The digest is a SECTION of a handoff; it must not carry a second handoff banner."""
    import agent.context_compressor as cc
    import engine

    def llm_with_banner(self, turns, focus_topic=None, memory_context="", bypass_cooldown=False):
        return cc.SUMMARY_PREFIX + "\nThe guard failures were environmental; the patch stands."

    monkeypatch.setattr(engine.ContextCompressor, "_generate_summary", llm_with_banner)
    eng = make_engine(config={"context": {"vcc": {"preserve_conclusions": True}}})
    text = eng._generate_summary(_prose_session())
    assert "the patch stands" in text
    assert text.count(cc.SUMMARY_PREFIX.split("\n")[0]) == 1  # only the real handoff banner, up top


def test_conclusions_leg_refuses_hermess_structural_fallback(monkeypatch, no_llm):
    """A refused summariser (oMLX 507 memory ceiling) inserts a structural summary, not a model's.

    Shipping that as a "conclusion" restates the projection we already have and charges the
    leg's latency for nothing, so the section must be dropped and the trigger must stop
    claiming a model ran.
    """
    import engine

    def structural(self, turns, focus_topic=None, memory_context="", bypass_cooldown=False):
        self._last_summary_error = ("summary model stalled on every route; "
                                    "deterministic fallback summary inserted")
        return "## Historical Task Snapshot None. # Detailed Session Log (oldest first) …"

    monkeypatch.setattr(engine.ContextCompressor, "_generate_summary", structural)
    eng = make_engine(config={"context": {"vcc": {"preserve_conclusions": True}}})
    eng._vcc_trigger = {"manual": True, "focus": None, "at": "2026-01-01 00:00:00"}
    text = eng._generate_summary(_prose_session())
    assert "## What the Agent Concluded" not in text
    assert "no LLM call" in text
    assert "## Detailed Session Log" in text  # the deterministic projection still shipped


def test_conclusions_leg_failure_ships_the_deterministic_handoff(monkeypatch, no_llm):
    import engine

    def explode(self, turns, focus_topic=None, memory_context="", bypass_cooldown=False):
        raise RuntimeError("aux model is down")

    monkeypatch.setattr(engine.ContextCompressor, "_generate_summary", explode)
    eng = make_engine(config={"context": {"vcc": {"preserve_conclusions": True}}})
    text = eng._generate_summary(_prose_session())
    assert "## What the Agent Concluded" not in text
    assert "## Detailed Session Log" in text  # the projection still shipped
