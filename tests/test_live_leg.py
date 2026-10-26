"""Live measurement of the rewritten conclusions leg (no stub): run with VCC_LIVE=1.

Why a test file: the terminal hook re-executes .py script paths with its own interpreter (no
ruamel), but leaves `python -m pytest` alone. The assertion is the point — the leg must come
back inside the wire cap, not just look like it did.
"""
import os
import sqlite3
import sys
import time

sys.path.insert(0, "/Users/danielgindin/.hermes/profiles/redde/plugins/hermes-vcc-engine")

import pytest

SESSION = "20261026_020308_8248bc"


@pytest.mark.skipif(not os.environ.get("VCC_LIVE"), reason="set VCC_LIVE=1 to hit the local server")
def test_live_leg_stays_inside_its_wire_cap():
    from vcc import engine

    db = sqlite3.connect("file:/Users/danielgindin/.hermes/profiles/redde/state.db?mode=ro", uri=True)
    rows = [r[0] for r in db.execute(
        "SELECT content FROM messages WHERE session_id=? AND role='assistant' "
        "AND length(content) BETWEEN 600 AND 4000 ORDER BY rowid DESC LIMIT 4", (SESSION,))]
    rows.reverse()
    material = [{"turn": i, "content": c[:3800]} for i, c in enumerate(rows, 1)]
    assert len(material) == 4, "need a real prose window to time the leg"

    eng = engine.VccEngine()
    # A compressor inside the gateway carries the session's real route on self.*; a bare test
    # engine has none, and `task="compression"` alone then falls to auto-detect and finds no
    # provider. Set the same fields production would supply.
    eng.model = os.environ.get("MODEL") or eng.model
    eng.provider = os.environ.get("PROVIDER", "custom")
    eng.base_url = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
    eng.api_key = os.environ.get("API_KEY", "")
    opts = dict(eng._vcc_options)
    print(f"\nmodel={eng.model}\ncap={opts['conclusions_wire_tokens']} keep={opts['conclusions_output_tokens']}"
          f" in={sum(len(m['content']) for m in material):,} chars")

    t0 = time.perf_counter()
    text = eng._vcc_digest_call(material, opts)
    dt = time.perf_counter() - t0
    print(f"LEG {dt:.2f} s (was 39.66 s) | out {len(text):,} chars (~{len(text) / 4:,.0f} tok)")
    print("--- first 5 lines ---")
    print("\n".join(l[:150] for l in text.strip().splitlines()[:5]))

    assert text.strip(), "the live leg returned nothing"
    assert not text.lstrip().startswith("#"), "the section must not carry its own heading"
    assert len(text) < opts["conclusions_wire_tokens"] * 5, f"wire cap ignored: {len(text):,} chars"
    assert dt < 30, f"leg still slow: {dt:.1f} s"
