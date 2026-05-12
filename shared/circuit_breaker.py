"""Per-tool circuit breaker.

closed   --(N failures)-->  open
open     --(cooldown elapsed)--> half_open (allows ONE probe)
half_open  --(success)--> closed       (counter reset)
half_open  --(failure)--> open         (cooldown doubled, capped at MAX_COOLDOWN)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from .db import get_conn

THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURES", "3"))
COOLDOWN_SECONDS = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "60"))
MAX_COOLDOWN = 300  # 5 min cap


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row(tool_name: str) -> dict | None:
    conn = get_conn()
    try:
        r = conn.execute(
            "SELECT * FROM circuit_state WHERE tool_name = ?", (tool_name,)
        ).fetchone()
    finally:
        conn.close()
    return dict(r) if r else None


def _ensure_row(tool_name: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO circuit_state
               (tool_name, state, consecutive_failures, opened_at, cooldown_seconds)
               VALUES (?, 'closed', 0, NULL, ?)""",
            (tool_name, COOLDOWN_SECONDS),
        )
        conn.commit()
    finally:
        conn.close()


def check_circuit(tool_name: str) -> tuple[bool, str | None]:
    """Returns (allowed, reason_if_denied). May transition open -> half_open."""
    _ensure_row(tool_name)
    row = _row(tool_name)
    if not row or row["state"] == "closed":
        return True, None

    if row["state"] == "open":
        if not row["opened_at"]:
            return False, "open"
        opened = datetime.fromisoformat(row["opened_at"])
        elapsed = (_now() - opened).total_seconds()
        if elapsed >= row["cooldown_seconds"]:
            # Atomically promote to half_open. Only one probe at a time.
            conn = get_conn()
            try:
                cur = conn.execute(
                    """UPDATE circuit_state SET state = 'half_open'
                       WHERE tool_name = ? AND state = 'open'""",
                    (tool_name,),
                )
                conn.commit()
                if cur.rowcount == 0:
                    return False, "half_open_probe_in_flight"
            finally:
                conn.close()
            return True, None
        return False, f"open (cooldown {int(row['cooldown_seconds'] - elapsed)}s remaining)"

    if row["state"] == "half_open":
        # Already probing; deny additional callers.
        return False, "half_open_probe_in_flight"

    return True, None


def record_success(tool_name: str) -> None:
    _ensure_row(tool_name)
    conn = get_conn()
    try:
        conn.execute(
            """UPDATE circuit_state
               SET state = 'closed',
                   consecutive_failures = 0,
                   opened_at = NULL,
                   cooldown_seconds = ?
               WHERE tool_name = ?""",
            (COOLDOWN_SECONDS, tool_name),
        )
        conn.commit()
    finally:
        conn.close()


def record_failure(tool_name: str) -> None:
    _ensure_row(tool_name)
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT consecutive_failures, cooldown_seconds, state FROM circuit_state WHERE tool_name = ?",
            (tool_name,),
        ).fetchone()
        fails = (row["consecutive_failures"] or 0) + 1
        current_state = row["state"]
        cooldown = row["cooldown_seconds"] or COOLDOWN_SECONDS

        if current_state == "half_open":
            # The probe failed; back to open with doubled cooldown.
            new_cooldown = min(cooldown * 2, MAX_COOLDOWN)
            conn.execute(
                """UPDATE circuit_state
                   SET state = 'open',
                       consecutive_failures = ?,
                       opened_at = ?,
                       cooldown_seconds = ?
                   WHERE tool_name = ?""",
                (fails, _now().isoformat(), new_cooldown, tool_name),
            )
        elif fails >= THRESHOLD:
            conn.execute(
                """UPDATE circuit_state
                   SET state = 'open',
                       consecutive_failures = ?,
                       opened_at = ?
                   WHERE tool_name = ?""",
                (fails, _now().isoformat(), tool_name),
            )
        else:
            conn.execute(
                "UPDATE circuit_state SET consecutive_failures = ? WHERE tool_name = ?",
                (fails, tool_name),
            )
        conn.commit()
    finally:
        conn.close()


def reset(tool_name: str) -> None:
    """Manual reset (admin / tests)."""
    conn = get_conn()
    try:
        conn.execute(
            """UPDATE circuit_state
               SET state = 'closed', consecutive_failures = 0, opened_at = NULL,
                   cooldown_seconds = ?
               WHERE tool_name = ?""",
            (COOLDOWN_SECONDS, tool_name),
        )
        conn.commit()
    finally:
        conn.close()
