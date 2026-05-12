from __future__ import annotations
import uuid


def new_id(prefix: str = "") -> str:
    """Short, URL-safe id. Prefix is for human-readable trace logs."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw}" if prefix else raw
