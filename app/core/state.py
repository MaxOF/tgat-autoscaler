"""Safety-state repositories with atomic JSON replacement."""

from __future__ import annotations
import copy
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Protocol


class StateStore(Protocol):
    def load(self) -> dict: ...
    def save(self, state: dict) -> None: ...


class MemoryStateStore:
    def __init__(self):
        self.state = {}

    def load(self):
        return copy.deepcopy(self.state)

    def save(self, state):
        self.state = copy.deepcopy(state)


class JSONStateStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("State must be an object")
            return value
        except (ValueError, OSError):
            logging.getLogger(__name__).exception("Unable to load safety state")
            return {}

    def save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as f:
                name = f.name
                json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path)
        finally:
            if name and Path(name).exists():
                Path(name).unlink()
