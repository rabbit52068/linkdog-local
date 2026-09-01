"""Persistent, non-secret settings for the LinkDog dashboard."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DashboardSettings:
    agent_name: str = "Xiaobin"
    system_prompt: str = ""
    model: str = "deepseek-v4-flash:0731"
    memory_enabled: bool = True
    max_history_turns: int = 6
    user_profile: str = ""
    context_memory: str = ""
    volume: int = 70

    def __post_init__(self) -> None:
        if not self.agent_name.strip():
            raise ValueError("agent_name cannot be blank")
        if not self.model.strip():
            raise ValueError("model cannot be blank")
        if not 0 <= self.max_history_turns <= 20:
            raise ValueError("max_history_turns must be between 0 and 20")
        if not 10 <= self.volume <= 100:
            raise ValueError("volume must be between 10 and 100")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DashboardSettings":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SettingsStore:
    """Load and atomically save versioned dashboard settings."""

    def __init__(
        self,
        path: Path | str,
        defaults: Optional[DashboardSettings] = None,
    ) -> None:
        self.path = Path(path)
        self.defaults = defaults or DashboardSettings()

    def load(self) -> DashboardSettings:
        if not self.path.exists():
            return self.defaults
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            data = payload.get("settings", payload)
            if not isinstance(data, dict):
                raise ValueError("settings payload must be an object")
            merged = self.defaults.to_dict()
            merged.update(data)
            return DashboardSettings.from_dict(merged)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid dashboard settings: {exc}") from exc

    def save(self, settings: DashboardSettings) -> DashboardSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "settings": settings.to_dict()}
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return settings
