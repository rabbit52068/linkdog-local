"""Full action catalog, mirrored from the official firmware.

Source: gitee.com/jeremywang0102/linkdog
  Third/Code/ESP32S3/xiaozhi-esp32-1.8.12/main/boards/linkdog/linkdog.cc,
  InitializeMcpAction().

This is the single source shared by the adapter (app/routes/control.py) and
the MCP client (app/hermes_tools.py), so the two allow-lists cannot drift.

Each action maps to the firmware MCP tool, a parameter type and a result type.

Parameter types (param_type):
  "none"     — no parameters; arguments carry the action name (self.action.group2)
  "duration" — seconds 1-10, default 4 (self.action.group1)
  "times"    — repetitions 1-5, default 3 (self.action.group3)
  "speed"    — speed 1-5 (self.action.set_speed)
  "angle"    — part + angle 0-180 (self.action.angle)
  "mode"     — 0 color / 1 black-and-white (self.screen.set_mode)
  "gesture"  — 1 rock / 2 scissors / 3 paper (self.game.rock_paper_scissors)
  "name"     — song title string (self.song.sing)
  "empty"    — truly no parameters; arguments is an empty dict
               (self.song.current / self.date.search)

Result types (result_type):
  "action" — "true" on success; "false" or an error string on failure
  "text"   — the string content on success (query tools)
"""

from typing import Any, Dict, Tuple

# action name -> (official MCP tool name, parameter type, return type)
ACTION_SPECS: Dict[str, Tuple[str, str, str]] = {
    # group1 — motion set 1 (duration 1-10, default 4)
    "forward": ("self.action.group1", "duration", "action"),
    "left": ("self.action.group1", "duration", "action"),
    "right": ("self.action.group1", "duration", "action"),
    "backward": ("self.action.group1", "duration", "action"),
    "left_right": ("self.action.group1", "duration", "action"),
    "front_back": ("self.action.group1", "duration", "action"),
    "shake_hands": ("self.action.group1", "duration", "action"),
    "crawl": ("self.action.group1", "duration", "action"),
    "wiggle": ("self.action.group1", "duration", "action"),
    "spin_around": ("self.action.group1", "duration", "action"),
    # group2 — motion set 2 (no parameters)
    "stand_up": ("self.action.group2", "none", "action"),
    "get_down": ("self.action.group2", "none", "action"),
    "sit_down": ("self.action.group2", "none", "action"),
    "stretch": ("self.action.group2", "none", "action"),
    "head_forward": ("self.action.group2", "none", "action"),
    "head_back": ("self.action.group2", "none", "action"),
    "pee_marking": ("self.action.group2", "none", "action"),
    "talent": ("self.action.group2", "none", "action"),
    "dance": ("self.action.group2", "none", "action"),
    "act_cute": ("self.action.group2", "none", "action"),
    "eat": ("self.action.group2", "none", "action"),
    "wronged": ("self.action.group2", "none", "action"),
    "sleep": ("self.action.group2", "none", "action"),
    "shiver": ("self.action.group2", "none", "action"),
    "wiggle_tail": ("self.action.group2", "none", "action"),
    # group3 — motion set 3 (times 1-5, default 3)
    "push_up": ("self.action.group3", "times", "action"),
    "greetings": ("self.action.group3", "times", "action"),
    "drink": ("self.action.group3", "times", "action"),
    "fart": ("self.action.group3", "times", "action"),
    # speed control
    "set_speed": ("self.action.set_speed", "speed", "action"),
    # S3 hardware speaker volume (official common MCP tools)
    "set_volume": ("self.audio_speaker.set_volume", "volume", "action"),
    "get_device_status": ("self.get_device_status", "empty", "text"),
    # limb/tail angle
    "angle": ("self.action.angle", "angle", "action"),
    # screen mode
    "set_screen_mode": ("self.screen.set_mode", "mode", "action"),
    # rock-paper-scissors
    "rock_paper_scissors": ("self.game.rock_paper_scissors", "gesture", "action"),
    # sing (returns true on success, error string on failure)
    "sing": ("self.song.sing", "name", "action"),
    # query type (returns string)
    "song_current": ("self.song.current", "empty", "text"),
    "date_search": ("self.date.search", "empty", "text"),
}

# parameter defaults
DEFAULTS: Dict[str, int] = {
    "duration": 4,
    "times": 3,
    "speed": 3,
}

# valid parameter ranges (inclusive)
RANGES: Dict[str, Tuple[int, int]] = {
    "duration": (1, 10),
    "times": (1, 5),
    "speed": (1, 5),
    "angle": (0, 180),
    "mode": (0, 1),
    "gesture": (1, 3),
    "volume": (10, 100),
}

# valid part values for self.action.angle
ANGLE_PARTS = {"left_hand", "right_hand", "left_leg", "right_leg", "tail"}


def tool_name(action: str) -> str:
    """Return the firmware MCP tool name for an action."""
    return ACTION_SPECS[action][0]


def param_type(action: str) -> str:
    """Return the action's parameter type."""
    return ACTION_SPECS[action][1]


def result_type(action: str) -> str:
    """Return the action's result type (action / text)."""
    return ACTION_SPECS[action][2]


def _clamp(value: int, ptype: str) -> int:
    lo, hi = RANGES[ptype]
    return max(lo, min(int(value), hi))


def build_arguments(action: str, **params: Any) -> Dict[str, Any]:
    """Build the firmware MCP tools/call arguments for an action.

    Missing parameters take DEFAULTS; out-of-range values are clamped.
    Required parameters (angle / mode / gesture) raise ValueError when missing.
    """
    ptype = param_type(action)

    if ptype == "none":
        return {"action": action}

    if ptype == "empty":
        return {}

    if ptype == "duration":
        duration = params.get("duration", DEFAULTS["duration"])
        return {"action": action, "duration": _clamp(duration, "duration")}

    if ptype == "times":
        times = params.get("times", DEFAULTS["times"])
        return {"action": action, "times": _clamp(times, "times")}

    if ptype == "speed":
        speed = params.get("speed", DEFAULTS["speed"])
        return {"speed": _clamp(speed, "speed")}

    if ptype == "volume":
        volume = params.get("volume")
        if volume is None:
            raise ValueError("set_volume requires 'volume'")
        return {"volume": _clamp(volume, "volume")}

    if ptype == "angle":
        part = params.get("part")
        angle = params.get("angle")
        if part not in ANGLE_PARTS:
            raise ValueError(f"invalid part: {part!r}")
        if angle is None:
            raise ValueError("angle requires 'angle'")
        return {"part": part, "angle": _clamp(angle, "angle")}

    if ptype == "mode":
        mode = params.get("mode")
        if mode is None:
            raise ValueError("set_screen_mode requires 'mode'")
        return {"mode": _clamp(mode, "mode")}

    if ptype == "gesture":
        gesture = params.get("gesture")
        if gesture is None:
            raise ValueError("rock_paper_scissors requires 'gesture'")
        return {"gesture": _clamp(gesture, "gesture")}

    if ptype == "name":
        name = params.get("name")
        if name is None:
            raise ValueError("sing requires 'name'")
        return {"name": str(name)}

    raise ValueError(f"unknown param type: {ptype}")
