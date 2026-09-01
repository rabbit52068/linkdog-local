"""對照官方 repo 的完整動作目錄。

來源：gitee.com/jeremywang0102/linkdog
  Third/Code/ESP32S3/xiaozhi-esp32-1.8.12/main/boards/linkdog/linkdog.cc
  的 InitializeMcpAction()。

這是 adapter（app/main.py）與 MCP client（app/hermes_tools.py）共用的
單一來源，避免兩邊各自維護一份 allow-list 造成漂移。

每個動作對應官方韌體暴露的 MCP tool、參數型別與回傳型別：

參數型別（param_type）：
  "none"     — 動作類無參數，arguments 含 action 欄位（self.action.group2）
  "duration" — 持續秒數 1-10，預設 4（self.action.group1）
  "times"    — 次數 1-5，預設 3（self.action.group3）
  "speed"    — 速度 1-5（self.action.set_speed）
  "angle"    — part + angle 0-180（self.action.angle）
  "mode"     — 0 彩色 / 1 黑白（self.screen.set_mode）
  "gesture"  — 1 石頭 / 2 剪刀 / 3 布（self.game.rock_paper_scissors）
  "name"     — 歌名字串（self.song.sing）
  "empty"    — 真正無參數，arguments 為空 dict（self.song.current / self.date.search）

回傳型別（result_type）：
  "action" — 成功回 "true"（bool），失敗回 "false" 或錯誤字串
  "text"   — 成功回字串內容（查詢類工具）
"""

from typing import Any, Dict, Tuple

# action name -> (官方 MCP tool name, 參數型別, 回傳型別)
ACTION_SPECS: Dict[str, Tuple[str, str, str]] = {
    # group1 — 運動動作集合1（duration 1-10，預設 4）
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
    # group2 — 運動動作集合2（無參數）
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
    # group3 — 運動動作集合3（times 1-5，預設 3）
    "push_up": ("self.action.group3", "times", "action"),
    "greetings": ("self.action.group3", "times", "action"),
    "drink": ("self.action.group3", "times", "action"),
    "fart": ("self.action.group3", "times", "action"),
    # 速度控制
    "set_speed": ("self.action.set_speed", "speed", "action"),
    # S3 hardware speaker volume (official common MCP tools)
    "set_volume": ("self.audio_speaker.set_volume", "volume", "action"),
    "get_device_status": ("self.get_device_status", "empty", "text"),
    # 四肢/尾巴角度
    "angle": ("self.action.angle", "angle", "action"),
    # 屏幕模式
    "set_screen_mode": ("self.screen.set_mode", "mode", "action"),
    # 石頭剪刀布
    "rock_paper_scissors": ("self.game.rock_paper_scissors", "gesture", "action"),
    # 唱歌（成功回 true，失敗回錯誤字串）
    "sing": ("self.song.sing", "name", "action"),
    # 查詢類（回字串）
    "song_current": ("self.song.current", "empty", "text"),
    "date_search": ("self.date.search", "empty", "text"),
}

# 參數預設值
DEFAULTS: Dict[str, int] = {
    "duration": 4,
    "times": 3,
    "speed": 3,
}

# 參數合法範圍（含端點）
RANGES: Dict[str, Tuple[int, int]] = {
    "duration": (1, 10),
    "times": (1, 5),
    "speed": (1, 5),
    "angle": (0, 180),
    "mode": (0, 1),
    "gesture": (1, 3),
    "volume": (10, 100),
}

# self.action.angle 的合法 part
ANGLE_PARTS = {"left_hand", "right_hand", "left_leg", "right_leg", "tail"}


def tool_name(action: str) -> str:
    """回傳動作對應的官方 MCP tool 名稱。"""
    return ACTION_SPECS[action][0]


def param_type(action: str) -> str:
    """回傳動作的參數型別。"""
    return ACTION_SPECS[action][1]


def result_type(action: str) -> str:
    """回傳動作的回傳型別（action / text）。"""
    return ACTION_SPECS[action][2]


def _clamp(value: int, ptype: str) -> int:
    lo, hi = RANGES[ptype]
    return max(lo, min(int(value), hi))


def build_arguments(action: str, **params: Any) -> Dict[str, Any]:
    """依動作建構官方 MCP tools/call 的 arguments。

    未提供的參數套用 DEFAULTS；超出範圍的數值 clamp 到合法區間。
    對 angle / mode / gesture 這類必填參數，缺漏時拋 ValueError。
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
