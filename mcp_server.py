import os

from fastmcp import FastMCP

from app.hermes_tools import LinkDogClient


ADAPTER_URL = os.environ.get("LINKDOG_ADAPTER_URL", "http://127.0.0.1:8003")
DEVICE_ID = os.environ.get("LINKDOG_DEVICE_ID", "")

mcp = FastMCP(
    "LinkDog",
    instructions=(
        "Control one LinkDog using the official firmware MCP actions. "
        "Never claim success unless the tool result says status=completed. "
        "Only call a motion tool when the user explicitly asks for that motion."
    ),
)


def client() -> LinkDogClient:
    return LinkDogClient(ADAPTER_URL, DEVICE_ID)


@mcp.tool
def linkdog_status() -> dict:
    """Check whether the self-hosted adapter and LinkDog are connected."""
    return client().status()


# --- group2：無參數動作 ---


@mcp.tool
def linkdog_sit() -> dict:
    """Make LinkDog sit down. Use only when the user explicitly asks for this motion."""
    return client().execute("sit_down")


@mcp.tool
def linkdog_stand() -> dict:
    """Make LinkDog stand up. Use only when the user explicitly asks for this motion."""
    return client().execute("stand_up")


@mcp.tool
def linkdog_get_down() -> dict:
    """Make LinkDog lie down. Use only when the user explicitly asks for this motion."""
    return client().execute("get_down")


@mcp.tool
def linkdog_wiggle_tail() -> dict:
    """Make LinkDog wiggle its tail. Use only when the user explicitly asks for this motion."""
    return client().execute("wiggle_tail")


@mcp.tool
def linkdog_stretch() -> dict:
    """Make LinkDog stretch (伸懒腰). Use only when the user explicitly asks for this motion."""
    return client().execute("stretch")


@mcp.tool
def linkdog_head_forward() -> dict:
    """Make LinkDog push its head forward (前顶). Use only when the user explicitly asks for this motion."""
    return client().execute("head_forward")


@mcp.tool
def linkdog_head_back() -> dict:
    """Make LinkDog push its head back (后顶). Use only when the user explicitly asks for this motion."""
    return client().execute("head_back")


@mcp.tool
def linkdog_pee_marking() -> dict:
    """Make LinkDog do a pee-marking motion (撒尿做标记). Use only when the user explicitly asks for this motion."""
    return client().execute("pee_marking")


@mcp.tool
def linkdog_talent() -> dict:
    """Make LinkDog perform a talent show (表演才艺). Use only when the user explicitly asks for this motion."""
    return client().execute("talent")


@mcp.tool
def linkdog_dance() -> dict:
    """Make LinkDog dance (跳舞). Use only when the user explicitly asks for this motion."""
    return client().execute("dance")


@mcp.tool
def linkdog_act_cute() -> dict:
    """Make LinkDog act cute (撒娇卖萌). Use only when the user explicitly asks for this motion."""
    return client().execute("act_cute")


@mcp.tool
def linkdog_eat() -> dict:
    """Make LinkDog do an eating motion (吃东西). Use only when the user explicitly asks for this motion."""
    return client().execute("eat")


@mcp.tool
def linkdog_wronged() -> dict:
    """Make LinkDog act wronged (受委屈). Use only when the user explicitly asks for this motion."""
    return client().execute("wronged")


@mcp.tool
def linkdog_sleep() -> dict:
    """Make LinkDog sleep and snore (睡觉打呼噜). Use only when the user explicitly asks for this motion."""
    return client().execute("sleep")


@mcp.tool
def linkdog_shiver() -> dict:
    """Make LinkDog shiver (发抖打哆嗦). Use only when the user explicitly asks for this motion."""
    return client().execute("shiver")


# --- group1：duration 動作（1-10 秒，預設 4） ---


@mcp.tool
def linkdog_forward(duration: int = 4) -> dict:
    """Make LinkDog walk forward. duration: 1-10 seconds (default 4)."""
    return client().execute("forward", duration=duration)


@mcp.tool
def linkdog_left(duration: int = 4) -> dict:
    """Make LinkDog turn left. duration: 1-10 seconds (default 4)."""
    return client().execute("left", duration=duration)


@mcp.tool
def linkdog_right(duration: int = 4) -> dict:
    """Make LinkDog turn right. duration: 1-10 seconds (default 4)."""
    return client().execute("right", duration=duration)


@mcp.tool
def linkdog_backward(duration: int = 4) -> dict:
    """Make LinkDog walk backward. duration: 1-10 seconds (default 4)."""
    return client().execute("backward", duration=duration)


@mcp.tool
def linkdog_left_right(duration: int = 4) -> dict:
    """Make LinkDog sway left and right (左右摇晃). duration: 1-10 seconds (default 4)."""
    return client().execute("left_right", duration=duration)


@mcp.tool
def linkdog_front_back(duration: int = 4) -> dict:
    """Make LinkDog sway front and back (前后摇晃). duration: 1-10 seconds (default 4)."""
    return client().execute("front_back", duration=duration)


@mcp.tool
def linkdog_shake_hands(duration: int = 4) -> dict:
    """Make LinkDog shake hands (握手). duration: 1-10 seconds (default 4)."""
    return client().execute("shake_hands", duration=duration)


@mcp.tool
def linkdog_crawl(duration: int = 4) -> dict:
    """Make LinkDog crawl (匍匐前进). duration: 1-10 seconds (default 4)."""
    return client().execute("crawl", duration=duration)


@mcp.tool
def linkdog_wiggle(duration: int = 4) -> dict:
    """Make LinkDog wiggle its butt (撅屁股/扭屁股). duration: 1-10 seconds (default 4)."""
    return client().execute("wiggle", duration=duration)


@mcp.tool
def linkdog_spin_around(duration: int = 4) -> dict:
    """Make LinkDog spin around (转圈圈). duration: 1-10 seconds (default 4)."""
    return client().execute("spin_around", duration=duration)


# --- group3：times 動作（1-5 次，預設 3） ---


@mcp.tool
def linkdog_push_up(times: int = 3) -> dict:
    """Make LinkDog do push-ups (俯卧撑). times: 1-5 (default 3)."""
    return client().execute("push_up", times=times)


@mcp.tool
def linkdog_greetings(times: int = 3) -> dict:
    """Make LinkDog bark as a greeting (学狗叫打招呼). times: 1-5 (default 3)."""
    return client().execute("greetings", times=times)


@mcp.tool
def linkdog_drink(times: int = 3) -> dict:
    """Make LinkDog drink water (喝水). times: 1-5 (default 3)."""
    return client().execute("drink", times=times)


@mcp.tool
def linkdog_fart(times: int = 3) -> dict:
    """Make LinkDog fart (放屁). times: 1-5 (default 3)."""
    return client().execute("fart", times=times)


# --- 速度 / 角度 / 屏幕 / 遊戲 ---


@mcp.tool
def linkdog_set_speed(speed: int = 3) -> dict:
    """Set LinkDog motion speed. speed: 1 (slowest) to 5 (fastest)."""
    return client().execute("set_speed", speed=speed)


@mcp.tool
def linkdog_angle(part: str, angle: int) -> dict:
    """Set a single limb/tail angle. part: left_hand/right_hand/left_leg/right_leg/tail. angle: 0-180."""
    return client().execute("angle", part=part, angle=angle)


@mcp.tool
def linkdog_set_screen_mode(mode: int) -> dict:
    """Set screen display mode. mode: 0 = color, 1 = black-and-white."""
    return client().execute("set_screen_mode", mode=mode)


@mcp.tool
def linkdog_rock_paper_scissors(gesture: int) -> dict:
    """Play rock-paper-scissors. gesture: 1=rock, 2=scissors, 3=paper."""
    return client().execute("rock_paper_scissors", gesture=gesture)


# --- 唱歌 / 查詢類 ---


@mcp.tool
def linkdog_sing(name: str) -> dict:
    """Make LinkDog sing a song by name (唱歌). name: the song title."""
    return client().execute("sing", name=name)


@mcp.tool
def linkdog_song_current() -> dict:
    """Ask what song LinkDog sang last (上一首唱了什麼)."""
    return client().execute("song_current")


@mcp.tool
def linkdog_date_search() -> dict:
    """Ask how many days since first meeting, or the first-meeting date (認識第幾天)."""
    return client().execute("date_search")


if __name__ == "__main__":
    mcp.run()
