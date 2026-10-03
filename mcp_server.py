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


# --- group2: no-parameter actions ---


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
    """Make LinkDog stretch (zh: 伸懶腰). Use only when the user explicitly asks for this motion."""
    return client().execute("stretch")


@mcp.tool
def linkdog_head_forward() -> dict:
    """Make LinkDog push its head forward (zh: 前頂). Use only when the user explicitly asks for this motion."""
    return client().execute("head_forward")


@mcp.tool
def linkdog_head_back() -> dict:
    """Make LinkDog push its head back (zh: 後頂). Use only when the user explicitly asks for this motion."""
    return client().execute("head_back")


@mcp.tool
def linkdog_pee_marking() -> dict:
    """Make LinkDog do a pee-marking motion (zh: 撒尿做標記). Use only when the user explicitly asks for this motion."""
    return client().execute("pee_marking")


@mcp.tool
def linkdog_talent() -> dict:
    """Make LinkDog perform a talent show (zh: 表演才藝). Use only when the user explicitly asks for this motion."""
    return client().execute("talent")


@mcp.tool
def linkdog_dance() -> dict:
    """Make LinkDog dance (zh: 跳舞). Use only when the user explicitly asks for this motion."""
    return client().execute("dance")


@mcp.tool
def linkdog_act_cute() -> dict:
    """Make LinkDog act cute (zh: 撒嬌賣萌). Use only when the user explicitly asks for this motion."""
    return client().execute("act_cute")


@mcp.tool
def linkdog_eat() -> dict:
    """Make LinkDog do an eating motion (zh: 吃東西). Use only when the user explicitly asks for this motion."""
    return client().execute("eat")


@mcp.tool
def linkdog_wronged() -> dict:
    """Make LinkDog act wronged (zh: 受委屈). Use only when the user explicitly asks for this motion."""
    return client().execute("wronged")


@mcp.tool
def linkdog_sleep() -> dict:
    """Make LinkDog sleep and snore (zh: 睡覺打呼嚕). Use only when the user explicitly asks for this motion."""
    return client().execute("sleep")


@mcp.tool
def linkdog_shiver() -> dict:
    """Make LinkDog shiver (zh: 發抖打哆嗦). Use only when the user explicitly asks for this motion."""
    return client().execute("shiver")


# --- group1: duration actions (1-10 seconds, default 4) ---


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
    """Make LinkDog sway left and right (zh: 左右搖晃). duration: 1-10 seconds (default 4)."""
    return client().execute("left_right", duration=duration)


@mcp.tool
def linkdog_front_back(duration: int = 4) -> dict:
    """Make LinkDog sway front and back (zh: 前後搖晃). duration: 1-10 seconds (default 4)."""
    return client().execute("front_back", duration=duration)


@mcp.tool
def linkdog_shake_hands(duration: int = 4) -> dict:
    """Make LinkDog shake hands (zh: 握手). duration: 1-10 seconds (default 4)."""
    return client().execute("shake_hands", duration=duration)


@mcp.tool
def linkdog_crawl(duration: int = 4) -> dict:
    """Make LinkDog crawl (zh: 匍匐前進). duration: 1-10 seconds (default 4)."""
    return client().execute("crawl", duration=duration)


@mcp.tool
def linkdog_wiggle(duration: int = 4) -> dict:
    """Make LinkDog wiggle its butt (zh: 撅屁股/扭屁股). duration: 1-10 seconds (default 4)."""
    return client().execute("wiggle", duration=duration)


@mcp.tool
def linkdog_spin_around(duration: int = 4) -> dict:
    """Make LinkDog spin around (zh: 轉圈圈). duration: 1-10 seconds (default 4)."""
    return client().execute("spin_around", duration=duration)


# --- group3: times actions (1-5 times, default 3) ---


@mcp.tool
def linkdog_push_up(times: int = 3) -> dict:
    """Make LinkDog do push-ups (zh: 伏地挺身). times: 1-5 (default 3)."""
    return client().execute("push_up", times=times)


@mcp.tool
def linkdog_greetings(times: int = 3) -> dict:
    """Make LinkDog bark as a greeting (zh: 學狗叫打招呼). times: 1-5 (default 3)."""
    return client().execute("greetings", times=times)


@mcp.tool
def linkdog_drink(times: int = 3) -> dict:
    """Make LinkDog drink water (zh: 喝水). times: 1-5 (default 3)."""
    return client().execute("drink", times=times)


@mcp.tool
def linkdog_fart(times: int = 3) -> dict:
    """Make LinkDog fart (zh: 放屁). times: 1-5 (default 3)."""
    return client().execute("fart", times=times)


# --- speed / angle / screen / game ---


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


# --- sing / query ---


@mcp.tool
def linkdog_sing(name: str) -> dict:
    """Make LinkDog sing a song by name (zh: 唱歌). name: the song title."""
    return client().execute("sing", name=name)


@mcp.tool
def linkdog_song_current() -> dict:
    """Ask what song LinkDog sang last (zh: 上一首唱了什麼)."""
    return client().execute("song_current")


@mcp.tool
def linkdog_date_search() -> dict:
    """Ask how many days since first meeting, or the first-meeting date (zh: 認識第幾天)."""
    return client().execute("date_search")


if __name__ == "__main__":
    mcp.run()
