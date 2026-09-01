import unittest

from fastmcp import Client

from mcp_server import mcp


class McpServerSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_full_official_action_set(self):
        async with Client(mcp) as client:
            tools = await client.list_tools()
        self.assertEqual(
            {tool.name for tool in tools},
            {
                "linkdog_status",
                # group2（無參數）
                "linkdog_sit",
                "linkdog_stand",
                "linkdog_get_down",
                "linkdog_wiggle_tail",
                "linkdog_stretch",
                "linkdog_head_forward",
                "linkdog_head_back",
                "linkdog_pee_marking",
                "linkdog_talent",
                "linkdog_dance",
                "linkdog_act_cute",
                "linkdog_eat",
                "linkdog_wronged",
                "linkdog_sleep",
                "linkdog_shiver",
                # group1（duration）
                "linkdog_forward",
                "linkdog_left",
                "linkdog_right",
                "linkdog_backward",
                "linkdog_left_right",
                "linkdog_front_back",
                "linkdog_shake_hands",
                "linkdog_crawl",
                "linkdog_wiggle",
                "linkdog_spin_around",
                # group3（times）
                "linkdog_push_up",
                "linkdog_greetings",
                "linkdog_drink",
                "linkdog_fart",
                # 速度 / 角度 / 屏幕 / 遊戲
                "linkdog_set_speed",
                "linkdog_angle",
                "linkdog_set_screen_mode",
                "linkdog_rock_paper_scissors",
                # 唱歌 / 查詢類
                "linkdog_sing",
                "linkdog_song_current",
                "linkdog_date_search",
            },
        )


if __name__ == "__main__":
    unittest.main()
