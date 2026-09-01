import unittest

from app.actions import (
    ACTION_SPECS,
    ANGLE_PARTS,
    build_arguments,
    param_type,
    result_type,
    tool_name,
)


class ActionSpecTests(unittest.TestCase):
    def test_group2_actions_are_parameterless(self):
        for action in (
            "stand_up", "get_down", "sit_down", "stretch", "head_forward",
            "head_back", "pee_marking", "talent", "dance", "act_cute",
            "eat", "wronged", "sleep", "shiver", "wiggle_tail",
        ):
            self.assertEqual(param_type(action), "none", action)
            self.assertEqual(tool_name(action), "self.action.group2", action)

    def test_group1_actions_take_duration(self):
        for action in (
            "forward", "left", "right", "backward", "left_right",
            "front_back", "shake_hands", "crawl", "wiggle", "spin_around",
        ):
            self.assertEqual(param_type(action), "duration", action)
            self.assertEqual(tool_name(action), "self.action.group1", action)

    def test_group3_actions_take_times(self):
        for action in ("push_up", "greetings", "drink", "fart"):
            self.assertEqual(param_type(action), "times", action)
            self.assertEqual(tool_name(action), "self.action.group3", action)

    def test_special_actions(self):
        self.assertEqual(tool_name("set_speed"), "self.action.set_speed")
        self.assertEqual(tool_name("angle"), "self.action.angle")
        self.assertEqual(tool_name("set_screen_mode"), "self.screen.set_mode")
        self.assertEqual(tool_name("rock_paper_scissors"), "self.game.rock_paper_scissors")

    def test_query_actions(self):
        self.assertEqual(tool_name("sing"), "self.song.sing")
        self.assertEqual(param_type("sing"), "name")
        self.assertEqual(result_type("sing"), "action")
        self.assertEqual(tool_name("song_current"), "self.song.current")
        self.assertEqual(param_type("song_current"), "empty")
        self.assertEqual(result_type("song_current"), "text")
        self.assertEqual(tool_name("date_search"), "self.date.search")
        self.assertEqual(param_type("date_search"), "empty")
        self.assertEqual(result_type("date_search"), "text")

    def test_all_actions_have_action_result_type_except_queries(self):
        for action, (_, _, rtype) in ACTION_SPECS.items():
            if action in ("song_current", "date_search", "get_device_status"):
                self.assertEqual(rtype, "text", action)
            else:
                self.assertEqual(rtype, "action", action)


class BuildArgumentsTests(unittest.TestCase):
    def test_group2_builds_action_only(self):
        self.assertEqual(build_arguments("sit_down"), {"action": "sit_down"})

    def test_duration_defaults_and_clamps(self):
        self.assertEqual(
            build_arguments("forward"),
            {"action": "forward", "duration": 4},
        )
        self.assertEqual(
            build_arguments("forward", duration=7),
            {"action": "forward", "duration": 7},
        )
        # clamp 到合法區間
        self.assertEqual(
            build_arguments("forward", duration=99),
            {"action": "forward", "duration": 10},
        )
        self.assertEqual(
            build_arguments("forward", duration=0),
            {"action": "forward", "duration": 1},
        )

    def test_times_defaults_and_clamps(self):
        self.assertEqual(
            build_arguments("push_up"),
            {"action": "push_up", "times": 3},
        )
        self.assertEqual(
            build_arguments("push_up", times=5),
            {"action": "push_up", "times": 5},
        )
        self.assertEqual(
            build_arguments("push_up", times=99),
            {"action": "push_up", "times": 5},
        )

    def test_speed(self):
        self.assertEqual(build_arguments("set_speed"), {"speed": 3})
        self.assertEqual(build_arguments("set_speed", speed=5), {"speed": 5})

    def test_volume_requires_value_and_clamps_to_firmware_range(self):
        self.assertEqual(build_arguments("set_volume", volume=55), {"volume": 55})
        self.assertEqual(build_arguments("set_volume", volume=0), {"volume": 10})
        self.assertEqual(build_arguments("set_volume", volume=999), {"volume": 100})
        with self.assertRaises(ValueError):
            build_arguments("set_volume")

    def test_angle_requires_part_and_angle(self):
        self.assertEqual(
            build_arguments("angle", part="tail", angle=90),
            {"part": "tail", "angle": 90},
        )
        with self.assertRaises(ValueError):
            build_arguments("angle", angle=90)
        with self.assertRaises(ValueError):
            build_arguments("angle", part="tail")
        with self.assertRaises(ValueError):
            build_arguments("angle", part="bogus", angle=90)

    def test_angle_clamps(self):
        self.assertEqual(
            build_arguments("angle", part="left_hand", angle=999),
            {"part": "left_hand", "angle": 180},
        )

    def test_mode_requires_mode(self):
        self.assertEqual(build_arguments("set_screen_mode", mode=1), {"mode": 1})
        with self.assertRaises(ValueError):
            build_arguments("set_screen_mode")

    def test_gesture_requires_gesture(self):
        self.assertEqual(
            build_arguments("rock_paper_scissors", gesture=2),
            {"gesture": 2},
        )
        with self.assertRaises(ValueError):
            build_arguments("rock_paper_scissors")

    def test_sing_requires_name(self):
        self.assertEqual(
            build_arguments("sing", name="小星星"),
            {"name": "小星星"},
        )
        with self.assertRaises(ValueError):
            build_arguments("sing")

    def test_empty_actions_build_empty_arguments(self):
        self.assertEqual(build_arguments("song_current"), {})
        self.assertEqual(build_arguments("date_search"), {})
        self.assertEqual(build_arguments("get_device_status"), {})

    def test_angle_parts_match_official(self):
        self.assertEqual(ANGLE_PARTS, {"left_hand", "right_hand", "left_leg", "right_leg", "tail"})


if __name__ == "__main__":
    unittest.main()
