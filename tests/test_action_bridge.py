import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import state
from app.routes import control
from app.device_session import DeviceSession
from app.main import app


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, text):
        self.messages.append(json.loads(text))


class ActionValidationTests(unittest.TestCase):
    def setUp(self):
        state.ACTIVE_SESSIONS.clear()
        state.PENDING_ACTIONS.clear()
        state.ACTION_LOCKS.clear()
        self.client = TestClient(app)

    def test_rejects_action_when_device_is_offline(self):
        response = self.client.post("/xiaozhi/action", json={"action": "sit_down"})
        self.assertEqual(response.status_code, 409)

    def test_malformed_hello_closes_socket_without_registering_device(self):
        from starlette.websockets import WebSocketDisconnect

        for bad_hello in ("not json", "[1, 2]"):
            with self.client.websocket_connect(
                "/xiaozhi/ws", headers={"device-id": "BAD:HELLO"}
            ) as ws:
                ws.send_text(bad_hello)
                with self.assertRaises(WebSocketDisconnect) as ctx:
                    ws.receive_text()
            self.assertEqual(ctx.exception.code, 1003)
            self.assertNotIn("BAD:HELLO", state.ACTIVE_SESSIONS)

    def test_rejects_unknown_action(self):
        response = self.client.post("/xiaozhi/action", json={"action": "not_a_real_action"})
        self.assertEqual(response.status_code, 400)

    def test_rejects_angle_without_part(self):
        # angle is allow-listed, but a missing part must return 400.
        response = self.client.post("/xiaozhi/action", json={"action": "angle", "angle": 90})
        self.assertEqual(response.status_code, 400)

    def test_get_down_and_wiggle_tail_are_now_allow_listed(self):
        # Root cause fixed (state gate), so both actions are allow-listed again.
        # Offline must return 409 (connection), not 400 (not allow-listed).
        for action in ("get_down", "wiggle_tail"):
            response = self.client.post("/xiaozhi/action", json={"action": action})
            self.assertEqual(response.status_code, 409)


class ActionExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        state.ACTIVE_SESSIONS.clear()
        state.PENDING_ACTIONS.clear()
        state.ACTION_LOCKS.clear()
        self.ws = FakeWebSocket()
        self.session = DeviceSession("TEST:DOG", self.ws)
        state.ACTIVE_SESSIONS["TEST:DOG"] = self.session

    async def asyncTearDown(self):
        await self.session.close()

    async def _start_action(self, action="sit_down"):
        task = asyncio.create_task(control.send_action(control.ActionRequest(
            device_id="TEST:DOG",
            action=action,
        )))
        # The state gate sends tts:start / tts:stop first; wait for the MCP tools/call.
        while not any(m.get("type") == "mcp" for m in self.ws.messages):
            if task.done():
                await task
            await asyncio.sleep(0)
        mcp_message = next(
            m for m in self.ws.messages if m.get("type") == "mcp"
        )
        return task, mcp_message

    @staticmethod
    def _success_payload(request_id):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": "true"}],
                "isError": False,
            },
        }

    async def test_waits_for_successful_mcp_response(self):
        task, message = await self._start_action()
        request_id = message["payload"]["id"]
        self.assertFalse(task.done())
        self.assertEqual(message["payload"]["method"], "tools/call")
        self.assertEqual(
            message["payload"]["params"],
            {
                "name": "self.action.group2",
                "arguments": {"action": "sit_down"},
            },
        )

        resolved = control.resolve_mcp_response(
            "TEST:DOG", self._success_payload(request_id)
        )
        self.assertTrue(resolved)
        result = await task
        self.assertEqual(result["status"], "completed")

    async def test_returns_gateway_error_for_mcp_failure(self):
        task, message = await self._start_action()
        request_id = message["payload"]["id"]
        control.resolve_mcp_response("TEST:DOG", {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": "failed"}],
                "isError": True,
            },
        })
        with self.assertRaises(HTTPException) as caught:
            await task
        self.assertEqual(caught.exception.status_code, 502)

    async def test_times_out_without_mcp_response(self):
        with patch.object(state, "ACTION_TIMEOUT_SECONDS", 0.01):
            with self.assertRaises(HTTPException) as caught:
                await control.send_action(control.ActionRequest(
                    device_id="TEST:DOG",
                    action="sit_down",
                ))
        self.assertEqual(caught.exception.status_code, 504)

    async def test_rejects_second_action_while_first_is_running(self):
        first_task, first_message = await self._start_action()
        with self.assertRaises(HTTPException) as caught:
            await control.send_action(control.ActionRequest(
                device_id="TEST:DOG",
                action="stand_up",
            ))
        self.assertEqual(caught.exception.status_code, 409)

        request_id = first_message["payload"]["id"]
        control.resolve_mcp_response(
            "TEST:DOG", self._success_payload(request_id)
        )
        result = await first_task
        self.assertEqual(result["status"], "completed")

    async def test_voice_action_uses_existing_bridge_and_returns_confirmation(self):
        completed = {
            "status": "completed",
            "device_id": "TEST:DOG",
            "action": "sit_down",
        }
        with patch.object(control, "send_action", AsyncMock(return_value=completed)) as send:
            response = await control.execute_voice_action("TEST:DOG", "sit_down")

        request = send.await_args.args[0]
        self.assertEqual(request.device_id, "TEST:DOG")
        self.assertEqual(request.action, "sit_down")
        self.assertEqual(response, "Okay, I sat down.")

    async def test_voice_volume_up_reads_status_then_sets_hardware_volume(self):
        status = {
            "status": "completed",
            "device_id": "TEST:DOG",
            "action": "get_device_status",
            "text": '{"audio_speaker":{"volume":60}}',
        }
        completed = {
            "status": "completed",
            "device_id": "TEST:DOG",
            "action": "set_volume",
        }
        with patch.object(
            control,
            "send_action",
            AsyncMock(side_effect=[status, completed]),
        ) as send:
            response = await control.execute_voice_volume(
                "TEST:DOG", {"mode": "up"}
            )

        first, second = [call.args[0] for call in send.await_args_list]
        self.assertEqual(first.action, "get_device_status")
        self.assertEqual(second.action, "set_volume")
        self.assertEqual(second.volume, 70)
        self.assertEqual(response, "Volume set to 70 percent.")

    async def test_set_volume_uses_official_mcp_without_motion_state_gate(self):
        with patch.object(control, "ensure_listening_state", new=AsyncMock()) as gate:
            task = asyncio.create_task(control.send_action(control.ActionRequest(
                device_id="TEST:DOG",
                action="set_volume",
                volume=65,
            )))
            while not any(m.get("type") == "mcp" for m in self.ws.messages):
                await asyncio.sleep(0)

            message = next(m for m in self.ws.messages if m.get("type") == "mcp")
            self.assertEqual(message["payload"]["params"], {
                "name": "self.audio_speaker.set_volume",
                "arguments": {"volume": 65},
            })
            gate.assert_not_awaited()

            request_id = message["payload"]["id"]
            control.resolve_mcp_response(
                "TEST:DOG", self._success_payload(request_id)
            )
            result = await task

        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
