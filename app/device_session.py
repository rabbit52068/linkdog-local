"""Per-device WebSocket lifecycle and bounded audio transport state."""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set


class DeviceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


class SessionClosedError(RuntimeError):
    """Raised when output is attempted after a device session closes."""


class DeviceSession:
    """Own one device connection, its queues, output lock, and worker tasks."""

    def __init__(
        self,
        device_id: str,
        websocket: Any,
        audio_queue_size: int = 64,
        ip_address: Optional[str] = None,
    ) -> None:
        if audio_queue_size <= 0:
            raise ValueError("audio_queue_size must be positive")
        self.device_id = device_id
        self.websocket = websocket
        self.ip_address = ip_address
        self.state = DeviceState.IDLE
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=audio_queue_size)
        self.dropped_audio_frames = 0
        self._write_lock = asyncio.Lock()
        self._tasks: Set[asyncio.Task] = set()
        self._close_callbacks: List[Callable[[], Awaitable[Any]]] = []
        self._closed = False

    def enqueue_audio(self, packet: bytes) -> None:
        if self._closed:
            return
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                self.dropped_audio_frames += 1
            except asyncio.QueueEmpty:
                pass
        self.audio_queue.put_nowait(packet)

    async def next_audio(self) -> bytes:
        return await self.audio_queue.get()

    def discard_queued_audio(self) -> int:
        discarded = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                discarded += 1
            except asyncio.QueueEmpty:
                break
        return discarded

    async def send_json(self, message: Dict[str, Any]) -> None:
        self._ensure_open()
        encoded = json.dumps(message, ensure_ascii=False)
        async with self._write_lock:
            self._ensure_open()
            await self.websocket.send_text(encoded)

    async def send_audio(self, packet: bytes) -> None:
        self._ensure_open()
        async with self._write_lock:
            self._ensure_open()
            await self.websocket.send_bytes(packet)

    def start_task(self, awaitable: Awaitable[Any]) -> asyncio.Task:
        self._ensure_open()
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def add_close_callback(self, callback: Callable[[], Awaitable[Any]]) -> None:
        self._ensure_open()
        self._close_callbacks.append(callback)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.state = DeviceState.CLOSED
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        callbacks = self._close_callbacks
        self._close_callbacks = []
        for callback in callbacks:
            try:
                await callback()
            except Exception:
                pass
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                break

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionClosedError(f"device session {self.device_id} is closed")
