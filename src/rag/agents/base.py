"""Agent base classes and in-memory message bus."""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    BROADCAST = "broadcast"


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_agent: str = ""
    to_agent: str = ""
    message_type: MessageType = MessageType.REQUEST
    content: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def reply(self, content: dict[str, Any], *, error: bool = False) -> AgentMessage:
        return AgentMessage(
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            message_type=MessageType.ERROR if error else MessageType.RESPONSE,
            content=content,
            correlation_id=self.correlation_id,
        )


@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """Async in-memory pub/sub message bus for agent communication."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[AgentMessage], asyncio.Future]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, handler: Callable[[AgentMessage], Any]) -> None:
        async with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

    async def unsubscribe(self, topic: str, handler: Callable) -> None:
        async with self._lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [h for h in self._subscribers[topic] if h != handler]

    async def publish(self, topic: str, message: AgentMessage) -> None:
        handlers = self._subscribers.get(topic, [])
        await asyncio.gather(*[asyncio.ensure_future(h(message)) for h in handlers])

    async def request(self, topic: str, message: AgentMessage, timeout: float = 30.0) -> AgentMessage:
        """Publish and wait for a single response on the correlation ID."""
        fut: asyncio.Future[AgentMessage] = asyncio.get_event_loop().create_future()

        async def _collector(msg: AgentMessage) -> None:
            if msg.correlation_id == message.correlation_id and not fut.done():
                fut.set_result(msg)

        reply_topic = f"reply:{message.correlation_id}"
        await self.subscribe(reply_topic, _collector)
        try:
            await self.publish(topic, message)
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            await self.unsubscribe(reply_topic, _collector)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, name: str, bus: MessageBus) -> None:
        self.name = name
        self.bus = bus
        self._running = False

    @property
    @abstractmethod
    def capabilities(self) -> list[AgentCapability]:
        ...

    @abstractmethod
    async def handle(self, message: AgentMessage) -> AgentMessage:
        ...

    async def start(self) -> None:
        self._running = True
        await self.bus.subscribe(self.name, self._dispatch)
        await self.on_start()

    async def stop(self) -> None:
        self._running = False
        await self.bus.unsubscribe(self.name, self._dispatch)
        await self.on_stop()

    async def on_start(self) -> None:
        pass

    async def on_stop(self) -> None:
        pass

    async def publish(self, topic: str, message: AgentMessage) -> None:
        await self.bus.publish(topic, message)

    async def _dispatch(self, message: AgentMessage) -> None:
        response = await self.handle(message)
        reply_topic = f"reply:{message.correlation_id}"
        await self.bus.publish(reply_topic, response)
