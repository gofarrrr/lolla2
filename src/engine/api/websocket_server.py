#!/usr/bin/env python3
"""
METIS WebSocket Server for Real-Time Collaboration
Provides real-time engagement updates, phase progression, and collaboration features
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, Any
from uuid import uuid4
from dataclasses import dataclass, asdict

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.routing import APIRouter
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocket = None
    APIRouter = None


@dataclass
class WebSocketMessage:
    """Standard WebSocket message format"""

    type: str
    data: Any
    timestamp: str
    source: str
    message_id: str = None

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid4())


@dataclass
class ConnectedClient:
    """Connected WebSocket client information"""

    websocket: WebSocket
    connection_id: str
    user_id: str
    user_name: str
    engagement_id: Optional[str]
    connected_at: datetime
    last_heartbeat: datetime


class EngagementWebSocketManager:
    """Manages WebSocket connections and real-time collaboration for engagements"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Connection management
        self.connections: Dict[str, ConnectedClient] = {}  # connection_id -> client
        self.engagement_connections: Dict[str, Set[str]] = (
            {}
        )  # engagement_id -> connection_ids
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids

        # Message queues for offline delivery
        self.message_queues: Dict[str, List[WebSocketMessage]] = (
            {}
        )  # connection_id -> messages

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self.total_connections = 0
        self.total_messages = 0
        self.bytes_sent = 0
        self.bytes_received = 0

    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("🚀 WebSocket background tasks started")

    async def stop_background_tasks(self):
        """Stop background maintenance tasks"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self._cleanup_task:
            self._cleanup_task.cancel()

        self.logger.info("🛑 WebSocket background tasks stopped")

    async def connect_client(
        self,
        websocket: WebSocket,
        user_id: str,
        user_name: str = None,
        engagement_id: str = None,
    ) -> str:
        """Connect a new WebSocket client"""

        connection_id = str(uuid4())
        user_name = user_name or f"User-{user_id[:8]}"

        # Accept WebSocket connection
        await websocket.accept()

        # Create client record
        client = ConnectedClient(
            websocket=websocket,
            connection_id=connection_id,
            user_id=user_id,
            user_name=user_name,
            engagement_id=engagement_id,
            connected_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
        )

        # Store connection
        self.connections[connection_id] = client

        # Track by user
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)

        # Track by engagement if specified
        if engagement_id:
            if engagement_id not in self.engagement_connections:
                self.engagement_connections[engagement_id] = set()
            self.engagement_connections[engagement_id].add(connection_id)

        self.total_connections += 1

        self.logger.info(
            f"✅ Client connected: {connection_id} (user: {user_id}, engagement: {engagement_id})"
        )

        # Send connection established message
        await self.send_to_client(
            connection_id,
            WebSocketMessage(
                type="connection_established",
                data={
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "engagement_id": engagement_id,
                    "server_time": datetime.now(timezone.utc).isoformat(),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="server",
            ),
        )

        # Send queued messages if any
        await self._deliver_queued_messages(connection_id)

        # Notify engagement about new user
        if engagement_id:
            await self.broadcast_to_engagement(
                engagement_id,
                WebSocketMessage(
                    type="collaboration_event",
                    data={
                        "engagement_id": engagement_id,
                        "event_type": "user_joined",
                        "user_id": user_id,
                        "user_name": user_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="server",
                ),
                exclude_connection=connection_id,
            )

            # Send active users count
            await self._update_active_users_count(engagement_id)

        return connection_id

    async def disconnect_client(self, connection_id: str):
        """Disconnect a WebSocket client"""

        client = self.connections.get(connection_id)
        if not client:
            return

        # Remove from tracking
        self.connections.pop(connection_id, None)

        # Remove from user connections
        if client.user_id in self.user_connections:
            self.user_connections[client.user_id].discard(connection_id)
            if not self.user_connections[client.user_id]:
                del self.user_connections[client.user_id]

        # Remove from engagement connections
        if client.engagement_id and client.engagement_id in self.engagement_connections:
            self.engagement_connections[client.engagement_id].discard(connection_id)
            if not self.engagement_connections[client.engagement_id]:
                del self.engagement_connections[client.engagement_id]
            else:
                # Notify remaining users
                await self.broadcast_to_engagement(
                    client.engagement_id,
                    WebSocketMessage(
                        type="collaboration_event",
                        data={
                            "engagement_id": client.engagement_id,
                            "event_type": "user_left",
                            "user_id": client.user_id,
                            "user_name": client.user_name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="server",
                    ),
                )

                # Update active users count
                await self._update_active_users_count(client.engagement_id)

        # Clean up message queue
        self.message_queues.pop(connection_id, None)

        self.logger.info(
            f"❌ Client disconnected: {connection_id} (user: {client.user_id})"
        )

    async def handle_client_message(self, connection_id: str, message_data: dict):
        """Handle incoming message from client"""

        client = self.connections.get(connection_id)
        if not client:
            return

        # Update heartbeat
        client.last_heartbeat = datetime.now(timezone.utc)
        self.total_messages += 1

        message_type = message_data.get("type")
        data = message_data.get("data", {})

        self.logger.debug(f"📨 Message from {connection_id}: {message_type}")

        try:
            if message_type == "heartbeat":
                await self.send_to_client(
                    connection_id,
                    WebSocketMessage(
                        type="heartbeat_response",
                        data={"server_time": datetime.now(timezone.utc).isoformat()},
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="server",
                    ),
                )

            elif message_type == "join_engagement":
                await self._handle_join_engagement(client, data)

            elif message_type == "leave_engagement":
                await self._handle_leave_engagement(client, data)

            elif message_type == "add_comment":
                await self._handle_add_comment(client, data)

            elif message_type == "assumption_changed":
                await self._handle_assumption_changed(client, data)

            elif message_type == "export_requested":
                await self._handle_export_requested(client, data)

            else:
                self.logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            self.logger.error(f"Error handling message from {connection_id}: {e}")
            await self.send_to_client(
                connection_id,
                WebSocketMessage(
                    type="error",
                    data={"message": f"Failed to process message: {str(e)}"},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="server",
                ),
            )

    async def send_to_client(
        self, connection_id: str, message: WebSocketMessage
    ) -> bool:
        """Send message to specific client"""

        client = self.connections.get(connection_id)
        if not client:
            # Queue message for later delivery
            if connection_id not in self.message_queues:
                self.message_queues[connection_id] = []
            self.message_queues[connection_id].append(message)
            return False

        try:
            message_str = json.dumps(asdict(message))
            await client.websocket.send_text(message_str)
            self.bytes_sent += len(message_str)
            return True

        except Exception as e:
            self.logger.error(f"Failed to send message to {connection_id}: {e}")
            await self.disconnect_client(connection_id)
            return False

    async def broadcast_to_engagement(
        self,
        engagement_id: str,
        message: WebSocketMessage,
        exclude_connection: str = None,
    ):
        """Broadcast message to all clients in an engagement"""

        connection_ids = self.engagement_connections.get(engagement_id, set())

        for (
            connection_id
        ) in connection_ids.copy():  # Copy to avoid modification during iteration
            if connection_id != exclude_connection:
                await self.send_to_client(connection_id, message)

    async def broadcast_to_all(self, message: WebSocketMessage):
        """Broadcast message to all connected clients"""

        for connection_id in list(self.connections.keys()):  # Copy keys
            await self.send_to_client(connection_id, message)

    async def send_engagement_update(self, engagement_id: str, engagement_data: dict):
        """Send engagement status update to all clients in engagement"""

        message = WebSocketMessage(
            type="engagement_update",
            data=engagement_data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="metis_system",
        )

        await self.broadcast_to_engagement(engagement_id, message)

    async def send_phase_progress(self, engagement_id: str, phase_data: dict):
        """Send phase progress update to all clients in engagement"""

        message = WebSocketMessage(
            type="phase_progress",
            data=phase_data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="metis_system",
        )

        await self.broadcast_to_engagement(engagement_id, message)

    async def send_system_alert(self, alert_data: dict, engagement_id: str = None):
        """Send system alert to clients"""

        message = WebSocketMessage(
            type="system_alert",
            data=alert_data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="metis_system",
        )

        if engagement_id:
            await self.broadcast_to_engagement(engagement_id, message)
        else:
            await self.broadcast_to_all(message)

    # Private helper methods

    async def _handle_join_engagement(self, client: ConnectedClient, data: dict):
        """Handle client joining an engagement"""

        engagement_id = data.get("engagement_id")
        if not engagement_id:
            return

        # Remove from old engagement if any
        if client.engagement_id:
            old_connections = self.engagement_connections.get(
                client.engagement_id, set()
            )
            old_connections.discard(client.connection_id)

        # Add to new engagement
        client.engagement_id = engagement_id
        if engagement_id not in self.engagement_connections:
            self.engagement_connections[engagement_id] = set()
        self.engagement_connections[engagement_id].add(client.connection_id)

        # Notify other users
        await self.broadcast_to_engagement(
            engagement_id,
            WebSocketMessage(
                type="collaboration_event",
                data={
                    "engagement_id": engagement_id,
                    "event_type": "user_joined",
                    "user_id": client.user_id,
                    "user_name": client.user_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="server",
            ),
            exclude_connection=client.connection_id,
        )

        await self._update_active_users_count(engagement_id)

    async def _handle_leave_engagement(self, client: ConnectedClient, data: dict):
        """Handle client leaving an engagement"""

        if not client.engagement_id:
            return

        engagement_id = client.engagement_id

        # Remove from engagement
        connections = self.engagement_connections.get(engagement_id, set())
        connections.discard(client.connection_id)

        # Notify other users
        await self.broadcast_to_engagement(
            engagement_id,
            WebSocketMessage(
                type="collaboration_event",
                data={
                    "engagement_id": engagement_id,
                    "event_type": "user_left",
                    "user_id": client.user_id,
                    "user_name": client.user_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="server",
            ),
            exclude_connection=client.connection_id,
        )

        client.engagement_id = None
        await self._update_active_users_count(engagement_id)

    async def _handle_add_comment(self, client: ConnectedClient, data: dict):
        """Handle adding a comment"""

        if not client.engagement_id:
            return

        comment = data.get("comment", "")

        await self.broadcast_to_engagement(
            client.engagement_id,
            WebSocketMessage(
                type="collaboration_event",
                data={
                    "engagement_id": client.engagement_id,
                    "event_type": "comment_added",
                    "user_id": client.user_id,
                    "user_name": client.user_name,
                    "data": {"comment": comment},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="server",
            ),
        )

    async def _handle_assumption_changed(self, client: ConnectedClient, data: dict):
        """Handle assumption change notification"""

        if not client.engagement_id:
            return

        await self.broadcast_to_engagement(
            client.engagement_id,
            WebSocketMessage(
                type="collaboration_event",
                data={
                    "engagement_id": client.engagement_id,
                    "event_type": "assumption_changed",
                    "user_id": client.user_id,
                    "user_name": client.user_name,
                    "data": {
                        "assumption_id": data.get("assumption_id"),
                        "old_value": data.get("old_value"),
                        "new_value": data.get("new_value"),
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="server",
            ),
        )

    async def _handle_export_requested(self, client: ConnectedClient, data: dict):
        """Handle export request notification"""

        if not client.engagement_id:
            return

        await self.broadcast_to_engagement(
            client.engagement_id,
            WebSocketMessage(
                type="collaboration_event",
                data={
                    "engagement_id": client.engagement_id,
                    "event_type": "export_requested",
                    "user_id": client.user_id,
                    "user_name": client.user_name,
                    "data": {"format": data.get("format")},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="server",
            ),
        )

    async def _update_active_users_count(self, engagement_id: str):
        """Update and broadcast active users count for engagement"""

        count = len(self.engagement_connections.get(engagement_id, set()))

        await self.broadcast_to_engagement(
            engagement_id,
            WebSocketMessage(
                type="active_users_update",
                data={"count": count},
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="server",
            ),
        )

    async def _deliver_queued_messages(self, connection_id: str):
        """Deliver queued messages to newly connected client"""

        messages = self.message_queues.pop(connection_id, [])
        for message in messages:
            await self.send_to_client(connection_id, message)

    async def _heartbeat_loop(self):
        """Background task to monitor client heartbeats"""

        while True:
            try:
                current_time = datetime.now(timezone.utc)
                stale_connections = []

                for connection_id, client in self.connections.items():
                    time_since_heartbeat = current_time - client.last_heartbeat

                    if time_since_heartbeat.total_seconds() > 90:  # 90 second timeout
                        stale_connections.append(connection_id)

                # Clean up stale connections
                for connection_id in stale_connections:
                    self.logger.info(
                        f"⏰ Cleaning up stale connection: {connection_id}"
                    )
                    await self.disconnect_client(connection_id)

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self):
        """Background task for general cleanup"""

        while True:
            try:
                # Clean up empty engagement connections
                empty_engagements = [
                    eng_id
                    for eng_id, connections in self.engagement_connections.items()
                    if not connections
                ]

                for eng_id in empty_engagements:
                    del self.engagement_connections[eng_id]

                # Clean up empty user connections
                empty_users = [
                    user_id
                    for user_id, connections in self.user_connections.items()
                    if not connections
                ]

                for user_id in empty_users:
                    del self.user_connections[user_id]

                # Clean up old message queues
                old_queues = [
                    conn_id
                    for conn_id, messages in self.message_queues.items()
                    if messages
                    and (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(
                            messages[0].timestamp.replace("Z", "+00:00")
                        )
                    ).total_seconds()
                    > 3600
                ]

                for conn_id in old_queues:
                    del self.message_queues[conn_id]

                await asyncio.sleep(300)  # Clean up every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)

    def get_statistics(self) -> dict:
        """Get WebSocket server statistics"""

        return {
            "active_connections": len(self.connections),
            "total_connections": self.total_connections,
            "total_messages": self.total_messages,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "engagements_active": len(self.engagement_connections),
            "users_active": len(self.user_connections),
            "queued_messages": sum(
                len(queue) for queue in self.message_queues.values()
            ),
        }


# Global manager instance
_websocket_manager: Optional[EngagementWebSocketManager] = None


def get_websocket_manager() -> EngagementWebSocketManager:
    """Get or create global WebSocket manager"""
    global _websocket_manager

    if _websocket_manager is None:
        _websocket_manager = EngagementWebSocketManager()

    return _websocket_manager
