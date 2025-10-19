"""
WebSocket Connection Manager for METIS Engagement API
"""

from typing import Dict, Any
from fastapi import WebSocket


class ConnectionManager:
    """WebSocket connection manager for real-time engagement updates"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, engagement_id: str):
        """Connect and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[engagement_id] = websocket

    def disconnect(self, engagement_id: str):
        """Disconnect and unregister a WebSocket connection"""
        if engagement_id in self.active_connections:
            del self.active_connections[engagement_id]

    async def send_update(self, engagement_id: str, data: Dict[str, Any]):
        """Send update to a specific engagement's WebSocket connection"""
        if engagement_id in self.active_connections:
            try:
                await self.active_connections[engagement_id].send_json(data)
            except:
                self.disconnect(engagement_id)

    async def broadcast_system_update(self, data: Dict[str, Any]):
        """Broadcast system update to all active connections"""
        disconnected = []
        for engagement_id, connection in self.active_connections.items():
            try:
                await connection.send_json(data)
            except:
                disconnected.append(engagement_id)

        for engagement_id in disconnected:
            self.disconnect(engagement_id)
