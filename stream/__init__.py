"""
IndexTTS Streaming Server Package

Provides WebSocket-based streaming TTS capabilities for IndexTTS2.
"""

from stream.streaming_client import StreamingTTSClient
from stream.streaming_server import StreamingIndexTTS

__all__ = ["StreamingTTSClient", "StreamingIndexTTS"]
