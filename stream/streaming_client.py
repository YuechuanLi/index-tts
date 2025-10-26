#!/usr/bin/env python3
"""
IndexTTS Streaming Client Example
Demonstrates how to connect and receive streaming audio
"""

import asyncio
import base64
import json
import wave
from pathlib import Path

import numpy as np
import websockets


class StreamingTTSClient:
    """Client for IndexTTS streaming server"""

    def __init__(self, server_url: str = "ws://localhost:8000/ws/stream"):
        """
        Initialize streaming client

        Args:
            server_url: WebSocket server URL
        """
        self.server_url = server_url
        self.audio_chunks = []
        self.sample_rate = 24000
        self.total_chunks = 0

    async def synthesize_streaming(
        self,
        text: str,
        spk_audio_prompt: str,
        output_path: str = "output_stream.wav",
        emo_audio_prompt: str = None,
        emo_vector: list = None,
        emo_text: str = None,
        use_emo_text: bool = False,
        emo_alpha: float = 1.0,
        use_random: bool = False,
        on_chunk_callback=None,
    ):
        """
        Synthesize speech with streaming

        Args:
            text: Text to synthesize
            spk_audio_prompt: Speaker reference audio path
            output_path: Output WAV file path
            emo_audio_prompt: Emotion reference audio (optional)
            emo_vector: 8-dim emotion vector (optional)
            emo_text: Emotion description text (optional)
            use_emo_text: Use text-based emotion
            emo_alpha: Emotion intensity
            use_random: Use stochastic sampling
            on_chunk_callback: Callback for each audio chunk (chunk_id, chunk_data)
        """
        print(f"Connecting to {self.server_url}...")

        async with websockets.connect(self.server_url) as websocket:
            # Build request
            request = {
                "text": text,
                "spk_audio_prompt": spk_audio_prompt,
                "emo_alpha": emo_alpha,
                "use_random": use_random,
            }

            if emo_audio_prompt:
                request["emo_audio_prompt"] = emo_audio_prompt
            if emo_vector:
                request["emo_vector"] = emo_vector
            if emo_text:
                request["emo_text"] = emo_text
                request["use_emo_text"] = True
            elif use_emo_text:
                request["use_emo_text"] = True

            # Send request
            await websocket.send(json.dumps(request))
            print(f"Sent request: text='{text[:50]}...'")

            self.audio_chunks = []
            self.total_chunks = 0

            # Receive streaming response
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "metadata":
                    self.sample_rate = data.get("sample_rate", 24000)
                    print(
                        f"Metadata: sample_rate={self.sample_rate}Hz, chunk_size={data.get('chunk_size')}"
                    )

                elif msg_type == "audio":
                    chunk_id = data["chunk_id"]
                    chunk_b64 = data["data"]

                    # Decode audio chunk
                    chunk_bytes = base64.b64decode(chunk_b64)
                    chunk_array = np.frombuffer(chunk_bytes, dtype=np.int16)

                    self.audio_chunks.append(chunk_array)
                    self.total_chunks += 1

                    print(
                        f"Received chunk {chunk_id}: {len(chunk_array)} samples", end="\r"
                    )

                    # Call callback if provided
                    if on_chunk_callback:
                        on_chunk_callback(chunk_id, chunk_array)

                elif msg_type == "complete":
                    print(f"\nGeneration complete!")
                    print(f"  Total chunks: {data['total_chunks']}")
                    print(f"  Total samples: {data['total_samples']}")
                    print(f"  Audio duration: {data['total_duration']:.2f}s")
                    print(f"  Generation time: {data['generation_time']:.2f}s")
                    print(
                        f"  Real-time factor: {data['generation_time'] / data['total_duration']:.2f}x"
                    )

                    # Save combined audio
                    self._save_audio(output_path)
                    print(f"  Saved to: {output_path}")

                elif msg_type == "error":
                    print(f"Error: {data['message']}")
                    raise Exception(data["message"])

    def _save_audio(self, output_path: str):
        """Save combined audio chunks to WAV file"""
        if not self.audio_chunks:
            print("No audio chunks to save")
            return

        # Combine all chunks
        combined = np.concatenate(self.audio_chunks)

        # Convert to float32
        audio_float = combined.astype(np.float32) / 32767.0

        # Save as WAV
        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(combined.tobytes())

    async def synthesize_with_playback(
        self,
        text: str,
        spk_audio_prompt: str,
        output_path: str = "output_stream.wav",
        **kwargs,
    ):
        """
        Synthesize with real-time playback (requires pyaudio)

        Args:
            text: Text to synthesize
            spk_audio_prompt: Speaker reference
            output_path: Output file
            **kwargs: Additional arguments for synthesize_streaming
        """
        try:
            import pyaudio

            # Initialize PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=4096,
            )

            def play_chunk(chunk_id, chunk_data):
                """Callback to play chunk in real-time"""
                stream.write(chunk_data.tobytes())

            # Synthesize with playback
            await self.synthesize_streaming(
                text=text,
                spk_audio_prompt=spk_audio_prompt,
                output_path=output_path,
                on_chunk_callback=play_chunk,
                **kwargs,
            )

            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()

        except ImportError:
            print(
                "PyAudio not installed. Install with: pip install pyaudio"
            )
            print("Falling back to file-only mode...")
            await self.synthesize_streaming(
                text=text,
                spk_audio_prompt=spk_audio_prompt,
                output_path=output_path,
                **kwargs,
            )


async def main():
    """Example usage"""
    client = StreamingTTSClient(server_url="ws://localhost:8000/ws/stream")

    # Example 1: Basic streaming
    print("=" * 60)
    print("Example 1: Basic Voice Cloning with Streaming")
    print("=" * 60)
    await client.synthesize_streaming(
        text="Hello! This is a test of the streaming TTS system. The audio is being generated and transmitted in real-time.",
        spk_audio_prompt="examples/voice_01.wav",
        output_path="output_stream1.wav",
    )

    # Example 2: With emotion control
    print("\n" + "=" * 60)
    print("Example 2: With Emotion Vector")
    print("=" * 60)
    await client.synthesize_streaming(
        text="Wow! This is amazing! I can't believe how well this works!",
        spk_audio_prompt="examples/voice_10.wav",
        emo_vector=[0, 0, 0, 0, 0, 0, 0.7, 0],  # Surprised
        emo_alpha=0.8,
        output_path="output_stream2.wav",
    )

    # Example 3: With text-based emotion
    print("\n" + "=" * 60)
    print("Example 3: With Text-Based Emotion")
    print("=" * 60)
    await client.synthesize_streaming(
        text="快躲起来！他要来了！",
        spk_audio_prompt="examples/voice_12.wav",
        use_emo_text=True,
        emo_alpha=0.6,
        output_path="output_stream3.wav",
    )

    # Example 4: With real-time playback (if pyaudio available)
    print("\n" + "=" * 60)
    print("Example 4: With Real-Time Playback")
    print("=" * 60)
    await client.synthesize_with_playback(
        text="This audio will play in real-time as it streams from the server!",
        spk_audio_prompt="examples/voice_01.wav",
        output_path="output_stream4.wav",
    )


if __name__ == "__main__":
    asyncio.run(main())
