# IndexTTS Streaming Server Guide

Real-time audio streaming for IndexTTS using WebSocket protocol.

## Features

- 🎵 **Real-time streaming** - Audio chunks transmitted as they're generated
- 🌐 **WebSocket protocol** - Bi-directional, low-latency communication
- 🎨 **Web UI** - Beautiful browser-based demo interface
- 🐍 **Python client** - Easy integration into Python applications
- 🔊 **Live playback** - Play audio while it's still generating (with PyAudio)
- 📊 **Progress tracking** - Real-time generation statistics
- 🎛️ **Full emotion control** - Audio, vector, and text-based emotion

## Architecture

```
┌─────────────┐         WebSocket         ┌──────────────────┐
│             │ ◄────────────────────────► │                  │
│   Client    │    JSON Messages           │  Streaming       │
│ (Browser/   │                             │  Server          │
│  Python)    │ ◄────────────────────────► │  (FastAPI)       │
│             │   Base64 Audio Chunks      │                  │
└─────────────┘                             └──────────────────┘
                                                     │
                                                     ↓
                                            ┌──────────────────┐
                                            │   IndexTTS2      │
                                            │   Model          │
                                            └──────────────────┘
```

### Message Flow

1. **Client → Server**: JSON request with text and parameters
2. **Server → Client**: Metadata (sample rate, chunk size)
3. **Server → Client**: Audio chunks (base64 encoded, streamed)
4. **Server → Client**: Completion message with statistics

## Installation

### Core Dependencies

```bash
# Install streaming server dependencies
uv pip install fastapi uvicorn websockets

# Optional: For real-time playback in Python client
uv pip install pyaudio  # May require system dependencies
```

### System Requirements

**For PyAudio (optional, for real-time playback):**

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# macOS
brew install portaudio

# Windows
# PyAudio wheels available from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

## Quick Start

### 1. Start the Server

```bash
# Quick start with script (from project root)
./stream/start_streaming.sh

# With custom configuration
TTS_FP16=true TTS_DEEPSPEED=true ./stream/start_streaming.sh

# Or with uvicorn directly (from project root)
uv run uvicorn stream.streaming_server:app --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`

### 2. Use Web UI

Open your browser and navigate to:
```
http://localhost:8000
```

The web interface provides:
- Text input for synthesis
- Speaker audio selection
- Emotion control (multiple modes)
- Real-time progress bar
- Live audio playback
- Generation statistics

### 3. Use Python Client

```python
import asyncio
from stream.streaming_client import StreamingTTSClient

async def test():
    client = StreamingTTSClient()

    await client.synthesize_streaming(
        text="Hello! This is streaming TTS.",
        spk_audio_prompt="examples/voice_01.wav",
        output_path="output.wav"
    )

asyncio.run(test())
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `TTS_CONFIG` | `checkpoints/config.yaml` | Model config path |
| `TTS_MODEL_DIR` | `checkpoints` | Model directory |
| `TTS_FP16` | `true` | Use FP16 inference |
| `TTS_CUDA_KERNEL` | `true` | Use CUDA kernels |
| `TTS_DEEPSPEED` | `false` | Use DeepSpeed |
| `TTS_CHUNK_SIZE` | `4096` | Samples per audio chunk |

### Example Configurations

**Development (Fast startup):**
```bash
TTS_FP16=false TTS_CUDA_KERNEL=false uv run streaming_server.py
```

**Production (Best performance):**
```bash
TTS_FP16=true TTS_CUDA_KERNEL=true TTS_DEEPSPEED=true uv run streaming_server.py
```

**Low latency (Smaller chunks):**
```bash
TTS_CHUNK_SIZE=2048 uv run streaming_server.py
```

## API Reference

### WebSocket Endpoint

**URL:** `ws://localhost:8000/ws/stream`

#### Request Format

```json
{
  "text": "Text to synthesize",
  "spk_audio_prompt": "path/to/speaker.wav",

  // Optional: Emotion control
  "emo_audio_prompt": "path/to/emotion.wav",
  "emo_vector": [0, 0, 0, 0, 0, 0, 0.5, 0],
  "emo_text": "Surprised and excited",
  "use_emo_text": false,
  "emo_alpha": 1.0,

  // Optional: Generation parameters
  "use_random": false
}
```

#### Response Messages

**1. Metadata**
```json
{
  "type": "metadata",
  "sample_rate": 24000,
  "chunk_size": 4096,
  "text": "Text to synthesize"
}
```

**2. Audio Chunk**
```json
{
  "type": "audio",
  "data": "base64_encoded_pcm16_audio",
  "chunk_id": 0,
  "sample_rate": 24000,
  "chunk_samples": 4096
}
```

**3. Completion**
```json
{
  "type": "complete",
  "total_chunks": 50,
  "total_samples": 204800,
  "total_duration": 8.53,
  "generation_time": 3.21
}
```

**4. Error**
```json
{
  "type": "error",
  "message": "Error description"
}
```

### REST Endpoints

#### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

## Python Client API

### StreamingTTSClient

```python
from stream.streaming_client import StreamingTTSClient

client = StreamingTTSClient(server_url="ws://localhost:8000/ws/stream")
```

#### Methods

##### `synthesize_streaming()`

Generate audio with streaming.

```python
await client.synthesize_streaming(
    text: str,
    spk_audio_prompt: str,
    output_path: str = "output_stream.wav",
    emo_audio_prompt: str = None,
    emo_vector: list = None,
    emo_text: str = None,
    use_emo_text: bool = False,
    emo_alpha: float = 1.0,
    use_random: bool = False,
    on_chunk_callback = None  # Optional: callback(chunk_id, chunk_data)
)
```

##### `synthesize_with_playback()`

Generate with real-time audio playback (requires PyAudio).

```python
await client.synthesize_with_playback(
    text: str,
    spk_audio_prompt: str,
    output_path: str = "output_stream.wav",
    **kwargs  # Same as synthesize_streaming
)
```

## Usage Examples

### Example 1: Basic Streaming

```python
import asyncio
from stream.streaming_client import StreamingTTSClient

async def basic_example():
    client = StreamingTTSClient()

    await client.synthesize_streaming(
        text="Hello world! This is streaming TTS.",
        spk_audio_prompt="examples/voice_01.wav",
        output_path="output.wav"
    )

asyncio.run(basic_example())
```

### Example 2: With Emotion Vector

```python
await client.synthesize_streaming(
    text="Wow! This is amazing!",
    spk_audio_prompt="examples/voice_10.wav",
    emo_vector=[0, 0, 0, 0, 0, 0, 0.7, 0],  # Surprised
    emo_alpha=0.8,
    output_path="surprised.wav"
)
```

### Example 3: Custom Chunk Callback

```python
import numpy as np

chunks_received = []

def on_chunk(chunk_id, chunk_data):
    chunks_received.append(chunk_data)
    print(f"Chunk {chunk_id}: {len(chunk_data)} samples, "
          f"RMS: {np.sqrt(np.mean(chunk_data**2)):.2f}")

await client.synthesize_streaming(
    text="Processing chunks in real-time...",
    spk_audio_prompt="examples/voice_01.wav",
    on_chunk_callback=on_chunk,
    output_path="output.wav"
)
```

### Example 4: Real-Time Playback

```python
# Requires: pip install pyaudio
await client.synthesize_with_playback(
    text="Listen to this as it generates!",
    spk_audio_prompt="examples/voice_01.wav",
    output_path="playback.wav"
)
```

### Example 5: Web Browser (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => {
    ws.send(JSON.stringify({
        text: "Hello from the browser!",
        spk_audio_prompt: "examples/voice_01.wav"
    }));
};

const audioChunks = [];

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'audio') {
        // Decode base64 audio
        const audioData = atob(msg.data);
        const audioArray = new Int16Array(audioData.length / 2);
        for (let i = 0; i < audioArray.length; i++) {
            const byte1 = audioData.charCodeAt(i * 2);
            const byte2 = audioData.charCodeAt(i * 2 + 1);
            audioArray[i] = (byte2 << 8) | byte1;
        }
        audioChunks.push(audioArray);

    } else if (msg.type === 'complete') {
        // Combine and play audio
        const combined = Int16Array.from(audioChunks.flat());
        playAudio(combined, msg.sample_rate);
    }
};
```

## Performance Tuning

### Chunk Size

Smaller chunks = lower latency, higher overhead
Larger chunks = higher latency, better throughput

| Chunk Size | Latency | Use Case |
|------------|---------|----------|
| 2048 | ~85ms | Interactive applications |
| 4096 | ~170ms | **Recommended balance** |
| 8192 | ~340ms | Batch processing |

### Model Optimization

```bash
# Fastest inference (recommended)
TTS_FP16=true TTS_CUDA_KERNEL=true TTS_DEEPSPEED=true

# Balanced
TTS_FP16=true TTS_CUDA_KERNEL=true

# Maximum compatibility
TTS_FP16=false TTS_CUDA_KERNEL=false
```

### Server Configuration

For production deployment:

```bash
# Use multiple workers (CPU-bound preprocessing)
uvicorn stream.streaming_server:app --workers 2 --host 0.0.0.0 --port 8000

# With SSL/TLS
uvicorn stream.streaming_server:app --ssl-keyfile key.pem --ssl-certfile cert.pem

# Behind nginx/proxy
uvicorn stream.streaming_server:app --proxy-headers --forwarded-allow-ips='*'
```

## Troubleshooting

### Issue: WebSocket connection fails

**Solution:**
```bash
# Check server is running
curl http://localhost:8000/health

# Check firewall allows port 8000
sudo ufw allow 8000
```

### Issue: Audio is choppy or distorted

**Solution:**
- Increase `TTS_CHUNK_SIZE` (larger chunks)
- Enable `TTS_FP16` and `TTS_CUDA_KERNEL` for faster generation
- Check network bandwidth (streaming requires ~384 kbps)

### Issue: High latency

**Solution:**
- Decrease `TTS_CHUNK_SIZE` (smaller chunks)
- Use local server (avoid network latency)
- Enable DeepSpeed for faster generation

### Issue: PyAudio installation fails

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-dev
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio‑0.2.11‑cp312‑cp312‑win_amd64.whl
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install uv && \
    uv pip install --system fastapi uvicorn websockets

# Copy application
COPY . .

# Download models
RUN git lfs pull

EXPOSE 8000

CMD ["uvicorn", "stream.streaming_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t indextts-streaming .
docker run -p 8000:8000 --gpus all indextts-streaming
```

### Systemd Service

Create `/etc/systemd/system/indextts-streaming.service`:

```ini
[Unit]
Description=IndexTTS Streaming Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/index-tts
Environment="TTS_FP16=true"
Environment="TTS_CUDA_KERNEL=true"
ExecStart=/opt/index-tts/.venv/bin/uvicorn stream.streaming_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable indextts-streaming
sudo systemctl start indextts-streaming
```

## Benchmarks

Tested on NVIDIA RTX 4090, FP16 + CUDA kernels:

| Text Length | Generation Time | Audio Duration | Real-Time Factor | Chunks | Latency (first chunk) |
|-------------|----------------|----------------|------------------|--------|-----------------------|
| 20 words | 1.2s | 3.5s | 0.34x | 21 | ~1.2s |
| 50 words | 2.8s | 8.7s | 0.32x | 53 | ~1.3s |
| 100 words | 5.1s | 17.4s | 0.29x | 105 | ~1.4s |

Network overhead: ~50-100ms per chunk (localhost)

## Security Considerations

1. **Authentication**: Add API key validation in production
2. **Rate limiting**: Limit requests per client
3. **Input validation**: Sanitize file paths
4. **CORS**: Configure allowed origins
5. **SSL/TLS**: Use HTTPS/WSS in production

Example with authentication:

```python
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    # Receive API key
    auth_msg = await websocket.receive_text()
    api_key = json.loads(auth_msg).get("api_key")

    if not validate_api_key(api_key):
        await websocket.close(code=4001, reason="Invalid API key")
        return

    # Continue with streaming...
```

## License

Same as IndexTTS main project. See [LICENSE](LICENSE).

## Support

- GitHub Issues: https://github.com/index-tts/index-tts/issues
- Email: indexspeech@bilibili.com
- Discord: https://discord.gg/uT32E7KDmy

---

**Last Updated:** 2025-10-25
**Version:** 1.0.0
