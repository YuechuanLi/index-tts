#!/usr/bin/env python3
"""
IndexTTS Streaming Server with WebSocket
Provides real-time audio streaming during generation
"""

import asyncio
import base64
import json
import logging
import os
import queue
import tempfile
import threading
import time
import wave
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from indextts.infer_v2 import IndexTTS2


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StreamingIndexTTS:
    """Wrapper for IndexTTS2 with streaming capabilities"""

    def __init__(
        self,
        cfg_path: str = "checkpoints/config.yaml",
        model_dir: str = "checkpoints",
        use_fp16: bool = True,
        use_cuda_kernel: bool = True,
        use_deepspeed: bool = False,
        device: Optional[str] = None,
        chunk_size: int = 4096,  # Audio samples per chunk
    ):
        """
        Initialize streaming TTS model

        Args:
            cfg_path: Path to config file
            model_dir: Path to model directory
            use_fp16: Use FP16 inference
            use_cuda_kernel: Use CUDA kernels
            use_deepspeed: Use DeepSpeed
            device: Target device
            chunk_size: Audio samples per streaming chunk
        """
        logger.info("Initializing StreamingIndexTTS...")
        self.tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed,
            device=device,
        )
        self.chunk_size = chunk_size
        self.sample_rate = 24000  # IndexTTS output sample rate
        logger.info(f"StreamingIndexTTS initialized (chunk_size={chunk_size})")

    async def stream_generate(
        self,
        text: str,
        spk_audio_prompt: str,
        emo_audio_prompt: Optional[str] = None,
        emo_vector: Optional[list] = None,
        emo_text: Optional[str] = None,
        use_emo_text: bool = False,
        emo_alpha: float = 1.0,
        use_random: bool = False,
    ):
        """
        Generate audio with streaming output

        Yields:
            dict: Audio chunk with metadata
                {
                    "type": "audio" | "metadata" | "complete" | "error",
                    "data": base64 encoded audio or metadata,
                    "chunk_id": int,
                    "sample_rate": int,
                    "total_duration": float (only in complete message)
                }
        """
        chunk_id = 0
        start_time = time.time()
        total_samples = 0

        sample_rate_holder = [self.sample_rate]

        try:
            # Send metadata first
            yield {
                "type": "metadata",
                "sample_rate": sample_rate_holder[0],
                "chunk_size": self.chunk_size,
                "text": text,
            }

            # Generate audio in background thread
            audio_queue = queue.Queue(maxsize=10)
            exception_holder = [None]

            def generate_audio():
                try:
                    # Use a temporary file for full audio generation (auto-cleaned)
                    fd, temp_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)

                    try:
                        # Generate full audio
                        self.tts.infer(
                            spk_audio_prompt=spk_audio_prompt,
                            text=text,
                            output_path=temp_path,
                            emo_audio_prompt=emo_audio_prompt,
                            emo_vector=emo_vector,
                            emo_text=emo_text,
                            use_emo_text=use_emo_text,
                            emo_alpha=emo_alpha,
                            use_random=use_random,
                            verbose=False,
                        )

                        # Load generated audio
                        with wave.open(temp_path, "rb") as wav_file:
                            sr = wav_file.getframerate()
                            channels = wav_file.getnchannels()
                            frames = wav_file.getnframes()
                            audio_bytes = wav_file.readframes(frames)

                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                        if channels > 1:
                            # average stereo channels to mono to match downstream expectations
                            audio_np = audio_np.reshape(-1, channels).mean(axis=1)
                        audio_np = audio_np.astype(np.float32) / 32767.0
                        sample_rate_holder[0] = int(sr)
                    finally:
                        # Clean up temp file
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

                    # Split into chunks and queue
                    for i in range(0, len(audio_np), self.chunk_size):
                        chunk = audio_np[i : i + self.chunk_size]
                        audio_queue.put(chunk)

                    audio_queue.put(None)  # Signal completion

                except Exception as e:
                    exception_holder[0] = e
                    audio_queue.put(None)

            # Start generation thread
            gen_thread = threading.Thread(target=generate_audio, daemon=True)
            gen_thread.start()

            # Stream chunks as they become available
            while True:
                # Check for exception
                if exception_holder[0]:
                    raise exception_holder[0]

                # Get next chunk (with timeout)
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if chunk is None:
                    break

                # Convert to bytes and encode
                chunk_bytes = (chunk * 32767).astype(np.int16).tobytes()
                current_sample_rate = sample_rate_holder[0]
                chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")

                total_samples += len(chunk)

                yield {
                    "type": "audio",
                    "data": chunk_b64,
                    "chunk_id": chunk_id,
                    "sample_rate": current_sample_rate,
                    "chunk_samples": len(chunk),
                }

                chunk_id += 1

                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.001)

            # Send completion message
            duration = time.time() - start_time
            yield {
                "type": "complete",
                "total_chunks": chunk_id,
                "total_samples": total_samples,
                "total_duration": total_samples / max(sample_rate_holder[0], 1),
                "generation_time": duration,
            }

            logger.info(
                f"Streaming complete: {chunk_id} chunks, {duration:.2f}s generation time"
            )

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield {"type": "error", "message": str(e)}


# FastAPI application
app = FastAPI(title="IndexTTS Streaming Server", version="1.0.0")

# Global model state
streaming_tts: Optional[StreamingIndexTTS] = None
model_ready_event: asyncio.Event | None = None
model_load_error: Optional[Exception] = None


async def _load_model_async(config: dict):
    """Background task that loads the model without blocking startup."""
    global streaming_tts, model_load_error
    assert model_ready_event is not None, "model_ready_event must be initialized"

    try:
        logger.info("Starting IndexTTS Streaming Server...")
        streaming_tts = await asyncio.to_thread(StreamingIndexTTS, **config)
        logger.info("IndexTTS model loaded and ready!")
    except Exception as exc:
        model_load_error = exc
        logger.error("Failed to load IndexTTS model", exc_info=True)
    finally:
        model_ready_event.set()


@app.on_event("startup")
async def startup_event():
    """Kick off asynchronous model initialization."""
    global model_ready_event
    logger.info("Scheduling IndexTTS model load task...")
    model_ready_event = asyncio.Event()

    config = {
        "cfg_path": os.getenv("TTS_CONFIG", "checkpoints/config.yaml"),
        "model_dir": os.getenv("TTS_MODEL_DIR", "checkpoints"),
        "use_fp16": os.getenv("TTS_FP16", "true").lower() == "true",
        "use_cuda_kernel": os.getenv("TTS_CUDA_KERNEL", "true").lower() == "true",
        "use_deepspeed": os.getenv("TTS_DEEPSPEED", "false").lower() == "true",
        "chunk_size": int(os.getenv("TTS_CHUNK_SIZE", "4096")),
    }

    asyncio.create_task(_load_model_async(config))


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS"""
    await websocket.accept()
    logger.info(f"WebSocket connection established: {websocket.client}")

    if model_ready_event is None:
        await websocket.close(code=1011)
        return

    await model_ready_event.wait()

    if model_load_error:
        await websocket.send_json(
            {"type": "error", "message": f"Model failed to load: {model_load_error}"}
        )
        await websocket.close(code=1011)
        return

    if streaming_tts is None:
        await websocket.send_json(
            {"type": "error", "message": "Model not available"}
        )
        await websocket.close(code=1011)
        return

    try:
        while True:
            # Receive request from client
            data = await websocket.receive_text()
            request = json.loads(data)

            logger.info(f"Received request: text='{request.get('text', '')[:50]}...'")

            # Validate request
            if "text" not in request or "spk_audio_prompt" not in request:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Missing required fields: text, spk_audio_prompt",
                    }
                )
                continue

            # Stream audio generation
            async for chunk in streaming_tts.stream_generate(
                text=request["text"],
                spk_audio_prompt=request["spk_audio_prompt"],
                emo_audio_prompt=request.get("emo_audio_prompt"),
                emo_vector=request.get("emo_vector"),
                emo_text=request.get("emo_text"),
                use_emo_text=request.get("use_emo_text", False),
                emo_alpha=request.get("emo_alpha", 1.0),
                use_random=request.get("use_random", False),
            ):
                await websocket.send_json(chunk)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


@app.get("/")
async def get_demo_page():
    """Serve demo HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IndexTTS Streaming Demo</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                text-align: center;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                font-weight: bold;
                margin-bottom: 5px;
                color: #333;
            }
            input, textarea, select {
                width: 100%;
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                box-sizing: border-box;
            }
            textarea {
                min-height: 100px;
                font-family: inherit;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                margin-right: 10px;
            }
            button:hover {
                opacity: 0.9;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            #status {
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
                display: none;
            }
            #status.info {
                background: #e3f2fd;
                color: #1976d2;
                border-left: 4px solid #1976d2;
            }
            #status.success {
                background: #e8f5e9;
                color: #388e3c;
                border-left: 4px solid #388e3c;
            }
            #status.error {
                background: #ffebee;
                color: #c62828;
                border-left: 4px solid #c62828;
            }
            #progress {
                width: 100%;
                height: 30px;
                background: #f0f0f0;
                border-radius: 15px;
                overflow: hidden;
                margin: 20px 0;
                display: none;
            }
            #progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                width: 0%;
                transition: width 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }
            .audio-player {
                margin: 20px 0;
                display: none;
            }
            audio {
                width: 100%;
            }
            .stats {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                display: none;
            }
            .stats h3 {
                margin-top: 0;
            }
            .stat-item {
                margin: 5px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ IndexTTS Streaming Demo</h1>

            <div class="form-group">
                <label>Text to synthesize:</label>
                <textarea id="text" placeholder="Enter text here...">Hello! This is a streaming test of IndexTTS. The audio will play in real-time as it generates.</textarea>
            </div>

            <div class="form-group">
                <label>Speaker audio prompt (path):</label>
                <input type="text" id="spk_audio" value="examples/voice_01.wav" placeholder="examples/voice_01.wav">
            </div>

            <div class="form-group">
                <label>Emotion control (optional):</label>
                <select id="emo_mode">
                    <option value="none">None</option>
                    <option value="audio">Audio reference</option>
                    <option value="vector">Vector (8-dim)</option>
                    <option value="text">Text description</option>
                </select>
            </div>

            <div id="emo_audio_group" class="form-group" style="display:none;">
                <label>Emotion audio path:</label>
                <input type="text" id="emo_audio" placeholder="examples/emo_sad.wav">
            </div>

            <div id="emo_vector_group" class="form-group" style="display:none;">
                <label>Emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]:</label>
                <input type="text" id="emo_vector" value="0,0,0,0,0,0,0.5,0" placeholder="0,0,0,0,0,0,0.5,0">
            </div>

            <div id="emo_text_group" class="form-group" style="display:none;">
                <label>Emotion description:</label>
                <input type="text" id="emo_text" placeholder="Surprised and excited">
            </div>

            <div class="form-group">
                <label>Emotion intensity (0.0 - 1.0):</label>
                <input type="number" id="emo_alpha" value="1.0" min="0" max="1" step="0.1">
            </div>

            <button id="generateBtn" onclick="startStreaming()">🚀 Start Streaming</button>
            <button id="stopBtn" onclick="stopStreaming()" disabled>⏹️ Stop</button>

            <div id="status"></div>
            <div id="progress">
                <div id="progress-bar">0%</div>
            </div>

            <div class="audio-player" id="audioContainer">
                <h3>🔊 Streaming Audio</h3>
                <audio id="audioPlayer" controls></audio>
            </div>

            <div class="stats" id="stats">
                <h3>📊 Statistics</h3>
                <div class="stat-item"><strong>Chunks received:</strong> <span id="stat-chunks">0</span></div>
                <div class="stat-item"><strong>Total samples:</strong> <span id="stat-samples">0</span></div>
                <div class="stat-item"><strong>Duration:</strong> <span id="stat-duration">0.0s</span></div>
                <div class="stat-item"><strong>Generation time:</strong> <span id="stat-gentime">0.0s</span></div>
            </div>
        </div>

        <script>
            let ws = null;
            let audioContext = null;
            let audioBuffer = [];
            let mediaSource = null;
            let sourceBuffer = null;

            // Show/hide emotion controls
            document.getElementById('emo_mode').addEventListener('change', function() {
                const mode = this.value;
                document.getElementById('emo_audio_group').style.display = mode === 'audio' ? 'block' : 'none';
                document.getElementById('emo_vector_group').style.display = mode === 'vector' ? 'block' : 'none';
                document.getElementById('emo_text_group').style.display = mode === 'text' ? 'block' : 'none';
            });

            function showStatus(message, type) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = type;
                status.style.display = 'block';
            }

            function updateProgress(percent, text) {
                const progress = document.getElementById('progress');
                const bar = document.getElementById('progress-bar');
                progress.style.display = 'block';
                bar.style.width = percent + '%';
                bar.textContent = text || (percent + '%');
            }

            async function startStreaming() {
                const text = document.getElementById('text').value;
                const spk_audio = document.getElementById('spk_audio').value;
                const emo_mode = document.getElementById('emo_mode').value;
                const emo_alpha = parseFloat(document.getElementById('emo_alpha').value);

                if (!text || !spk_audio) {
                    showStatus('Please fill in required fields', 'error');
                    return;
                }

                // Build request
                const request = {
                    text: text,
                    spk_audio_prompt: spk_audio,
                    emo_alpha: emo_alpha
                };

                if (emo_mode === 'audio') {
                    request.emo_audio_prompt = document.getElementById('emo_audio').value;
                } else if (emo_mode === 'vector') {
                    const vec_str = document.getElementById('emo_vector').value;
                    request.emo_vector = vec_str.split(',').map(v => parseFloat(v.trim()));
                } else if (emo_mode === 'text') {
                    request.use_emo_text = true;
                    request.emo_text = document.getElementById('emo_text').value;
                }

                // Initialize WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);

                audioBuffer = [];
                let chunkCount = 0;

                document.getElementById('generateBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('stats').style.display = 'none';
                showStatus('Connecting to server...', 'info');

                ws.onopen = () => {
                    showStatus('Generating audio...', 'info');
                    updateProgress(0, 'Starting...');
                    ws.send(JSON.stringify(request));
                };

                ws.onmessage = async (event) => {
                    const message = JSON.parse(event.data);

                    if (message.type === 'metadata') {
                        showStatus(`Sample rate: ${message.sample_rate}Hz, Chunk size: ${message.chunk_size}`, 'info');
                    } else if (message.type === 'audio') {
                        chunkCount++;
                        updateProgress(Math.min(95, chunkCount * 5), `Chunk ${chunkCount}`);

                        // Decode base64 audio
                        const audioData = atob(message.data);
                        const audioArray = new Int16Array(audioData.length / 2);
                        for (let i = 0; i < audioArray.length; i++) {
                            audioArray[i] = (audioData.charCodeAt(i * 2 + 1) << 8) | audioData.charCodeAt(i * 2);
                        }
                        audioBuffer.push(audioArray);

                        // Update stats
                        document.getElementById('stat-chunks').textContent = chunkCount;
                        document.getElementById('stat-samples').textContent = message.chunk_samples * chunkCount;

                    } else if (message.type === 'complete') {
                        updateProgress(100, 'Complete!');
                        showStatus('Generation complete! Playing audio...', 'success');

                        // Combine all chunks
                        const totalLength = audioBuffer.reduce((sum, arr) => sum + arr.length, 0);
                        const combined = new Int16Array(totalLength);
                        let offset = 0;
                        for (const chunk of audioBuffer) {
                            combined.set(chunk, offset);
                            offset += chunk.length;
                        }

                        // Convert to float and create audio blob
                        const floatArray = new Float32Array(combined.length);
                        for (let i = 0; i < combined.length; i++) {
                            floatArray[i] = combined[i] / 32768.0;
                        }

                        // Create audio context and play
                        if (!audioContext) {
                            audioContext = new (window.AudioContext || window.webkitAudioContext)();
                        }

                        const audioBufferObj = audioContext.createBuffer(1, floatArray.length, message.sample_rate || 24000);
                        audioBufferObj.getChannelData(0).set(floatArray);

                        // Create WAV blob and set to audio player
                        const wavBlob = bufferToWave(audioBufferObj, audioBufferObj.length);
                        const audioUrl = URL.createObjectURL(wavBlob);
                        const audioPlayer = document.getElementById('audioPlayer');
                        audioPlayer.src = audioUrl;
                        document.getElementById('audioContainer').style.display = 'block';
                        audioPlayer.play();

                        // Update final stats
                        document.getElementById('stats').style.display = 'block';
                        document.getElementById('stat-chunks').textContent = message.total_chunks;
                        document.getElementById('stat-samples').textContent = message.total_samples;
                        document.getElementById('stat-duration').textContent = message.total_duration.toFixed(2) + 's';
                        document.getElementById('stat-gentime').textContent = message.generation_time.toFixed(2) + 's';

                        ws.close();
                        document.getElementById('generateBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;

                    } else if (message.type === 'error') {
                        showStatus('Error: ' + message.message, 'error');
                        ws.close();
                        document.getElementById('generateBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                    }
                };

                ws.onerror = (error) => {
                    showStatus('WebSocket error: ' + error, 'error');
                    document.getElementById('generateBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                };

                ws.onclose = () => {
                    document.getElementById('generateBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                };
            }

            function stopStreaming() {
                if (ws) {
                    ws.close();
                    showStatus('Streaming stopped', 'info');
                }
            }

            // Convert AudioBuffer to WAV blob
            function bufferToWave(abuffer, len) {
                const numOfChan = abuffer.numberOfChannels;
                const length = len * numOfChan * 2 + 44;
                const buffer = new ArrayBuffer(length);
                const view = new DataView(buffer);
                const channels = [];
                let sample;
                let offset = 0;
                let pos = 0;

                // Write WAV header
                setUint32(0x46464952); // "RIFF"
                setUint32(length - 8); // file length - 8
                setUint32(0x45564157); // "WAVE"

                setUint32(0x20746d66); // "fmt " chunk
                setUint32(16); // length = 16
                setUint16(1); // PCM (uncompressed)
                setUint16(numOfChan);
                setUint32(abuffer.sampleRate);
                setUint32(abuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
                setUint16(numOfChan * 2); // block-align
                setUint16(16); // 16-bit

                setUint32(0x61746164); // "data" - chunk
                setUint32(length - pos - 4); // chunk length

                // Write interleaved data
                for (let i = 0; i < abuffer.numberOfChannels; i++)
                    channels.push(abuffer.getChannelData(i));

                while (pos < length) {
                    for (let i = 0; i < numOfChan; i++) {
                        sample = Math.max(-1, Math.min(1, channels[i][offset]));
                        sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
                        view.setInt16(pos, sample, true);
                        pos += 2;
                    }
                    offset++;
                }

                return new Blob([buffer], { type: 'audio/wav' });

                function setUint16(data) {
                    view.setUint16(pos, data, true);
                    pos += 2;
                }

                function setUint32(data) {
                    view.setUint32(pos, data, true);
                    pos += 4;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": streaming_tts is not None,
        "model_ready": model_ready_event.is_set() if model_ready_event else False,
        "model_error": str(model_load_error) if model_load_error else None,
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(app, host=host, port=port, log_level="info")
