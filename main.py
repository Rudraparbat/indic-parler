import torch
import soundfile as sf
import io
import os
import re
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from huggingface_hub import login
from pydantic import BaseModel
from typing import Optional

# ─────────────────────────────────────────────
# HuggingFace Login
# ─────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged into HuggingFace!")
else:
    raise RuntimeError("HF_TOKEN environment variable not set!")

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
desc_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)
print("Model ready!")

# ─────────────────────────────────────────────
# Voice → Description mapping (OpenAI voices)
# ─────────────────────────────────────────────
VOICE_MAP = {
    "alloy":   "A male speaker with a clear and neutral tone, moderate speed. The recording is of very high quality.",
    "echo":    "A male speaker with a deep and calm voice, slow speed. The recording is of very high quality.",
    "fable":   "A female speaker with a warm and expressive voice, moderate speed. The recording is of very high quality.",
    "onyx":    "A male speaker with a deep authoritative voice, moderate speed. The recording is of very high quality.",
    "nova":    "A female speaker with a bright and energetic voice, fast speed. The recording is of very high quality.",
    "shimmer": "A female speaker with a soft and gentle voice, slow speed. The recording is of very high quality.",
    "rohit":   "Rohit speaks in a moderate speed with a clear and natural tone. The recording is of very high quality.",
    "leela":   "Leela speaks in a high-pitched cheerful tone with moderate speed. The recording is of very high quality.",
}
DEFAULT_VOICE = "alloy"

def get_description(voice: str) -> str:
    return VOICE_MAP.get(voice.lower(), VOICE_MAP[DEFAULT_VOICE])

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
SENTENCE_END = re.compile(r'[।.!?]')
MINIMUM_CHUNK_CHARS = 25

def synthesize_chunk(text: str, desc_ids) -> bytes:
    prompt_ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        generation = model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask
        )
    audio = generation.cpu().numpy().squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, model.config.sampling_rate, format="WAV")
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class TTSRequest(BaseModel):
    model: Optional[str] = "tts-1"
    input: str
    voice: Optional[str] = DEFAULT_VOICE
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0

# ─────────────────────────────────────────────
# OpenAI-compatible: POST /v1/audio/speech
# Used by: OpenAI SDK, curl, external clients
# ─────────────────────────────────────────────
@app.post("/v1/audio/speech")
def synthesize(req: TTSRequest):
    description = get_description(req.voice)
    desc_ids = desc_tokenizer(description, return_tensors="pt").to(device)
    audio_bytes = synthesize_chunk(req.input, desc_ids)
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )

# ─────────────────────────────────────────────
# Internal WebSocket: /v1/audio/stream
# Used by: your LLM backend pipeline only
# NOT OpenAI compatible — internal use only
#
# Flow:
#   1. LLM streams text tokens to this endpoint
#   2. Server buffers until sentence boundary hit
#   3. Server synthesizes and sends audio bytes back immediately
#   4. Repeat until LLM stream ends
#
# This endpoint is called by YOUR backend, not external clients
# External clients always use POST /v1/audio/speech above
# ─────────────────────────────────────────────
@app.websocket("/v1/audio/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    buffer = ""
    desc_ids = None
    loop = asyncio.get_event_loop()

    # Default desc_ids
    default_desc = get_description(DEFAULT_VOICE)

    try:
        async for message in websocket.iter_json():
            msg_type = message.get("type")

            # Step 1 — client sets voice (optional, send before text)
            # {"type": "config", "voice": "nova"}
            if msg_type == "config":
                voice = message.get("voice", DEFAULT_VOICE)
                desc_ids = desc_tokenizer(
                    get_description(voice), return_tensors="pt"
                ).to(device)

            # Step 2 — LLM token arrives
            # {"type": "text", "chunk": "नमस्ते, "}
            elif msg_type == "text":
                if desc_ids is None:
                    desc_ids = desc_tokenizer(
                        default_desc, return_tensors="pt"
                    ).to(device)

                buffer += message.get("chunk", "")

                # Synthesize as soon as a sentence is complete
                if SENTENCE_END.search(buffer) and len(buffer) >= MINIMUM_CHUNK_CHARS:
                    text_to_synth = buffer.strip()
                    buffer = ""
                    audio_bytes = await loop.run_in_executor(
                        None, synthesize_chunk, text_to_synth, desc_ids
                    )
                    await websocket.send_bytes(audio_bytes)

            # Step 3 — LLM stream finished, flush remaining buffer
            # {"type": "end"}
            elif msg_type == "end":
                if buffer.strip():
                    if desc_ids is None:
                        desc_ids = desc_tokenizer(
                            default_desc, return_tensors="pt"
                        ).to(device)
                    audio_bytes = await loop.run_in_executor(
                        None, synthesize_chunk, buffer.strip(), desc_ids
                    )
                    await websocket.send_bytes(audio_bytes)
                    buffer = ""
                # Tell client all audio is done
                await websocket.send_json({"type": "done"})
                break

    except WebSocketDisconnect:
        print("Client disconnected")

# ─────────────────────────────────────────────
# OpenAI-compatible: GET /v1/models
# ─────────────────────────────────────────────
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "owned_by": "ai4bharat"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "ai4bharat"},
        ]
    }

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "voices": list(VOICE_MAP.keys())
    }