import torch
import soundfile as sf
import io
import os
import re
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from huggingface_hub import login
from pydantic import BaseModel
from typing import AsyncIterator

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
# Helpers
# ─────────────────────────────────────────────

# Sentence boundary characters (includes Hindi/Indic ।)
SENTENCE_END = re.compile(r'(?<=[।.!?])\s+')
MINIMUM_CHUNK_CHARS = 30  # don't synthesize tiny fragments

def split_on_boundaries(text: str) -> list[str]:
    """Split text into speakable sentence chunks."""
    parts = SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

def synthesize_chunk(text: str, desc_ids) -> bytes:
    """Convert a single text chunk to WAV bytes."""
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
# Request Schemas
# ─────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    description: str = "Rohit speaks in a moderate speed with a clear and natural tone. The recording is of very high quality."


class StreamingTTSRequest(BaseModel):
    # List of text chunks coming from LLM stream
    chunks: list[str]
    description: str = "Rohit speaks in a moderate speed with a clear and natural tone. The recording is of very high quality."


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/v1/audio/speech")
def synthesize(req: TTSRequest):
    """Standard single request TTS."""
    desc_ids = desc_tokenizer(req.description, return_tensors="pt").to(device)
    audio_bytes = synthesize_chunk(req.text, desc_ids)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
async def synthesize_stream(req: StreamingTTSRequest):
    """
    Accepts chunks of text from an LLM stream.
    Buffers until a sentence boundary is hit, then
    synthesizes and streams audio immediately.
    Lowest possible TTFS.
    """
    desc_ids = desc_tokenizer(req.description, return_tensors="pt").to(device)

    async def audio_stream() -> AsyncIterator[bytes]:
        buffer = ""

        for chunk in req.chunks:
            buffer += chunk

            # Check if buffer has a complete sentence
            if SENTENCE_END.search(buffer) and len(buffer) >= MINIMUM_CHUNK_CHARS:
                sentences = split_on_boundaries(buffer)

                # Keep last incomplete sentence in buffer
                if not SENTENCE_END.search(sentences[-1]):
                    buffer = sentences[-1]
                    sentences = sentences[:-1]
                else:
                    buffer = ""

                # Synthesize and stream each complete sentence
                for sentence in sentences:
                    if sentence:
                        audio_bytes = await asyncio.get_event_loop().run_in_executor(
                            None, synthesize_chunk, sentence, desc_ids
                        )
                        yield audio_bytes
                        await asyncio.sleep(0)

        # Flush remaining buffer
        if buffer.strip():
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None, synthesize_chunk, buffer.strip(), desc_ids
            )
            yield audio_bytes

    return StreamingResponse(audio_stream(), media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok", "device": device}