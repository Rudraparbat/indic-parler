import torch
import soundfile as sf
import io
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from huggingface_hub import login
from pydantic import BaseModel

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
# Request Schema
# ─────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    description: str = "Rohit speaks in a moderate speed with a clear and natural tone. The recording is of very high quality."


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.post("/v1/audio/speech")
def synthesize(req: TTSRequest):
    desc_ids = desc_tokenizer(
        req.description, return_tensors="pt"
    ).to(device)
    prompt_ids = tokenizer(
        req.text, return_tensors="pt"
    ).to(device)

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
    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok", "device": device}