from fastapi.exceptions import HTTPException
import torch
import soundfile as sf
import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from pydantic import BaseModel

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
desc_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)
print("Model ready!")

class TTSRequest(BaseModel):
    text: str
    description: str = "Rohit speaks in a moderate speed with a clear and natural tone. The recording is of very high quality."

@app.post("/v1/audio/speech")
def synthesize(req: TTSRequest):
    try :
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
    except Exception as e :
        raise HTTPException(400 , str(e))

@app.get("/health")
def health():
    return {"status": "ok"}