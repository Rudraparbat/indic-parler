from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


# Indic Parler supported languages
class IndicLanguage(str, Enum):
    HINDI = "hi"
    BENGALI = "bn"
    TELUGU = "te"
    TAMIL = "ta"
    MALAYALAM = "ml"
    KANNADA = "kn"
    GUJARATI = "gu"
    MARATHI = "mr"
    PUNJABI = "pa"
    ODIA = "or"
    ENGLISH = "en"

VOICE_PRESETS = {
    # female voices
    "meera":  "Meera's voice is warm and expressive, speaking at a moderate pace with clear pronunciation.",
    "ananya": "Ananya speaks with a calm, soft tone, slightly slow-paced and very clear.",
    "priya":  "Priya has an energetic, high-pitched voice with a fast speaking pace.",
    # male voices
    "arjun":  "Arjun's voice is deep and authoritative, speaking at a moderate pace.",
    "rohan":  "Rohan speaks with a friendly, conversational tone at a natural pace.",
    "karan":  "Karan has a smooth, low-pitched voice with clear and slow articulation.",
    # neutral
    "default": "The speaker has a neutral tone, moderate pace, and clear pronunciation.",
}

# ─────────────────────────────────────────────
# CORE OpenAI-compatible Speech Request
# ─────────────────────────────────────────────
class OpenAISpeechRequest(BaseModel):
    """
    OpenAI-compatible /v1/audio/speech request.
    'voice' accepts either a preset name (e.g 'meera') 
    or a raw Parler description string for full control.
    """

    # --- OpenAI required ---
    model: str = Field(
        default="indic-parler-tts",
        description="Model to use. Accepted: indic-parler-tts, tts-1, tts-1-hd (all map to same model)"
    )
    input: str = Field(
        ...,
        description="The text to synthesize audio for"
    )
    voice: str = Field(
        default="meera",
        description=(
            "Voice preset name (meera, ananya, priya, arjun, rohan, karan) "
            "OR a raw Parler description prompt for custom voice control."
        )
    )

    # --- OpenAI optional ---
    response_format: Literal["mp3", "opus", "flac", "wav", "pcm"] = Field(
        default="wav",
        description="Output audio format. AAC not supported."
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed of speech. Note: Indic Parler does not natively support speed control — this will be applied via post-processing."
    )

    # --- Indic Parler extensions ---
    language: IndicLanguage = Field(
        default=IndicLanguage.HINDI,
        description="Language code for the input text."
    )
    stream: bool = Field(
        default=True,
        description="Stream audio chunks as they are generated, sentence by sentence."
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download URL in X-Download-Path header after stream completes."
    )
    chunk_size: Optional[int] = Field(
        default=None,
        description="Override default streaming chunk size in bytes."
    )


# ─────────────────────────────────────────────
# Word-level timestamp schemas (captioned speech)
# ─────────────────────────────────────────────
class WordTimestamp(BaseModel):
    word: str = Field(..., description="The word")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class CaptionedSpeechRequest(BaseModel):
    """For /v1/audio/speech/captioned — returns audio + word timestamps"""

    model: str = Field(default="indic-parler-tts")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="meera")
    response_format: Literal["mp3", "opus", "flac", "wav", "pcm"] = Field(default="mp3")
    language: IndicLanguage = Field(default=IndicLanguage.HINDI)
    return_timestamps: bool = Field(
        default=True,
        description="Include word-level timestamps in the response."
    )


class CaptionedSpeechResponse(BaseModel):
    """Response for captioned speech — base64 audio + timestamps"""
    audio: str = Field(..., description="Base64-encoded audio")
    audio_format: str = Field(..., description="Format of the audio")
    timestamps: Optional[list[WordTimestamp]] = Field(
        default=None,
        description="Word-level timestamps if return_timestamps=True"
    )


# ─────────────────────────────────────────────
# Models list endpoint schema
# ─────────────────────────────────────────────
class ModelObject(BaseModel):
    """OpenAI-compatible model object"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ai4bharat"


class ModelListResponse(BaseModel):
    """Response for GET /v1/models"""
    object: str = "list"
    data: list[ModelObject]


# ─────────────────────────────────────────────
# Voices list endpoint schema
# ─────────────────────────────────────────────
class VoiceObject(BaseModel):
    """A single voice entry"""
    voice_id: str
    name: str
    description: str
    language: str


class VoiceListResponse(BaseModel):
    """Response for GET /v1/audio/voices"""
    object: str = "list"
    data: list[VoiceObject]