import numpy as np
from ..schemas import VOICE_PRESETS
from loguru import logger
import librosa

def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Post-process speed since Parler has no native speed control"""
    logger.debug(f"Applying speed multiplier {speed}x via librosa time_stretch")
    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
    logger.debug(f"Speed applied: original length {len(audio)}, new length {len(stretched)}")
    return stretched


def resolve_voice(voice: str) -> str:
    """Resolve preset name → description, or pass through raw description"""
    if voice in VOICE_PRESETS:
        logger.debug(f"Voice preset '{voice}' resolved to description: '{VOICE_PRESETS[voice][:50]}...'")
        return VOICE_PRESETS[voice]
    logger.debug(f"Voice '{voice}' not in presets, treating as raw description")
    return voice


