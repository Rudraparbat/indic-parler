"""OpenAI-compatible router for text-to-speech"""
import asyncio
from threading import Thread
from typing import AsyncGenerator, Union

import numpy as np
import torch
import subprocess
import shlex
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSStreamer

from .schemas import VOICE_PRESETS, OpenAISpeechRequest, CaptionedSpeechRequest
from loguru import logger



openai_router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


import struct

def get_wav_header(sample_rate=44100):
    """
    Creates a 44-byte WAV header for 16-bit Mono PCM.
    """
    byte_rate = sample_rate * 2 
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 0xFFFFFFFF, b'WAVE', b'fmt ', 16, 1, 1,
        sample_rate, byte_rate, 2, 16, b'data', 0xFFFFFFFF
    )
    return header


def resolve_voice(voice: str) -> str:
    """Resolve preset name → description, or pass through raw description"""
    if voice in VOICE_PRESETS:
        logger.debug(f"Voice preset '{voice}' resolved to description: '{VOICE_PRESETS[voice][:50]}...'")
        return VOICE_PRESETS[voice]
    logger.debug(f"Voice '{voice}' not in presets, treating as raw description")
    return voice


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Post-process speed since Parler has no native speed control"""
    import librosa
    logger.debug(f"Applying speed multiplier {speed}x via librosa time_stretch")
    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
    logger.debug(f"Speed applied: original length {len(audio)}, new length {len(stretched)}")
    return stretched


def _run_generation_thread(model, generation_kwargs: dict) -> None:
    """Target function for the generation thread"""
    logger.debug("Generation thread started")
    try:
        model.generate(**generation_kwargs)
        logger.debug("Generation thread completed successfully")
    except Exception as e:
        logger.error(f"Generation thread failed: {e}")
        raise



async def stream_audio_chunks(
    request: Union[OpenAISpeechRequest, CaptionedSpeechRequest],
    client_request: Request,
    play_steps_in_s: float = 0.5,
) -> AsyncGenerator[bytes, None]:
    """
    Stream audio chunks using ParlerTTSStreamer.
    Yields audio at frame level as it's decoded in a background thread.
    """

    request_id = id(request)  # simple unique id per request for log tracing
    logger.info(f"[{request_id}] New streaming request | voice='{request.voice}' | format='{request.response_format}' | input_length={len(request.input)} chars")
    
    # dropping flac type streaming
    if request.stream and request.response_format == "flac":
        logger.warning(
            f"[{request_id}] FLAC requested with streaming=True — "
            f"FLAC buffers heavily and is not suitable for streaming. "
            f"Rejecting request."
        )
        raise ValueError(
            "FLAC format is not supported for streaming. "
            "Use wav, mp3, opus, or pcm for streaming. "
            "Set stream=False to get a complete FLAC file."
        )
    
    # --- Pull everything from app.state ---
    model = client_request.app.state.model
    tokenizer = client_request.app.state.tokenizer
    description_tokenizer = client_request.app.state.description_tokenizer
    device = client_request.app.state.device
    sampling_rate = client_request.app.state.sampling_rate
    frame_rate = client_request.app.state.frame_rate
    logger.debug(f"[{request_id}] Using device='{device}' | sampling_rate={sampling_rate} | frame_rate={frame_rate}")

    # --- Resolve voice description ---
    voice_description = resolve_voice(request.voice)
    logger.info(f"[{request_id}] Voice resolved | description='{voice_description[:80]}{'...' if len(voice_description) > 80 else ''}'")

    # --- Calculate play_steps ---
    play_steps = int(frame_rate * play_steps_in_s)
    logger.debug(f"[{request_id}] Chunk config | play_steps_in_s={play_steps_in_s}s | play_steps={play_steps} frames")

    logger.debug(
        f"[{request_id}] StreamingAudioWriter initialized | "
        f"format={request.response_format}"
    )

    try:
        # --- Tokenize description ---
        logger.debug(f"[{request_id}] Tokenizing voice description...")
        inputs = description_tokenizer(
            voice_description,
            return_tensors="pt"
        ).to(device)
        logger.debug(f"[{request_id}] Description tokenized | input_ids shape={inputs.input_ids.shape}")

        # --- Tokenize input text ---
        logger.debug(f"[{request_id}] Tokenizing input text...")
        prompt = tokenizer(
            request.input,
            return_tensors="pt"
        ).to(device)
        logger.debug(f"[{request_id}] Input text tokenized | prompt_ids shape={prompt.input_ids.shape}")

        # --- Set up native Parler streamer ---
        logger.debug(f"[{request_id}] Initializing ParlerTTSStreamer...")
        streamer = ParlerTTSStreamer(
            model,
            device=device,
            play_steps=play_steps
        )
        logger.debug(f"[{request_id}] ParlerTTSStreamer ready")

        # --- Build generation kwargs ---
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_attention_mask=prompt.attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10,
        )
        logger.debug(f"[{request_id}] Generation kwargs built | do_sample=True | temperature=1.0 | min_new_tokens=10")

        # --- Kick off generation thread ---
        logger.info(f"[{request_id}] Starting generation thread...")
        thread = Thread(
            target=_run_generation_thread,
            args=(model, generation_kwargs),
            daemon=True
        )
        thread.start()
        logger.info(f"[{request_id}] Generation thread started | thread_id={thread.ident}")

        # --- Stream chunks ---
        chunk_count = 0
        total_audio_seconds = 0.0

        logger.info(f"[{request_id}] Beginning to stream chunks to client...")

        yield get_wav_header(sampling_rate)
        for audio_chunk in streamer:
            if audio_chunk.shape[0] == 0:
                logger.info(f"[{request_id}] Received empty chunk — generation complete")
                break

            chunk_duration = round(audio_chunk.shape[0] / sampling_rate, 4)
            chunk_count += 1
            total_audio_seconds += chunk_duration
            logger.debug(f"[{request_id}] Chunk #{chunk_count} received | duration={chunk_duration}s | shape={audio_chunk.shape}")
            audio_np = audio_chunk.cpu().numpy().squeeze()
            if request.response_format == "pcm":
                logger.debug(f"[{request_id}] Yielding PCM float32 | {len(audio_np)} samples")
                yield audio_np.astype(np.float32).tobytes()
            elif request.response_format == "wav":
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
                logger.debug(f"[{request_id}] Yielding WAV int16 | max={audio_int16.max()} min={audio_int16.min()}")
                yield audio_int16.tobytes()
                
            else:  # mp3, opus - convert to int16 first
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
                yield audio_int16.tobytes()
                
    except Exception as e:
        logger.error(f"[{request_id}] Fatal error in stream_audio_chunks: {e}", exc_info=True)
        raise

    finally:
        # ─────────────────────────────────────────
        # cleanup — thread 
        # ─────────────────────────────────────────
        if thread and thread.is_alive():
            logger.warning(
                f"[{request_id}] Generation thread still alive — "
                f"joining with 5s timeout..."
            )
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.error(
                    f"[{request_id}] Generation thread did not terminate "
                    f"within timeout — possible resource leak!"
                )
            else:
                logger.debug(
                    f"[{request_id}] Generation thread joined cleanly"
                )

async def generate_full_audio(
    request: OpenAISpeechRequest,
    client_request: Request,
    request_id: int,
) -> bytes:
    """
    Direct full audio generation — no streaming, no chunking.
    Generates complete audio in one shot and returns raw bytes.
    """

    # --- Pull from app.state ---
    model = client_request.app.state.model
    tokenizer = client_request.app.state.tokenizer
    description_tokenizer = client_request.app.state.description_tokenizer
    device = client_request.app.state.device
    sampling_rate = client_request.app.state.sampling_rate

    logger.debug(
        f"[{request_id}] App state loaded | "
        f"device={device} | sampling_rate={sampling_rate}"
    )

    # --- Resolve voice ---
    voice_description = resolve_voice(request.voice)
    logger.info(
        f"[{request_id}] Voice resolved | "
        f"preview='{voice_description[:80]}{'...' if len(voice_description) > 80 else ''}'"
    )

    # --- Tokenize ---
    logger.debug(f"[{request_id}] Tokenizing voice description...")
    inputs = description_tokenizer(
        voice_description,
        return_tensors="pt"
    ).to(device)
    logger.debug(f"[{request_id}] Description tokenized | shape={inputs.input_ids.shape}")

    logger.debug(f"[{request_id}] Tokenizing input text...")
    prompt = tokenizer(
        request.input,
        return_tensors="pt"
    ).to(device)
    logger.debug(f"[{request_id}] Input tokenized | shape={prompt.input_ids.shape}")

    # --- Generate full audio in one shot ---
    logger.info(f"[{request_id}] Starting full generation...")
    with torch.no_grad():
        generation = model.generate(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_attention_mask=prompt.attention_mask,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10,
        )
    logger.info(
        f"[{request_id}] Generation complete | "
        f"output_shape={generation.shape}"
    )

    # --- Convert to numpy ---
    audio = generation.cpu().numpy().squeeze()
    logger.debug(f"[{request_id}] Converted to numpy | shape={audio.shape} | dtype={audio.dtype}")

    # --- Apply speed if needed ---
    if request.speed != 1.0:
        logger.debug(f"[{request_id}] Applying speed={request.speed}x")
        audio = apply_speed(audio, request.speed)

    return audio
 
    

@openai_router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    try :
        request_id = id(request)
        if request.model not in ['indic-parler-tts']:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_model",
                    "message": f"Unsupported model: {request.model}",
                    "type": "invalid_request_error",
                },
            )
        
        logger.info(
            f"[{request_id}] POST /v1/audio/speech | "
            f"model='{request.model}' | "
            f"voice='{request.voice}' | "
            f"format='{request.response_format}' | "
            f"stream={request.stream} | "
            f"input_length={len(request.input)} chars"
        )

        if not request.input.strip():
            logger.warning(f"[{request_id}] Empty input text received")
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
        content_type_map = {
            "mp3":  "audio/mpeg",
            "opus": "audio/opus",
            "wav":  "audio/wav",
            "pcm":  "audio/pcm",
            "flac": "audio/flac",
        }
        content_type = content_type_map[request.response_format]
        logger.debug(f"[{request_id}] Content type: '{content_type}'")

        # Asks For Streaming Response
        if request.stream:
            logger.info(f"[{request_id}] Streaming mode → StreamingResponse")
            generator  =  stream_audio_chunks(
                    request=request,
                    client_request=client_request,
                )
            logger.info(f"[{request_id}] Stream finished cleanly")
            sampling_rate = client_request.app.state.sampling_rate
            return StreamingResponse(
                generator ,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Request-ID": str(request_id),
                    "Accept-Ranges": "bytes",
                    "X-Sample-Rate": str(sampling_rate),  # Essential for clients (PCM/float32)
                }
            )
        else :
            logger.info(f"[{request_id}] Non-streaming mode → direct generation")
            full_audio = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: asyncio.run(
                        generate_full_audio(request, client_request, request_id)
                    )
            )

            logger.info(
                    f"[{request_id}] Returning full audio | "
                    f"size={len(full_audio)} bytes"
            )

            return Response(
                content=full_audio,
                media_type=content_type,
                headers={
                        "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                        "Content-Length": str(len(full_audio)),
                        "X-Request-ID": str(request_id),
                }
            )
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )

    except RuntimeError as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )



@openai_router.get("/models")
async def list_models():
    """List all available models"""
    try:
        # Create standard model list
        models = [
            {
                "id": "indic-parler-tts",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            }
        ]

        return {"object": "list", "data": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model list",
                "type": "server_error",
            },
        )
