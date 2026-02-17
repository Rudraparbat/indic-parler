from threading import Thread
from typing import AsyncGenerator, Union
from fastapi import Request
from parler_tts import ParlerTTSStreamer
from services.streaming_writer import StreamingAudioWriter
from ..schemas import OpenAISpeechRequest, CaptionedSpeechRequest
from loguru import logger
from src.services.utils import  resolve_voice


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
    logger.info("Initialized audio converter..")
    writer= StreamingAudioWriter(str(request.response_format) , sampling_rate)
    logger.info(f"Audio writer init successfully")
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

        for audio_chunk in streamer:
            if audio_chunk.shape[0] == 0:
                logger.info(f"[{request_id}] Received empty chunk — generation complete")
                break

            chunk_duration = round(audio_chunk.shape[0] / sampling_rate, 4)
            chunk_count += 1
            total_audio_seconds += chunk_duration
            logger.debug(f"[{request_id}] Chunk #{chunk_count} received | duration={chunk_duration}s | shape={audio_chunk.shape}")
            logger.info(f"chunk recived {audio_chunk}")
            chunk_bytes = writer.write_chunk(audio_chunk)
            logger.info(f"Converted chunks {chunk_bytes}")
            yield chunk_bytes
        
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
        if 'writer' in locals():  
            final_bytes = writer.write_chunk(finalize=True)  # Write trailer
            logger.debug(f"[{request_id}] Writer finalized: {len(final_bytes)} bytes")
            yield final_bytes  # Send final playable chunk!
            writer.close()
        

