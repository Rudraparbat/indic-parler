import torch
from src.schemas import  OpenAISpeechRequest
from loguru import logger
from fastapi import  Request
from src.services.utils import apply_speed , resolve_voice



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