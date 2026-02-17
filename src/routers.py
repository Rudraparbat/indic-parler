"""OpenAI-compatible router for text-to-speech"""
import asyncio
from threading import Thread
from typing import AsyncGenerator, Union
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from .schemas import OpenAISpeechRequest
from loguru import logger
from src.services.full_audio import generate_full_audio
from src.services.audio_streaming import stream_audio_chunks

openai_router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)
    

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
            generator  =  await stream_audio_chunks(
                    request=request,
                    client_request=client_request,
                )
            logger.info(f"[{request_id}] Stream finished cleanly")
            return StreamingResponse(
                generator ,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked", 
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
                        "Cache-Control": "no-cache"
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
                "owned_by": "ai4bharat",
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
