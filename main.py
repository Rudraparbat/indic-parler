"""
FastAPI OpenAI Compatible API
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from huggingface_hub import login
from config import settings
from src.routers import openai_router
import subprocess
import shlex
import numpy as np
from dotenv import load_dotenv
load_dotenv()

def setup_logger():
    """Configure loguru logger with custom formatting"""
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    level = os.getenv("API_LOG_LEVEL", "DEBUG").upper()
    if level not in valid_levels:
        level = "DEBUG"
    print(f"Global API loguru logger level: {level}")
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
                "{level: <8} | "
                "<fg #4169E1>{module}:{line}</fg #4169E1> | "
                "{message}",
                "colorize": True,
                "level": level,
            },
        ],
    }
    logger.remove()
    logger.configure(**config)
    logger.level("ERROR", color="<red>")


# Configure logger
setup_logger()
HF_TOKEN = os.getenv('HF_TOKEN')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Indic Parler TTS model on startup, clean up on shutdown"""
    try:
        login(token=HF_TOKEN)
        print("✅ Hugging Face authentication successful")
    except Exception as e:
        print(f"❌ Hugging Face authentication failed: {e}")
        raise

    # --- STARTUP ---

    # 1. Detect device
    if torch.cuda.is_available():
        app.state.device = "cuda"
    elif torch.backends.mps.is_available():
        app.state.device = "mps"
    else:
        app.state.device = "cpu"

    logger.info(f"Loading Indic Parler TTS on {app.state.device}...")

    try:
        from transformers import AutoTokenizer
        from parler_tts import ParlerTTSForConditionalGeneration

        model_name = "ai4bharat/indic-parler-tts"

        # 2. Load tokenizers & model directly into app.state
        app.state.description_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
        )
        app.state.tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
        )
        app.state.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if app.state.device == "cuda" else torch.float32,
            token=HF_TOKEN,
            attn_implementation="eager",  
        ).to(app.state.device)


        app.state.model.eval()
        if app.state.device == "cuda":
            logger.info("Compiling model with torch.compile for additional speedup...")
            app.state.model = torch.compile(
                app.state.model,
                mode="reduce-overhead"  
            )
            logger.info("Model compilation complete")
        app.state.sampling_rate = app.state.model.audio_encoder.config.sampling_rate
        app.state.frame_rate = app.state.model.audio_encoder.config.frame_rate
        
    except Exception as e:
        logger.error(f"Failed to load Indic Parler TTS model: {e}")
        raise

    # --- STARTUP BANNER ---
    boundary = "░" * 2 * 12
    startup_msg = f"""
{boundary}

    ██╗███╗   ██╗██████╗ ██╗ ██████╗
    ██║████╗  ██║██╔══██╗██║██╔════╝
    ██║██╔██╗ ██║██║  ██║██║██║     
    ██║██║╚██╗██║██║  ██║██║██║     
    ██║██║ ╚████║██████╔╝██║╚██████╗
    ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝
    ██████╗  █████╗ ██████╗ ██╗     ███████╗██████╗ 
    ██╔══██╗██╔══██╗██╔══██╗██║     ██╔════╝██╔══██╗
    ██████╔╝███████║██████╔╝██║     █████╗  ██████╔╝
    ██╔═══╝ ██╔══██║██╔══██╗██║     ██╔══╝  ██╔══██╗
    ██║     ██║  ██║██║  ██║███████╗███████╗██║  ██║
    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
    ████████╗████████╗███████╗
    ╚══██╔══╝╚══██╔══╝██╔════╝
       ██║      ██║   ███████╗
       ██║      ██║   ╚════██║
       ██║      ██║   ███████║
       ╚═╝      ╚═╝   ╚══════╝

{boundary}"""

    startup_msg += f"\nModel loaded on  : {app.state.device} | {model_name}"

    if app.state.device == "mps":
        startup_msg += "\nBackend          : Apple Metal Performance Shaders (MPS)"
    elif app.state.device == "cuda":
        startup_msg += f"\nBackend          : CUDA {torch.version.cuda}"
    else:
        startup_msg += "\nBackend          : CPU"

    startup_msg += f"\nPrecision        : {'float16' if app.state.device == 'cuda' else 'float32'}"
    startup_msg += f"\n{boundary}\n"

    logger.info(startup_msg)

    yield  # 

    # --- SHUTDOWN ---
    logger.info("Shutting down, releasing model...")
    del app.state.model, app.state.tokenizer, app.state.description_tokenizer
    if app.state.device == "cuda":
        torch.cuda.empty_cache()


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    openapi_url="/openapi.json",  # Explicitly enable OpenAPI schema
)

# Add CORS middleware if enabled
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(openai_router, prefix="/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}
