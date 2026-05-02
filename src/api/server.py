import logging
import asyncio
from typing import Optional
from pathlib import Path
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.pipeline import AudioPipeline
from src.config import (
    API_HOST,
    API_PORT,
    API_DEBUG,
    RAG_COLLECTION_NAME,
    TEMP_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Customer Support Agent",
    description="End-to-end audio pipeline for customer support",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioPipeline] = None


# Request/Response models
class TextChatRequest(BaseModel):
    """Request model for text chat endpoint."""
    text: str


class TextChatResponse(BaseModel):
    """Response model for text chat endpoint."""
    status: str
    response: str
    query: str


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    components: dict


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    try:
        logger.info("Starting up API server...")

        # Initialize ChromaDB collection (placeholder)
        # In production, connect to actual ChromaDB instance
        import chromadb
        
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=RAG_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize pipeline
        pipeline = AudioPipeline(chroma_collection=collection)
        await pipeline.initialize()

        logger.info("API server startup complete")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")
    # Cleanup resources if needed
    logger.info("API server shutdown complete")


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status of all components
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        health = await pipeline.health_check()
        return HealthResponse(
            status="healthy",
            components=health,
        )

    except Exception as e:
        logger.error(f"Error during health check: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat/text", response_model=TextChatResponse)
async def chat_text(request: TextChatRequest):
    """
    Text chat endpoint.

    Args:
        request: TextChatRequest containing user query

    Returns:
        AI-generated response

    Raises:
        HTTPException: If processing fails
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    try:
        logger.info(f"Processing text request: {request.text[:100]}")

        response = await pipeline.process_text(request.text)

        return TextChatResponse(
            status="success",
            response=response,
            query=request.text,
        )

    except Exception as e:
        logger.error(f"Error processing text chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing text chat",
        )


@app.post("/chat/audio")
async def chat_audio(file: UploadFile = File(...)):
    """
    Audio chat endpoint.

    Args:
        file: Audio file upload (wav, mp3, m4a, etc.)

    Returns:
        Audio response as MP3 stream

    Raises:
        HTTPException: If processing fails
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if file.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    temp_path = None
    try:
        logger.info(f"Processing audio request: {file.filename}")

        # Read uploaded file
        audio_bytes = await file.read()

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Invalid audio file")

        # Process audio through pipeline
        response_audio = await pipeline.process_audio(audio_bytes)

        if len(response_audio) == 0:
            raise HTTPException(status_code=500, detail="No audio response generated")

        # Return audio as stream
        return StreamingResponse(
            iter([response_audio]),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=response.mp3"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing audio chat",
        )


@app.post("/chat/audio-response")
async def chat_audio_response(file: UploadFile = File(...)):
    """
    Audio chat endpoint that returns audio response.

    Args:
        file: Audio file upload

    Returns:
        Audio response with transcription
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        logger.info(f"Processing audio-response request: {file.filename}")

        # Read uploaded file
        audio_bytes = await file.read()

        # Process audio through pipeline
        response_audio = await pipeline.process_audio(audio_bytes)

        # Return MP3 audio response
        return StreamingResponse(
            iter([response_audio]),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=response.mp3"
            },
        )

    except Exception as e:
        logger.error(f"Error in audio-response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing audio")


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Audio Customer Support Agent",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "text_chat": "POST /chat/text",
            "audio_chat": "POST /chat/audio",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


def run_server(host: str = API_HOST, port: int = API_PORT, debug: bool = API_DEBUG):
    """Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()