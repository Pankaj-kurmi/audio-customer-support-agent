
import logging
import asyncio
from typing import Optional, Dict, Any
from src.stt import STTService
from src.tts import TTSService
from src.llm import CustomerSupportAgent

logger = logging.getLogger(__name__)

class AudioPipeline:
    """End-to-end audio pipeline for customer support."""

    def __init__(
        self,
        chroma_collection,
        stt_service: Optional[STTService] = None,
        tts_service: Optional[TTSService] = None,
        llm_agent: Optional[CustomerSupportAgent] = None,
    ):
        self.chroma_collection = chroma_collection
        self.stt_service = stt_service or STTService()
        self.tts_service = tts_service or TTSService()
        self.llm_agent = llm_agent or CustomerSupportAgent(chroma_collection)
        self.is_initialized = False
        self.pipeline_stats = {"total_audio_processed": 0, "total_text_processed": 0, "errors": 0}
        logger.info("AudioPipeline initialized")

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            logger.info("Initializing pipeline components...")
            logger.info("Initializing STT service...")
            await self.stt_service.initialize()
            logger.info("Initializing TTS service...")
            await self.tts_service.initialize()
            
            if self.llm_agent is None:
                raise ValueError("LLM agent not available")

            self.is_initialized = True
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}", exc_info=True)
            raise

    async def process_audio(self, audio_bytes: bytes) -> bytes:
        """Process audio end-to-end: STT → LLM → TTS."""
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Empty audio bytes provided")

        try:
            logger.info("Starting audio processing pipeline...")
            logger.info("Step 1: Converting audio to text (STT)...")
            transcript = await self.stt_service.transcribe(audio_bytes)

            if not transcript or not transcript.strip():
                logger.warning("STT produced empty transcript")
                return b""

            logger.info(f"Transcript: '{transcript}'")
            logger.info("Step 2: Generating response (LLM)...")
            response = await self.llm_agent.generate_response(transcript)

            if not response or not response.strip():
                logger.warning("LLM produced empty response")
                return b""

            logger.info(f"LLM Response: '{response[:100]}...'")
            logger.info("Step 3: Converting response to audio (TTS)...")
            audio_output = await self.tts_service.synthesize(response)

            self.pipeline_stats["total_audio_processed"] += 1
            logger.info("Audio processing pipeline complete")
            return audio_output

        except Exception as e:
            logger.error(f"Error in audio pipeline: {str(e)}", exc_info=True)
            self.pipeline_stats["errors"] += 1
            raise

    async def process_text(self, text: str) -> str:
        """Process text directly (bypass STT)."""
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        if not text or not text.strip():
            raise ValueError("Empty text provided")

        try:
            logger.info(f"Processing text: '{text}'")
            response = await self.llm_agent.generate_response(text)
            self.pipeline_stats["total_text_processed"] += 1
            logger.info("Text processing complete")
            return response
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}", exc_info=True)
            self.pipeline_stats["errors"] += 1
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all pipeline components."""
        stt_health = await self.stt_service.health_check()
        tts_health = await self.tts_service.health_check()

        return {
            "pipeline": {"status": "healthy" if self.is_initialized else "uninitialized", "initialized": self.is_initialized},
            "components": {
                "stt": stt_health,
                "tts": tts_health,
                "llm": {"service": "LLM", "status": "healthy"},
                "rag": {"service": "RAG", "status": "healthy"},
            },
            "stats": self.pipeline_stats,
        }
