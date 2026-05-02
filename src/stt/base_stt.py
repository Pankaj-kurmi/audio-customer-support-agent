import logging
import asyncio
import tempfile
from pathlib import Path
import whisper

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text service using OpenAI Whisper."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.is_initialized = False
        logger.info(f"STTService initialized with model_size={model_size}, device={device}")

    async def initialize(self) -> None:
        """Load and initialize the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, whisper.load_model, self.model_size, self.device
            )
            self.is_initialized = True
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}", exc_info=True)
            raise

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        """Transcribe audio bytes to text."""
        if not self.is_initialized:
            raise ValueError("STT service not initialized. Call initialize() first.")
        
        if not audio_bytes or len(audio_bytes) == 0:
            logger.warning("Empty audio_bytes provided to transcribe")
            return ""

        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/tmp")
            temp_file.write(audio_bytes)
            temp_file.close()

            logger.info(f"Transcribing audio from {temp_file.name}")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._sync_transcribe, temp_file.name, language
            )

            transcript = result.get("text", "").strip()
            logger.info(f"Transcription complete: '{transcript[:100]}...'")
            return transcript

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            raise
        finally:
            if temp_file and Path(temp_file.name).exists():
                try:
                    Path(temp_file.name).unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up temp file: {str(e)}")

    def _sync_transcribe(self, audio_path: str, language: str) -> dict:
        """Synchronous transcription wrapper."""
        try:
            result = self.model.transcribe(audio_path, language=language, fp16=False)
            return result
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {str(e)}", exc_info=True)
            raise

    async def health_check(self) -> dict:
        """Check health of STT service."""
        return {
            "service": "STT",
            "status": "healthy" if self.is_initialized else "uninitialized",
            "model": self.model_size,
            "device": self.device,
        }
