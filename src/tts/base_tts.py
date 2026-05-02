import logging
import asyncio
import edge_tts

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service using Microsoft Edge TTS."""

    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%", volume: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.is_initialized = False
        self.available_voices = []
        logger.info(f"TTSService initialized with voice={voice}, rate={rate}, volume={volume}")

    async def initialize(self) -> None:
        """Initialize TTS service and verify voice availability."""
        try:
            logger.info("Initializing Edge TTS service")
            loop = asyncio.get_event_loop()
            self.available_voices = await loop.run_in_executor(None, self._sync_get_voices)

            voice_names = [v["ShortName"] for v in self.available_voices]
            if self.voice not in voice_names:
                logger.warning(f"Voice {self.voice} not available, using first available: {voice_names[0]}")
                self.voice = voice_names[0]

            self.is_initialized = True
            logger.info(f"Edge TTS initialized successfully with voice: {self.voice}")

        except Exception as e:
            logger.error(f"Error initializing TTS service: {str(e)}", exc_info=True)
            raise

    def _sync_get_voices(self) -> list:
        """Synchronous voice list retrieval."""
        try:
            import asyncio as aio
            
            async def _get_voices():
                return await edge_tts.list_voices()
            
            loop = aio.new_event_loop()
            aio.set_event_loop(loop)
            voices = loop.run_until_complete(_get_voices())
            loop.close()
            return voices
        except Exception as e:
            logger.error(f"Error getting voice list: {str(e)}", exc_info=True)
            return []

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio."""
        if not self.is_initialized:
            raise ValueError("TTS service not initialized. Call initialize() first.")

        if not text or not text.strip():
            logger.warning("Empty text provided to synthesize")
            return b""

        try:
            logger.info(f"Synthesizing text: '{text[:100]}...'")

            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
            )

            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            audio_bytes = b"".join(audio_chunks)
            logger.info(f"Audio synthesis complete: {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}", exc_info=True)
            raise

    async def health_check(self) -> dict:
        """Check health of TTS service."""
        return {
            "service": "TTS",
            "status": "healthy" if self.is_initialized else "uninitialized",
            "voice": self.voice,
            "available_voices": len(self.available_voices),
        }
