# Configuration File

# This file holds the configuration settings for the audio customer support agent.

STT_ENGINE = 'Google'
LLM_ENGINE = 'OpenAI'
TTS_ENGINE = 'Google TTS'
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
TEMP_DIR = PROJECT_ROOT / "tmp"
TEMP_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# STT Configuration
STT_MODEL = os.getenv("STT_MODEL", "base")
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")

# TTS Configuration
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_VOLUME = os.getenv("TTS_VOLUME", "+0%")

# RAG Configuration
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "customer_support")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
LOG_FILE.parent.mkdir(exist_ok=True)
