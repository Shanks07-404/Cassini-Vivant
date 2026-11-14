"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_DIR = BACKEND_ROOT / "data"
RULES_DIR = BACKEND_ROOT / "rules"
CONTENT_DIR = PROJECT_ROOT / "content"


class Settings(BaseSettings):
    """Central configuration class."""
    
    # Gemini Assistant Config
    GEMINI_API_KEY: Optional[str] = ""
    GEMINI_MODEL: Optional[str] = ""
    
    # App Settings
    APP_ENV: str = "development"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEFAULT_LANGUAGE: str = "en"
    
    # Simulation & Model Parameters
    SIMULATION_SEED: int = 42
    FROST_THRESHOLD_C: float = 0.0
    DEMO_HERO_NIGHT: str = "night_001"
    DEMO_PARCEL_ID: str = "parcel_214"
    
    # Reference Geography (Synthetic Terroir Center)
    CENTER_LAT: float = 47.8100
    CENTER_LON: float = 3.8000
    TERRAIN_EXTENT_M: float = 4000.0  # 4km x 4km
    
    # Flags & Badges
    IS_SIMULATED_DATA: bool = True
    HONESTY_BADGE: str = "SIMULATED DATA"
    
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
