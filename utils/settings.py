"""
Application settings and configuration using environment variables
Centralizes all configuration management for production deployment
"""

import os
from pathlib import Path
from typing import Optional
from decouple import config, Csv


class Settings:
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = config('APP_NAME', default='VisionForge')
    APP_VERSION: str = config('APP_VERSION', default='2.0.0')
    ENVIRONMENT: str = config('ENVIRONMENT', default='development')
    DEBUG: bool = config('DEBUG', default=False, cast=bool)
    
    # Streamlit
    STREAMLIT_SERVER_PORT: int = config('STREAMLIT_SERVER_PORT', default=8501, cast=int)
    STREAMLIT_SERVER_ADDRESS: str = config('STREAMLIT_SERVER_ADDRESS', default='0.0.0.0')
    STREAMLIT_SERVER_HEADLESS: bool = config('STREAMLIT_SERVER_HEADLESS', default=True, cast=bool)
    STREAMLIT_SERVER_ENABLE_CORS: bool = config('STREAMLIT_SERVER_ENABLE_CORS', default=False, cast=bool)
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION: bool = config('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', default=True, cast=bool)
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE: int = config('STREAMLIT_SERVER_MAX_UPLOAD_SIZE', default=500, cast=int)
    
    # Security
    SECRET_KEY: str = config('SECRET_KEY', default='dev-secret-key-change-in-production')
    ALLOWED_ORIGINS: list = config('ALLOWED_ORIGINS', default='http://localhost:8501', cast=Csv())
    MAX_FILE_SIZE_MB: int = config('MAX_FILE_SIZE_MB', default=500, cast=int)
    RATE_LIMIT_REQUESTS: int = config('RATE_LIMIT_REQUESTS', default=100, cast=int)
    RATE_LIMIT_PERIOD: int = config('RATE_LIMIT_PERIOD', default=60, cast=int)
    
    # Logging
    LOG_LEVEL: str = config('LOG_LEVEL', default='INFO')
    LOG_FORMAT: str = config('LOG_FORMAT', default='json')
    LOG_FILE: str = config('LOG_FILE', default='logs/app.log')
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / config('DATA_DIR', default='data')
    UPLOAD_DIR: Path = BASE_DIR / config('UPLOAD_DIR', default='uploads')
    EXPERIMENTS_DIR: Path = BASE_DIR / config('EXPERIMENTS_DIR', default='experiments')
    MODELS_DIR: Path = BASE_DIR / config('MODELS_DIR', default='models')
    LOGS_DIR: Path = BASE_DIR / 'logs'
    
    # External Services
    HF_TOKEN: Optional[str] = config('HF_TOKEN', default=None)
    WANDB_API_KEY: Optional[str] = config('WANDB_API_KEY', default=None)
    WANDB_PROJECT: str = config('WANDB_PROJECT', default='ml-experiments')
    
    # Performance
    MAX_WORKERS: int = config('MAX_WORKERS', default=4, cast=int)
    BATCH_SIZE_LIMIT: int = config('BATCH_SIZE_LIMIT', default=128, cast=int)
    GPU_MEMORY_FRACTION: float = config('GPU_MEMORY_FRACTION', default=0.9, cast=float)
    
    # Monitoring
    PROMETHEUS_PORT: int = config('PROMETHEUS_PORT', default=9090, cast=int)
    METRICS_ENABLED: bool = config('METRICS_ENABLED', default=True, cast=bool)
    
    # Email Notifications (optional)
    SMTP_HOST: Optional[str] = config('SMTP_HOST', default=None)
    SMTP_PORT: int = config('SMTP_PORT', default=587, cast=int)
    SMTP_USER: Optional[str] = config('SMTP_USER', default=None)
    SMTP_PASSWORD: Optional[str] = config('SMTP_PASSWORD', default=None)
    NOTIFICATION_EMAIL: Optional[str] = config('NOTIFICATION_EMAIL', default=None)
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return cls.ENVIRONMENT.lower() == 'production'
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment"""
        return cls.ENVIRONMENT.lower() == 'development'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.UPLOAD_DIR, cls.EXPERIMENTS_DIR, 
                         cls.MODELS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_production_settings(cls) -> list:
        """Validate production settings and return warnings"""
        warnings = []
        
        if cls.is_production():
            if cls.SECRET_KEY == 'dev-secret-key-change-in-production':
                warnings.append("⚠️ Using default SECRET_KEY in production! Please set a secure random key.")
            
            if cls.DEBUG:
                warnings.append("⚠️ DEBUG mode is enabled in production!")
            
            if not cls.STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION:
                warnings.append("⚠️ XSRF protection is disabled in production!")
        
        return warnings


# Initialize settings instance
settings = Settings()

# Create required directories
settings.create_directories()
