"""
Logging utilities for the intelligent training system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


def setup_logger(name: Optional[str] = None, 
                level: int = logging.INFO,
                log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logger with rich formatting.
    
    Args:
        name: Logger name (uses root if None)
        level: Logging level
        log_file: Optional file to log to
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with rich formatting
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        "%(message)s",
        datefmt="[%X]"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingProgressBar:
    """Rich progress bar for training."""
    
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=Console(stderr=True)
        )
        self.task_id = None
    
    def start(self, description: str = "Training"):
        """Start progress bar."""
        self.progress.start()
        self.task_id = self.progress.add_task(description, total=None)
    
    def update(self, description: str):
        """Update progress description."""
        if self.task_id is not None:
            self.progress.update(self.task_id, description=description)
    
    def stop(self):
        """Stop progress bar."""
        if self.progress.live.is_started:
            self.progress.stop()


# Create default logger
logger = setup_logger()