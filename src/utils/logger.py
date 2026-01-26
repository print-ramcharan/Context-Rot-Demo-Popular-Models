import logging
import sys
from datetime import datetime
from pathlib import Path

class SystemLogger:
    """
    Centralized logging for the external memory system.
    """
    
    def __init__(self, name: str = "ExternalMemory",
                 log_dir: str = "logs",
                 level: str = "INFO"):
        """
        Initialize logger.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._setup_logging(log_dir, level)
    
    def _setup_logging(self, log_dir: str, level: str):
        """Configure logging handlers and formatters."""
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(
            Path(log_dir) / f"system_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def debug(self, msg): self.logger.debug(msg)
    def warning(self, msg): self.logger.warning(msg)

    def log_operation(self, operation: str, **kwargs):
        """Log a system operation with context."""
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"Operation: {operation} | {context}")

class ErrorHandler:
    """
    Centralized error handling with recovery strategies.
    """
    
    @staticmethod
    def handle_api_error(error: Exception, retry_count: int = 0) -> dict:
        """Handle API errors."""
        error_msg = str(error).lower()
        should_retry = False
        wait_time = 0
        
        if "rate limit" in error_msg or "429" in error_msg:
            should_retry = True
            wait_time = 2 * (retry_count + 1)
        elif "timeout" in error_msg or "connection" in error_msg:
            should_retry = True
            wait_time = 1
            
        return {
            'should_retry': should_retry,
            'wait_time': wait_time,
            'error_type': type(error).__name__,
            'message': str(error)
        }
