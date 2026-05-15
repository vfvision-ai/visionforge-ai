"""
Custom exception classes for VisionForge.
Provides structured error handling across the application.
"""


class MLPlatformException(Exception):
    """Base exception for all VisionForge errors"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DatasetError(MLPlatformException):
    """Raised when there are issues with dataset loading or processing"""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset cannot be found"""
    pass


class DatasetFormatError(DatasetError):
    """Raised when dataset format is invalid or unsupported"""
    pass


class DatasetTooLargeError(DatasetError):
    """Raised when dataset exceeds size limits"""
    pass


class ModelError(MLPlatformException):
    """Raised when there are issues with model operations"""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a model cannot be found"""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails"""
    pass


class ModelArchitectureError(ModelError):
    """Raised when there's an error in model architecture"""
    pass


class TrainingError(MLPlatformException):
    """Raised when there are issues during training"""
    pass


class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted"""
    pass


class ValidationError(MLPlatformException):
    """Raised when validation fails"""
    pass


class ConfigurationError(MLPlatformException):
    """Raised when there are configuration issues"""
    pass


class InferenceError(MLPlatformException):
    """Raised when there are issues during inference"""
    pass


class FileUploadError(MLPlatformException):
    """Raised when file upload fails"""
    pass


class SecurityError(MLPlatformException):
    """Raised when security checks fail"""
    pass


class RateLimitError(SecurityError):
    """Raised when rate limit is exceeded"""
    pass


class InvalidInputError(ValidationError):
    """Raised when input validation fails"""
    pass
