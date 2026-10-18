class WFMError(RuntimeError):
    """Base error for actionable pipeline failures."""


class ConfigurationError(WFMError):
    """Raised for an invalid or incompatible run configuration."""


class ArtifactError(WFMError):
    """Raised for unavailable or corrupt model artifacts."""


class StageError(WFMError):
    """Raised when a pipeline stage command fails."""
