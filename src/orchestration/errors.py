class RetryableError(Exception):
    """Errors that can be retried according to policy."""


class FatalStageError(Exception):
    """Non-retryable failure for a stage."""


class PipelineExecutionError(Exception):
    """Pipeline-level fatal error."""
