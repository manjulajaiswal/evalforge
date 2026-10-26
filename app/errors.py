"""Exception hierarchy.

The categories map directly onto how a failure is reported and which exit
code the evaluation command returns (see app/exit_codes.py).
"""

from __future__ import annotations


class EvalForgeError(Exception):
    """Base class for all EvalForge errors."""


class ConfigurationError(EvalForgeError):
    """Invalid or missing configuration (e.g. no API key, empty vector store)."""


class DatasetValidationError(EvalForgeError):
    """The golden dataset is malformed. ``issues`` holds one message per problem."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        preview = "\n".join(f"  - {issue}" for issue in issues[:20])
        more = f"\n  ... and {len(issues) - 20} more" if len(issues) > 20 else ""
        super().__init__(f"Golden dataset is invalid ({len(issues)} issue(s)):\n{preview}{more}")


class ExternalServiceError(EvalForgeError):
    """A call to an external service (Gemini) failed.

    ``transient`` is True for failures that might succeed on retry
    (503, 429, timeouts) and False for permanent ones (bad key, unknown model).
    """

    def __init__(
        self,
        message: str,
        *,
        transient: bool,
        status_code: int | None = None,
        status: str | None = None,
        attempts: int = 1,
    ):
        super().__init__(message)
        self.transient = transient
        self.status_code = status_code
        self.status = status
        self.attempts = attempts


class EmptyModelResponseError(ExternalServiceError):
    """The model returned no text (e.g. blocked by safety filters)."""

    def __init__(self, message: str, *, attempts: int = 1):
        super().__init__(message, transient=False, attempts=attempts)


class JudgeOutputError(EvalForgeError):
    """The LLM judge returned output that does not match the required schema."""
