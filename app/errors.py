"""Domain errors. Each carries the HTTP status and code the API returns for it."""


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class InvalidDocument(AppError):
    status_code = 422
    code = "invalid_document"


class EmbeddingError(AppError):
    status_code = 502
    code = "embedding_provider_error"


class LLMUnavailable(AppError):
    status_code = 502
    code = "llm_unavailable"


class LLMRateLimited(AppError):
    status_code = 429
    code = "llm_rate_limited"
