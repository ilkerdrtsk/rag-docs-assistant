import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health_router, router
from app.errors import AppError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="RAG Docs Assistant",
    version="1.0.0",
    description=(
        "Ask questions about your own documents. Hybrid retrieval (PostgreSQL full-text "
        "search + pgvector) and grounded answers from Claude with numbered citations."
    ),
)
app.include_router(router)
app.include_router(health_router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content={"code": exc.code, "detail": exc.detail}
    )
