from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dependencies import get_rag_service
from app.routes.rag import router as api_router
from app.core.init_setup import initialize_database
from app.config import settings


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        initialize_database(reset=settings.RESET)

        get_rag_service()

        yield

    app = FastAPI(
        lifespan=lifespan,
        title="demoRAG_API",
        version="0.1.0",
    )

    app.include_router(api_router, prefix="/api")
    return app

app = create_app()