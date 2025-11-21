import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from application.initializer import IncludeAPIRouter
from application.main.config import settings
from application.main.services.stub_worker import StubWorker


# Global worker instance
_worker: StubWorker = None


def get_application():
    _app = FastAPI(title=settings.API_NAME,
                   description=settings.API_DESCRIPTION,
                   version=settings.API_VERSION)
    _app.include_router(IncludeAPIRouter())
    _app.add_middleware(
        CORSMiddleware,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return _app


app = get_application()


@app.on_event("startup")
async def startup_event():
    """Start the worker on application startup."""
    global _worker
    import os
    
    ml_service_url = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
    _worker = StubWorker(ml_service_url=ml_service_url)
    
    # Start worker in background
    asyncio.create_task(_worker.run(poll_interval=3))
    print(f"Worker started in background (ML Service: {ml_service_url})")


@app.on_event("shutdown")
async def app_shutdown():
    """Stop the worker on application shutdown."""
    global _worker
    if _worker:
        _worker.stop()
    print("On App Shutdown i will be called.")


#uvicorn.run("manage:app", host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL, use_colors=True,reload=True)
