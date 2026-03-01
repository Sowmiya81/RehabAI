from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.core.config import get_settings
from api.routers.analyze import router as analyze_router, init_agent

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {settings.app_name} API — initializing agent...")
    init_agent(max_steps=settings.max_agent_steps)
    print("Agent ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="RehabAI API",
    description="Multi-Agent Exercise Coaching System — REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
