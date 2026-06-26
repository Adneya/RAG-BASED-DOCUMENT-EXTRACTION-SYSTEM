from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent.pipeline import AgentPipeline
from app.config import settings
from app.schemas import AgentResponse


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
pipeline = AgentPipeline()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": settings.app_name},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=AgentResponse)
async def chat(
    query: str = Form(default=""),
    files: list[UploadFile] | None = File(default=None),
) -> AgentResponse:
    return await pipeline.run(query=query, files=files)

