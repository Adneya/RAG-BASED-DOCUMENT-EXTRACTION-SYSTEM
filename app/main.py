from __future__ import annotations

from pathlib import Path
import sqlite3

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent.pipeline import AgentPipeline, build_cost_estimate
from app.auth import authenticate, create_session, create_user, delete_session, user_from_session
from app.agent.responses import answer_from_rag
from app.config import settings
from app.schemas import (
    AgentResponse,
    ExtractedContent,
    PlanStep,
    RagDocument,
    RagIngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    AnalyticsResponse,
    AuthUser,
    LoginRequest,
    RegisterRequest,
)
from app.tools.analytics import analyze_table
from app.tools.extraction import extract_upload


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
pipeline = AgentPipeline()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"app_name": settings.app_name},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/auth/me", response_model=AuthUser)
async def auth_me(request: Request) -> AuthUser:
    user = user_from_session(request.cookies.get("session"))
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return AuthUser(**user)


@app.post("/api/auth/register", response_model=AuthUser)
async def auth_register(payload: RegisterRequest, response: Response) -> AuthUser:
    if "@" not in payload.email or "." not in payload.email.rsplit("@", 1)[-1]:
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    try:
        user = create_user(payload.name, payload.email, payload.password)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    response.set_cookie("session", create_session(user["id"]), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return AuthUser(**user)


@app.post("/api/auth/login", response_model=AuthUser)
async def auth_login(payload: LoginRequest, response: Response) -> AuthUser:
    user = authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    response.set_cookie("session", create_session(user["id"]), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return AuthUser(**user)


@app.post("/api/auth/logout")
async def auth_logout(request: Request, response: Response) -> dict[str, str]:
    delete_session(request.cookies.get("session"))
    response.delete_cookie("session")
    return {"status": "ok"}


@app.post("/api/chat", response_model=AgentResponse)
async def chat(
    query: str = Form(default=""),
    files: list[UploadFile] | None = File(default=None),
) -> AgentResponse:
    return await pipeline.run(query=query, files=files)


@app.post("/api/analytics/analyze", response_model=AnalyticsResponse)
async def analyze_analytics_table(
    file: UploadFile = File(...),
    date_column: str | None = Form(default=None),
    measure_column: str | None = Form(default=None),
    category_column: str | None = Form(default=None),
) -> AnalyticsResponse:
    filename = file.filename or "uploaded-table.csv"
    try:
        result = analyze_table(
            filename,
            await file.read(),
            date_column=date_column,
            measure_column=measure_column,
            category_column=category_column,
        )
    except ValueError as exc:
        return AnalyticsResponse(status="partial", filename=filename, warnings=[str(exc)])
    return AnalyticsResponse(status="completed", filename=filename, **result)


@app.post("/api/rag/ingest", response_model=RagIngestResponse)
async def rag_ingest(
    files: list[UploadFile] | None = File(default=None),
) -> RagIngestResponse:
    plan: list[PlanStep] = []
    extracted: list[ExtractedContent] = []
    uploads = [file for file in files or [] if file.filename]

    plan.append(
        PlanStep(
            tool="input_extractor",
            action=f"Extract content from {len(uploads)} uploaded file(s)",
            status="running" if uploads else "skipped",
        )
    )
    for upload in uploads:
        extracted.append(await extract_upload(upload))
    if uploads:
        plan[-1].status = "completed"
        plan[-1].detail = "Parsed upload text for RAG ingestion."
    else:
        plan[-1].detail = "No uploaded files were provided."

    plan.append(
        PlanStep(
            tool="rag_indexer",
            action="Chunk content, embed chunks, and persist them in Chroma",
            status="running",
        )
    )
    result = pipeline.rag_store.ingest_contents(extracted)
    if result.chunks_indexed:
        plan[-1].status = "completed"
        plan[-1].detail = (
            f"Indexed {result.chunks_indexed} chunk(s) across "
            f"{len(result.documents)} document(s)."
        )
    else:
        plan[-1].status = "skipped"
        plan[-1].detail = "No readable text was available to index."

    status = "partial" if any(item.warnings for item in extracted) else "completed"
    return RagIngestResponse(
        status=status,
        documents=result.documents,
        chunks_indexed=result.chunks_indexed,
        extracted_texts=extracted,
        plan_trace=plan,
    )


@app.post("/api/rag/query", response_model=RagQueryResponse)
async def rag_query(request: RagQueryRequest) -> RagQueryResponse:
    plan = [
        PlanStep(
            tool="rag_retriever",
            action="Retrieve relevant indexed chunks",
            status="running",
        )
    ]
    rewritten_query = pipeline.rag_store.rewrite_query(request.query)
    contexts = pipeline.rag_store.retrieve(rewritten_query, top_k=request.top_k)
    if contexts:
        plan[-1].status = "completed"
        plan[-1].detail = f"Retrieved {len(contexts)} chunk(s)."
        final_output = answer_from_rag(rewritten_query, contexts)
    else:
        plan[-1].status = "skipped"
        plan[-1].detail = "No relevant indexed chunks were found."
        final_output = "I could not find relevant indexed context for that question."

    plan.append(
        PlanStep(
            tool="response_builder",
            action="Generate grounded RAG answer",
            status="completed",
            detail="Final text-only output generated.",
        )
    )
    return RagQueryResponse(
        status="completed" if contexts else "partial",
        final_output=final_output,
        retrieved_contexts=contexts,
        plan_trace=plan,
        cost_estimate=build_cost_estimate(rewritten_query, []),
    )


@app.get("/api/rag/documents", response_model=list[RagDocument])
async def rag_documents() -> list[RagDocument]:
    return pipeline.rag_store.list_documents()


@app.get("/api/rag/documents/{document_id}/summary")
async def rag_document_summary(document_id: str) -> dict[str, str]:
    return {"document_id": document_id, "summary": pipeline.rag_store.summarize_document(document_id)}


@app.delete("/api/rag/documents/{document_id}")
async def rag_delete_document(document_id: str) -> dict[str, int | str]:
    deleted = pipeline.rag_store.delete_document(document_id)
    return {"document_id": document_id, "deleted_chunks": deleted}
