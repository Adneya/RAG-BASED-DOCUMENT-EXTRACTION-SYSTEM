# Multi-Input Agent

A FastAPI application that accepts text, images, PDFs, audio files, and multiple files in one request, extracts text, infers the user's goal, runs the minimum useful tool sequence, and returns text-only results with extracted text and a visible plan trace.

## Features

- Chat-like UI with one text box and multi-file upload.
- Image OCR for JPG and PNG files with confidence when available.
- PDF text extraction with OCR fallback for scanned pages.
- Audio transcription for MP3, WAV, and M4A through Gemini when `GEMINI_API_KEY` is set, with duration reporting when WAV metadata or `ffprobe` is available.
- Open-ended conversational answers use Gemini when `GEMINI_API_KEY` is set.
- YouTube URL detection from any extracted input and transcript fetching through `youtube-transcript-api`.
- RAG indexing with local chunking, deterministic embeddings, persistent ChromaDB storage, retrieval, and source citations.
- Intent handling for extraction, YouTube transcript summary, conversational answering, summaries, sentiment, code explanation, audio summary, action items, and cross-input comparison.
- Mandatory follow-up behavior when the user goal is unclear.
- Plan trace, extracted text panel, warnings, and token/cost estimate.
- Dockerfile and Render deployment configuration.

## Target GUI Stack

The current UI is served by FastAPI with Jinja templates, vanilla JavaScript, and CSS. For the professional dashboard upgrade, use this frontend stack:

| Layer | Choice |
| --- | --- |
| Frontend app | React with Vite |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Component system | shadcn/ui |
| Icons | Lucide React |
| API data fetching | TanStack Query |
| Local UI state | Zustand |
| Tables | TanStack Table |
| Charts and metrics | Recharts |
| File upload UX | react-dropzone |
| PDF preview | PDF.js |
| Frontend tests | Vitest, React Testing Library, Playwright |
| Backend integration | FastAPI remains the API service through `/api/chat` |
| Deployment | FastAPI on Render, frontend on Render Static Site or Vercel |

This stack keeps the backend stable while giving the GUI a modern dashboard foundation with strong typing, reusable components, document previews, and production-quality state/data handling.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

On Linux or macOS, replace the activation command with:

```bash
source .venv/bin/activate
```

## Environment Variables

Copy `.env.example` and set values in your deployment provider.

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Recommended | Enables Gemini answers and audio transcription. |
| `GEMINI_CHAT_MODEL` | No | Gemini model for conversational/context answers. |
| `GEMINI_AUDIO_MODEL` | No | Gemini model for audio transcription. |
| `MAX_UPLOAD_MB` | No | Per-file upload guard. Defaults to 25. |
| `YOUTUBE_TRANSCRIPT_LANGS` | No | Comma-separated transcript language preference. |
| `INPUT_PRICE_PER_MILLION_TOKENS` | No | Optional cost estimator input rate. |
| `OUTPUT_PRICE_PER_MILLION_TOKENS` | No | Optional cost estimator output rate. |
| `RAG_STORE_DIR` | No | Persistent ChromaDB directory. Defaults to `data/chroma`. |
| `RAG_COLLECTION_NAME` | No | Chroma collection name. Defaults to `documents`. |
| `RAG_CHUNK_TOKENS` | No | Approximate chunk size for indexed text. Defaults to `260`. |
| `RAG_CHUNK_OVERLAP_TOKENS` | No | Approximate chunk overlap. Defaults to `45`. |
| `RAG_TOP_K` | No | Number of chunks retrieved for grounded answers. Defaults to `5`. |
| `RAG_EMBEDDING_DIMENSIONS` | No | Size of the local hashing embedding vector. Defaults to `384`. |

## RAG API

The normal `/api/chat` endpoint now indexes uploaded readable content and uses retrieved chunks for document questions. Dedicated RAG endpoints are also available:

- `POST /api/rag/ingest` with multipart `files` to extract, chunk, embed, and persist documents.
- `POST /api/rag/query` with JSON `{"query": "...", "top_k": 5}` to answer from indexed documents with citations.
- `GET /api/rag/documents` to list indexed documents.
- `DELETE /api/rag/documents/{document_id}` to remove one indexed document.

## Docker

```bash
docker build -t multi-input-agent .
docker run --env-file .env -p 8000:8000 multi-input-agent
```

The Docker image installs Tesseract for OCR and ffmpeg for common media handling.

## Deployment

Render is the shortest path with the included `render.yaml`.

1. Push this repository to GitHub.
2. Create a new Render Blueprint from the repository.
3. Add `GEMINI_API_KEY` as a secret environment variable.
4. Deploy and submit the generated public URL.

Live deployment URL: https://rag-based-document-extraction-system.onrender.com

## Required Sample Checks

Use the UI or `/api/chat` endpoint for these:

- Audio lecture plus "Summarize this" should return transcription-derived summary and duration when the API key is configured.
- PDF meeting notes plus "What are the action items?" should return only action items.
- Image screenshot containing code plus "Explain" should OCR, explain the code, flag risks, and include time complexity.
- PDF containing a YouTube URL plus "Hit the YT URL in this PDF and give me a summary of it" should extract the URL, fetch transcript, and summarize it.
- Audio plus PDF plus "Do the audio and the document discuss the same topic?" should transcribe, parse, compare, and return a verdict.

## Tests

```bash
python -m unittest discover -s tests
```

`pytest` is included in `requirements.txt`, so this also works after installing dependencies:

```bash
pytest
```

Run the executable checks for the required assignment workflows:

```bash
python -m unittest tests.test_required_cases
```

## Design Decisions

- The app uses a deterministic local planner first so ambiguous requests trigger a follow-up instead of guessing.
- Optional external tools fail gracefully and report warnings in the UI.
- All tool outputs are normalized into `ExtractedContent`, which keeps source type, text, confidence, duration, and warnings together.
- Final answers are text-only, while the UI separately displays extracted text and plan trace for explainability.
- Open-ended general answering and audio transcription use Gemini when configured; task-specific flows have local fallbacks for reliable demos.

