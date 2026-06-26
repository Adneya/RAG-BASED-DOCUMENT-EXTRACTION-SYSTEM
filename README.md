# Multi-Input Agent

A FastAPI application that accepts text, images, PDFs, audio files, and multiple files in one request, extracts text, infers the user's goal, runs the minimum useful tool sequence, and returns text-only results with extracted text and a visible plan trace.

## Features

- Chat-like UI with one text box and multi-file upload.
- Image OCR for JPG and PNG files with confidence when available.
- PDF text extraction with OCR fallback for scanned pages.
- Audio transcription for MP3, WAV, and M4A through Gemini when `GEMINI_API_KEY` is set, with duration reporting when WAV metadata or `ffprobe` is available.
- Open-ended conversational answers use Gemini when `GEMINI_API_KEY` is set.
- YouTube URL detection from any extracted input and transcript fetching through `youtube-transcript-api`.
- Intent handling for extraction, YouTube transcript summary, conversational answering, summaries, sentiment, code explanation, audio summary, action items, and cross-input comparison.
- Mandatory follow-up behavior when the user goal is unclear.
- Plan trace, extracted text panel, warnings, and token/cost estimate.
- Dockerfile and Render deployment configuration.

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

