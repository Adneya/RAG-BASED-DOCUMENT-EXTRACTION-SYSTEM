# Architecture Diagram

```mermaid
flowchart TD
    UI[Chat UI] --> API[FastAPI /api/chat]
    API --> Extractor[Input Extractor]
    Extractor --> Text[Text Decoder]
    Extractor --> Image[Image OCR]
    Extractor --> PDF[PDF Parser and OCR Fallback]
    Extractor --> Audio[Audio Transcription]
    Text --> Context[Unified Context]
    Image --> Context
    PDF --> Context
    Audio --> Context
    Context --> Router[Intent Router and Follow-up Guard]
    Router --> ToolRegistry[Tool Registry]
    ToolRegistry --> YT[YouTube Transcript Tool]
    ToolRegistry --> Summary[Summarizer]
    ToolRegistry --> Sentiment[Sentiment Analyzer]
    ToolRegistry --> Code[Code Explainer]
    ToolRegistry --> Compare[Cross-input Reasoner]
    YT --> Response[Text-only Response Builder]
    Summary --> Response
    Sentiment --> Response
    Code --> Response
    Compare --> Response
    Router --> Response
    Response --> Trace[Plan Trace and Cost Estimate]
    Trace --> UI
    Response --> UI
```

## Notes

- The extractor normalizes all supported inputs into text plus source metadata.
- The intent router applies the mandatory follow-up rule before running task tools.
- YouTube transcript fetching is triggered when a URL appears in any input and the query asks to use that URL.
- Audio transcription and open-ended conversational answers use OpenAI when `OPENAI_API_KEY` is configured.
- OCR uses Tesseract through `pytesseract`; the Dockerfile installs the Tesseract binary.

