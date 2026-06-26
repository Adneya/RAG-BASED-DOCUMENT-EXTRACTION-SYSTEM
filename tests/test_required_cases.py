import unittest
from unittest.mock import AsyncMock, patch

from app.agent.pipeline import AgentPipeline
from app.schemas import ExtractedContent
from app.tools.youtube import YouTubeTranscript


class DummyUpload:
    def __init__(self, filename: str):
        self.filename = filename


class RequiredWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_lecture_returns_summary_sections_and_duration(self):
        transcript = (
            "The lecture introduces retrieval augmented generation for documents. "
            "It explains how extracted text is chunked and embedded for search. "
            "The speaker describes using retrieved passages to answer user questions. "
            "The lecture highlights that source grounding reduces hallucinations. "
            "It ends with deployment guidance for a FastAPI document agent."
        )

        with patch("app.agent.pipeline.extract_upload", new_callable=AsyncMock) as extract_upload:
            extract_upload.return_value = ExtractedContent(
                filename="lecture.wav",
                source_type="audio",
                text=transcript,
                duration_seconds=300,
            )

            response = await AgentPipeline().run(
                query="Summarize this",
                files=[DummyUpload("lecture.wav")],
            )

        self.assertEqual(response.status, "completed")
        self.assertIn("1-line summary:", response.final_output)
        self.assertIn("3 bullets:", response.final_output)
        self.assertIn("5-sentence summary:", response.final_output)
        self.assertIn("Duration:", response.final_output)
        self.assertIn("lecture.wav: 5m 00s (300 seconds)", response.final_output)

    async def test_pdf_action_items_returns_action_items_only(self):
        notes = (
            "[Page 1]\nMeeting notes. "
            "Owner: Maya will send the launch checklist by Friday. "
            "Ravi should update the dashboard before Monday. "
            "Budget discussion was closed."
        )

        with patch("app.agent.pipeline.extract_upload", new_callable=AsyncMock) as extract_upload:
            extract_upload.return_value = ExtractedContent(
                filename="meeting-notes.pdf",
                source_type="pdf",
                text=notes,
            )

            response = await AgentPipeline().run(
                query="What are the action items?",
                files=[DummyUpload("meeting-notes.pdf")],
            )

        self.assertEqual(response.status, "completed")
        self.assertTrue(response.final_output.startswith("Action items:"))
        self.assertIn("Maya will send the launch checklist", response.final_output)
        self.assertIn("Ravi should update the dashboard", response.final_output)
        self.assertNotIn("1-line summary:", response.final_output)
        self.assertNotIn("Budget discussion was closed", response.final_output)

    async def test_image_code_explain_detects_language_and_bugs(self):
        code = "def divide(total):\n    return total / 0"

        with patch("app.agent.pipeline.extract_upload", new_callable=AsyncMock) as extract_upload:
            extract_upload.return_value = ExtractedContent(
                filename="code-screenshot.png",
                source_type="image",
                text=code,
                ocr_confidence=0.92,
            )

            response = await AgentPipeline().run(
                query="Explain",
                files=[DummyUpload("code-screenshot.png")],
            )

        self.assertEqual(response.status, "completed")
        self.assertIn("Language: Python", response.final_output)
        self.assertIn("Potential bugs or risks:", response.final_output)
        self.assertIn("division by zero", response.final_output)

    async def test_pdf_youtube_url_fetches_transcript_without_prompting(self):
        pdf_text = (
            "[Page 1]\nWatch this source: "
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        transcript = (
            "The video explains retrieval augmented generation. "
            "It shows how a document is parsed and embedded. "
            "It retrieves relevant passages for a question. "
            "It uses those passages to produce a grounded answer. "
            "It recommends showing the plan trace to users."
        )

        with (
            patch("app.agent.pipeline.extract_upload", new_callable=AsyncMock) as extract_upload,
            patch("app.agent.pipeline.fetch_youtube_transcript") as fetch_transcript,
        ):
            extract_upload.return_value = ExtractedContent(
                filename="source-link.pdf",
                source_type="pdf",
                text=pdf_text,
            )
            fetch_transcript.return_value = YouTubeTranscript(
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                video_id="dQw4w9WgXcQ",
                text=transcript,
                warnings=[],
            )

            response = await AgentPipeline().run(
                query="Hit the YT URL in this PDF and give me a summary of it",
                files=[DummyUpload("source-link.pdf")],
            )

        fetch_transcript.assert_called_once()
        self.assertEqual(response.status, "completed")
        self.assertIsNone(response.follow_up_question)
        self.assertIn("youtube_transcript", [step.tool for step in response.plan_trace])
        self.assertIn("1-line summary:", response.final_output)
        self.assertIn("3 bullets:", response.final_output)
        self.assertIn("5-sentence summary:", response.final_output)
        self.assertNotIn("youtube.com/watch", response.final_output)

    async def test_audio_and_pdf_same_topic_returns_comparison(self):
        audio_text = (
            "The audio lecture discusses retrieval augmented generation. "
            "It explains vector search and document grounding for question answering."
        )
        pdf_text = (
            "The PDF note discusses retrieval augmented generation. "
            "It covers vector search, document grounding, and question answering."
        )

        with patch("app.agent.pipeline.extract_upload", new_callable=AsyncMock) as extract_upload:
            extract_upload.side_effect = [
                ExtractedContent(
                    filename="lecture.wav",
                    source_type="audio",
                    text=audio_text,
                    duration_seconds=120,
                ),
                ExtractedContent(
                    filename="notes.pdf",
                    source_type="pdf",
                    text=pdf_text,
                ),
            ]

            response = await AgentPipeline().run(
                query="Do the audio and the document discuss the same topic?",
                files=[DummyUpload("lecture.wav"), DummyUpload("notes.pdf")],
            )

        self.assertEqual(response.status, "completed")
        self.assertIn("Verdict:", response.final_output)
        self.assertIn("same broad topic", response.final_output)
        self.assertIn("Shared terms:", response.final_output)


if __name__ == "__main__":
    unittest.main()
