import unittest

from app.agent.intent import infer_intent
from app.agent.responses import render_response
from app.schemas import ExtractedContent
from app.tools.youtube import detect_youtube_urls, extract_video_id


class AgentIntentTests(unittest.TestCase):
    def test_ambiguous_file_query_requires_clarification(self):
        decision = infer_intent(
            "analyze",
            [ExtractedContent(filename="notes.pdf", source_type="pdf", text="Meeting notes")],
            [],
        )

        self.assertEqual(decision.intent, "clarify")
        self.assertIsNotNone(decision.follow_up_question)

    def test_summary_intent(self):
        decision = infer_intent(
            "Summarize this",
            [ExtractedContent(filename="notes.txt", source_type="text", text="Alpha. Beta.")],
            [],
        )

        self.assertEqual(decision.intent, "summary")

    def test_youtube_detection(self):
        text = "See https://www.youtube.com/watch?v=dQw4w9WgXcQ for context."
        urls = detect_youtube_urls(text)

        self.assertEqual(len(urls), 1)
        self.assertEqual(extract_video_id(urls[0]), "dQw4w9WgXcQ")


class AgentResponseTests(unittest.TestCase):
    def test_summary_has_required_sections(self):
        output = render_response(
            "summary",
            "summarize",
            [
                ExtractedContent(
                    filename="demo.txt",
                    source_type="text",
                    text=(
                        "The project extracts text from files. "
                        "It routes the user goal to a small tool chain. "
                        "It shows extracted text and plan trace. "
                        "It supports summaries and comparisons. "
                        "It asks a follow-up when the task is unclear."
                    ),
                )
            ],
        )

        self.assertIn("1-line summary:", output)
        self.assertIn("3 bullets:", output)
        self.assertIn("5-sentence summary:", output)

    def test_sentiment_response_shape(self):
        output = render_response(
            "sentiment",
            "",
            [
                ExtractedContent(
                    filename="feedback.txt",
                    source_type="text",
                    text="The result is excellent, clear, and useful.",
                )
            ],
        )

        self.assertIn("Label: Positive", output)
        self.assertIn("Confidence:", output)
        self.assertIn("Justification:", output)

    def test_code_explanation_mentions_complexity(self):
        code = "def total(items):\n    s = 0\n    for item in items:\n        s += item\n    return s"
        output = render_response(
            "code_explanation",
            "Explain",
            [ExtractedContent(filename="code.py", source_type="text", text=code)],
        )

        self.assertIn("Language: Python", output)
        self.assertIn("Time complexity:", output)


if __name__ == "__main__":
    unittest.main()

