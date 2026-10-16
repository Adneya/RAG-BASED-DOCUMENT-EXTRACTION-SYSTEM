import sys
import types
import unittest
from unittest.mock import patch

from app.tools.youtube import fetch_youtube_transcript


class YouTubeTranscriptTests(unittest.TestCase):
    def test_fetch_transcript_supports_current_api(self):
        class FetchedTranscript:
            def to_raw_data(self):
                return [{"text": "Hello"}, {"text": "from current API"}]

        class FakeYouTubeTranscriptApi:
            call = None

            def fetch(self, video_id, languages):
                FakeYouTubeTranscriptApi.call = (video_id, languages)
                return FetchedTranscript()

        fake_module = types.SimpleNamespace(YouTubeTranscriptApi=FakeYouTubeTranscriptApi)

        with patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            transcript = fetch_youtube_transcript(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )

        self.assertEqual(transcript.video_id, "dQw4w9WgXcQ")
        self.assertEqual(transcript.text, "Hello from current API")
        self.assertEqual(transcript.warnings, [])
        self.assertEqual(FakeYouTubeTranscriptApi.call[0], "dQw4w9WgXcQ")

    def test_fetch_transcript_supports_legacy_api(self):
        class FakeYouTubeTranscriptApi:
            call = None

            @staticmethod
            def get_transcript(video_id, languages):
                FakeYouTubeTranscriptApi.call = (video_id, languages)
                return [{"text": "Hello"}, {"text": "from legacy API"}]

        fake_module = types.SimpleNamespace(YouTubeTranscriptApi=FakeYouTubeTranscriptApi)

        with patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            transcript = fetch_youtube_transcript("https://youtu.be/dQw4w9WgXcQ")

        self.assertEqual(transcript.video_id, "dQw4w9WgXcQ")
        self.assertEqual(transcript.text, "Hello from legacy API")
        self.assertEqual(transcript.warnings, [])
        self.assertEqual(FakeYouTubeTranscriptApi.call[0], "dQw4w9WgXcQ")


if __name__ == "__main__":
    unittest.main()
