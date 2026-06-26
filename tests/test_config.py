import os
import tempfile
import unittest
from pathlib import Path

from app.config import load_dotenv_file


class ConfigTests(unittest.TestCase):
    def test_load_dotenv_file_sets_missing_values_without_overriding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "GEMINI_API_KEY=from_file\n"
                "GEMINI_CHAT_MODEL=file_model\n",
                encoding="utf-8",
            )

            original_key = os.environ.pop("GEMINI_API_KEY", None)
            original_model = os.environ.get("GEMINI_CHAT_MODEL")
            os.environ["GEMINI_CHAT_MODEL"] = "existing_model"
            try:
                load_dotenv_file(env_path)

                self.assertEqual(os.environ["GEMINI_API_KEY"], "from_file")
                self.assertEqual(os.environ["GEMINI_CHAT_MODEL"], "existing_model")
            finally:
                if original_key is None:
                    os.environ.pop("GEMINI_API_KEY", None)
                else:
                    os.environ["GEMINI_API_KEY"] = original_key

                if original_model is None:
                    os.environ.pop("GEMINI_CHAT_MODEL", None)
                else:
                    os.environ["GEMINI_CHAT_MODEL"] = original_model


if __name__ == "__main__":
    unittest.main()
