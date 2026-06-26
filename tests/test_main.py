import unittest

from fastapi.testclient import TestClient

from app.main import app


class MainRouteTests(unittest.TestCase):
    def test_index_renders(self):
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Multi-Input Agent", response.text)

    def test_health_check(self):
        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
