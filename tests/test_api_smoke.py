import unittest

import main


class ApiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = main.app.test_client()

    def test_state_endpoint(self) -> None:
        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("tasks", body)
        self.assertIn("memory", body)

    def test_integrations_health(self) -> None:
        response = self.client.get("/api/integrations/health")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("tools", body)


if __name__ == "__main__":
    unittest.main()
