import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.server as server


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.prompt = None

    def __call__(self, prompt, return_tensors):
        self.prompt = prompt
        return FakeInputs(input_ids=[1])

    def decode(self, output, skip_special_tokens):
        return "generated text"


class FakeModel:
    def __init__(self):
        self.device = None
        self.max_new_tokens = None

    def generate(self, input_ids, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        return ["generated tokens"]


class ServerApiTests(unittest.TestCase):
    def setUp(self):
        self.original_model = server._model
        self.original_tokenizer = server._tokenizer
        self.original_device = server._device
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        server._model = self.model
        server._tokenizer = self.tokenizer
        server._device = "cpu"

        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                get_device_name=lambda device_index: "test-gpu",
            ),
            no_grad=nullcontext,
        )
        self.torch_patch = patch.dict(sys.modules, {"torch": self.torch})
        self.torch_patch.start()

        self.loader_patch = patch.object(server, "_load_model")
        self.loader_patch.start()
        self.client = TestClient(server.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.loader_patch.stop()
        self.torch_patch.stop()
        server._model = self.original_model
        server._tokenizer = self.original_tokenizer
        server._device = self.original_device

    def test_root_reports_service_identity(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"service": "gpu-model-serving", "status": "ok"},
        )

    def test_health_and_readiness_report_loaded_model(self):
        health = self.client.get("/healthz")
        readiness = self.client.get("/readyz")

        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["model_loaded"])
        self.assertEqual(readiness.json()["status"], "ready")

    def test_readiness_reports_loading_when_model_is_not_loaded(self):
        server._model = None
        server._tokenizer = None

        response = self.client.get("/readyz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "loading")

    def test_device_reports_cpu_when_cuda_is_unavailable(self):
        response = self.client.get("/device")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"device": "cpu", "cuda_available": False, "gpu_name": None},
        )

    def test_generate_uses_model_and_returns_text_and_latency(self):
        response = self.client.post(
            "/generate",
            json={"prompt": "  Explain Kubernetes.  ", "max_new_tokens": 20},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "generated text")
        self.assertEqual(response.json()["device"], "cpu")
        self.assertIsInstance(response.json()["latency_ms"], float)
        self.assertEqual(self.tokenizer.prompt, "Explain Kubernetes.")
        self.assertEqual(self.model.max_new_tokens, 20)

    def test_generate_rejects_blank_prompt(self):
        response = self.client.post("/generate", json={"prompt": "   "})

        self.assertEqual(response.status_code, 422)

    def test_generate_rejects_token_count_outside_allowed_range(self):
        for token_count in (0, 257):
            with self.subTest(max_new_tokens=token_count):
                response = self.client.post(
                    "/generate",
                    json={"prompt": "hello", "max_new_tokens": token_count},
                )

                self.assertEqual(response.status_code, 422)

    def test_generate_rejects_extra_fields(self):
        response = self.client.post(
            "/generate",
            json={"prompt": "hello", "unexpected": True},
        )

        self.assertEqual(response.status_code, 422)

    def test_generate_loads_model_lazily(self):
        server._model = None
        server._tokenizer = None
        tokenizer = FakeTokenizer()
        model = FakeModel()

        def load_model():
            server._model = model
            server._tokenizer = tokenizer

        with patch.object(server, "_load_model", side_effect=load_model) as loader:
            response = self.client.post(
                "/generate",
                json={"prompt": "hello", "max_new_tokens": 5},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "generated text")
        loader.assert_called_once_with()

    def test_generate_returns_service_unavailable_when_model_loading_fails(self):
        server._model = None
        server._tokenizer = None

        with patch.object(
            server,
            "_load_model",
            side_effect=RuntimeError("model download failed"),
        ):
            response = self.client.post(
                "/generate",
                json={"prompt": "hello"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("model download failed", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
