from io import BytesIO
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import abort
from hmm_segmenter.model import HMMSegmenter
from werkzeug.exceptions import BadRequest, MethodNotAllowed, RequestEntityTooLarge

from web.app import create_app
from web.service import MAX_UPLOAD_BYTES, SegmentationService, ServiceError


def sample_model():
    pi = {"B": 0.3, "M": 0.0, "E": 0.2, "S": 0.5}
    A = {
        "B": {"B": 0.0, "M": 0.5, "E": 0.5, "S": 0.0},
        "M": {"B": 0.0, "M": 0.5, "E": 0.5, "S": 0.0},
        "E": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
        "S": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
    }
    B = {
        "B": {"自": 1.0},
        "M": {"然": 1.0},
        "E": {"语": 1.0},
        "S": {"好": 1.0},
    }
    return HMMSegmenter(pi, A, B)


class SegmentationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = SegmentationService(sample_model(), self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_segment_upload_returns_preview_and_download_metadata(self):
        result = self.service.segment_upload("demo.txt", "自然语。好！".encode())
        self.assertEqual(result["output_name"], "demo_seg.txt")
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["word_count"], 2)
        path, name = self.service.resolve_result(result["job_id"])
        self.assertEqual(path.read_text(encoding="utf-8"), "自然语\n好\n")
        self.assertEqual(name, "demo_seg.txt")

    def test_rejects_wrong_extension(self):
        with self.assertRaises(ServiceError) as context:
            self.service.segment_upload("demo.pdf", b"text")
        self.assertEqual(context.exception.code, "INVALID_FILE_TYPE")

    def test_rejects_empty_input(self):
        with self.assertRaises(ServiceError) as context:
            self.service.segment_upload("demo.txt", b"")
        self.assertEqual(context.exception.code, "EMPTY_FILE")

    def test_rejects_oversized_input(self):
        service = SegmentationService(
            sample_model(),
            self.temp.name,
            max_upload_bytes=3,
        )
        with self.assertRaises(ServiceError) as context:
            service.segment_upload("demo.txt", b"1234")
        self.assertEqual(context.exception.code, "FILE_TOO_LARGE")

    def test_rejects_unknown_encoding(self):
        with self.assertRaises(ServiceError) as context:
            self.service.segment_upload("demo.txt", b"\xff")
        self.assertEqual(context.exception.code, "DECODE_ERROR")

    def test_preview_is_limited_to_twenty_sentences(self):
        result = self.service.segment_upload(
            "demo.txt",
            "好。".join([""] * 22).encode(),
        )
        self.assertEqual(result["sentence_count"], 21)
        self.assertEqual(len(result["preview"]), 20)

    def test_output_name_removes_path_components(self):
        result = self.service.segment_upload("../../demo.txt", "好".encode())
        self.assertEqual(result["output_name"], "demo_seg.txt")

    def test_cleanup_removes_expired_result(self):
        result = self.service.segment_upload("demo.txt", "好".encode())
        path, _ = self.service.resolve_result(result["job_id"])
        job_id = result["job_id"]
        old = time.time() - 7200
        path.touch()
        import os
        os.utime(path, (old, old))
        count = self.service.cleanup_expired(now=time.time())
        self.assertEqual(count, 1)
        with self.assertRaises(ServiceError) as context:
            self.service.resolve_result(job_id)
        self.assertEqual(context.exception.code, "RESULT_EXPIRED")

    def test_segment_upload_cleans_expired_results_before_writing_new_result(self):
        expired = self.service.segment_upload("old.txt", "好".encode())
        expired_path, _ = self.service.resolve_result(expired["job_id"])
        old = time.time() - 7200
        expired_path.touch()
        import os
        os.utime(expired_path, (old, old))

        current = self.service.segment_upload("new.txt", "好".encode())

        self.assertFalse(expired_path.exists())
        with self.assertRaises(ServiceError) as context:
            self.service.resolve_result(expired["job_id"])
        self.assertEqual(context.exception.code, "RESULT_EXPIRED")
        self.service.resolve_result(current["job_id"])


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(sample_model(), result_dir=self.temp.name)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def test_upload_returns_json_and_download_works(self):
        response = self.client.post(
            "/api/segment",
            data={"file": (BytesIO("自然语。好！".encode()), "demo.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        with self.client.get(payload["download_url"]) as download:
            self.assertEqual(download.status_code, 200)
            self.assertEqual(download.data.decode("utf-8"), "自然语\n好\n")

    def test_upload_error_has_consistent_shape(self):
        response = self.client.post(
            "/api/segment",
            data={"file": (BytesIO(b"hello"), "demo.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "INVALID_FILE_TYPE")

    def test_missing_model_returns_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(None, result_dir=tmp)
            response = app.test_client().post(
                "/api/segment",
                data={"file": (BytesIO(b"hello"), "demo.txt")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"]["code"], "MODEL_UNAVAILABLE")

    def test_missing_result_returns_consistent_error(self):
        response = self.client.get("/api/results/not-a-uuid/download")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["code"], "RESULT_NOT_FOUND")

    def test_method_not_allowed_returns_consistent_error(self):
        with self.client.get("/api/segment") as response:
            self.assertEqual(response.status_code, 405)
            self.assertTrue(response.is_json)
            self.assertEqual(
                response.get_json()["error"],
                {
                    "code": "METHOD_NOT_ALLOWED",
                    "message": MethodNotAllowed.description,
                },
            )

    def test_request_entity_too_large_returns_consistent_error(self):
        with self.client.post(
            "/api/segment",
            data=b"x" * (MAX_UPLOAD_BYTES + 1),
            content_type="text/plain",
        ) as response:
            self.assertEqual(response.status_code, 413)
            self.assertTrue(response.is_json)
            self.assertEqual(
                response.get_json()["error"],
                {
                    "code": "FILE_TOO_LARGE",
                    "message": RequestEntityTooLarge.description,
                },
            )

    def test_other_api_http_error_returns_consistent_error(self):
        @self.app.get("/api/test-error")
        def api_test_error():
            abort(400)

        with self.client.get("/api/test-error") as response:
            self.assertEqual(response.status_code, 400)
            self.assertTrue(response.is_json)
            self.assertEqual(
                response.get_json()["error"],
                {
                    "code": "HTTP_ERROR",
                    "message": BadRequest.description,
                },
            )

    def test_unexpected_api_exception_logs_and_returns_internal_error(self):
        @self.app.get("/api/test-runtime-error")
        def api_runtime_error():
            raise RuntimeError("boom")

        with patch.object(self.app.logger, "exception") as mock_exception:
            response = self.client.get("/api/test-runtime-error")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "INTERNAL_ERROR",
        )
        mock_exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
