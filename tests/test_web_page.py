import tempfile
import unittest

from hmm_segmenter.model import HMMSegmenter
from web.app import create_app


class WebPageTests(unittest.TestCase):
    def test_workbench_contains_required_controls(self):
        pi = {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0}
        A = {
            "B": {"B": 0.0, "M": 0.0, "E": 1.0, "S": 0.0},
            "M": {"B": 0.0, "M": 0.0, "E": 1.0, "S": 0.0},
            "E": {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0},
            "S": {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0},
        }
        B = {state: {} for state in ("B", "M", "E", "S")}
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(HMMSegmenter(pi, A, B), result_dir=tmp)
            html = app.test_client().get("/").get_data(as_text=True)
        for element_id in (
            "upload-form",
            "file-input",
            "drop-zone",
            "analyze-button",
            "status-region",
            "file-name",
            "sentence-count",
            "word-count",
            "preview",
            "download-button",
        ):
            self.assertIn(f'id="{element_id}"', html)


if __name__ == "__main__":
    unittest.main()
