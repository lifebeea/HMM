from html.parser import HTMLParser
import re
import tempfile
import unittest

from hmm_segmenter.model import HMMSegmenter
from web.app import create_app


class ElementByIdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.elements[element_id] = (tag, attributes)


class WebPageTests(unittest.TestCase):
    def setUp(self):
        pi = {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0}
        A = {
            "B": {"B": 0.0, "M": 0.0, "E": 1.0, "S": 0.0},
            "M": {"B": 0.0, "M": 0.0, "E": 1.0, "S": 0.0},
            "E": {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0},
            "S": {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0},
        }
        B = {state: {} for state in ("B", "M", "E", "S")}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        app = create_app(HMMSegmenter(pi, A, B), result_dir=self.temp.name)
        self.client = app.test_client()
        self.html = self.client.get("/").get_data(as_text=True)
        parser = ElementByIdParser()
        parser.feed(self.html)
        self.elements = parser.elements

    def test_workbench_contains_required_controls(self):
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
            self.assertIn(element_id, self.elements)

    def test_file_input_is_visually_hidden_without_hidden_attribute(self):
        tag, attributes = self.elements["file-input"]
        self.assertEqual(tag, "input")
        self.assertEqual(attributes.get("type"), "file")
        self.assertEqual(attributes.get("class"), "visually-hidden")
        self.assertNotIn("hidden", attributes)

    def test_download_control_is_a_disabled_button(self):
        tag, attributes = self.elements["download-button"]
        self.assertEqual(tag, "button")
        self.assertEqual(attributes.get("type"), "button")
        self.assertEqual(attributes.get("class"), "primary-button")
        self.assertIn("disabled", attributes)

    def test_reduced_motion_disables_loading_animation(self):
        with self.client.get("/static/styles.css") as response:
            css = response.get_data(as_text=True)
        self.assertRegex(
            css,
            re.compile(
                r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{"
                r".*?\.progress\s+span\s*\{[^}]*animation:\s*none\s*;?[^}]*\}"
                r"\s*\}",
                re.DOTALL,
            ),
        )

    def test_page_contains_no_cdn_urls(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)

    def test_page_loads_javascript_without_inline_result_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            pi = {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0}
            A = {
                "B": {"B": 0.0, "M": 0.0, "E": 1.0, "S": 0.0},
                "M": {"B": 0.0, "M": 0.0, "E": 1.0, "S": 0.0},
                "E": {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0},
                "S": {"B": 1.0, "M": 0.0, "E": 0.0, "S": 0.0},
            }
            B = {state: {} for state in ("B", "M", "E", "S")}
            app = create_app(HMMSegmenter(pi, A, B), result_dir=tmp)
            with app.test_client().get("/static/app.js") as response:
                script = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("formData.append", script)
        self.assertNotIn("preview.innerHTML =", script)

    def test_javascript_locks_upload_controls_while_processing(self):
        with self.client.get("/static/app.js") as response:
            script = response.get_data(as_text=True)

        self.assertRegex(script, r"\blet\s+processing\s*=\s*false\s*;")
        self.assertRegex(
            script,
            r"function\s+chooseFile\s*\([^)]*\)\s*\{\s*"
            r"if\s*\(processing\)\s*return\s*;",
        )
        self.assertRegex(
            script,
            r"processing\s*=\s*true\s*;[\s\S]*?"
            r"dropZone\.classList\.add\(\"disabled\"\)\s*;[\s\S]*?"
            r"dropZone\.setAttribute\(\"aria-disabled\",\s*\"true\"\)\s*;",
        )
        self.assertRegex(
            script,
            r"processing\s*=\s*true\s*;[\s\S]*?"
            r"fileInput\.disabled\s*=\s*true\s*;",
        )
        self.assertRegex(
            script,
            r"processing\s*=\s*true\s*;[\s\S]*?"
            r"analyzeButton\.disabled\s*=\s*true\s*;",
        )
        self.assertRegex(
            script,
            r"finally\s*\{[\s\S]*?"
            r"processing\s*=\s*false\s*;[\s\S]*?"
            r"dropZone\.classList\.remove\(\"disabled\"\)\s*;[\s\S]*?"
            r"dropZone\.setAttribute\(\"aria-disabled\",\s*\"false\"\)\s*;"
            r"[\s\S]*?analyzeButton\.disabled\s*=\s*!selectedFile\s*;",
        )


if __name__ == "__main__":
    unittest.main()
