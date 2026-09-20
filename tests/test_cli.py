from io import StringIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hmm_segmenter import cli
from hmm_segmenter.cli import segment_file, segment_text
from hmm_segmenter.model import HMMSegmenter


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
        "M": {"然": 0.5, "语": 0.5},
        "E": {"言": 1.0},
        "S": {"好": 1.0},
    }
    return HMMSegmenter(pi, A, B)


class CLITests(unittest.TestCase):
    def test_segment_text_handles_multiple_sentences(self):
        self.assertEqual(segment_text(sample_model(), "自然语言。好！"), ["自然语言", "好"])

    def test_segment_file_writes_one_sentence_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "demo.txt"
            source.write_text("自然语言。好！", encoding="utf-8")
            output = segment_file(sample_model(), source)
            self.assertEqual(output.name, "demo_seg.txt")
            self.assertEqual(output.read_text(encoding="utf-8"), "自然语言\n好\n")

    def test_interactive_retraining_failure_keeps_menu_running(self):
        model = sample_model()
        output = StringIO()

        with patch(
            "hmm_segmenter.cli.train_hmm",
            side_effect=OSError("disk full"),
        ), patch("builtins.input", side_effect=["3", "0"]), patch(
            "sys.stdout",
            output,
        ):
            cli.interactive_menu(model, Path("corpus.txt"), Path("model.json"))

        captured = output.getvalue()
        self.assertIn("训练失败：disk full", captured)
        self.assertEqual(captured.count("0 - 退出"), 2)


if __name__ == "__main__":
    unittest.main()
