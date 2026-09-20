import json
import tempfile
import unittest
from pathlib import Path

from hmm_segmenter.model import HMMSegmenter, ModelError


def sample_model():
    pi = {"B": 0.3, "M": 0.0, "E": 0.2, "S": 0.5}
    A = {
        "B": {"B": 0.0, "M": 0.5, "E": 0.5, "S": 0.0},
        "M": {"B": 0.0, "M": 0.5, "E": 0.5, "S": 0.0},
        "E": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
        "S": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
    }
    B = {
        "B": {"自": 0.5, "语": 0.1, "g": 1.0},
        "M": {"然": 1.0, "言": 0.5, "o": 1.0},
        "E": {"言": 0.5, "好": 1.0, "d": 1.0},
        "S": {"好": 1.0},
    }
    return HMMSegmenter(pi, A, B, {"corpus": "unit"})


class HMMSegmenterTests(unittest.TestCase):
    def test_cut_returns_words_without_losing_characters(self):
        model = sample_model()
        words = model.cut("自然语言")
        self.assertEqual("".join(words), "自然语言")

    def test_cut_treats_whitespace_as_boundary(self):
        model = sample_model()
        self.assertEqual(model.cut("好 good"), ["好", "good"])

    def test_save_and_load_round_trip(self):
        model = sample_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = HMMSegmenter.load(path)
            self.assertEqual(loaded.cut("好"), ["好"])
            self.assertEqual(loaded.metadata["corpus"], "unit")

    def test_load_rejects_missing_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"pi": {}, "A": {}, "B": {}}), encoding="utf-8")
            with self.assertRaises(ModelError):
                HMMSegmenter.load(path)


if __name__ == "__main__":
    unittest.main()
