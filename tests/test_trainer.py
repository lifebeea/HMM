import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hmm_segmenter.trainer import read_text_file, train_hmm


class TrainerTests(unittest.TestCase):
    def test_read_text_file_supports_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("自然语言", encoding="utf-8")
            self.assertEqual(read_text_file(path), "自然语言")

    def test_train_hmm_builds_normalized_probabilities(self):
        words = {
            "我喜欢自然语言": ["我", "喜欢", "自然语言"],
            "今天下雨": ["今天", "下雨"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            model_path = Path(tmp) / "model.json"
            corpus.write_text("我喜欢自然语言。今天下雨。", encoding="utf-8")
            with patch(
                "hmm_segmenter.trainer.jieba.cut",
                side_effect=lambda sentence: words[sentence],
            ):
                model = train_hmm(corpus, model_path, max_chars=100)
            self.assertAlmostEqual(sum(model.pi.values()), 1.0)
            for state in model.A:
                self.assertAlmostEqual(sum(model.A[state].values()), 1.0)
            self.assertTrue(model_path.exists())
            self.assertEqual(model.cut("我喜欢"), ["我", "喜欢"])


if __name__ == "__main__":
    unittest.main()
