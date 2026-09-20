import json
import math
import tempfile
import unittest
from pathlib import Path

from hmm_segmenter.model import HMMSegmenter, ModelError
from hmm_segmenter.viterbi import STATES


def sample_model():
    pi = {"B": 0.3, "M": 0.0, "E": 0.2, "S": 0.5}
    A = {
        "B": {"B": 0.0, "M": 0.5, "E": 0.5, "S": 0.0},
        "M": {"B": 0.0, "M": 0.5, "E": 0.5, "S": 0.0},
        "E": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
        "S": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
    }
    B = {
        "B": {"自": 0.5, "g": 0.5},
        "M": {"然": 0.5, "o": 0.5},
        "E": {"语": 0.5, "d": 0.5},
        "S": {"言": 0.5, "好": 0.5},
    }
    return HMMSegmenter(pi, A, B, {"corpus": "unit"})


class HMMSegmenterTests(unittest.TestCase):
    def assert_model_rejected(self, model):
        with self.assertRaises(ModelError):
            HMMSegmenter(model.pi, model.A, model.B)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            model.save(path)
            with self.assertRaises(ModelError):
                HMMSegmenter.load(path)

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

    def test_constructor_and_load_reject_malformed_emission_row(self):
        model = sample_model()
        model.B["B"] = list(STATES)

        self.assert_model_rejected(model)

    def test_constructor_and_load_reject_negative_probability(self):
        for section, state, value in (
            ("pi", "B", -0.1),
            ("A", "B", -0.1),
            ("B", "B", -0.1),
        ):
            with self.subTest(section=section):
                model = sample_model()
                if section == "pi":
                    model.pi[state] = value
                else:
                    row = getattr(model, section)[state]
                    row[next(iter(row))] = value

                self.assert_model_rejected(model)

    def test_constructor_and_load_reject_non_finite_probability(self):
        for section, value in (
            ("pi", math.nan),
            ("A", math.inf),
            ("B", -math.inf),
        ):
            with self.subTest(section=section):
                model = sample_model()
                if section == "pi":
                    model.pi["B"] = value
                else:
                    row = getattr(model, section)["B"]
                    row[next(iter(row))] = value

                self.assert_model_rejected(model)

    def test_constructor_and_load_reject_probability_greater_than_one(self):
        for section, value in (
            ("pi", 1.1),
            ("A", 1.1),
            ("B", 1.1),
        ):
            with self.subTest(section=section):
                model = sample_model()
                if section == "pi":
                    model.pi["B"] = value
                else:
                    row = getattr(model, section)["B"]
                    row[next(iter(row))] = value

                self.assert_model_rejected(model)

    def test_constructor_and_load_reject_non_normalized_initial_probabilities(self):
        model = sample_model()
        model.pi["B"] = 0.4

        self.assert_model_rejected(model)

    def test_constructor_and_load_reject_non_normalized_transition_row(self):
        model = sample_model()
        model.A["B"]["M"] = 0.6

        self.assert_model_rejected(model)

    def test_constructor_and_load_reject_non_normalized_non_empty_emission_row(self):
        model = sample_model()
        model.B["B"]["自"] = 0.8

        self.assert_model_rejected(model)


if __name__ == "__main__":
    unittest.main()
