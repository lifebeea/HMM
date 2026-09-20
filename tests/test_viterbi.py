import unittest

from hmm_segmenter.viterbi import tags_to_words, viterbi_decode, words_to_tags


def perfect_model():
    states = ("B", "M", "E", "S")
    pi = {"B": 0.25, "M": 0.25, "E": 0.25, "S": 0.25}
    A = {
        "B": {"B": 0.0, "M": 0.5, "E": 0.5, "S": 0.0},
        "M": {"B": 0.0, "M": 0.5, "E": 0.5, "S": 0.0},
        "E": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
        "S": {"B": 0.5, "M": 0.0, "E": 0.0, "S": 0.5},
    }
    B = {
        state: {"你": 1.0, "好": 1.0, "世": 1.0, "界": 1.0}
        for state in states
    }
    return pi, A, B


class ViterbiTests(unittest.TestCase):
    def test_words_to_tags(self):
        self.assertEqual(
            words_to_tags(["我", "喜欢", "自然语言"]),
            ["S", "B", "E", "B", "M", "M", "E"],
        )

    def test_tags_to_words_keeps_characters(self):
        chars = list("我喜欢自然语言")
        tags = ["S", "B", "E", "B", "M", "M", "E"]
        self.assertEqual(tags_to_words(chars, tags), ["我", "喜欢", "自然语言"])

    def test_viterbi_decodes_bmes_path(self):
        pi, A, B = perfect_model()
        self.assertEqual(viterbi_decode(list("你好"), pi, A, B), ["B", "E"])

    def test_viterbi_handles_zero_probabilities(self):
        pi = {"B": 0.0, "M": 0.0, "E": 0.0, "S": 1.0}
        A = {
            "B": {"M": 1.0, "E": 0.0},
            "M": {"M": 0.0, "E": 1.0},
            "E": {"B": 0.0, "S": 1.0},
            "S": {"B": 0.0, "S": 1.0},
        }
        B = {"B": {"你": 1.0}, "E": {"好": 1.0}, "M": {}, "S": {}}
        tags = viterbi_decode(list("你好"), pi, A, B)
        self.assertEqual(tags, ["S", "S"])
        self.assertEqual("".join(tags_to_words(list("你好"), tags)), "你好")


if __name__ == "__main__":
    unittest.main()
