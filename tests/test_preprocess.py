import unittest

from hmm_segmenter.preprocess import clean_text, split_sentences


class PreprocessTests(unittest.TestCase):
    def test_clean_text_keeps_supported_content(self):
        raw = "<p>你好&nbsp;世界！</p>  ABC-123 #@"
        self.assertEqual(clean_text(raw), "你好 世界！ ABC-123")

    def test_clean_text_collapses_whitespace(self):
        self.assertEqual(clean_text("自然\n\n  语言\t处理"), "自然 语言 处理")

    def test_split_sentences_handles_chinese_and_english(self):
        text = "你好！今天下雨吗？It works.继续。"
        self.assertEqual(
            split_sentences(text),
            ["你好", "今天下雨吗", "It works", "继续"],
        )

    def test_split_sentences_preserves_decimal(self):
        self.assertEqual(split_sentences("价格是3.14元。结束"), ["价格是3.14元", "结束"])

    def test_split_sentences_filters_empty_parts(self):
        self.assertEqual(split_sentences("你好。。。！！再见"), ["你好", "再见"])


if __name__ == "__main__":
    unittest.main()
