# HMM Chinese Segmentation and Web Upload Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete HMM Chinese word segmentation project with training, CLI interaction, Flask file processing, browser preview, and result download.

**Architecture:** A standalone `hmm_segmenter` package owns preprocessing, BMES statistics, model persistence, and Viterbi decoding. A thin CLI and Flask web layer reuse the same loaded model; the web service stores UUID-named result files and returns preview metadata to a plain HTML/CSS/JavaScript page.

**Tech Stack:** Python 3.10+, standard-library `re/json/math/uuid/unittest`, `jieba` for training labels only, Flask 3 for the local web server, Jinja templates, vanilla CSS/JavaScript.

## Global Constraints

- Python version: 3.10 or newer.
- Runtime segmentation must use the project's HMM and Viterbi implementation, never `jieba`.
- Training may use `jieba>=0.42.1` only to create BMES labels.
- Web dependency: `Flask>=3.0,<4`.
- Upload format: `.txt` only.
- Maximum upload size: `5 MiB`.
- Result retention: 1 hour.
- Preview limit: at most 20 sentences or 2000 characters.
- Web server default bind address: `127.0.0.1:5000`.
- Result files: `instance/results/<UUID>.txt`.
- Tests use `unittest`; run them with `PYTHONPATH=src:. python3 -m unittest discover -s tests -v`.
- Use ASCII in source code except Chinese fixture strings and user-facing interface copy.
- Do not add a frontend build system or runtime CDN dependency.

---

## File Map

```text
项目二/
├── data/raw/train.txt
├── docs/superpowers/
│   ├── plans/2026-09-20-hmm-segmentation-and-web.md
│   └── specs/2026-09-20-*.md
├── instance/results/.gitkeep
├── models/hmm_model.json
├── src/hmm_segmenter/
│   ├── __init__.py
│   ├── cli.py
│   ├── model.py
│   ├── preprocess.py
│   ├── trainer.py
│   └── viterbi.py
├── tests/
│   ├── test_cli.py
│   ├── test_model.py
│   ├── test_preprocess.py
│   ├── test_trainer.py
│   ├── test_viterbi.py
│   ├── test_web.py
│   └── test_web_page.py
├── web/
│   ├── __init__.py
│   ├── app.py
│   ├── service.py
│   ├── static/app.js
│   ├── static/styles.css
│   └── templates/index.html
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

### Task 1: Project Skeleton and Text Preprocessing

**Files:**
- Create: `requirements.txt`
- Create: `src/hmm_segmenter/__init__.py`
- Create: `src/hmm_segmenter/preprocess.py`
- Create: `tests/test_preprocess.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Python standard library only.
- Produces: `clean_text(text: str) -> str`, `split_sentences(text: str) -> list[str]`.

- [ ] **Step 1: Write the failing preprocessing tests**

```python
# tests/test_preprocess.py
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
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_preprocess -v
```

Expected: `ModuleNotFoundError` for `hmm_segmenter` or `preprocess.py`.

- [ ] **Step 3: Implement the package and preprocessing**

```python
# src/hmm_segmenter/preprocess.py
import html
import re


_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BOUNDARY = re.compile(r"[。！？!?；;]|(?<!\d)\.(?!\d)")
_KEEP = re.compile(
    r"[^一-龥A-Za-z0-9\s"
    r"，。！？；：、,.!?;:()（）【】《》“”‘’'\"\[\]—…·_-]"
)


def clean_text(text: str) -> str:
    """Normalize supported text while removing HTML and noisy symbols."""
    text = html.unescape(text)
    text = _HTML_TAG.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = _KEEP.sub("", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries and discard empty segments."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
```

```python
# src/hmm_segmenter/__init__.py
from .preprocess import clean_text, split_sentences

__all__ = ["clean_text", "split_sentences"]
```

```text
# requirements.txt
jieba>=0.42.1
Flask>=3.0,<4
```

Update `.gitignore` to include:

```gitignore
instance/results/*
!instance/results/.gitkeep
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_preprocess -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt src/hmm_segmenter/__init__.py src/hmm_segmenter/preprocess.py tests/test_preprocess.py
git commit -m "feat: add text preprocessing"
```

---

### Task 2: BMES Conversion and Viterbi Decoder

**Files:**
- Create: `src/hmm_segmenter/viterbi.py`
- Create: `tests/test_viterbi.py`

**Interfaces:**
- Consumes: plain dictionaries containing `pi`, `A`, and `B`.
- Produces:
  - `STATES: tuple[str, ...]`
  - `MIN_FLOAT: float`
  - `words_to_tags(words: list[str]) -> list[str]`
  - `tags_to_words(chars: list[str], tags: list[str]) -> list[str]`
  - `viterbi_decode(chars: list[str], pi: dict, A: dict, B: dict) -> list[str]`

- [ ] **Step 1: Write the failing Viterbi tests**

```python
# tests/test_viterbi.py
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
            ["S", "B", "E", "B", "M", "E"],
        )

    def test_tags_to_words_keeps_characters(self):
        chars = list("我喜欢自然语言")
        tags = ["S", "B", "E", "B", "M", "E"]
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
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_viterbi -v
```

Expected: FAIL because `hmm_segmenter.viterbi` does not exist.

- [ ] **Step 3: Implement BMES conversion and Viterbi**

```python
# src/hmm_segmenter/viterbi.py
import math


STATES = ("B", "M", "E", "S")
MIN_FLOAT = 1e-10
ALLOWED_TRANSITIONS = {
    "B": {"M", "E"},
    "M": {"M", "E"},
    "E": {"B", "S"},
    "S": {"B", "S"},
}


def words_to_tags(words: list[str]) -> list[str]:
    tags: list[str] = []
    for word in words:
        if not word:
            continue
        if len(word) == 1:
            tags.append("S")
        else:
            tags.append("B")
            tags.extend("M" for _ in range(len(word) - 2))
            tags.append("E")
    return tags


def tags_to_words(chars: list[str], tags: list[str]) -> list[str]:
    words: list[str] = []
    current = ""
    for char, tag in zip(chars, tags):
        if tag == "B":
            if current:
                words.append(current)
            current = char
        elif tag == "M":
            current += char
        elif tag == "E":
            current += char
            words.append(current)
            current = ""
        else:
            if current:
                words.append(current)
                current = ""
            words.append(char)
    if current:
        words.append(current)
    return words


def _log(value: float, boost: float = 1.0) -> float:
    return math.log(max(value, MIN_FLOAT) * boost)


def viterbi_decode(
    chars: list[str],
    pi: dict[str, float],
    A: dict[str, dict[str, float]],
    B: dict[str, dict[str, float]],
) -> list[str]:
    if not chars:
        return []

    scores: list[dict[str, float]] = [
        {
            state: _log(pi.get(state, 0.0))
            + _log(
                B.get(state, {}).get(chars[0], 0.0),
                10.0 if state == "S" else 1.0,
            )
            for state in STATES
        }
    ]
    backpointers: list[dict[str, str]] = [{} for _ in chars]

    for index in range(1, len(chars)):
        current_scores: dict[str, float] = {}
        for state in STATES:
            best_previous = ""
            best_score = -math.inf
            for previous in STATES:
                if state not in ALLOWED_TRANSITIONS[previous]:
                    continue
                score = (
                    scores[index - 1][previous]
                    + _log(A.get(previous, {}).get(state, 0.0))
                    + _log(
                        B.get(state, {}).get(chars[index], 0.0),
                        10.0 if state == "S" else 1.0,
                    )
                )
                if score > best_score:
                    best_score = score
                    best_previous = previous
            current_scores[state] = best_score
            backpointers[index][state] = best_previous
        scores.append(current_scores)

    last = max(("E", "S"), key=lambda state: scores[-1][state])
    path = [last]
    for index in range(len(chars) - 1, 0, -1):
        last = backpointers[index][last]
        path.append(last)
    path.reverse()
    return path
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_viterbi -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hmm_segmenter/viterbi.py tests/test_viterbi.py
git commit -m "feat: add BMES Viterbi decoder"
```

---

### Task 3: HMM Model Object and Persistence

**Files:**
- Create: `src/hmm_segmenter/model.py`
- Create: `tests/test_model.py`
- Modify: `src/hmm_segmenter/__init__.py`

**Interfaces:**
- Consumes: `viterbi_decode`, `tags_to_words`, `STATES`.
- Produces:
  - `ModelError(ValueError)`
  - `HMMSegmenter(pi, A, B, metadata=None)`
  - `HMMSegmenter.cut(sentence: str) -> list[str]`
  - `HMMSegmenter.save(path: str | Path) -> None`
  - `HMMSegmenter.load(path: str | Path) -> HMMSegmenter`

- [ ] **Step 1: Write the failing model tests**

```python
# tests/test_model.py
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
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_model -v
```

Expected: FAIL because `model.py` does not exist.

- [ ] **Step 3: Implement the model object**

```python
# src/hmm_segmenter/model.py
import json
import re
from pathlib import Path

from .viterbi import STATES, tags_to_words, viterbi_decode


class ModelError(ValueError):
    """Raised when an HMM model is malformed or unreadable."""


class HMMSegmenter:
    def __init__(self, pi, A, B, metadata=None):
        self.pi = pi
        self.A = A
        self.B = B
        self.metadata = metadata or {}
        self._validate()

    def _validate(self) -> None:
        if set(self.pi) != set(STATES):
            raise ModelError("initial probability states are incomplete")
        if set(self.A) != set(STATES):
            raise ModelError("transition states are incomplete")
        if set(self.B) != set(STATES):
            raise ModelError("emission states are incomplete")
        for state in STATES:
            if set(self.A[state]) != set(STATES):
                raise ModelError(f"transition row is incomplete: {state}")
            if sum(self.A[state].values()) <= 0:
                raise ModelError(f"transition row has no probability mass: {state}")
        if sum(self.pi.values()) <= 0:
            raise ModelError("initial probabilities have no probability mass")

    def cut(self, sentence: str) -> list[str]:
        if not sentence or not sentence.strip():
            return []
        words: list[str] = []
        for chunk in (part for part in re.split(r"\s+", sentence.strip()) if part):
            chars = list(chunk)
            tags = viterbi_decode(chars, self.pi, self.A, self.B)
            words.extend(tags_to_words(chars, tags))
        return words

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "pi": self.pi,
            "A": self.A,
            "B": self.B,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "HMMSegmenter":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            return cls(
                pi=payload["pi"],
                A=payload["A"],
                B=payload["B"],
                metadata=payload.get("metadata", {}),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelError(f"cannot load HMM model: {source}") from exc
```

Update `src/hmm_segmenter/__init__.py`:

```python
from .model import HMMSegmenter, ModelError
from .preprocess import clean_text, split_sentences
from .viterbi import tags_to_words, viterbi_decode, words_to_tags

__all__ = [
    "HMMSegmenter",
    "ModelError",
    "clean_text",
    "split_sentences",
    "tags_to_words",
    "viterbi_decode",
    "words_to_tags",
]
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_model -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hmm_segmenter/__init__.py src/hmm_segmenter/model.py tests/test_model.py
git commit -m "feat: add HMM model persistence"
```

---

### Task 4: Corpus Reader and HMM Training

**Files:**
- Create: `src/hmm_segmenter/trainer.py`
- Create: `tests/test_trainer.py`
- Modify: `src/hmm_segmenter/__init__.py`

**Interfaces:**
- Consumes: `clean_text`, `split_sentences`, `words_to_tags`, `HMMSegmenter`.
- Produces:
  - `read_text_file(path: str | Path) -> str`
  - `train_hmm(corpus_path, output_path=None, max_chars=2_000_000) -> HMMSegmenter`

- [ ] **Step 1: Write the failing trainer tests**

```python
# tests/test_trainer.py
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
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_trainer -v
```

Expected: FAIL because `trainer.py` does not exist.

- [ ] **Step 3: Implement training**

```python
# src/hmm_segmenter/trainer.py
from pathlib import Path

import jieba

from .model import HMMSegmenter
from .preprocess import clean_text, split_sentences
from .viterbi import STATES, words_to_tags


DEFAULT_MAX_CHARS = 2_000_000
ENCODINGS = ("utf-8", "gbk", "gb18030")


def read_text_file(path: str | Path) -> str:
    data = Path(path).read_bytes()
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"cannot decode text file with {', '.join(ENCODINGS)}")


def _normalize(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: value / total for key, value in counts.items()}


def train_hmm(
    corpus_path: str | Path,
    output_path: str | Path | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> HMMSegmenter:
    text = read_text_file(corpus_path)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    sentences = split_sentences(clean_text(text))
    if not sentences:
        raise ValueError("training corpus contains no usable sentences")

    pi_counts = {state: 0 for state in STATES}
    transition_counts = {
        state: {target: 0 for target in STATES} for state in STATES
    }
    emission_counts = {state: {} for state in STATES}

    for sentence in sentences:
        tagged_chars: list[tuple[str, str]] = []
        for chunk in (part for part in sentence.split() if part):
            words = [word for word in jieba.cut(chunk) if word.strip()]
            chars = list("".join(words))
            tags = words_to_tags(words)
            if len(chars) == len(tags):
                tagged_chars.extend(zip(chars, tags))
        if not tagged_chars:
            continue
        pi_counts[tagged_chars[0][1]] += 1
        for (_, previous), (_, current) in zip(tagged_chars, tagged_chars[1:]):
            transition_counts[previous][current] += 1
        for char, state in tagged_chars:
            emissions = emission_counts[state]
            emissions[char] = emissions.get(char, 0) + 1

    pi = _normalize(pi_counts)
    A = {state: _normalize(transition_counts[state]) for state in STATES}
    B = {
        state: _normalize(emission_counts[state])
        for state in STATES
    }
    metadata = {
        "states": list(STATES),
        "smoothing": 1e-10,
        "corpus": str(corpus_path),
        "sentence_count": len(sentences),
    }
    model = HMMSegmenter(pi, A, B, metadata)
    if output_path is not None:
        model.save(output_path)
    return model
```

Update `src/hmm_segmenter/__init__.py` to export `read_text_file` and `train_hmm`:

```python
from .model import HMMSegmenter, ModelError
from .preprocess import clean_text, split_sentences
from .trainer import read_text_file, train_hmm
from .viterbi import tags_to_words, viterbi_decode, words_to_tags

__all__ = [
    "HMMSegmenter",
    "ModelError",
    "clean_text",
    "read_text_file",
    "split_sentences",
    "tags_to_words",
    "train_hmm",
    "viterbi_decode",
    "words_to_tags",
]
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_trainer -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hmm_segmenter/__init__.py src/hmm_segmenter/trainer.py tests/test_trainer.py
git commit -m "feat: train HMM probabilities"
```

---

### Task 5: CLI, Main Entry Point, and Local Training Data

**Files:**
- Create: `src/hmm_segmenter/cli.py`
- Create: `main.py`
- Create: `tests/test_cli.py`
- Create: `data/raw/train.txt`
- Create: `models/` directory through generated model

**Interfaces:**
- Consumes: `HMMSegmenter`, `ModelError`, `train_hmm`, `read_text_file`, `clean_text`, `split_sentences`.
- Produces:
  - `default_model_path(root: Path) -> Path`
  - `load_or_train(model_path: Path, corpus_path: Path) -> HMMSegmenter`
  - `segment_text(model, text) -> list[str]`
  - `segment_file(model, input_path, output_path=None) -> Path`
  - `interactive_menu(model, corpus_path, model_path) -> None`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing CLI tests**

```python
# tests/test_cli.py
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_cli -v
```

Expected: FAIL because `cli.py` does not exist.

- [ ] **Step 3: Implement the CLI and entry point**

```python
# src/hmm_segmenter/cli.py
import argparse
import sys
from pathlib import Path

from .model import HMMSegmenter, ModelError
from .preprocess import clean_text, split_sentences
from .trainer import read_text_file, train_hmm


def default_model_path(root: Path) -> Path:
    return root / "models" / "hmm_model.json"


def load_or_train(model_path: Path, corpus_path: Path) -> HMMSegmenter:
    if model_path.exists():
        return HMMSegmenter.load(model_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"training corpus not found: {corpus_path}")
    print(f"Training HMM model from {corpus_path}...")
    return train_hmm(corpus_path, model_path)


def segment_text(model: HMMSegmenter, text: str) -> list[str]:
    sentences = split_sentences(clean_text(text))
    return [" ".join(model.cut(sentence)) for sentence in sentences if sentence]


def segment_file(
    model: HMMSegmenter,
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    source = Path(input_path)
    destination = (
        Path(output_path)
        if output_path is not None
        else source.with_name(f"{source.stem}_seg.txt")
    )
    sentences = split_sentences(clean_text(read_text_file(source)))
    lines = [" ".join(model.cut(sentence)) for sentence in sentences if sentence]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def interactive_menu(model, corpus_path: Path, model_path: Path) -> None:
    while True:
        print("\n1 - 语句分词\n2 - 篇章分词\n3 - 重新训练\n0 - 退出")
        choice = input(">>> ").strip()
        if choice == "0":
            return
        if choice == "1":
            while True:
                sentence = input("句子>>> ").strip()
                if not sentence:
                    break
                for line in segment_text(model, sentence):
                    print(line)
        elif choice == "2":
            raw_path = input("文件路径：").strip()
            if not raw_path:
                continue
            try:
                print(f"结果文件：{segment_file(model, raw_path)}")
            except (OSError, UnicodeError) as exc:
                print(f"处理失败：{exc}")
        elif choice == "3":
            model = train_hmm(corpus_path, model_path)
        else:
            print("无效选择")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HMM Chinese word segmenter")
    parser.add_argument("--model", default="models/hmm_model.json")
    parser.add_argument("--corpus", default="data/raw/train.txt")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train")
    text_parser = subparsers.add_parser("segment-text")
    text_parser.add_argument("text")
    file_parser = subparsers.add_parser("segment-file")
    file_parser.add_argument("input")
    file_parser.add_argument("-o", "--output")
    web_parser = subparsers.add_parser("web")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=5000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model).resolve()
    corpus_path = Path(args.corpus).resolve()
    try:
        if args.command == "train":
            train_hmm(corpus_path, model_path)
            print(f"模型已保存：{model_path}")
            return 0
        model = load_or_train(model_path, corpus_path)
        if args.command is None:
            interactive_menu(model, corpus_path, model_path)
        elif args.command == "segment-text":
            for line in segment_text(model, args.text):
                print(line)
        elif args.command == "segment-file":
            print(f"结果文件：{segment_file(model, args.input, args.output)}")
        elif args.command == "web":
            from web.app import create_app

            app = create_app(segmenter=model)
            app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
        return 0
    except (FileNotFoundError, ModelError, OSError, UnicodeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# main.py
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hmm_segmenter.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Copy the approved local corpus and train the real model**

Run:

```bash
mkdir -p data/raw models
```

Run:

```bash
cp ../NLP学习/金庸-倚天屠龙记.txt data/raw/train.txt
```

Run:

```bash
python3 main.py train
```

Expected: `models/hmm_model.json` is created and the command exits with status 0.

- [ ] **Step 5: Run CLI tests and a smoke command**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_cli -v
```

Expected: 2 tests pass.

Run:

```bash
python3 main.py segment-text "自然语言处理很有趣。"
```

Expected: one line containing the segmented words, with every source character retained.

- [ ] **Step 6: Commit**

```bash
git add src/hmm_segmenter/cli.py main.py tests/test_cli.py data/raw/train.txt models/hmm_model.json
git commit -m "feat: add CLI and trained HMM model"
```

---

### Task 6: Web File Processing Service

**Files:**
- Create: `web/__init__.py`
- Create: `web/service.py`
- Create: `tests/test_web.py`
- Create: `instance/results/.gitkeep`

**Interfaces:**
- Consumes: `HMMSegmenter`, `clean_text`, `split_sentences`, `read_text_file`.
- Produces:
  - `ServiceError(message, code, status_code=400)`
  - `SegmentationService(segmenter, result_dir, max_upload_bytes=5 * 1024 * 1024, ttl_seconds=3600)`
  - `SegmentationService.cleanup_expired(now=None) -> int`
  - `SegmentationService.segment_upload(filename: str, data: bytes) -> dict`
  - `SegmentationService.resolve_result(job_id: str) -> tuple[Path, str]`

- [ ] **Step 1: Write the failing service tests**

```python
# tests/test_web.py
import tempfile
import time
import unittest
from pathlib import Path

from hmm_segmenter.model import HMMSegmenter
from web.service import SegmentationService, ServiceError


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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_web -v
```

Expected: FAIL because `web.service` does not exist.

- [ ] **Step 3: Implement the service**

```python
# web/service.py
import json
import time
import uuid
from pathlib import Path

from hmm_segmenter.preprocess import clean_text, split_sentences


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
RESULT_TTL_SECONDS = 3600
MAX_PREVIEW_SENTENCES = 20
MAX_PREVIEW_CHARS = 2000


class ServiceError(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class SegmentationService:
    def __init__(
        self,
        segmenter,
        result_dir: str | Path,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
        ttl_seconds: int = RESULT_TTL_SECONDS,
    ):
        self.segmenter = segmenter
        self.result_dir = Path(result_dir)
        self.max_upload_bytes = max_upload_bytes
        self.ttl_seconds = ttl_seconds
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_expired()

    def cleanup_expired(self, now: float | None = None) -> int:
        threshold = (time.time() if now is None else now) - self.ttl_seconds
        removed = 0
        for path in self.result_dir.glob("*.txt"):
            if path.stat().st_mtime >= threshold:
                continue
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            removed += 1
        return removed

    def segment_upload(self, filename: str, data: bytes) -> dict:
        if not filename:
            raise ServiceError("请选择文本文件", "EMPTY_FILE")
        if Path(filename).suffix.lower() != ".txt":
            raise ServiceError("仅支持 .txt 文本文件", "INVALID_FILE_TYPE")
        if not data:
            raise ServiceError("文件内容为空", "EMPTY_FILE")
        if len(data) > self.max_upload_bytes:
            raise ServiceError("文件超过 5 MiB 限制", "FILE_TOO_LARGE", 413)
        try:
            text = read_text_file_bytes(data)
        except UnicodeError as exc:
            raise ServiceError(
                "无法识别文件编码，请使用 UTF-8、GBK 或 GB18030",
                "DECODE_ERROR",
            ) from exc

        sentences = split_sentences(clean_text(text))
        if not sentences:
            raise ServiceError("清洗后没有可分词文本", "EMPTY_TEXT")

        lines: list[str] = []
        word_count = 0
        for sentence in sentences:
            words = self.segmenter.cut(sentence)
            word_count += len(words)
            lines.append(" ".join(words))

        job_id = str(uuid.uuid4())
        safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        safe_stem = Path(safe_filename).stem or "result"
        output_name = f"{safe_stem}_seg.txt"
        result_path = self.result_dir / f"{job_id}.txt"
        metadata_path = self.result_dir / f"{job_id}.json"
        result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        metadata_path.write_text(
            json.dumps({"output_name": output_name, "created_at": time.time()}),
            encoding="utf-8",
        )
        return {
            "job_id": job_id,
            "input_name": filename,
            "output_name": output_name,
            "sentence_count": len(lines),
            "word_count": word_count,
            "preview": self._preview(lines),
            "download_url": f"/api/results/{job_id}/download",
        }

    @staticmethod
    def _preview(lines: list[str]) -> list[str]:
        preview: list[str] = []
        char_count = 0
        for line in lines[:MAX_PREVIEW_SENTENCES]:
            added = len(line) + (1 if preview else 0)
            if char_count + added > MAX_PREVIEW_CHARS:
                break
            preview.append(line)
            char_count += added
        return preview

    def resolve_result(self, job_id: str) -> tuple[Path, str]:
        try:
            canonical = str(uuid.UUID(job_id))
        except (ValueError, AttributeError) as exc:
            raise ServiceError("结果不存在", "RESULT_NOT_FOUND", 404) from exc
        result_path = self.result_dir / f"{canonical}.txt"
        metadata_path = self.result_dir / f"{canonical}.json"
        if not result_path.is_file() or not metadata_path.is_file():
            raise ServiceError("结果已过期或不存在", "RESULT_EXPIRED", 404)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            output_name = metadata["output_name"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ServiceError("结果元数据不可用", "RESULT_EXPIRED", 404) from exc
        return result_path, output_name


def read_text_file_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError("unsupported encoding")
```

Create `web/__init__.py`:

```python
"""Flask web layer for HMM segmentation."""
```

Create `instance/results/.gitkeep` as an empty file.

- [ ] **Step 4: Run service tests and verify they pass**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_web -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/__init__.py web/service.py tests/test_web.py instance/results/.gitkeep
git commit -m "feat: add web segmentation service"
```

---

### Task 7: Flask Routes and API Error Handling

**Files:**
- Create: `web/app.py`
- Modify: `tests/test_web.py`

**Interfaces:**
- Consumes: `SegmentationService`, `ServiceError`, an injected `HMMSegmenter`.
- Produces:
  - `create_app(segmenter, model_path=None, result_dir=None) -> Flask`
  - `GET /`
  - `POST /api/segment`
  - `GET /api/results/<job_id>/download`

- [ ] **Step 1: Add failing route tests**

Append:

```python
from io import BytesIO

from web.app import create_app


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
        download = self.client.get(payload["download_url"])
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


if __name__ == "__main__":
    unittest.main()
```

Ensure `unittest.main()` appears only once at the end of the file.

- [ ] **Step 2: Run route tests and verify they fail**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_web.WebAppTests -v
```

Expected: FAIL because `web.app` does not exist.

- [ ] **Step 3: Implement the Flask application**

```python
# web/app.py
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from web.service import MAX_UPLOAD_BYTES, SegmentationService, ServiceError


def create_app(segmenter=None, model_path=None, result_dir=None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["MODEL_PATH"] = str(model_path) if model_path else None
    root = Path(__file__).resolve().parents[1]
    service = SegmentationService(
        segmenter,
        result_dir or root / "instance" / "results",
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/segment")
    def segment():
        if service.segmenter is None:
            raise ServiceError("模型不可用", "MODEL_UNAVAILABLE", 503)
        upload = request.files.get("file")
        if upload is None:
            raise ServiceError("请选择文本文件", "EMPTY_FILE")
        return jsonify(service.segment_upload(upload.filename or "", upload.read()))

    @app.get("/api/results/<job_id>/download")
    def download(job_id: str):
        path, output_name = service.resolve_result(job_id)
        return send_file(
            path,
            as_attachment=True,
            download_name=output_name,
            mimetype="text/plain; charset=utf-8",
        )

    @app.errorhandler(ServiceError)
    def handle_service_error(error: ServiceError):
        return (
            jsonify({"error": {"code": error.code, "message": error.message}}),
            error.status_code,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_error):
        return (
            jsonify(
                {
                    "error": {
                        "code": "FILE_TOO_LARGE",
                        "message": "文件超过 5 MiB 限制",
                    }
                }
            ),
            413,
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "error": {
                            "code": "RESULT_NOT_FOUND",
                            "message": error.description,
                        }
                    }
                ),
                error.code,
            )
        return error

    @app.errorhandler(Exception)
    def handle_internal_error(_error):
        return (
            jsonify(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "服务器处理失败",
                    }
                }
            ),
            500,
        )

    return app
```

- [ ] **Step 4: Run route tests and verify they pass**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_web.WebAppTests -v
```

Expected: 4 API tests pass. The index page is tested after its template is added in Task 8.

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/test_web.py
git commit -m "feat: add Flask segmentation API"
```

---

### Task 8: Accessible Workbench Template and Responsive CSS

**Files:**
- Create: `web/templates/index.html`
- Create: `web/static/styles.css`
- Create: `tests/test_web_page.py`

**Interfaces:**
- Consumes: `GET /`.
- Produces: DOM IDs `upload-form`, `file-input`, `drop-zone`, `analyze-button`, `status-region`, `file-name`, `sentence-count`, `word-count`, `preview`, `download-button`.

- [ ] **Step 1: Write the failing page-structure test**

```python
# tests/test_web_page.py
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
```

- [ ] **Step 2: Run the page test and verify it fails**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_web_page -v
```

Expected: FAIL because `web/templates/index.html` does not exist.

- [ ] **Step 3: Implement the template and CSS**

Use the confirmed result-first layout. The template must contain the IDs listed above, inline Lucide-compatible SVG symbols, and no CDN references. Its main structure is:

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HMM 中文分词</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
</head>
<body>
  <svg hidden aria-hidden="true">
    <symbol id="icon-upload" viewBox="0 0 24 24">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>
    </symbol>
    <symbol id="icon-download" viewBox="0 0 24 24">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>
    </symbol>
  </svg>

  <header class="app-header">
    <div>
      <p class="eyebrow">HMM / BMES / VITERBI</p>
      <h1>中文分词分析</h1>
    </div>
    <span class="model-state">模型已就绪</span>
  </header>

  <main class="workspace">
    <form id="upload-form" class="task-bar">
      <label id="drop-zone" class="drop-zone" for="file-input">
        <svg class="icon"><use href="#icon-upload"></use></svg>
        <span id="selected-file">选择或拖入 .txt 文件</span>
      </label>
      <input id="file-input" name="file" type="file" accept=".txt,text/plain" hidden>
      <button id="analyze-button" class="primary-button" type="submit" disabled>
        开始分析
      </button>
    </form>

    <div id="status-region" class="status" role="status" aria-live="polite"></div>
    <div id="progress" class="progress" hidden><span></span></div>

    <section class="stats" aria-label="分析统计">
      <article><span>文件</span><strong id="file-name">--</strong></article>
      <article><span>句子</span><strong id="sentence-count">--</strong></article>
      <article><span>词数</span><strong id="word-count">--</strong></article>
    </section>

    <section class="result-panel">
      <div class="result-heading">
        <div>
          <p class="eyebrow">RESULT</p>
          <h2>分词结果预览</h2>
        </div>
        <a id="download-button" class="primary-button disabled" aria-disabled="true">
          <svg class="icon"><use href="#icon-download"></use></svg>
          下载结果
        </a>
      </div>
      <pre id="preview" class="preview">上传文件后，结果显示在这里。</pre>
    </section>
  </main>

  <script src="{{ url_for('static', filename='app.js') }}"></script>
</body>
</html>
```

Implement `styles.css` with these exact rules:

```css
:root {
  color-scheme: light;
  --background: #f4f6f8;
  --surface: #ffffff;
  --text: #18212f;
  --muted: #667085;
  --line: #d8dee8;
  --accent: #087f8c;
  --accent-strong: #066874;
  --danger: #c4322b;
  --success: #247a4b;
  --radius: 8px;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  min-width: 320px;
  background: var(--background);
  color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.app-header,
.workspace {
  width: min(1120px, calc(100% - 32px));
  margin-inline: auto;
}
.app-header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  padding: 28px 0 18px;
}
h1, h2, p { margin: 0; }
h1 { font-size: 28px; line-height: 1.2; }
h2 { font-size: 18px; }
.eyebrow {
  margin-bottom: 5px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
}
.model-state {
  color: var(--success);
  font-size: 13px;
}
.workspace { padding-bottom: 40px; }
.task-bar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  padding: 12px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
}
.drop-zone {
  display: flex;
  align-items: center;
  gap: 9px;
  min-width: 0;
  padding: 11px 13px;
  border: 1px dashed #98a2b3;
  border-radius: 6px;
  color: var(--muted);
  cursor: pointer;
}
.drop-zone.dragging { border-color: var(--accent); color: var(--accent); }
#selected-file {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.icon { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2; }
.primary-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  min-height: 44px;
  padding: 0 18px;
  border: 0;
  border-radius: 6px;
  background: var(--accent);
  color: white;
  font: inherit;
  font-weight: 650;
  text-decoration: none;
  cursor: pointer;
}
.primary-button:hover { background: var(--accent-strong); }
.primary-button:disabled,
.primary-button.disabled { opacity: .45; pointer-events: none; }
.status { min-height: 24px; padding: 7px 2px; color: var(--muted); }
.status.error { color: var(--danger); }
.status.success { color: var(--success); }
.progress {
  height: 4px;
  overflow: hidden;
  border-radius: 4px;
  background: #dfe5ec;
}
.progress span {
  display: block;
  width: 34%;
  height: 100%;
  background: var(--accent);
  animation: loading 1.1s ease-in-out infinite;
}
.stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 14px 0;
}
.stats article,
.result-panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
}
.stats article { padding: 14px; min-width: 0; }
.stats span { display: block; color: var(--muted); font-size: 12px; }
.stats strong {
  display: block;
  margin-top: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.result-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px;
  border-bottom: 1px solid var(--line);
}
.preview {
  min-height: 280px;
  margin: 0;
  padding: 18px;
  overflow: auto;
  color: #344054;
  font: 14px/1.9 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
@keyframes loading {
  from { transform: translateX(-110%); }
  to { transform: translateX(300%); }
}
@media (max-width: 640px) {
  .app-header { align-items: start; flex-direction: column; gap: 10px; }
  .task-bar { grid-template-columns: 1fr; }
  .stats { grid-template-columns: 1fr; }
  .result-heading { align-items: stretch; flex-direction: column; }
  .primary-button { width: 100%; }
}
@media (prefers-reduced-motion: reduce) {
  .progress span { animation-duration: 2.5s; }
}
```

- [ ] **Step 4: Run page and route tests**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_web_page tests.test_web.WebAppTests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/templates/index.html web/static/styles.css tests/test_web_page.py
git commit -m "feat: add responsive upload workbench"
```

---

### Task 9: Browser Upload and Download Behavior

**Files:**
- Create: `web/static/app.js`
- Modify: `tests/test_web_page.py`

**Interfaces:**
- Consumes: DOM IDs from Task 8 and `POST /api/segment`.
- Produces: browser states `idle`, `ready`, `processing`, `complete`, `error`, plus drag/drop and safe text preview.

- [ ] **Step 1: Add a failing JavaScript integration test**

Add this method inside `WebPageTests`, before the final `if __name__ == "__main__":` block:

```python
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
            response = app.test_client().get("/static/app.js")
        script = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("formData.append", script)
        self.assertNotIn("preview.innerHTML =", script)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_web_page.WebPageTests.test_page_loads_javascript_without_inline_result_injection -v
```

Expected: FAIL because `app.js` does not exist.

- [ ] **Step 3: Implement browser behavior**

```javascript
// web/static/app.js
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const fileName = document.getElementById("selected-file");
const analyzeButton = document.getElementById("analyze-button");
const statusRegion = document.getElementById("status-region");
const progress = document.getElementById("progress");
const statFileName = document.getElementById("file-name");
const sentenceCount = document.getElementById("sentence-count");
const wordCount = document.getElementById("word-count");
const preview = document.getElementById("preview");
const downloadButton = document.getElementById("download-button");

let selectedFile = null;
let downloadUrl = null;

function setStatus(message = "", type = "") {
  statusRegion.textContent = message;
  statusRegion.className = `status ${type}`.trim();
}

function resetResult() {
  statFileName.textContent = "--";
  sentenceCount.textContent = "--";
  wordCount.textContent = "--";
  preview.textContent = "上传文件后，结果显示在这里。";
  downloadButton.classList.add("disabled");
  downloadButton.setAttribute("aria-disabled", "true");
  downloadButton.removeAttribute("href");
  downloadButton.removeAttribute("download");
  downloadUrl = null;
}

function chooseFile(file) {
  resetResult();
  setStatus();
  if (!file) {
    selectedFile = null;
    fileName.textContent = "选择或拖入 .txt 文件";
    analyzeButton.disabled = true;
    return;
  }
  if (!file.name.toLowerCase().endsWith(".txt")) {
    selectedFile = null;
    fileName.textContent = "选择或拖入 .txt 文件";
    analyzeButton.disabled = true;
    setStatus("仅支持 .txt 文本文件", "error");
    return;
  }
  selectedFile = file;
  fileName.textContent = `${file.name} (${formatBytes(file.size)})`;
  analyzeButton.disabled = false;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", (event) => {
  chooseFile(event.dataTransfer.files[0]);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  fileInput.disabled = true;
  progress.hidden = false;
  setStatus("正在分析文件...");

  const formData = new FormData();
  formData.append("file", selectedFile, selectedFile.name);
  try {
    const response = await fetch("/api/segment", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error?.message || "处理失败");
    }
    statFileName.textContent = payload.input_name;
    sentenceCount.textContent = payload.sentence_count;
    wordCount.textContent = payload.word_count;
    preview.textContent = payload.preview.join("\n");
    downloadUrl = payload.download_url;
    downloadButton.href = downloadUrl;
    downloadButton.download = payload.output_name;
    downloadButton.classList.remove("disabled");
    downloadButton.setAttribute("aria-disabled", "false");
    setStatus("结果文件已生成", "success");
  } catch (error) {
    setStatus(error.message || "处理失败", "error");
  } finally {
    progress.hidden = true;
    fileInput.disabled = false;
    analyzeButton.disabled = !selectedFile;
  }
});

resetResult();
```

- [ ] **Step 4: Validate JavaScript and run all web tests**

Run:

```bash
node --check web/static/app.js
```

Expected: no output and exit status 0.

Run:

```bash
PYTHONPATH=src:. python3 -m unittest tests.test_web tests.test_web_page -v
```

Expected: all web tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/static/app.js tests/test_web_page.py
git commit -m "feat: connect upload and download workflow"
```

---

### Task 10: Documentation and Full End-to-End Verification

**Files:**
- Create: `README.md`
- Modify: `.gitignore`
- Modify: source files only if verification exposes a defect.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: reproducible setup, training, CLI, web, and test instructions.

- [ ] **Step 1: Install dependencies**

Run:

```bash
python3 -m pip install -r requirements.txt
```

Expected: `jieba` and Flask are available. If sandbox/network restrictions block installation, request approval and rerun the same command outside the sandbox.

- [ ] **Step 2: Run the complete automated test suite**

Run:

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests -v
```

Expected: all preprocessing, Viterbi, model, trainer, CLI, service, route, and page tests pass.

- [ ] **Step 3: Add README**

```markdown
# HMM 中文分词与 Web 工作台

## 功能

- 使用 BMES 状态和 Viterbi 算法实现中文分词。
- 支持语句分词、篇章分词和结果文件输出。
- 提供本地上传页面，可在线预览并下载 `_seg.txt`。

## 安装

```bash
python3 -m pip install -r requirements.txt
```

## 训练

默认语料位于 `data/raw/train.txt`：

```bash
python3 main.py train
```

模型输出到 `models/hmm_model.json`。

## 使用

启动交互模式：

```bash
python3 main.py
```

启动 Web 页面：

```bash
python3 main.py web
```

浏览器访问 `http://127.0.0.1:5000`，上传 `.txt` 文件后查看预览并下载结果。

## 测试

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests -v
```
```

- [ ] **Step 4: Start the web server and verify the real model**

Run:

```bash
python3 main.py web
```

Expected: Flask reports that it is running on `http://127.0.0.1:5000`.

In a second terminal, verify the health of the page:

```bash
curl -I http://127.0.0.1:5000/
```

Expected: HTTP `200`.

Verify a real upload:

```bash
curl -sS -F "file=@data/raw/train.txt;type=text/plain" http://127.0.0.1:5000/api/segment
```

Expected: JSON containing `job_id`, `output_name`, `sentence_count`, `word_count`, `preview`, and `download_url`.

Use the returned `download_url`:

```bash
curl -sS -o /tmp/train_seg.txt http://127.0.0.1:5000/api/results/JOB_ID/download
```

Expected: `/tmp/train_seg.txt` is a UTF-8 file with one segmented sentence per line.

- [ ] **Step 5: Verify desktop and mobile layouts**

Open `http://127.0.0.1:5000` at widths `1440px` and `390px`. Confirm:

- Upload control remains visible.
- Filename and button text do not overlap.
- Statistics stack correctly on mobile.
- Processing, success, and error messages are visible.
- Download produces `_seg.txt`.

Stop the Flask process after verification.

- [ ] **Step 6: Commit**

```bash
git add README.md .gitignore
git commit -m "docs: add setup and verification guide"
```
