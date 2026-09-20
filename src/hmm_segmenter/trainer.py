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
