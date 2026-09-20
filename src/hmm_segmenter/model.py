import json
import math
import re
from collections.abc import Mapping
from numbers import Real
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

    @staticmethod
    def _validate_probability_values(values, context) -> None:
        for value in values.values():
            if not isinstance(value, Real):
                raise ModelError(f"{context} contains a non-numeric probability")
            if not math.isfinite(value):
                raise ModelError(f"{context} contains a non-finite probability")
            if value < 0:
                raise ModelError(f"{context} contains a negative probability")
            if value > 1:
                raise ModelError(f"{context} contains a probability greater than one")

    def _validate(self) -> None:
        if not isinstance(self.pi, Mapping):
            raise ModelError("initial probabilities are malformed")
        if not isinstance(self.A, Mapping):
            raise ModelError("transition probabilities are malformed")
        if not isinstance(self.B, Mapping):
            raise ModelError("emission probabilities are malformed")
        if set(self.pi) != set(STATES):
            raise ModelError("initial probability states are incomplete")
        if set(self.A) != set(STATES):
            raise ModelError("transition states are incomplete")
        if set(self.B) != set(STATES):
            raise ModelError("emission states are incomplete")
        self._validate_probability_values(self.pi, "initial probabilities")
        for state in STATES:
            if not isinstance(self.A[state], Mapping):
                raise ModelError(f"transition row is malformed: {state}")
            if set(self.A[state]) != set(STATES):
                raise ModelError(f"transition row is incomplete: {state}")
            self._validate_probability_values(
                self.A[state],
                f"transition probabilities for state {state}",
            )
            if not math.isclose(
                sum(self.A[state].values()),
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ModelError(f"transition row does not sum to one: {state}")
        for state in STATES:
            if not isinstance(self.B[state], Mapping):
                raise ModelError(f"emission row is malformed: {state}")
            self._validate_probability_values(
                self.B[state],
                f"emission probabilities for state {state}",
            )
            if self.B[state] and not math.isclose(
                sum(self.B[state].values()),
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ModelError(f"emission row does not sum to one: {state}")
        if not math.isclose(
            sum(self.pi.values()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ModelError("initial probabilities do not sum to one")

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
