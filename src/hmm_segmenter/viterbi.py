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


def _emission_score(
    state: str,
    char: str,
    emissions: dict[str, dict[str, float]],
) -> float:
    value = emissions.get(state, {}).get(char, 0.0)
    boost = 10.0 if state == "S" and value == 0.0 else 1.0
    return _log(value, boost)


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
            + _emission_score(state, chars[0], B)
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
                    + _emission_score(state, chars[index], B)
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
