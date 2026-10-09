"""Local exact-match / ANLS-style scoring, not an official benchmark adapter."""
import unicodedata


def normalize_answer(text: str) -> str:
    # Preserve punctuation and digits: silently dropping them corrupts field extraction.
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def score_answer(prediction: str, answers: list[str]) -> dict[str, float]:
    if not answers or not all(isinstance(answer, str) for answer in answers):
        raise ValueError('answers must be a nonempty list of strings')
    prediction = normalize_answer(prediction)
    exact, anls = 0.0, 0.0
    for answer in answers:
        answer = normalize_answer(answer)
        exact = max(exact, float(prediction == answer))
        distance = edit_distance(prediction, answer) / max(len(prediction), len(answer), 1)
        anls = max(anls, 1.0 - distance if distance < 0.5 else 0.0)
    return {'normalized_em': exact, 'anls_style': anls}
