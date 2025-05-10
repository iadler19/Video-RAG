import re

filler_words = [
    "uh",
    "um",
    "like",
    "you",
    "know",
    "so",
    "actually",
    "basically",
    "definitely",
    "sorry",
]


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)

    if filler_words:
        for word in filler_words:
            # Word boundaries (\b) ensure that we're removing whole words
            text = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)

    return text.strip()
