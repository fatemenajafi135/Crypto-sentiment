import re
import regex
import unicodedata


def replace_urls(text: str) -> str:

    tco_pattern = r'https://t\.co/\w+'
    text = re.sub(tco_pattern, ' TWITTER URL ', text)
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, 'URL', text)

    return text


def normalize_text(text: str) -> str:
    # Normalize stylistic variants (e.g., "𝘵" → "t") using NFKC
    text = unicodedata.normalize('NFKC', text)
    # Decompose characters into base + combining marks (e.g., "í" → "i" + "◌́")
    text = unicodedata.normalize('NFKD', text)
    # Remove combining diacritical marks (e.g., accents, graves)
    text = regex.sub(r'\p{Mn}', '', text)

    return text


def normalize_whitespace(text):

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' {3,}', ' ', text)

    return text.strip()


def clean(text: str) -> str:
    text = replace_urls(text)
    text = normalize_text(text)
    text = normalize_whitespace(text)
    return text
