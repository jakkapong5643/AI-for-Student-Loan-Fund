from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

def normalize_thai_text(text: str) -> str:
    text = normalize(text)
    tokens = word_tokenize(text, engine="newmm")
    return ' '.join(tokens)
