import unicodedata


def clean_sentence(text):
    result = ""
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            if depth > 0:
                depth -= 1
        elif depth == 0:
            result += char

    result = " ".join(result.split())
    result = unicodedata.normalize("NFKC", result)

    return result


def compute_windows_even(n):
    return [(i * n // 5, (i + 1) * n // 5 - 1) for i in range(5)]
