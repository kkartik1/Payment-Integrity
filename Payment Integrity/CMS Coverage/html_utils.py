from bs4 import BeautifulSoup

def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text("\n").strip()