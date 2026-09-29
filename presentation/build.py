import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = (ROOT / "relatorio.md").read_text(encoding="utf-8")
deck_path = ROOT / "presentation" / "pitch.html"
deck = deck_path.read_text(encoding="utf-8")


def use_case(number: int) -> str:
    section = report.split(f"### Caso de uso {number} ")[1]
    source = section.split("```mermaid\n", 1)[1].split("```", 1)[0]
    source = source.replace(
        "classDef future fill:#f1f5f9,stroke:#94a3b8,stroke-dasharray:5 4,color:#64748b",
        "classDef future fill:#0f1626,stroke:#64748b,stroke-dasharray:5 4,color:#93a3bd",
    )

    def label(match: re.Match) -> str:
        opening, body, closing = match.groups()
        return f'{opening}"`{body.replace("<br/>", chr(10))}`"{closing}'

    source = re.sub(r"(\[\(|\[|\|)([^\[\]\(\)\|\"]*<br/>[^\[\]\(\)\|\"]*)(\)\]|\]|\|)", label, source)
    return source


def notes(slide: str) -> str:
    block = report.split(f"— Slide {slide}:", 1)[1].split("\n### ", 1)[0]
    block = block.split("\n", 1)[1].strip()
    block = re.sub(r"`([^`]*)`", r"\1", block).replace("**", "")
    return html.escape(block)


SLIDES = (ROOT / "presentation" / "slides_template.html").read_text(encoding="utf-8")
for number in range(1, 5):
    SLIDES = SLIDES.replace(f"@@UC{number}@@", use_case(number))
SLIDES = re.sub(r"@@NOTES(\d+)@@", lambda m: notes(m.group(1)), SLIDES)

start = deck.index('<div class="slides">') + len('<div class="slides">')
end = deck.index("</div>\n</div>\n\n<script")
deck_path.write_text(deck[:start] + "\n" + SLIDES + "\n" + deck[end:], encoding="utf-8")
print("slides rebuilt")
