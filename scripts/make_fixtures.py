import shutil
from pathlib import Path

from docx import Document
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = ROOT / "data/fixtures"
    directory.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "eval/data/atlas_policy.txt").read_text(encoding="utf-8")
    (directory / "atlas.txt").write_text(text, encoding="utf-8")
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(str(directory / "atlas.docx"))
    canvas = Canvas(str(directory / "atlas.pdf"))
    for index, line in enumerate(text.splitlines()):
        canvas.drawString(40, 760 - index * 24, line)
    canvas.save()
    shutil.copy(ROOT / "tests/fixtures/atlas_legacy.doc", directory / "atlas_legacy.doc")
    (directory / "empty.txt").write_bytes(b"")
    (directory / "unsupported.csv").write_text("column,value\n", encoding="utf-8")
    print(f"Upload fixtures from {directory}")


if __name__ == "__main__":
    main()
