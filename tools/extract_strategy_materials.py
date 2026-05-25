from pathlib import Path
from zipfile import ZipFile
import re
import sys
import xml.etree.ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def natural_key(path: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path)]


def text_from_pptx(path: Path) -> str:
    chunks = [f"# {path.name}"]
    with ZipFile(path) as zf:
        slide_names = sorted(
            [name for name in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", name)],
            key=natural_key,
        )
        for idx, name in enumerate(slide_names, start=1):
            root = ET.fromstring(zf.read(name))
            texts = []
            for node in root.findall(".//a:t", NS):
                if node.text:
                    texts.append(node.text.strip())
            text = " ".join(t for t in texts if t)
            if text:
                chunks.append(f"\n## Slide {idx}\n{text}")
    return "\n".join(chunks)


def text_from_docx(path: Path) -> str:
    chunks = [f"# {path.name}"]
    with ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
        for para in root.findall(".//w:p", NS):
            texts = [node.text for node in para.findall(".//w:t", NS) if node.text]
            text = "".join(texts).strip()
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: extract_strategy_materials.py OUT_FILE FILE...", file=sys.stderr)
        return 2
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for raw in sys.argv[2:]:
        path = Path(raw)
        suffix = path.suffix.lower()
        if suffix == ".pptx":
            sections.append(text_from_pptx(path))
        elif suffix == ".docx":
            sections.append(text_from_docx(path))
        else:
            sections.append(f"# {path.name}\n[unsupported by zip extractor: {suffix}]")
    out.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
