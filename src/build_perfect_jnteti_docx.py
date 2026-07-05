"""Compatibility wrapper for the final JNTETI DOCX builder."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "build_jnteti_final.py"), run_name="__main__")
