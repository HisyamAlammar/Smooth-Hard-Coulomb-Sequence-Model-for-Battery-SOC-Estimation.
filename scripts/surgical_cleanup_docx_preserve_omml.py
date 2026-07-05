from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "drafts" / "JNTETI_SOC_Hard_Coulomb_Definitif_Final.docx"
BACKUP = ROOT / "drafts" / "JNTETI_SOC_Hard_Coulomb_Definitif_Final.before_surgical_cleanup.docx"
OUTDIR = ROOT / "drafts" / "surgical_cleanup_verification"
DOC_XML = "word/document.xml"


OMATH_RE = re.compile(rb"<m:oMath[\s>].*?</m:oMath>", re.S)


FONT_RPR = (
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
    '<w:sz w:val="18"/>'
)
BASE_RPR = f"<w:rPr>{FONT_RPR}<w:i/></w:rPr>"
SUB_RPR = f'<w:rPr>{FONT_RPR}<w:vertAlign w:val="subscript"/></w:rPr>'


VARIABLES = {
    "V_proxy": ("V", "proxy"),
    "V_terminal": ("V", "terminal"),
    "R_int": ("R", "int"),
    "Q_actual": ("Q", "actual"),
}


def read_package(path: Path):
    with zipfile.ZipFile(path, "r") as zin:
        infos = zin.infolist()
        data = {info.filename: zin.read(info.filename) for info in infos}
    return infos, data


def write_package(path: Path, infos, data):
    tmp = path.with_suffix(".surgical.tmp.docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in infos:
            zout.writestr(info, data[info.filename])
    tmp.replace(path)


def omath_blocks(xml_bytes: bytes) -> list[bytes]:
    return OMATH_RE.findall(xml_bytes)


def wt(text: str) -> str:
    return f'<w:t xml:space="preserve">{text}</w:t>'


def normal_run(text: str, rpr: str) -> str:
    if not text:
        return ""
    return f"<w:r>{rpr}{wt(text)}</w:r>"


def variable_run(base: str, sub: str) -> str:
    return (
        f"<w:r>{BASE_RPR}{wt(base)}</w:r>"
        f"<w:r>{SUB_RPR}{wt(sub)}</w:r>"
    )


def split_physical_variables_in_text(text: str, rpr: str) -> str:
    if not any(token in text for token in VARIABLES):
        return f"<w:r>{rpr}{wt(text)}</w:r>"

    parts: list[str] = []
    i = 0
    token_re = re.compile("|".join(re.escape(k) for k in sorted(VARIABLES, key=len, reverse=True)))
    for m in token_re.finditer(text):
        parts.append(normal_run(text[i:m.start()], rpr))
        base, sub = VARIABLES[m.group(0)]
        parts.append(variable_run(base, sub))
        i = m.end()
    parts.append(normal_run(text[i:], rpr))
    return "".join(parts)


def split_physical_variables_in_paragraph(match: re.Match[str]) -> str:
    paragraph = match.group(0)
    if "<m:oMath" in paragraph or not any(token in paragraph for token in VARIABLES):
        return paragraph

    token_re = "|".join(re.escape(k) for k in sorted(VARIABLES, key=len, reverse=True))

    def run_repl(run_match: re.Match[str]) -> str:
        run_xml = run_match.group(0)
        if not any(token in run_xml for token in VARIABLES):
            return run_xml
        text_match = re.search(
            rf"<w:t(?P<attrs>[^>]*)>(?P<text>[^<]*(?:{token_re})[^<]*)</w:t>",
            run_xml,
            re.S,
        )
        if not text_match:
            return run_xml
        rpr_match = re.search(r"<w:rPr>.*?</w:rPr>", run_xml, re.S)
        rpr = rpr_match.group(0) if rpr_match else ""
        return split_physical_variables_in_text(text_match.group("text"), rpr)

    return re.sub(r"<w:r>(?:(?!</w:r>).)*?</w:r>", run_repl, paragraph, flags=re.S)


def replace_count(xml: str, old: str, new: str, label: str, required: int = 1) -> tuple[str, int]:
    count = xml.count(old)
    if count != required:
        raise RuntimeError(f"{label}: expected {required}, found {count}")
    return xml.replace(old, new), count


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)

    infos, data = read_package(DOCX)
    before_xml_bytes = data[DOC_XML]
    before_omath = omath_blocks(before_xml_bytes)
    (OUTDIR / "document_before.xml").write_bytes(before_xml_bytes)
    for i, block in enumerate(before_omath, 1):
        (OUTDIR / f"omath_before_{i:02d}.xml").write_bytes(block)
    if len(before_omath) != 12:
        raise RuntimeError(f"Expected 12 OMML blocks, found {len(before_omath)}")

    xml = before_xml_bytes.decode("utf-8")
    replacement_counts: dict[str, int] = {}

    replacements = [
        (
            "fix1a",
            "tetap gagal mencapai invariansi struktural. Ledger historis Sprint 44 menunjukkan baseline Soft-PINN PI-TCN masih memiliki PVR 17,02%",
            "tetap gagal mencapai invariansi struktural. Evaluasi historis menunjukkan bahwa baseline Soft-PINN PI-TCN masih memiliki PVR 17,02%",
        ),
        (
            "fix1b",
            "Ledger Soft-PINN historis berhasil ditelusuri pada logs/sprint44_results_v3.json. Baseline PI-TCN berbasis penalti lunak mencapai PVR 17,02%",
            "Analisis retrospektif terhadap log eksperimen mengonfirmasi bahwa baseline Soft-PINN (PI-TCN) tetap mencapai PVR 17,02%",
        ),
        (
            "fix1c",
            "Gbr. 4 memperlihatkan sapuan η langsung dari eta_gamma_sweep.csv.",
            "Gbr. 4 memperlihatkan sapuan η berdasarkan data eksperimen kalibrasi.",
        ),
        (
            "fix3",
            "Ketiga, isolasi temperatur pada Skenario A: train = 25°C, 10°C, validation = 0°C, test = 40°C, -10°C, -20°C.",
            "Ketiga, isolasi temperatur pada Skenario A: train = {25°C, 10°C}, validation = {0°C}, test = {40°C, -10°C, -20°C}.",
        ),
        (
            "fix4",
            "Guna menjaga keadilan pembandingan (fair baseline), model Vanilla LSTM dibangun",
            "Guna menjaga keadilan pembandingan, model Vanilla LSTM dibangun",
        ),
    ]

    for label, old, new in replacements:
        xml, count = replace_count(xml, old, new, label)
        replacement_counts[label] = count

    xml, variable_paragraph_count = re.subn(
        r"<w:p(?=\s|>).*?</w:p>",
        split_physical_variables_in_paragraph,
        xml,
        flags=re.S,
    )
    replacement_counts["physical_variable_paragraphs_scanned"] = variable_paragraph_count

    after_xml_bytes = xml.encode("utf-8")
    after_omath = omath_blocks(after_xml_bytes)
    (OUTDIR / "document_after.xml").write_bytes(after_xml_bytes)
    for i, block in enumerate(after_omath, 1):
        (OUTDIR / f"omath_after_{i:02d}.xml").write_bytes(block)

    omath_identical = before_omath == after_omath
    if len(after_omath) != len(before_omath) or not omath_identical:
        raise RuntimeError("OMML protection failed: count or byte content changed")

    data[DOC_XML] = after_xml_bytes
    write_package(DOCX, infos, data)

    report = {
        "docx": str(DOCX),
        "backup": str(BACKUP),
        "preflight_omath_count": len(before_omath),
        "post_edit_omath_count": len(after_omath),
        "omath_byte_identical": omath_identical,
        "replacement_counts": replacement_counts,
    }
    (OUTDIR / "surgical_cleanup_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
