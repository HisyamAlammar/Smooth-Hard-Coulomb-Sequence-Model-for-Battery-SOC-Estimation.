from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "drafts" / "JNTETI_SOC_Hard_Coulomb_Definitif_Final.docx"
WORK = ROOT / "drafts" / "equation_forensics_after"
BACKUP = ROOT / "drafts" / "JNTETI_SOC_Hard_Coulomb_Definitif_Final.before_equation_omml.docx"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "xml": "http://www.w3.org/XML/1998/namespace",
}


def qn(prefix_name: str) -> str:
    prefix, name = prefix_name.split(":")
    return f"{{{NS[prefix]}}}{name}"


def w_el(name: str, **attrs):
    el = etree.Element(qn(f"w:{name}"))
    for key, val in attrs.items():
        el.set(qn(f"w:{key}"), str(val))
    return el


def m_el(name: str, **attrs):
    el = etree.Element(qn(f"m:{name}"))
    for key, val in attrs.items():
        el.set(qn(f"m:{key}"), str(val))
    return el


def paragraph_text(p) -> str:
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def run(text: str, *, normal: bool = False):
    r = m_el("r")
    if normal:
        rpr = m_el("rPr")
        rpr.append(m_el("nor"))
        r.append(rpr)
    t = m_el("t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def wrap(tag: str, children):
    el = m_el(tag)
    for child in children:
        el.append(child)
    return el


def e(children):
    return wrap("e", children)


def sub(base, subscript):
    el = m_el("sSub")
    el.append(e(base))
    el.append(wrap("sub", subscript))
    return el


def sup(base, superscript):
    el = m_el("sSup")
    el.append(e(base))
    el.append(wrap("sup", superscript))
    return el


def subsup(base, subscript, superscript):
    el = m_el("sSubSup")
    el.append(e(base))
    el.append(wrap("sub", subscript))
    el.append(wrap("sup", superscript))
    return el


def frac(num, den):
    el = m_el("f")
    fpr = m_el("fPr")
    ftype = m_el("type")
    ftype.set(qn("m:val"), "bar")
    fpr.append(ftype)
    el.append(fpr)
    el.append(wrap("num", num))
    el.append(wrap("den", den))
    return el


def rad(children):
    el = m_el("rad")
    rpr = m_el("radPr")
    hide = m_el("degHide")
    hide.set(qn("m:val"), "1")
    rpr.append(hide)
    el.append(rpr)
    el.append(m_el("deg"))
    el.append(e(children))
    return el


def acc_hat(children):
    el = m_el("acc")
    apr = m_el("accPr")
    chr_el = m_el("chr")
    chr_el.set(qn("m:val"), "\u0302")
    apr.append(chr_el)
    el.append(apr)
    el.append(e(children))
    return el


def nary_sum(subscript, superscript, body):
    el = m_el("nary")
    npr = m_el("naryPr")
    chr_el = m_el("chr")
    chr_el.set(qn("m:val"), "∑")
    loc = m_el("limLoc")
    loc.set(qn("m:val"), "undOvr")
    npr.append(chr_el)
    npr.append(loc)
    el.append(npr)
    el.append(wrap("sub", subscript))
    el.append(wrap("sup", superscript))
    el.append(e(body))
    return el


def cases(rows):
    d = m_el("d")
    dpr = m_el("dPr")
    beg = m_el("begChr")
    beg.set(qn("m:val"), "{")
    end = m_el("endChr")
    end.set(qn("m:val"), "")
    grow = m_el("grow")
    grow.set(qn("m:val"), "1")
    dpr.extend([beg, end, grow])
    d.append(dpr)
    arr = m_el("eqArr")
    for row in rows:
        arr.append(e(row))
    d.append(e([arr]))
    return d


def comma():
    return run(", ")


def space():
    return run(" ")


def text(s: str):
    return run(s, normal=True)


def var(s: str):
    return run(s)


def sig_arg_delta_t():
    return [
        run("σ"),
        run("("),
        subsup([run("ℓ")], [run("t")], [run("δ")]),
        run(")"),
    ]


def y_hat_sub(sub_txt: str):
    return sub([acc_hat([run("y")])], [run(sub_txt)])


def soc_hat_sub(sub_txt: str):
    return sub([acc_hat([text("SOC")])], [run(sub_txt)])


def build_equation(num: int):
    R_B_100_5 = sup([run("R")], [run("B×100×5")])
    R_B_100_64 = sup([run("R")], [run("B×100×64")])
    R_B_100_1 = sup([run("R")], [run("B×100×1")])
    R_B_1 = sup([run("R")], [run("B×1")])

    if num == 1:
        return [
            run("("), run("𝐡"), comma(), run("𝐜"), run(") = "), text("LSTM"), run("("), run("x"), run("), "),
            run("x"), run(" ∈ "), R_B_100_5, run(" → "), run("𝐡"), run(" ∈ "), R_B_100_64,
        ]
    if num == 2:
        return [
            sup([run("ℓ")], [run("δ")]), run(" = "), sub([run("f")], [run("δ")]),
            run("("), run("𝐡"), run(") ∈ "), R_B_100_1, comma(),
            sup([run("ℓ")], [run("a")]), run(" = "), sub([run("f")], [text("anchor")]),
            run("("), run("𝐡"), run("[:, -1, :]"), run(") ∈ "), R_B_1,
        ]
    if num == 3:
        return [
            sub([text("limit")], [run("t")]), run(" = |"), sub([run("I")], [run("t")]),
            run("| · η · γ,    γ = "),
            frac([run("Δt")], [sub([run("Q")], [text("nom")]), run(" · 3600")]),
            run(" = 9,259 × "), sup([run("10")], [run("-5")]), run(" "), text("SOC/A/s"),
        ]
    if num == 4:
        rows = [
            [run("-"), sub([text("limit")], [run("t")]), run(" · "), *sig_arg_delta_t(), text(", if "), sub([run("I")], [run("t")]), run(" < -τ")],
            [run("+"), sub([text("limit")], [run("t")]), run(" · "), *sig_arg_delta_t(), text(", if "), sub([run("I")], [run("t")]), run(" > τ")],
            [run("0"), text(", if "), run("|"), sub([run("I")], [run("t")]), run("| ≤ τ")],
        ]
        return [sub([run("δ")], [run("t")]), run(" = "), cases(rows)]
    if num == 5:
        return [
            sub([run("C")], [run("t")]), run(" = "),
            nary_sum([run("k=1")], [run("t")], [sub([run("δ")], [run("k")])]),
        ]
    if num == 6:
        return [
            sub([run("z")], [text("lo")]), run(" = "), text("clamp"), run("(-"),
            sub([text("min")], [run("t")]), run(" "), sub([run("C")], [run("t")]),
            run(", 0, 1),    "), sub([run("z")], [text("hi")]), run(" = "), text("clamp"),
            run("(1 - "), sub([text("max")], [run("t")]), run(" "), sub([run("C")], [run("t")]),
            run(", 0, 1)"),
        ]
    if num == 7:
        return [
            sub([text("SOC")], [text("anchor")]), run(" = "), sub([run("z")], [text("lo")]),
            run(" + "), text("max"), run("("), sub([run("z")], [text("hi")]), run(" - "),
            sub([run("z")], [text("lo")]), run(", ε) · σ("), sup([run("ℓ")], [run("a")]),
            run("),    ε = "), sup([run("10")], [run("-6")]),
        ]
    if num == 8:
        return [
            soc_hat_sub("t"), run(" = "), sub([text("SOC")], [text("anchor")]),
            run(" + "), sub([run("C")], [run("t")]), comma(), text("for all "),
            run("t ∈ {1, ..., T}"),
        ]
    if num == 9:
        return [
            subsup([text("limit")], [run("t")], [run("*")]), run(" = |"),
            sub([run("I")], [run("t")]), run("| · "), sup([run("η")], [run("*")]),
            run(" · γ,    "), sup([run("η")], [run("*")]), run(" = 2,0"),
        ]
    if num == 10:
        return [
            sub([run("r")], [run("Δ")]), run(" = "),
            frac([text("mean"), run("(|Δ"), acc_hat([run("s")]), run("|)")], [text("mean"), run("(|Δs|)")]),
            run(" ≈ 1,0 "), text("pada "), sup([run("η")], [run("*")]), run(" = 2,0"),
        ]
    if num == 11:
        diff_squared = sup([
            run("("), sub([run("y")], [run("n,t")]), run(" - "), y_hat_sub("n,t"), run(")"),
        ], [run("2")])
        summand = [diff_squared]
        return [
            text("RMSE"), run(" = "),
            rad([
                frac([run("1")], [run("N · T")]), run(" · "),
                nary_sum([run("n,t")], [run("")], summand),
            ]),
            run(" × 100%"),
        ]
    if num == 12:
        def indicator_discharge():
            return [run("𝟙"), run("("), sub([run("I")], [run("t")]), run(" < -τ)")]

        ind2 = [run("𝟙"), run("(Δ"), y_hat_sub("t"), run(" > 0)")]
        numerator = [
            sub([run("∑")], [run("t")]), run(" "), *indicator_discharge(), run(" · "), *ind2,
        ]
        denominator = [sub([run("∑")], [run("t")]), run(" "), *indicator_discharge()]
        return [sub([text("PVR")], [text("dis")]), run(" = "), frac(numerator, denominator)]
    raise ValueError(num)


def equation_paragraph(num: int):
    p = etree.Element(qn("w:p"))
    ppr = w_el("pPr")
    jc = w_el("jc", val="center")
    spacing = w_el("spacing", after="60", line="240", lineRule="auto")
    ppr.extend([jc, spacing])
    p.append(ppr)
    omath = m_el("oMath")
    for child in build_equation(num):
        omath.append(child)
    p.append(omath)
    wr = w_el("r")
    rpr = w_el("rPr")
    rfonts = w_el("rFonts", ascii="Times New Roman", hAnsi="Times New Roman", cs="Times New Roman")
    sz = w_el("sz", val="18")
    rpr.extend([rfonts, sz])
    wt = w_el("t")
    wt.set(qn("xml:space"), "preserve")
    wt.text = f"    ({num})"
    wr.extend([rpr, wt])
    p.append(wr)
    return p


def capture_invariants(root):
    text_all = "\n".join(
        "".join(p.xpath(".//w:t/text()", namespaces=NS))
        for p in root.xpath("//w:p", namespaces=NS)
    )
    invariants = {
        "title_id": "Smooth Hard-Coulomb Constraint untuk Estimasi SOC Baterai Li-Ion pada Suhu Ekstrem",
        "title_en": "Smooth Hard-Coulomb Constraint for Physics-Constrained Li-Ion Battery SOC Estimation",
        "heading_dataset": "A. SPESIFIKASI DATASET DAN PRA-PEMROSESAN",
        "heading_arch": "C. ARSITEKTUR SMOOTH HARD-COULOMB CONSTRAINT",
        "fairness": "model Vanilla LSTM dibangun menggunakan backbone dan anggaran pelatihan yang sebanding, memiliki ukuran sebesar 53.569 parameter. Selisih tipis sekitar ~1,9% (1.057 parameter) pada model usulan Smooth Hard-Coulomb (54.626 parameter)",
        "placeholder_email": "[ISI EMAIL]",
        "placeholder_dates": "Received: DD MM YY",
        "placeholder_supervisor": "[ISI NAMA DOSEN PEMBIMBING]",
        "doi_count": str(text_all.lower().count("doi:")),
    }
    return {k: (v in text_all if k != "doi_count" else v) for k, v in invariants.items()}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)

    with zipfile.ZipFile(DOCX, "r") as zin:
        items = {info.filename: zin.read(info.filename) for info in zin.infolist()}
    xml = items["word/document.xml"]
    (WORK / "document_before.xml").write_bytes(xml)
    root = etree.fromstring(xml)
    before_invariants = capture_invariants(root)

    paragraphs = root.xpath("//w:body/w:p", namespaces=NS)
    markers = {
        1: "LSTM(x)",
        2: "l_delta",
        3: "limit_t =",
        4: "δ_t =",
        5: "C_t =",
        6: "z_lo =",
        7: "SOC_anchor =",
        8: "SOC_hat,t",
        9: "limit*_t",
        10: "r_Δ =",
        11: "RMSE =",
        12: "PVR_dis",
    }
    originals = {}
    replacements = {}
    for num, marker in markers.items():
        matched = []
        for p in paragraphs:
            txt = paragraph_text(p)
            if marker in txt and re.search(rf"\({num}\)\s*$", txt):
                matched.append(p)
        if len(matched) != 1:
            raise RuntimeError(f"Equation {num} matches={len(matched)} marker={marker!r}")
        old = matched[0]
        originals[num] = paragraph_text(old)
        replacements[num] = etree.tostring(old, encoding="unicode")
        new = equation_paragraph(num)
        parent = old.getparent()
        parent.replace(old, new)

    after_xml = etree.tostring(root, encoding="utf-8", xml_declaration=True, standalone=True)
    items["word/document.xml"] = after_xml
    tmp = DOCX.with_suffix(".omml.tmp.docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for filename, data in items.items():
            zout.writestr(filename, data)
    tmp.replace(DOCX)

    after_root = etree.fromstring(after_xml)
    after_invariants = capture_invariants(after_root)
    (WORK / "document_after.xml").write_bytes(after_xml)
    for num, txt in originals.items():
        (WORK / f"eq{num:02d}_original_text.txt").write_text(txt, encoding="utf-8")
        nodes = after_root.xpath(
            f"(//w:body/w:p[.//w:t[contains(., '({num})')]])[1]",
            namespaces=NS,
        )
        if nodes:
            (WORK / f"eq{num:02d}_after.xml").write_text(
                etree.tostring(nodes[0], encoding="unicode"), encoding="utf-8"
            )
    report = {
        "docx": str(DOCX),
        "backup": str(BACKUP),
        "original_equation_text": originals,
        "before_invariants": before_invariants,
        "after_invariants": after_invariants,
        "invariants_unchanged": before_invariants == after_invariants,
    }
    (WORK / "patch_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
