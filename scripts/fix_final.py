import pandas as pd
from pathlib import Path
import re

f_in = Path('scripts/build_jnteti_final.py')
text = f_in.read_text(encoding='utf-8')

# Fix A: ISO 26262 in Abstract
old_abs_target = 'Pelanggaran ini menjadikan luaran model physically inadmissible untuk BMS di bawah kerangka ISO 26262.'
new_abs = 'Pelanggaran ini merepresentasikan ketidaksesuaian luaran model dengan prinsip konsistensi arah aliran arus yang disyaratkan secara prinsip oleh kerangka keselamatan fungsional seperti ISO 26262 dan PAS 8800, meskipun penelitian ini tidak melakukan proses sertifikasi ASIL formal terhadap sistem yang diusulkan.'
text = text.replace(old_abs_target, new_abs)

# Fix B: Equation 4 - remove LaTeX, use unicode piecewise
old_eq4 = '    add_equation(doc, "δ_t = \\\\begin{cases} -limit_t \\\\cdot \\\\sigma(ℓ_t^δ) & \\\\text{jika } I_t < -τ \\\\\\\\ +limit_t \\\\cdot \\\\sigma(ℓ_t^δ) & \\\\text{jika } I_t > τ \\\\\\\\ 0 & \\\\text{jika } |I_t| \\\\le τ \\\\end{cases}", 4)'
new_eq4 = '    add_para(doc, "δ_t = { −limit_t · σ(ℓ_t^δ),  jika I_t < −τ", align=WD_ALIGN_PARAGRAPH.CENTER, font="Cambria Math", size=9.5)\n    add_para(doc, "          { +limit_t · σ(ℓ_t^δ),  jika I_t > τ", align=WD_ALIGN_PARAGRAPH.CENTER, font="Cambria Math", size=9.5)\n    add_para(doc, "          { 0,                    jika |I_t| ≤ τ                    (4)", align=WD_ALIGN_PARAGRAPH.CENTER, font="Cambria Math", size=9.5)'
text = text.replace(old_eq4, new_eq4)

# Fix C: Duplicate paragraph Shift/Leak
target_dup = '    add_para(doc, "Kolom \\\'Shift (%)\\\' pada Tabel I melaporkan persentase perubahan nilai SOC rata-rata akibat koreksi ohmic bias relatif terhadap label asli, sedangkan kolom \\\'Leak\\\' melaporkan jumlah timestamp yang teridentifikasi tumpang tindih antar partisi train/validation/test melalui enam asersi interseksi (Bagian II.B); nilai nol pada seluruh varian mengonfirmasi tidak adanya temporal leakage.")\n\n'

# Count occurrences before replacing
print("Duplicate paragraphs before removal:", text.count(target_dup))

text = text.replace(target_dup, '') # remove all instances
# Now carefully add it back right before the table I call
insert_marker = '    add_dataframe_table(\n        doc,\n        "Tabel I. Varian dataset dari Protokol Dasar ke Protokol Final'
text = text.replace(insert_marker, target_dup + insert_marker)

f_in.write_text(text, encoding='utf-8')

# Fix D: Table VII claim text
df7 = pd.read_csv('outputs/tables/table07_claim_summary.csv')
# Look for ID 17 and fix the text
df7.loc[df7['id'] == 17, 'claim'] = 'Bagian dari narasi MaxE katastropik pada draf awal adalah artefak koreksi label'
df7.to_csv('outputs/tables/table07_claim_summary.csv', index=False)

print('Executed fix script successfully.')
