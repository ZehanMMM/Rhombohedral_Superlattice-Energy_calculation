from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "V620_JACS_SI"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCX_PATH = OUT_DIR / "V620_JACS_SI_16nm_concise_with_Zeeman_MA.docx"


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa, indent_dxa=120):
    total = sum(widths_dxa)
    table.autofit = False
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        grid.append(grid_col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.first_child_found_in("w:shd")
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_run_font(run, name="Times New Roman", size=11, bold=None, italic=None, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    pf = normal.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15

    h1 = styles["Heading 1"]
    h1.font.name = "Times New Roman"
    h1._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    h1._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    h1.font.size = Pt(14)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0, 0, 0)
    h1.paragraph_format.space_before = Pt(0)
    h1.paragraph_format.space_after = Pt(8)
    h1.paragraph_format.keep_with_next = True

    h2 = styles["Heading 2"]
    h2.font.name = "Times New Roman"
    h2._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    h2._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    h2.font.size = Pt(11)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0, 0, 0)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)
    h2.paragraph_format.keep_with_next = True

    cap = styles["Caption"]
    cap.font.name = "Times New Roman"
    cap._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    cap._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    cap.font.size = Pt(9)
    cap.font.italic = False
    cap.font.color.rgb = RGBColor(0, 0, 0)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    cap.paragraph_format.line_spacing = 1.05


def add_para(doc, text="", bold_lead=None):
    p = doc.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        r1 = p.add_run(bold_lead)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_lead):])
        set_run_font(r2)
    else:
        r = p.add_run(text)
        set_run_font(r)
    return p


def add_para_segments(doc, segments):
    p = doc.add_paragraph()
    for item in segments:
        if isinstance(item, str):
            text, opts = item, {}
        else:
            text, opts = item
        run = p.add_run(text)
        set_run_font(
            run,
            size=opts.get("size", 11),
            bold=opts.get("bold", False),
            italic=opts.get("italic", False),
        )
        run.font.subscript = opts.get("sub", False)
        run.font.superscript = opts.get("sup", False)
    return p


def add_reference(doc, number, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(f"{number}. ")
    set_run_font(r, size=9)
    r = p.add_run(text)
    set_run_font(r, size=9)
    return p


def add_equation(doc, segments):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_together = True
    for item in segments:
        if isinstance(item, str):
            text, opts = item, {}
        else:
            text, opts = item
        run = p.add_run(text)
        set_run_font(run, name="Cambria Math", size=11, italic=opts.get("italic", False))
        run.font.subscript = opts.get("sub", False)
        run.font.superscript = opts.get("sup", False)
    return p


def add_table_caption(doc, text):
    p = doc.add_paragraph(style="Caption")
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=9, bold=False)
    return p


def style_table(table, font_size=9, header_fill="EDEDED", alignments=None):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.style = "Table Grid"
    for ridx, row in enumerate(table.rows):
        if ridx == 0:
            set_repeat_table_header(row)
        for cidx, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if ridx == 0:
                set_cell_shading(cell, header_fill)
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                p.alignment = alignments[cidx] if alignments else WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    set_run_font(run, size=font_size, bold=(ridx == 0))


def add_placeholder(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = True
    p_pr = p._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "8")
        el.set(qn("w:space"), "8")
        el.set(qn("w:color"), "A6A6A6")
        p_bdr.append(el)
    p_pr.append(p_bdr)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F7F7F7")
    p_pr.append(shd)
    r = p.add_run("Insert the supplied Figure S14 artwork here")
    set_run_font(r, size=10, italic=True, color="666666")


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    set_run_font(run, size=9)


def add_hyperlink(paragraph, text, url):
    part = paragraph.part
    rel_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), "Times New Roman")
    r_fonts.set(qn("w:hAnsi"), "Times New Roman")
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "22")
    size_cs = OxmlElement("w:szCs")
    size_cs.set(qn("w:val"), "22")
    r_pr.extend([r_fonts, color, underline, size, size_cs])
    run.append(r_pr)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    return hyperlink


def build_docx():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    configure_styles(doc)
    add_page_number(section.footer.paragraphs[0])

    title = doc.add_paragraph(style="Heading 1")
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(10)
    run = title.add_run("V6.20 calculation for 16 nm nanocubes")
    set_run_font(run, size=14, bold=True)

    add_para(
        doc,
        "The V6.20 model gives the energy per particle of a rhombohedral lattice of 16 nm Fe₃O₄ nanocubes. A trial structure is defined by the rhombohedral angle α, particle rotation γ, and lattice constant a. The central particle and its neighbors in a 3 × 3 × 3 lattice shell are included (Figure S14)."
    )

    doc.add_paragraph("Lattice geometry and fixed separation", style="Heading 2")
    add_para(
        doc,
        "The particle centers form a rhombohedral lattice with primitive vector length a and common angle α. The primitive vectors are"
    )
    add_equation(
        doc,
        [
            "a₁ = a(c, s, 0),    a₂ = a(c, −s/2, √3s/2),    a₃ = a(c, −s/2, −√3s/2),",
        ],
    )
    add_equation(
        doc,
        [
            "c = [(1 + 2 cos α)/3]",
            ("1/2", {"sup": True}),
            ",    s = (1 − c",
            ("2", {"sup": True}),
            ")",
            ("1/2", {"sup": True}),
            ".",
        ],
    )
    add_para(
        doc,
        "All cubes have the same orientation, Q(γ) = Rₓ(γ)QSC. The magnetic moment lies along the field and a cube ⟨111⟩ axis. The structure has a 120° period in γ."
    )
    add_para(
        doc,
        "Each cube has edge length L = 16 nm and rounding radius rround = 1.5 nm. For center vector rj, let p = Qᵀrj, b = L − 2rround, and z = |p| − b(1,1,1). The signed surface gap is"
    )
    add_equation(
        doc,
        [
            "D",
            ("j", {"sub": True}),
            " = ‖max(z,0)‖",
            ("2", {"sub": True}),
            " + min[max(z",
            ("x", {"sub": True}),
            ",z",
            ("y", {"sub": True}),
            ",z",
            ("z", {"sub": True}),
            "),0] − 2r",
            ("round", {"sub": True}),
            ".",
        ],
    )
    add_para(
        doc,
        "Dmin is the smallest gap among the 26 neighboring sites. The reference structure, a = 20.31 nm, α = 74.0°, and γ = −15.0°, gives Dref = 3.377736 nm. For each (α,γ), a is obtained from Dmin = Dref. No soft steric potential is used. The rounding radius describes particle shape, not ligand thickness."
    )

    add_placeholder(doc)
    cap = doc.add_paragraph(style="Caption")
    r = cap.add_run("Figure S14. ")
    set_run_font(r, size=9, bold=True)
    r = cap.add_run(
        "Schematic of the van der Waals calculation. (A) Each cube contains 4 × 4 × 4 voxels. (B) The central particle is evaluated in a 3 × 3 × 3 lattice shell. The dipolar sum uses 26 neighbors. The VdW sum uses six primitive neighbors."
    )
    set_run_font(r, size=9, bold=False)

    interaction_heading = doc.add_paragraph("Interaction Hamiltonian", style="Heading 2")
    interaction_heading.paragraph_format.page_break_before = True
    add_para(
        doc,
        "The energy per particle is"
    )
    add_equation(
        doc,
        [
            "E = E",
            ("Z", {"sub": True}),
            " + E",
            ("ani", {"sub": True}),
            " + E",
            ("dd", {"sub": True}),
            "(26) + E",
            ("VdW", {"sub": True}),
            "(6).",
        ],
    )
    add_para_segments(
        doc,
        [
            "Edd(26) and EVdW(6) denote the 26 neighbor dipolar sum and six neighbor VdW sum. Energies are reported in kBT at T = 298.15 K. The measured value Ms = 2.85 × 10⁵ A m⁻¹ and the applied field B = 0.50 T are used. The calculation uses Kani = 2.00 × 10⁴ J m⁻³. Values of this order have been reported for magnetite nanocrystals.",
            ("1,2", {"sup": True}),
            " The magnetic moment is μ = MsL³. EZ and Eani are constant during the structural scan. The minimized energy is Eint = Edd(26) + EVdW(6). The dipolar term is",
        ],
    )
    add_equation(
        doc,
        [
            "E",
            ("dd", {"sub": True}),
            "(26) = ½ Σ",
            ("j=1,…,26", {"sub": True}),
            " μ",
            ("0", {"sub": True}),
            "/(4πr",
            ("j", {"sub": True}),
            ("3", {"sup": True}),
            ")[μ",
            ("i", {"sub": True}),
            "·μ",
            ("j", {"sub": True}),
            " − 3(μ",
            ("i", {"sub": True}),
            "·r̂",
            ("j", {"sub": True}),
            ")(μ",
            ("j", {"sub": True}),
            "·r̂",
            ("j", {"sub": True}),
            ")].",
        ],
    )
    add_para(
        doc,
        "The factor 1/2 avoids double counting. The sum is limited to the 3 × 3 × 3 lattice shell."
    )
    add_para_segments(
        doc,
        [
            "Each sharp cube is represented by 64 equal voxels with ΔV = (L/4)³. The Hamaker constant is AH = 2.00 × 10⁻²⁰ J. This value lies within the 9 to 29 zJ range calculated for iron oxide particles in nonpolar solvents.",
            ("3", {"sup": True}),
            " The pairwise additive Hamaker form is used.",
            ("4", {"sup": True}),
            " The VdW term is",
        ],
    )
    add_equation(
        doc,
        [
            "E",
            ("VdW", {"sub": True}),
            "(6) = −A",
            ("H", {"sub": True}),
            "/(2π",
            ("2", {"sup": True}),
            ") Σ",
            ("j∈P₆", {"sub": True}),
            " Σ",
            ("p,q", {"sub": True}),
            " (ΔV)",
            ("2", {"sup": True}),
            "/{max[|r",
            ("j", {"sub": True}),
            " + x",
            ("q", {"sub": True}),
            " − x",
            ("p", {"sub": True}),
            "|",
            ("2", {"sup": True}),
            ",d",
            ("floor", {"sub": True}),
            ("2", {"sup": True}),
            "]}",
            ("3", {"sup": True}),
            ".",
        ],
    )
    add_para(
        doc,
        "P₆ = {±a₁, ±a₂, ±a₃}. xp and xq are voxel centers. dfloor² = 1.00 × 10⁻¹⁹ m² is a numerical cutoff. Rounded geometry is used for Dmin. Sharp voxels are used for the VdW term."
    )

    doc.add_paragraph("Numerical minimization and result", style="Heading 2")
    add_para(
        doc,
        "α was scanned from 60° to 90° in 0.5° steps. For each α, 61 values of γ covered one 120° period. At every (α,γ), a was found from Dmin = Dref. The lowest grid point was refined in γ and α."
    )
    add_para(
        doc,
        "The minimum occurs at α = 74.177°, γ = −16.517°, and a = 20.1963 nm. At this point, Edd(26) = −3.5877 kBT and EVdW(6) = −2.2559 kBT. Thus, Eint = −5.8436 kBT per particle. The constant terms are EZ = −141.7936 kBT and Eani = −6.6336 kBT. The total energy is E = −154.2708 kBT per particle."
    )
    add_para(
        doc,
        "The model excludes a soft ligand free energy, Ewald summation, many body dispersion, defects, and nonuniform magnetization."
    )

    references_heading = doc.add_paragraph("References", style="Heading 2")
    references = [
        "Moreno, R., Poyser, S., Meilak, D., Meo, A., Jenkins, S., Lazarov, V. K., Vallejo-Fernandez, G., Majetich, S., Evans, R. F. L. The Role of Faceting and Elongation on the Magnetic Anisotropy of Magnetite Fe₃O₄ Nanocrystals. Sci. Rep. 2020, 10, 2722. https://doi.org/10.1038/s41598-020-58976-7.",
        "Greculeasa, S. G., Palade, P., Schinteie, G., Leca, A., Dumitrache, F., Lungu, I., Prodan, G., Kuncser, A., Kuncser, V. Tuning Structural and Magnetic Properties of Fe Oxide Nanoparticles by Specific Hydrogenation Treatments. Sci. Rep. 2020, 10, 17174. https://doi.org/10.1038/s41598-020-74188-5.",
        "Faure, B., Salazar-Alvarez, G., Bergström, L. Hamaker Constants of Iron Oxide Nanoparticles. Langmuir 2011, 27, 8659-8664. https://doi.org/10.1021/la201387d.",
        "Hamaker, H. C. The London van der Waals Attraction between Spherical Particles. Physica 1937, 4, 1058-1072. https://doi.org/10.1016/S0031-8914(37)80203-7.",
    ]
    for number, reference in enumerate(references, start=1):
        add_reference(doc, number, reference)

    doc.add_paragraph("Code availability", style="Heading 2")
    code_p = doc.add_paragraph()
    lead = code_p.add_run(
        "The MATLAB code is available at "
    )
    set_run_font(lead)
    add_hyperlink(
        code_p,
        "GitHub repository",
        "https://github.com/ZehanMMM/Rhombohedral-Superlattice-energy-calculation",
    )
    code_p.add_run(".")

    core = doc.core_properties
    core.title = "V6.20 calculation for 16 nm nanocubes"
    core.subject = "JACS Supporting Information insert"
    core.author = ""
    core.keywords = "nanocube, rhombohedral superlattice, dipole, van der Waals, V6.20"
    doc.save(DOCX_PATH)


if __name__ == "__main__":
    build_docx()
    print(DOCX_PATH)
