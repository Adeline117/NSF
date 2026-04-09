"""
Markdown to LaTeX converter for academic papers.
Handles: headers, tables, bold/italic, math, code blocks, BibTeX.
Does NOT handle images (figures need manual \includegraphics).

Usage:
    python md_to_latex.py input.md output.tex --template acm|ieee
"""

import argparse
import re
import sys
from pathlib import Path

ACM_PREAMBLE = r"""\documentclass[sigconf,nonacm=false]{acmart}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}

\begin{document}
"""

IEEE_PREAMBLE = r"""\documentclass[conference]{IEEEtran}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{cite}
\usepackage{url}

\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""


def convert_md_to_latex(md_text: str) -> str:
    """Convert markdown body to LaTeX body (no preamble)."""
    lines = md_text.split("\n")
    out = []
    in_code = False
    in_table = False
    in_bibtex = False
    table_rows = []
    bibtex_content = []

    for line in lines:
        # Code blocks
        if line.strip().startswith("```bibtex"):
            in_bibtex = True
            continue
        if line.strip().startswith("```") and in_bibtex:
            in_bibtex = False
            continue
        if in_bibtex:
            bibtex_content.append(line)
            continue
        if line.strip().startswith("```"):
            if in_code:
                out.append(r"\end{verbatim}")
                in_code = False
            else:
                out.append(r"\begin{verbatim}")
                in_code = True
            continue
        if in_code:
            out.append(line)
            continue

        # Tables
        if "|" in line and not in_table:
            # Check if it's a table header
            if re.match(r"^\s*\|.*\|", line):
                in_table = True
                table_rows = [line]
                continue
        if in_table:
            if "|" in line:
                table_rows.append(line)
                continue
            else:
                # End of table
                out.append(_convert_table(table_rows))
                in_table = False
                table_rows = []

        # Skip front matter markers
        if line.strip() == "---":
            continue

        # Headers
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            title = _convert_inline(m.group(2))
            if level == 1:
                out.append(f"\\section{{{title}}}")
            elif level == 2:
                out.append(f"\\section{{{title}}}")
            elif level == 3:
                out.append(f"\\subsection{{{title}}}")
            elif level == 4:
                out.append(f"\\subsubsection{{{title}}}")
            else:
                out.append(f"\\paragraph{{{title}}}")
            continue

        # Regular text
        out.append(_convert_inline(line))

    # Handle trailing table
    if in_table and table_rows:
        out.append(_convert_table(table_rows))

    return "\n".join(out), bibtex_content


def _convert_inline(text: str) -> str:
    """Convert inline markdown to LaTeX."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"\\textit{\1}", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    # Links
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Escape special chars (but not in math mode)
    # Only escape & and % outside of math
    text = text.replace("&", r"\&") if "$" not in text else text
    text = text.replace("%", r"\%") if "$" not in text else text
    # Fix double escapes
    text = text.replace(r"\\&", r"\&")
    return text


def _convert_table(rows: list[str]) -> str:
    """Convert markdown table to LaTeX tabular."""
    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        parsed.append(cells)

    if len(parsed) < 2:
        return ""

    # Skip separator row (contains ---)
    data_rows = [r for r in parsed if not all(
        re.match(r"^[-:]+$", c.strip()) for c in r
    )]

    if not data_rows:
        return ""

    ncols = max(len(r) for r in data_rows)
    col_spec = "l" * ncols

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    for i, row in enumerate(data_rows):
        cells = [_convert_inline(c) for c in row]
        while len(cells) < ncols:
            cells.append("")
        line = " & ".join(cells) + r" \\"
        lines.append(line)
        if i == 0:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def convert_file(input_path: str, output_path: str, template: str = "acm"):
    md = Path(input_path).read_text()

    # Extract title (first # heading)
    title_match = re.search(r"^#\s+(.+)", md, re.MULTILINE)
    title = title_match.group(1) if title_match else "Untitled"

    # Extract abstract
    abstract = ""
    abs_match = re.search(
        r"##\s*Abstract\s*\n\n(.+?)(?=\n\n##|\n\n\*\*Keywords)",
        md, re.DOTALL,
    )
    if abs_match:
        abstract = abs_match.group(1).strip()

    # Extract keywords
    kw_match = re.search(r"\*\*Keywords?:\*\*\s*(.+)", md)
    keywords = kw_match.group(1).strip() if kw_match else ""

    body, bibtex = convert_md_to_latex(md)

    preamble = ACM_PREAMBLE if template == "acm" else IEEE_PREAMBLE

    tex = preamble
    tex += f"\\title{{{_convert_inline(title)}}}\n"
    tex += "\\author{NSF Project Team}\n"
    if template == "acm":
        tex += "\\affiliation{\\institution{University}}\n"
    tex += "\\maketitle\n\n"

    if abstract:
        tex += "\\begin{abstract}\n"
        tex += _convert_inline(abstract) + "\n"
        tex += "\\end{abstract}\n\n"

    if keywords:
        if template == "acm":
            tex += f"\\keywords{{{_convert_inline(keywords)}}}\n\n"

    tex += body
    tex += "\n\n\\bibliographystyle{ACM-Reference-Format}\n"

    if bibtex:
        bib_path = Path(output_path).with_suffix(".bib")
        bib_path.write_text("\n".join(bibtex))
        tex += f"\\bibliography{{{bib_path.stem}}}\n"

    tex += POSTAMBLE

    Path(output_path).write_text(tex)
    print(f"Saved {output_path} ({len(tex)} chars)")
    if bibtex:
        print(f"Saved {bib_path} ({len(bibtex)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input markdown file")
    parser.add_argument("output", help="Output .tex file")
    parser.add_argument("--template", choices=["acm", "ieee"], default="acm")
    args = parser.parse_args()
    convert_file(args.input, args.output, args.template)
