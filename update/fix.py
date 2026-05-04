#!/usr/bin/env python3
r"""
fix_citations.py
================
Post-processes a LaTeX Math IA source file in four passes:

  1. PREAMBLE    – adds \usepackage{cite} if not already present.
  2. CITATIONS   – replaces WIP (citation) / author-year / wrong cite-key
                   placeholders with proper \cite{key} commands.
  3. DISPLAY MATH – converts standalone inline-math lines ($...$) that sit
                   alone on a line to centred display math \[...\].
  4. BIBLIOGRAPHY – parses the .bib file and inlines a formatted
                   \begin{thebibliography}...\end{thebibliography} block,
                   producing a single self-contained .tex file (no bibtex
                   run needed — just pdflatex twice).

No external packages required — pure Python 3.8+.
"""

import re
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# HARD-CODED PATHS
# ══════════════════════════════════════════════════════════════════════════════

def main():
    input_path  = Path("/Users/leo/PycharmProjects/srs-benchmark/update/main-7.tex")
    output_path = input_path.with_stem(input_path.stem + "_fixed")
    bib_path    = Path("/Users/leo/PycharmProjects/srs-benchmark/update/Math IA-4.bib")

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not bib_path.exists():
        raise FileNotFoundError(f"Bib file not found: {bib_path}")

    src = input_path.read_text(encoding="utf-8")
    log: list[str] = []

    _banner(input_path, output_path, bib_path)

    # ── pass 1: preamble ─────────────────────────────────────────────────────
    src, notes = _patch_preamble(src)
    log.extend(notes)

    # ── pass 2: citations ────────────────────────────────────────────────────
    log.append("\n─── Citation replacements ───────────────────────────────────")
    for label, pattern, replacement, note in _CITATION_RULES:
        new_src, n = re.subn(pattern, replacement, src,
                             flags=re.IGNORECASE | re.DOTALL)
        tag = f"  [REPLACED x{n}]" if n else "  [no match ]"
        log.append(f"{tag}  {label}")
        if n:
            src = new_src

    # ── pass 3: centre standalone math ───────────────────────────────────────
    src, n_math = _center_standalone_math(src)
    log.append(
        f"\n  [REPLACED x{n_math}]  "
        r"Standalone $...$ lines -> display \[...\] (centred)"
    )

    # ── pass 4: inline bibliography ───────────────────────────────────────────
    log.append("\n─── Bibliography ────────────────────────────────────────────")
    entries = _parse_bib(bib_path)
    log.append(f"  Parsed {len(entries)} entries from '{bib_path.name}':")
    for e in entries:
        log.append(f"    [{e['type']:7s}]  {e['key']}")

    bib_block = _build_thebibliography(entries)
    src, bib_note = _replace_bibliography(src, bib_block)
    log.append(f"\n  {bib_note}")

    # ── pass 5: flag unresolved placeholders ──────────────────────────────────
    residual = list(re.finditer(r"\(citation\)", src, re.IGNORECASE))
    if residual:
        log.append(
            f"\n  WARNING: {len(residual)} unresolved (citation) "
            "placeholder(s) left — marked \\cite{TODO}:"
        )
        for m in residual:
            snip = src[max(0, m.start() - 70): m.end() + 70].replace("\n", " ")
            log.append(f"       ...{snip}...")
        src = re.sub(
            r"\(citation\)",
            r"\\cite{TODO}  % <- UNRESOLVED: replace with correct key",
            src, flags=re.IGNORECASE,
        )

    # ── write output ──────────────────────────────────────────────────────────
    output_path.write_text(src, encoding="utf-8")

    for line in log:
        print(line)

    print(f"\nWritten to {output_path}")
    print("Compile with:  pdflatex -> pdflatex  "
          "(bibliography is inlined; no bibtex step needed)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CITATION RULES
#   Each tuple: (human_label, regex_pattern, replacement, note)
#   Rules are applied in order — put more-specific patterns first.
# ══════════════════════════════════════════════════════════════════════════════

_CITATION_RULES = [

    # ── already-present cite keys that use wrong bib labels ──────────────────
    ("ibm_lstm -> noble_2025_long",
     r"\\cite\{ibm_lstm\}",
     r"\\cite{noble_2025_long}",
     "IBM LSTM figure caption is Noble (2025) in the bib."),

    ("ibm_attention -> bergmann_2024_attention",
     r"\\cite\{ibm_attention\}",
     r"\\cite{bergmann_2024_attention}",
     "IBM Attention figure caption is Bergmann & Stryker (2024) in the bib."),

    ("desmos -> a2026_partial",
     r"\\cite\{desmos\}",
     r"\\cite{a2026_partial}",
     "Desmos 3D graph cite updated to a2026_partial."),

    # ── inline author-year citations ──────────────────────────────────────────
    ("(Ye, Su & Cao 2022)",
     r"\(Ye,\s*Su\s*[&and]+\s*Cao\s*2022\)",
     r"\\cite{ye_2022_a}",
     "ACM KDD 2022 spaced-repetition paper."),

    ("(Jeon 2021)",
     r"\(Jeon\s*2021\)",
     r"\\cite{jeon_2021_last}",
     "Last Query Transformer RNN (arXiv 2021)."),

    ("(Sutton, 2019)",
     r"\(Sutton,?\s*2019\)",
     r"\\cite{sutton_2019_the}",
     "The Bitter Lesson — Rich Sutton 2019."),

    ("(Corbacıoglu & Aksel 2023)",
     r"\((?:Çorbacıoğlu|C.orbac.o.lu)\s*[&and]+\s*Aksel\s*2023\)",
     r"\\cite{orbacolu_2023_receiver}",
     "Turkish J. Emergency Medicine AUC guide 2023."),

    # Covers spelling variants: 'repetition', 'repitition', 'repititon' (typo in source)
    # Pattern: rep[ei]tit[io]?on  — the second vowel before 'on' is optional
    ("(open spaced repetition, 2025) — with or without comma",
     r"\(open\s+spaced\s+rep[ei]tit[io]?on,?\s*2025\)",
     r"\\cite{a2025_openspacedrepetitionsrsbenchmark}",
     "GitHub srs-benchmark repo."),

    # conclusion variant e.g. '( (open spaced repetition 2025)' — no comma after name
    ("(open spaced repetition 2025) — bare/no comma",
     r"\(\s*open\s+spaced\s+rep[ei]tit[io]?on\s+2025\s*\)",
     r"\\cite{a2025_openspacedrepetitionsrsbenchmark}",
     "GitHub srs-benchmark repo (conclusion variant)."),

    ("(github issue on FSRS 7)",
     r"\(github\s+issue\s+on\s+FSRS\s*[67]\)",
     r"\\cite{expertium_2025_outofdistribution}",
     "Expertium (2025) out-of-distribution GitHub issue."),

    # ── context-sensitive (citation) placeholders ─────────────────────────────
    ("(citation) — Ebbinghaus forgetting-rate claim",
     r"(Research by Eddinghouse[^(]*?)\(citation\)",
     r"\1\\cite{murre_2015_replication}",
     "Murre & Dros (2015) replication of Ebbinghaus."),

    ("(citation) — after 'fit the model\\'s parameters to the dataset'",
     r"(fit the model.s parameters to the dataset\s*)\(citation\)",
     r"\1\\cite{openspacedrepetition_2026_ankirevlogs10k}",
     "HuggingFace Anki-revlogs-10k dataset."),

    # ── new: scipy and PyTorch ────────────────────────────────────────────────
    # matches  "scipy.optimize's minimize\_scalar function"  (LaTeX-escaped _)
    ("scipy.optimize mention",
     r"(scipy\.optimize.s\s+minimize[_\\]+scalar\s+function)",
     r"\1~\\cite{scipyoptimizeminimize}",
     "SciPy minimize_scalar reference."),

    ("PyTorch mention",
     r"(I used PyTorch,\s*a specialized machine learning library)",
     r"\1~\\cite{pytorch}",
     "PyTorch reference."),
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DISPLAY MATH (centering)
# ══════════════════════════════════════════════════════════════════════════════

def _center_standalone_math(src: str) -> tuple[str, int]:
    r"""
    Convert lines that contain ONLY a single $...$ expression to display math.

    Before:
        $FPR = \frac{FP}{FP+TN} \approx 0.734$

    After:
        \[
            FPR = \frac{FP}{FP+TN} \approx 0.734
        \]

    Leaves lines with multiple $...$ spans or surrounding text untouched.
    """
    # Matches: optional indent + exactly one $...$ (no $ or newline inside) + optional trailing space
    pattern = re.compile(r"^([ \t]*)\$([^$\n]+)\$([ \t]*)$", re.MULTILINE)

    def _repl(m: re.Match) -> str:
        indent  = m.group(1)
        content = m.group(2).strip()
        return f"{indent}\\[\n{indent}    {content}\n{indent}\\]"

    return pattern.subn(_repl, src)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BIBTEX PARSER  (no external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def _brace_span(s: str, start: int) -> tuple[str, int]:
    """Return (inner_content, closing_brace_index) for the brace-group at start."""
    depth = 0
    i = start
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1: i], i
        i += 1
    return s[start + 1:], len(s) - 1


def _parse_fields(body: str) -> dict[str, str]:
    """Extract field = value pairs from a BibTeX entry body string."""
    fields: dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        eq = body.find("=", i)
        if eq == -1:
            break
        name = body[i:eq].strip().lower()
        i = eq + 1
        while i < n and body[i] in " \t\n\r":
            i += 1
        if i >= n:
            break
        if body[i] == "{":
            value, end = _brace_span(body, i)
            i = end + 1
        elif body[i] == '"':
            end = body.find('"', i + 1)
            value = body[i + 1: end] if end != -1 else body[i + 1:]
            i = (end + 1) if end != -1 else n
        else:
            end = i
            while end < n and body[end] not in ",\n}":
                end += 1
            value = body[i:end].strip()
            i = end
        if name:
            fields[name] = value.strip()
    return fields


def _parse_bib(bib_path: Path) -> list[dict]:
    """Parse a .bib file; return list of {'type', 'key', 'fields'} dicts."""
    text = bib_path.read_text(encoding="utf-8")
    entries = []
    i = 0
    while i < len(text):
        at = text.find("@", i)
        if at == -1:
            break
        brace = text.find("{", at)
        if brace == -1:
            break
        etype = text[at + 1: brace].strip().lower()
        body_content, end = _brace_span(text, brace)
        comma = body_content.find(",")
        key   = body_content[:comma].strip() if comma != -1 else body_content.strip()
        body  = body_content[comma + 1:] if comma != -1 else ""
        entries.append({"type": etype, "key": key, "fields": _parse_fields(body)})
        i = end + 1
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BIBLIOGRAPHY FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_MAP = {
    "1":  "Jan.", "01": "Jan.", "2":  "Feb.", "02": "Feb.",
    "3":  "Mar.", "03": "Mar.", "4":  "Apr.", "04": "Apr.",
    "5":  "May",  "05": "May",  "6":  "Jun.", "06": "Jun.",
    "7":  "Jul.", "07": "Jul.", "8":  "Aug.", "08": "Aug.",
    "9":  "Sep.", "09": "Sep.", "10": "Oct.", "11": "Nov.", "12": "Dec.",
}


def _format_authors(raw: str) -> str:
    """
    Convert a BibTeX author string to IEEE-style abbreviated names.

      'Murre, Jaap M. J. and Dros, Joeri'  ->  'J. M. J. Murre and J. Dros'
      'Google Developers'                   ->  'Google Developers'
      'Ye, Junyao and Su, Jingyong and Cao, Yilong'  ->  'J. Ye, J. Su, and Y. Cao'
    """
    parts = re.split(r"\s+and\s+", raw.strip(), flags=re.IGNORECASE)
    formatted = []
    for p in parts:
        p = p.strip()
        if "," in p:
            last, _, first = p.partition(",")
            last = last.strip()
            tokens = first.strip().split()
            # Preserve existing periods (e.g. 'M.' stays 'M.'); otherwise abbreviate
            initials = " ".join(
                t if t.endswith(".") else t[0] + "."
                for t in tokens if t
            )
            formatted.append(f"{initials} {last}")
        else:
            formatted.append(p)   # organisation / single-name — keep as-is

    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


def _strip_braces(s: str) -> str:
    """Remove LaTeX brace groups: {Title Case} -> Title Case."""
    return re.sub(r"\{([^}]*)\}", r"\1", s)


def _safe_pages(s: str) -> str:
    """Normalise page-range dashes to LaTeX double-dash."""
    return re.sub(r"[–—]", "--", s)


def _format_entry(e: dict) -> str:
    """
    Return a single LaTeX-formatted reference string for one BibTeX entry,
    following a condensed IEEE-like style appropriate for an IB exploration.
    """
    f     = e["fields"]
    etype = e["type"]

    raw_author = f.get("author", "")
    author  = _format_authors(raw_author) if raw_author else ""
    title   = _strip_braces(f.get("title",   ""))
    year    = f.get("year",   "")
    month   = _MONTH_MAP.get(f.get("month", ""), "")
    url     = f.get("url",    "")
    doi     = f.get("doi",    "")
    journal = _strip_braces(f.get("journal", ""))
    volume  = f.get("volume", "")
    pages   = f.get("pages",  "")
    org     = f.get("organization", "")

    date = " ".join(filter(None, [month, year]))
    parts: list[str] = []

    # Author / organisation credit
    if author:
        parts.append(author + ",")
    # (if no author, the title takes the leading position)

    # Title — quoted for all types
    parts.append(f"``{title},''")

    if etype == "article":
        # Journal, volume, pages, date
        venue = f"\\textit{{{journal}}}"
        if volume:
            venue += f", vol.~{volume}"
        if pages:
            venue += f", pp.~{_safe_pages(pages)}"
        if date:
            venue += f", {date}"
        parts.append(venue + ".")
        # Prefer DOI link; fall back to URL
        if doi:
            parts.append(
                f"doi:~\\href{{https://doi.org/{doi}}}{{{doi}}}."
            )
        elif url:
            parts.append(f"[Online]. Available: \\url{{{url}}}")
    else:
        # @misc / @book / other
        venue_parts: list[str] = []
        # Include organisation only if it adds new information
        if org and org.lower() != raw_author.lower():
            venue_parts.append(org)
        if date:
            venue_parts.append(date)
        if venue_parts:
            parts.append(", ".join(venue_parts) + ".")
        if url:
            parts.append(f"[Online]. Available: \\url{{{url}}}")

    return " ".join(parts)


def _build_thebibliography(entries: list[dict]) -> str:
    """Build a complete \\begin{thebibliography}...\\end{thebibliography} block."""
    width = str(max(len(entries), 9))    # LaTeX label-width hint
    lines = [
        "% ── Bibliography inlined by fix_citations.py ─────────────────────────────",
        "% Compile with:  pdflatex  ->  pdflatex  (no bibtex step needed)",
        f"\\begin{{thebibliography}}{{{width}}}",
    ]
    for e in entries:
        lines.append("")
        lines.append(f"\\bibitem{{{e['key']}}}")
        lines.append(_format_entry(e))
    lines += ["", "\\end{thebibliography}"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — REPLACE BIBLIOGRAPHY BLOCK IN SOURCE
# ══════════════════════════════════════════════════════════════════════════════

_THEBIB_RE = re.compile(
    r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL)
_BIB_CMD_RE = re.compile(
    r"(?:\\bibliographystyle\{[^}]*\}\s*\n?)?\\bibliography\{[^}]*\}")


def _replace_bibliography(src: str, block: str) -> tuple[str, str]:
    """Replace an existing bibliography block, or append before \\end{document}.

    Uses a lambda replacement so that backslashes inside the inlined bibliography
    (\\url, \\textit, etc.) are never misinterpreted as regex escape sequences.
    """
    for pat, tag in (
        (_THEBIB_RE,  "existing thebibliography environment"),
        (_BIB_CMD_RE, "existing \\bibliography{} command"),
    ):
        if pat.search(src):
            # lambda avoids re interpreting \u, \n, etc. in the replacement
            result = pat.sub(lambda _: block, src, count=1)
            return result, \
                   f"Replaced {tag} with inlined thebibliography ({len(block.splitlines())} lines)."
    src = src.replace(r"\end{document}", block + "\n\n" + r"\end{document}")
    return src, "No bibliography found — appended block before \\end{document}."


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PREAMBLE PATCHES
# ══════════════════════════════════════════════════════════════════════════════

def _patch_preamble(src: str) -> tuple[str, list[str]]:
    """
    Add \\usepackage{cite} if no citation-management package is loaded.
    hyperref (already present) covers \\url{} and \\href{} — no extra package needed.
    """
    notes: list[str] = []
    has_cite_pkg = any(
        re.search(r"\\usepackage(?:\[.*?\])?\{" + pkg + r"\}", src)
        for pkg in ("natbib", "biblatex", "cite")
    )
    if not has_cite_pkg:
        src = re.sub(
            r"(\\usepackage\{lastpage\})",
            r"\1\n\\usepackage{cite}   % added by fix_citations.py",
            src, count=1,
        )
        notes.append("  Added \\usepackage{cite} to preamble.")
    return src, notes


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _banner(inp: Path, out: Path, bib: Path) -> None:
    w = 64
    print("\n" + "=" * w)
    print("  fix_citations.py")
    print(f"  Input  : {inp}")
    print(f"  Output : {out}")
    print(f"  Bib    : {bib}")
    print("=" * w)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()