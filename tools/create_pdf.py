"""Tool to create PDF files from markdown/HTML content."""
import os
import re
from pathlib import Path

TOOL_DEFINITION = {
    "name": "create_pdf",
    "description": "Create a PDF file from text/markdown content. Returns the file path. Use this when asked to generate papers, reports, or documents as PDF.",
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Output filename (e.g. 'metabolism_paper.pdf')"
            },
            "content": {
                "type": "string",
                "description": "The full document content in markdown or HTML. Use HTML for best formatting: <h1>, <h2>, <p>, <ul>, <li>, <em>, <strong>, <blockquote>, <table>, etc."
            },
            "title": {
                "type": "string",
                "description": "Document title for the header"
            },
            "author": {
                "type": "string",
                "description": "Author name(s)"
            }
        },
        "required": ["filename", "content"]
    }
}

SAFE = True


def execute(inp: dict) -> str:
    try:
        from weasyprint import HTML
    except ImportError:
        return "Error: weasyprint not installed. Run: pip install weasyprint"

    filename = inp["filename"]
    content = inp["content"]
    title = inp.get("title", "")
    author = inp.get("author", "")

    # Convert markdown-style to HTML if needed
    if not content.strip().startswith("<"):
        lines = content.split("\n")
        html_lines = []
        in_list = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<h1>{}</h1>".format(stripped[2:]))
            elif stripped.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<h2>{}</h2>".format(stripped[3:]))
            elif stripped.startswith("### "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<h3>{}</h3>".format(stripped[4:]))
            elif stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append("<li>{}</li>".format(stripped[2:]))
            elif stripped.startswith("> "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<blockquote>{}</blockquote>".format(stripped[2:]))
            elif stripped == "":
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
            elif stripped.startswith("**") and stripped.endswith("**"):
                html_lines.append("<p><strong>{}</strong></p>".format(stripped[2:-2]))
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                stripped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
                stripped = re.sub(r'\*(.+?)\*', r'<em>\1</em>', stripped)
                html_lines.append("<p>{}</p>".format(stripped))
        if in_list:
            html_lines.append("</ul>")
        content = "\n".join(html_lines)

    # Build full HTML document with academic styling
    header = ""
    if title:
        header += '<h1 class="doc-title">{}</h1>'.format(title)
    if author:
        header += '<p class="doc-author">{}</p>'.format(author)

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {
        size: A4;
        margin: 2.5cm 2.5cm 3cm 2.5cm;
        @bottom-center {
            content: counter(page);
            font-size: 10pt;
            color: #666;
        }
    }
    body {
        font-family: 'Times New Roman', 'DejaVu Serif', Georgia, serif;
        font-size: 12pt;
        line-height: 1.6;
        color: #1a1a1a;
        text-align: justify;
    }
    h1 { font-size: 18pt; margin-top: 24pt; margin-bottom: 12pt; text-align: center; }
    h1.doc-title { font-size: 22pt; margin-top: 60pt; margin-bottom: 6pt; }
    p.doc-author { text-align: center; font-size: 14pt; color: #444; margin-bottom: 40pt; }
    h2 { font-size: 14pt; margin-top: 18pt; margin-bottom: 8pt; }
    h3 { font-size: 12pt; font-style: italic; margin-top: 14pt; margin-bottom: 6pt; }
    p { margin-bottom: 8pt; text-indent: 2em; }
    p:first-of-type { text-indent: 0; }
    blockquote {
        margin: 12pt 2em;
        padding: 8pt 16pt;
        border-left: 3pt solid #ccc;
        font-style: italic;
        color: #444;
    }
    ul, ol { margin: 8pt 0; padding-left: 2em; }
    li { margin-bottom: 4pt; }
    table { border-collapse: collapse; width: 100%; margin: 12pt 0; }
    th, td { border: 1pt solid #999; padding: 6pt 8pt; text-align: left; font-size: 10pt; }
    th { background: #f0f0f0; font-weight: bold; }
    .abstract {
        margin: 20pt 2em;
        padding: 12pt;
        border: 1pt solid #ccc;
        font-size: 11pt;
    }
    .abstract h2 { text-align: center; font-size: 12pt; }
</style>
</head>
<body>
HEADER_PLACEHOLDER
CONTENT_PLACEHOLDER
</body>
</html>""".replace("HEADER_PLACEHOLDER", header).replace("CONTENT_PLACEHOLDER", content)

    # Determine output path — always relative to bridge.py location
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename

    # Generate PDF
    HTML(string=html).write_pdf(str(output_path))

    return "PDF created: {} ({} bytes)".format(output_path, output_path.stat().st_size)
