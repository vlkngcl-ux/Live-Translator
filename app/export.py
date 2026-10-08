"""TXT ve DOCX dışa aktarma. DOCX ek kütüphane olmadan (zip + XML) üretilir."""
import datetime as _dt
import zipfile
from xml.sax.saxutils import escape

from engine import lang_name, lang_short


def _ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def build_lines(results, include_source: bool, include_time: bool):
    """[(zaman satırı veya None, çeviri, '(XX) kaynak metin' veya None)]"""
    rows = []
    for r in results:
        head = f"[{_ts(r.t_start)} - {_ts(r.t_end)}]" if include_time else None
        src = f"({lang_short(r.src)}) {r.source}" if include_source else None
        rows.append((head, r.target, src))
    return rows


def default_title(results=()):
    pairs = {(r.src, r.tgt) for r in results}
    if len(pairs) == 1:
        s, t = next(iter(pairs))
        head = f"{lang_name(s)} → {lang_name(t)} çeviri"
    else:
        head = "Çeviri"
    return head + " — " + _dt.datetime.now().strftime("%d.%m.%Y %H:%M")


def save_txt(path, results, include_source=False, include_time=True, title=None):
    title = title or default_title(results)
    lines = [title, ""]
    for head, target, source in build_lines(results, include_source, include_time):
        if head:
            lines.append(head)
        lines.append(target)
        if source:
            lines.append(source)
        lines.append("")
    # BOM: Windows Not Defteri'nde Türkçe/Romence karakterlerin doğru görünmesi için
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(lines))


_CT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def _p(text, bold=False, italic=False, size=None, color=None, space_after=None):
    rpr = ""
    if bold:
        rpr += "<w:b/>"
    if italic:
        rpr += "<w:i/>"
    if color:
        rpr += f'<w:color w:val="{color}"/>'
    if size:
        rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    ppr = f'<w:pPr><w:spacing w:after="{space_after}"/></w:pPr>' if space_after is not None else ""
    rpr_xml = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    run = f'<w:r>{rpr_xml}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    return f"<w:p>{ppr}{run}</w:p>"


def save_docx(path, results, include_source=False, include_time=True, title=None):
    title = title or default_title(results)
    body = [_p(title, bold=True, size=32, space_after=240)]
    for head, target, source in build_lines(results, include_source, include_time):
        if head:
            body.append(_p(head, size=18, color="808080", space_after=0))
        body.append(_p(target, size=24, space_after=60 if source else 200))
        if source:
            body.append(_p(source, italic=True, size=20, color="666666", space_after=200))
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           "<w:body>" + "".join(body) +
           '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
           'w:header="709" w:footer="709" w:gutter="0"/></w:sectPr>'
           "</w:body></w:document>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/document.xml", doc)
