"""Collect verified submission documents without research data or code archives."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

import fitz
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '00_REVISION_UPLOAD'


def main():
    receipt = json.loads((ROOT / 'latest_light_delivery.json').read_text(encoding='utf-8'))
    archive_path = Path(receipt['zip'])
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == receipt['sha256']
    OUT.mkdir(exist_ok=True)
    files = {
        'Manuscript_Revised_Clean.pdf': '01_Manuscripts/',
        'Manuscript_Revised_Marked.pdf': '01_Manuscripts/',
        'Response_to_Associate_Editor_and_Reviewers.pdf': '02_Response/',
        'Cover_Letter_Revised.pdf': '05_Submission_Files/',
        'Highlights_Revised.docx': '05_Submission_Files/',
    }
    with zipfile.ZipFile(archive_path) as archive:
        for name, prefix in files.items():
            payload = archive.read(prefix + name)
            assert payload == (ROOT / name).read_bytes(), name
            (OUT / name).write_bytes(payload)
        prefix = '03_LaTeX_Source/'
        main_name = 'Manuscript_Revised_Clean.tex'
        manuscript = archive.read(prefix + main_name).decode('utf-8')
        sources = {main_name, 'Manuscript_Revised_Clean.bbl',
                   'cas-dc.cls', 'cas-common.sty', 'revision-num.bst',
                   'GAT_SAC_PTDF_LODF_SCED_Restoration.bib', 'revision_references.bib'}
        sources.update(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', manuscript))
        sources.update(n.removeprefix(prefix) for n in archive.namelist()
                       if n.startswith(prefix + 'thumbnails/'))
        assert not re.search(r'\\input\{revision_', manuscript)
        with zipfile.ZipFile(OUT / 'Manuscript_Source.zip', 'w', zipfile.ZIP_DEFLATED) as output:
            for name in sorted(sources):
                assert '..' not in Path(name).parts
                output.writestr(name, archive.read(prefix + name))
            output.writestr('COMPILE.txt',
                'Main file: Manuscript_Revised_Clean.tex\n'
                'Compile: latexmk -pdf -interaction=nonstopmode -halt-on-error Manuscript_Revised_Clean.tex\n'
                'This archive contains the clean manuscript and its required assets only.\n')
        figure = archive.read(prefix + 'figures/framework_architecture.pdf')
        assert figure == (ROOT / 'figures/framework_architecture.pdf').read_bytes()
        (OUT / 'Graphical_Abstract.pdf').write_bytes(figure)
        source_folder = OUT / 'LaTeX_Source'
        editable = sources | {'Manuscript_Revised_Marked.tex',
                              'Manuscript_Revised_Marked.bbl',
                              'Response_to_Associate_Editor_and_Reviewers.tex'}
        for name in sorted(editable):
            target = source_folder / name
            assert target.resolve().is_relative_to(source_folder.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(prefix + name))
        (source_folder / 'Cover_Letter_Revised.tex').write_bytes(
            archive.read('05_Submission_Files/Cover_Letter_Revised.tex'))
        (source_folder / 'COMPILE.txt').write_text(
            'Editable LaTeX sources corresponding to the PDFs in the parent folder.\n'
            'The response embeds checked manuscript numbers/pages and can compile independently.\n'
            'No manuscript .aux file is required for the response.\n\n'
            'latexmk -pdf Manuscript_Revised_Marked.tex\n'
            'latexmk -pdf Manuscript_Revised_Clean.tex\n'
            'latexmk -pdf Response_to_Associate_Editor_and_Reviewers.tex\n'
            'latexmk -pdf Cover_Letter_Revised.tex\n\n'
            'Manuscript_Source.zip in the parent folder contains only the clean manuscript sources for submission.\n',
            encoding='utf-8')

    for name in files:
        if name.endswith('.pdf'):
            with fitz.open(OUT / name) as pdf:
                assert not pdf.is_encrypted and len(pdf) > 0
                assert all('??' not in page.get_text() for page in pdf), name
    highlights = [p.text for p in Document(OUT / 'Highlights_Revised.docx').paragraphs if p.text][1:]
    assert len(highlights) == 4 and all(len(text) <= 85 for text in highlights)
    with zipfile.ZipFile(OUT / 'Manuscript_Source.zip') as archive:
        assert archive.testzip() is None
        assert all('\\' not in name for name in archive.namelist())
    instructions = '''返修上传文件（本目录仅含当前版本）

Manuscript_Revised_Clean.pdf：干净修改稿。
Manuscript_Revised_Marked.pdf：标蓝修改稿。
Response_to_Associate_Editor_and_Reviewers.pdf：给AE及审稿人的逐条回复。
Cover_Letter_Revised.pdf：返修附信。
Highlights_Revised.docx：可编辑亮点，四条。
Manuscript_Source.zip：干净修改稿的LaTeX源文件、参考文献、图片及模板。
Graphical_Abstract.pdf：与当前正文图1完全相同的图形摘要，按系统栏目选择上传。

本说明不要上传。请按系统实际文件类型分别选择上述文件，不要上传整个工作目录。
源文件压缩包主文件为 Manuscript_Revised_Clean.tex，不含实验数据、模型或运行代码。
如系统要求单独的利益冲突声明，请使用作者确认或系统生成的文件；本目录不冒充已生成该声明。
其他目录是生成文稿、复核结果和保留旧版本所需的工作材料，不属于本次上传文件。
'''
    instructions += '\nLaTeX_Source：可直接打开的干净稿、标蓝稿、回复信和Cover Letter的.tex文件，附图片、参考文献、模板及编译顺序。\n'
    (OUT / '00_READ_BEFORE_UPLOAD.txt').write_text(instructions, encoding='utf-8-sig')
    manifest = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in OUT.rglob('*') if p.is_file()}
    (ROOT / 'submission_upload_manifest.json').write_text(json.dumps({
        'source_delivery': str(archive_path), 'files': manifest,
        'data_models_included': False, 'graphical_abstract_matches_figure_1': True,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'folder': str(OUT), 'files': sorted(manifest)}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
