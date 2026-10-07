"""Include maintained examples and source downloads; check the documentation."""
from pathlib import Path
import re
import sys
import tomllib

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from docs_contract import EXAMPLES, audit_docs, render_markdown


def on_config(config):
    # The version and the supported Mojo series come from pixi.toml; the theme
    # shows both.
    pixi = tomllib.loads((ROOT / 'pixi.toml').read_text())
    mojo = re.match(r'[=<>~^ ]*(\d+\.\d+)', pixi['dependencies']['mojo'])
    config.extra['version'] = pixi['workspace']['version']
    config.extra['mojo'] = mojo[1]
    return config


def on_pre_build(config):
    audit_docs()


def on_files(files, config):
    # Theme assets are static files of the custom theme directory; MkDocs copies them.
    sources = list((ROOT / 'src/apn_mojo').rglob('*.mojo')) + list(EXAMPLES.rglob('*.mojo'))
    for path in sorted(sources):
        files.append(File.generated(config, 'downloads/' + path.relative_to(ROOT).as_posix(), abs_src_path=str(path)))
    return files


def on_page_markdown(markdown, page, config, files):
    return render_markdown(markdown, page.file.src_uri)
