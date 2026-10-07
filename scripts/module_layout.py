"""Public packages, their exported names and the modules that declare them.

Paths are relative to src/apn_mojo. A package's public names are exactly what
its `__init__.mojo` imports. A name belongs to the first subpackage in
PUBLIC_PACKAGES that exports it; names only the top-level package exports
belong to `apn_mojo`, and the top level lists the rest as re-exports.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src/apn_mojo'


def _appledouble_files(root):
    """macOS writes binary "._name" companions when it copies files to a
    non-Apple volume. `mojo doc` and every source or document glob would read
    them, so the tooling refuses to run rather than build a wrong inventory."""
    found = []
    for folder, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d not in ('.git', '.pixi', '.cache', 'build', '__pycache__')]
        found += [Path(folder, name) for name in files if name.startswith('._')]
    return found


if _stray := _appledouble_files(ROOT):
    raise SystemExit(f'{len(_stray)} macOS AppleDouble files (for example {_stray[0].relative_to(ROOT)}) '
                     "would be read as project files; remove them with: find . -name '._*' -not -path './.pixi/*' -delete")
TOP = 'apn_mojo'
PUBLIC_PACKAGES = (TOP, 'integer', 'rational', 'float', 'complex', 'exact_complex', 'ball', 'complex_ball', 'batch', 'common')
_IMPORT = re.compile(r'^from (\.[\w.]*) import (\([^)]*\)|[^\n]*)', re.M)
_DECLARATION = r'^(?:def|struct|trait|comptime) `?{}\b'


def _module_path(init, relative):
    """The module a relative import in `init` names, as a path under src/apn_mojo."""
    base = init.parent.relative_to(SRC).parts
    return '/'.join((*base, *relative.lstrip('.').split('.')))


def package_exports(package):
    """{name: module} for the names a package's `__init__.mojo` imports."""
    init = SRC / '__init__.mojo' if package == TOP else SRC / package / '__init__.mojo'
    exports = {}
    for module, names in _IMPORT.findall(init.read_text()):
        for name in re.findall(r'\w+', names):
            exports[name] = _module_path(init, module)
    return exports


def provider(module, name):
    """The module that declares `name`, following re-exports from `module`."""
    path = SRC / f'{module}.mojo'
    text = path.read_text()
    if re.search(_DECLARATION.format(re.escape(name)), text, re.M):
        return module
    for source, names in _IMPORT.findall(text):
        if name in re.findall(r'\w+', names):
            return provider(_module_path(path, source), name)
    raise ValueError(f'No declaration of {name} reachable from {module}')


def public_names():
    """{package.name: provider} for every public name. Each subpackage owns
    its exports; the top level owns names no subpackage exports from the same
    declaration, such as the dispatching `add`."""
    result = {}
    declared = {}
    for package in PUBLIC_PACKAGES[1:]:
        for name, module in package_exports(package).items():
            source = provider(module, name)
            result[f'{package}.{name}'] = source
            declared.setdefault((name, source), package)
    for name, module in package_exports(TOP).items():
        source = provider(module, name)
        if (name, source) not in declared:
            result[f'{TOP}.{name}'] = source
    return dict(sorted(result.items()))


def top_level_reexports():
    """{name: owning subpackage} for the top-level names re-exported from one."""
    declared = {}
    for package in PUBLIC_PACKAGES[1:]:
        for name, module in package_exports(package).items():
            declared.setdefault((name, provider(module, name)), package)
    result = {}
    for name, module in package_exports(TOP).items():
        owner = declared.get((name, provider(module, name)))
        if owner:
            result[name] = owner
    return result
