"""Reviewed public declarations: functions, types, their methods and constructors."""
import hashlib
import re

from module_layout import SRC, public_names

# A constructor overload is public unless it is the copy/move machinery or
# takes a private keyword (`_rounded=`, `_validated=`).
_PRIVATE_CONSTRUCTOR = re.compile(r'\b(?:copy\s*:|deinit\b|_\w+\s*:)')


def _header(lines, start):
    """A declaration header from its first line to the colon that ends it."""
    header = []
    for line in lines[start:]:
        header.append(line.strip())
        if line.rstrip().endswith(':'):
            break
    return ' '.join(header)


def _public_method(name, header):
    if name == '__init__':
        arguments = header[header.find('('):]
        return not _PRIVATE_CONSTRUCTOR.search(arguments)
    return not name.startswith('_')


def declaration_sites():
    """Every public signature: (operation, signature) in source order."""
    for full, module in public_names().items():
        package, name = full.split('.', 1)
        lines = (SRC / f'{module}.mojo').read_text().splitlines()
        owner = None
        for i, line in enumerate(lines):
            found = re.match(r'^(?:struct|trait) (\w+)', line)
            if found:
                owner = found[1]
                continue
            if re.match(r'^(?:def|comptime|@)', line) and not line.startswith('@'):
                owner = None
            method = re.match(r'(    )?def `?(\w+)`?', line)
            if not method:
                continue
            if method[1]:
                if owner != name or not _public_method(method[2], _header(lines, i)):
                    continue
                yield f'{full}.{method[2]}', _header(lines, i)
            elif method[2] == name:
                yield full, _header(lines, i)


def declarations():
    result = {}
    for operation, signature in declaration_sites():
        result.setdefault(operation, []).append(
            dict(signature=signature, sha256=hashlib.sha256(signature.encode()).hexdigest()))
    return result


def public_types():
    """{package.Name: [declaration line]} for the public structs and traits."""
    result = {}
    for full, module in public_names().items():
        name = full.split('.', 1)[1]
        for line in (SRC / f'{module}.mojo').read_text().splitlines():
            if re.match(r'^(?:struct|trait) ' + re.escape(name) + r'\b', line):
                result.setdefault(full, []).append(line.strip())
    return result
