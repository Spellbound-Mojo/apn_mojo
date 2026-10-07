"""The reviewed inventory of public names, signatures and providers."""
import hashlib
import json

from declaration_contract import declarations, public_types
from module_layout import PUBLIC_PACKAGES, ROOT, public_names

INVENTORY = ROOT / 'docs/public-exports.json'


def export_inventory():
    operations = declarations()
    types = public_types()
    providers = public_names()
    exports = {}
    for name, headers in types.items():
        exports[name] = dict(kind='type', declarations=headers)
    for name, overloads in operations.items():
        package, rest = name.split('.', 1)
        if '.' not in rest:
            exports[name] = dict(kind='function', overloads=overloads)
        else:
            owner, method = rest.split('.', 1)
            exports[f'{package}.{owner}'].setdefault('methods', {})[method] = overloads
    missing = set(providers) - set(exports)
    if missing:
        raise ValueError('Public names without a declaration: ' + ', '.join(sorted(missing)))
    for name, value in exports.items():
        value['provider'] = providers[name]
    return dict(version=1, packages=list(PUBLIC_PACKAGES), exports=dict(sorted(exports.items())))


def audit_exports():
    """Fail on any difference from the reviewed inventory."""
    expected = json.loads(INVENTORY.read_text())
    if expected != export_inventory():
        raise ValueError('Unreviewed public export, provider or signature change; review '
                         'export_contract.export_inventory() and update docs/public-exports.json')
    return dict(exports=len(expected['exports']),
                sha256=hashlib.sha256(INVENTORY.read_bytes()).hexdigest())


if __name__ == '__main__':
    print(json.dumps(export_inventory(), indent=2))
