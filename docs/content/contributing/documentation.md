# Writing and maintaining the docs

All documentation for the project should reside within this website to
avoid searching for loose notes throughout the repo. Keep user guidance,
API contracts, design decisions, and performance records in this site.
When behavior changes, update the pages that explain it. AI-generated
documentation is allowed as long it is concise, informative, and
accurate.


## Build and review

Run documentation commands from the directory containing `pixi.toml`:

```sh
pixi run --locked -e docs docs-serve
pixi run --locked -e docs docs-check
pixi run --locked docs-test  # Linux
```

Use `docs-serve` to preview the site locally. `docs-check` audits the content
and API inventory, builds the site in strict mode, and checks local links,
anchors, assets, and the search inventory.

To verify the examples on Linux, run `docs-test`. It uses `--Werror` to compile
one driver for all maintained examples, runs it, and compares the output with
`docs/examples.json`. These checks are separate from the functional test
suites.

Choose the checks that cover your change. A successful site build verifies
the documentation structure; example compilation and numerical correctness
need their own checks. For layout or theme changes, also review desktop and
mobile rendering, search, keyboard navigation, and both color themes.

The `docs` environment pins MkDocs and Pygments. Mojo comes from the default
environment.

## Where things live

| Path | Purpose |
|---|---|
| `docs/content/` | Handwritten pages |
| `docs/examples/`, `docs/examples.json` | Maintained programs and expected output |
| `docs/api.json`, `docs/public-exports.json` | Extracted docstrings and public symbol inventory |
| `mkdocs.yml` | Navigation, site configuration, and plugins |
| `docs/theme/`, `docs/hooks.py` | Rendering, assets, directives, and downloads |
| `scripts/docs_contract.py` | Content and API audits, directive expansion |
| `scripts/check_docs.py` | Rendered link and search checks |
| `scripts/verify_docs_examples.py` | Example compilation and output checks |
| `scripts/api_docs.py`, `scripts/api_reference.py` | Docstring extraction and API rendering |
| `scripts/module_layout.py`, `scripts/export_contract.py` | Module and export inventories |

## Mathematics

Use LaTeX notation for equations: `$x^2$` for inline math, and `$$` on separate
lines for a displayed equation. Leave a blank line before and after a display
block:

```text
$$
\frac{a}{b} + \frac{c}{d} = \frac{ad + bc}{bd}
$$
```

The Markdown extension preserves the notation, and MathJax renders it in the
browser. Math inside code spans and fenced blocks stays literal. MathJax loads
from a CDN, so rendering equations requires an internet connection.

## Directives

Use HTML-comment directives to include API declarations and maintained
examples in a page:

```text
<!-- api: integer -->
<!-- example: docs/examples/first_integer.mojo -->
<!-- source: docs/examples/<file>.mojo -->
```

`api` inserts a package's generated declarations. `example` shows a program
alongside its run command and recorded output. Use `source` for an example
module that does not run as a standalone program. Inside a fenced block,
directives appear as literal text, as they do above.

## Rules the audit enforces

- Every page starts with a level-one heading.
- Mojo blocks come from `example` or `source` directives, not handwritten
  fenced Mojo code. Use a `text` fence for a signature or an abstract sketch.
- Each public package has exactly one `api` directive.
- The extracted API matches the public export inventory and has the required
  summaries, argument descriptions, and error documentation.
- Every runnable example is in the manifest and is referenced by a page.
  Recorded output ends with a newline.

The audit catches structural problems and inconsistencies. Read the prose
as well, and check its factual claims against the code.

When adding a public capability, also check its route from the homepage,
package index, tutorials, support guide, and example gallery. A generated
declaration does not replace a runnable introduction. Keep a new example in
the gallery as well as on its teaching page, and update overlapping capability
and shape descriptions together.

## Documenting an API

Edit source docstrings to change the generated API reference. Start each
function docstring with a short summary, then explain its parameters,
arguments, results, errors, and limits. Every overload needs a summary, and
at least one declaration of each function needs complete documentation.

Use type docstrings to describe operator behavior and constructor docstrings
to explain accepted inputs.

After a public API change, refresh the export inventory with
`python3 scripts/export_contract.py` and extract docstrings with
`pixi run --locked python3 scripts/api_docs.py`. Review the generated changes
and run the documentation and example checks.

## Where decisions go

Document algorithms, proofs, and measurements that explain a design choice
in the relevant architecture chapter. [Benchmark results](../architecture/benchmark-results.md)
contains the generated performance summary and full report download. Explain
workflows in [Development](development.md) or [Benchmarks](benchmarks.md).

Describe the current implementation. Historical measurements are useful when
they explain a design choice; include their setup so readers can judge where
the results apply.

## Adding an example

1. Add a self-contained program with a `def main() raises` entry point under
   `docs/examples/`.
2. Run the program and record its output in `docs/examples.json`.
3. Add an `example` directive and an entry in the [gallery](../examples/index.md).
4. Run `docs-test` and `docs-check`.

## Theme

The site's templates and styles live under `docs/theme/`. The theme uses local
assets and relative links, and supports search and both light and dark colors.
The example below uses the same directive as examples elsewhere in the site:

<!-- example: docs/examples/first_integer.mojo -->
