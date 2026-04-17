"""Customise pdoc's generated HTML titles.

pdoc uses the bare Python package name (``pinn_reef_thermal``) for both the
browser tab title and the top-of-page H1 on the index page. That reads as a
package slug rather than describing the project, so after pdoc finishes we
rewrite those two strings on the package index page to match the paper title.
Submodule pages keep their fully-qualified module names unchanged.

Invoked from ``.github/workflows/docs.yml`` as ``python docs/postprocess.py site``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TITLE = "Depth-Resolved Coral Reef Thermal Fields"
SUBTITLE = (
    "from Satellite SST and Sparse In-Situ Loggers Using Physics-Informed Neural Networks"
)
TAB_TITLE = f"{TITLE} - pinn-reef-thermal"


def postprocess(site_dir: Path) -> int:
    """Rewrite the package index page's tab title and H1. Return number of files touched."""
    touched = 0
    for html_file in site_dir.rglob("*.html"):
        original = html_file.read_text(encoding="utf-8")
        text = original

        # Browser tab title on the package index.
        text = text.replace(
            "<title>pinn_reef_thermal API documentation</title>",
            f"<title>{TAB_TITLE}</title>",
        )

        # H1 on the package index page. pdoc emits the module name verbatim
        # inside "<h1 class='modulename'>...</h1>" with whitespace; the regex
        # matches that exact shape but leaves submodule pages (where the H1
        # contains a dotted name) untouched.
        text = re.sub(
            r'(<h1 class="modulename">)\s*pinn_reef_thermal\s*(</h1>)',
            (
                r'\1'
                f'{TITLE}<br>'
                f'<small style="font-size:0.58em;color:var(--muted);'
                f'font-weight:300;line-height:1.4;display:inline-block;'
                f'margin-top:0.3em;">{SUBTITLE}</small>'
                r'\2'
            ),
            text,
        )

        if text != original:
            html_file.write_text(text, encoding="utf-8")
            touched += 1
            print(f"Rewrote titles in {html_file}")

    return touched


def main() -> int:
    site_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    if not site_dir.is_dir():
        print(f"error: {site_dir} is not a directory", file=sys.stderr)
        return 2
    touched = postprocess(site_dir)
    if touched == 0:
        print("warning: no HTML files were rewritten; pdoc output shape may have changed",
              file=sys.stderr)
        return 1
    print(f"postprocess: rewrote {touched} file(s) under {site_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
