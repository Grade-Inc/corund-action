"""`python3 -m corund_checks --matrix` prints the generated support/detection matrix that
checks/README.md must contain verbatim."""
from __future__ import annotations

import sys

from .c2_skip_audit import render_matrix

if "--matrix" in sys.argv[1:]:
    sys.stdout.write(render_matrix())
    raise SystemExit(0)
sys.stderr.write("usage: python3 -m corund_checks --matrix\n")
raise SystemExit(2)
