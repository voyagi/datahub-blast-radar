"""Make the package importable when pytest runs outside the project venv.

`uv run pytest` gets an editable install and needs none of this. A bare
`pytest` - from a CI step, a git hook, or a reviewer who just cloned the repo -
does not, and a src layout is not on sys.path by default. Adding it here means
the suite runs the same way from anywhere.
"""

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"

if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
