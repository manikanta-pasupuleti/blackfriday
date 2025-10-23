"""Top-level runner for the backend app.

Allows running the project with `python app.py` from the repository root
by delegating execution to `backend/app.py` (so its `if __name__ == '__main__'`
block runs as intended).
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    repo_root = os.path.dirname(__file__)
    backend_script = os.path.join(repo_root, 'backend', 'app.py')
    if not os.path.exists(backend_script):
        print(f"Could not find backend script at: {backend_script}")
        sys.exit(2)

    # Execute backend/app.py as if it were run directly so that its
    # `if __name__ == '__main__'` block executes.
    sys.argv[0] = backend_script
    runpy.run_path(backend_script, run_name='__main__')


if __name__ == '__main__':
    main()
