"""Desktop entry point for EchoSight 2.0."""

from __future__ import annotations

import sys

from . import __version__


def main() -> int:
    if "--version" in sys.argv[1:]:
        print(f"EchoSight {__version__}")
        return 0
    if "--diagnostics" in sys.argv[1:]:
        from .diagnostics import main as diagnostics_main

        return diagnostics_main()

    from .desktop import main as desktop_main

    return desktop_main()

if __name__ == "__main__":
    raise SystemExit(main())
