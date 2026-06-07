"""Enable ``python -m taosmd`` to run the CLI.

The console-script entry point (``taosmd``) maps to :func:`taosmd.cli.main`,
but executing the package as a module (``python -m taosmd``) needs an explicit
``__main__``. Service supervisors installed by ``taosmd serve --install-service``
(systemd / launchd) invoke ``python -m taosmd serve``, so this module must exist
for the background service to start.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
