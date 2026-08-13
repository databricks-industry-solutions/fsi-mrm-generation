#!/usr/bin/env python
"""Backwards-compatible entry point.

The CLI now lives in ``databricks/cli.py`` and is exposed as the ``mrm-generate``
console script (see pyproject.toml). This shim preserves the historical
``python databricks.py ...`` invocation.
"""
from databricks.cli import main

if __name__ == '__main__':
    main()
