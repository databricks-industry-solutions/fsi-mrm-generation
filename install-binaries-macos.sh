#!/usr/bin/env bash
#
# Installs the NATIVE libraries this project needs on macOS. These are NOT
# Python packages and cannot be installed by uv/pip:
#
#   * graphviz  -> provides `dot`, used by the `graphviz` python package to
#                  render model signature / lineage diagrams.
#   * pango     -> the text/graphics rendering stack (Pango, cairo, GDK-PixBuf,
#                  HarfBuzz, fontconfig) used by the `weasyprint` python package
#                  to turn HTML into PDF.
#
# Both are normal Homebrew formulae.
#
# Usage:  ./install-binaries-macos.sh
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew is required. Install it from https://brew.sh first." >&2
  exit 1
fi

echo "==> Installing graphviz (dot) via Homebrew"
if command -v dot >/dev/null 2>&1; then
  echo "    graphviz already installed: $(dot -V 2>&1)"
else
  brew install graphviz
fi

echo "==> Installing pango (WeasyPrint rendering libraries) via Homebrew"
if brew list pango >/dev/null 2>&1; then
  echo "    pango already installed: $(brew list --versions pango)"
else
  brew install pango
fi

echo "==> Verifying"
dot -V
brew list --versions pango
echo "==> Native libraries ready."
echo
echo "NOTE: on Apple Silicon, export the Homebrew library path so WeasyPrint can"
echo "      find these libraries (add to your shell profile to persist):"
echo "        export DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix)/lib"
