#!/usr/bin/env bash
#
# Installs the two NATIVE binaries this project needs on macOS. These are NOT
# Python packages and cannot be installed by uv/pip:
#
#   * graphviz     -> provides `dot`, used by the `graphviz` python package to
#                     render model signature / lineage diagrams.
#   * wkhtmltopdf  -> used by the `pdfkit` python package to turn HTML into PDF.
#
# graphviz is a normal Homebrew formula. wkhtmltopdf's Homebrew formula/cask was
# REMOVED after the upstream project was archived, so we install its official
# macOS .pkg directly. (The .pkg is an Intel build; on Apple Silicon it runs via
# Rosetta 2 — install that first with `softwareupdate --install-rosetta` if
# needed.)
#
# Usage:  ./install-binaries-macos.sh
set -euo pipefail

WKHTMLTOPDF_PKG_URL="https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-2/wkhtmltox-0.12.6-2.macos-cocoa.pkg"

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

echo "==> Installing wkhtmltopdf"
if command -v wkhtmltopdf >/dev/null 2>&1; then
  echo "    wkhtmltopdf already installed: $(wkhtmltopdf --version 2>&1 | head -1)"
else
  tmp_pkg="$(mktemp -t wkhtmltox).pkg"
  echo "    Downloading $WKHTMLTOPDF_PKG_URL"
  curl -fL "$WKHTMLTOPDF_PKG_URL" -o "$tmp_pkg"
  echo "    Installing (requires sudo)"
  sudo installer -pkg "$tmp_pkg" -target /
  rm -f "$tmp_pkg"
fi

echo "==> Verifying"
dot -V
wkhtmltopdf --version
echo "==> Native binaries ready."
