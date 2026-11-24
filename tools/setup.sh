#!/usr/bin/env bash
# Local git config this project relies on (git does not version it): run once after `git init` or `git clone`.
cd "$(dirname "$0")/.." || exit 1
git config core.hooksPath .githooks
# the .venv interpreter (ISS-018): on Magerit there is no `python` outside the venv and the system python3 is 3.6
if [ -x .venv/bin/python ]; then py=.venv/bin/python; else py=python; fi
git config diff.ipynb.textconv "$py tools/nb_textconv.py"
echo "OK: pre-commit hook activo y diff de notebooks configurado."
