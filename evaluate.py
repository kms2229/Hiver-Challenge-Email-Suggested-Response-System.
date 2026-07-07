#!/usr/bin/env python
"""
evaluate.py — Top-level script for running the full evaluation pipeline.

Usage:
  uv run python evaluate.py
  uv run python evaluate.py --sample 20
  uv run python evaluate.py --output results/my_report.json

This is a thin wrapper around `src/cli.py evaluate` that makes it easy to
run without installing the package.
"""

from src.cli import evaluate

if __name__ == "__main__":
    evaluate()
