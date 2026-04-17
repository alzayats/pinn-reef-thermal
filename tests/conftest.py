"""Shared fixtures and CPU-enforcing configuration for the test suite."""

from __future__ import annotations

import os

# Force CPU before JAX is imported anywhere in the package. CI is CPU-only.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
