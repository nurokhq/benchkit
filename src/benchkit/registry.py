"""Benchmark discovery.

Benchmarks live beside the engine, one directory each, and are loaded by key on
demand. Registration is a plain mapping rather than an entry-point plugin system:
with a handful of benchmarks, an explicit list is easier to read and debug than
discovery magic, and it keeps the failure mode obvious when a key is wrong.
"""

import importlib.util
import sys
from pathlib import Path

#: key -> module path relative to the repository's `benchmarks/` directory.
REGISTRY = {
    "us-startup-programs": "us-startup-programs/benchmark.py",
}


def benchmarks_dir() -> Path:
    """The repository's `benchmarks/` directory.

    Derived from this file's location, so it works from a source checkout without any
    packaging step: <repo>/src/benchkit/registry.py -> <repo>/benchmarks.
    """
    return Path(__file__).resolve().parents[2] / "benchmarks"


def available() -> list[str]:
    return sorted(REGISTRY)


def load_benchmark(key: str):
    """Import and instantiate a benchmark by key.

    The module is loaded from its file path rather than as a package, because
    benchmark directories are not importable Python package names (they contain
    dashes) and making them so would force a layout choice on every new benchmark.
    """
    if key not in REGISTRY:
        raise SystemExit(f"unknown benchmark {key!r}; known: {', '.join(available())}")
    path = benchmarks_dir() / REGISTRY[key]
    if not path.is_file():
        raise SystemExit(f"benchmark module missing: {path}")
    module_name = f"benchkit_benchmark_{key.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load benchmark module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing so a benchmark can import its own sibling modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    factory = getattr(module, "build", None) or getattr(module, "Benchmark", None)
    if factory is None:
        raise SystemExit(f"{path} must expose build() or Benchmark")
    return factory() if callable(factory) else factory
