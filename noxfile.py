from __future__ import annotations
import nox

# Default sessions when you run plain `nox`
nox.options.sessions = ["lint", "tests"]
# Speed up local iteration
nox.options.reuse_venv = True

PY_VERS = ["3.10", "3.11", "3.12"]


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff>=0.5.0", "black>=24.0")
    session.run("ruff", "check", ".")
    session.run("black", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[fast]")
    session.install("mypy>=1.5")
    session.run("mypy", "dual_clocking", "dualdrive")


@nox.session(python=PY_VERS)
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev,fast]")
    session.run("pytest", "-q")


@nox.session(python=PY_VERS)
def bench(session: nox.Session) -> None:
    """Run micro/macro benchmarks (pytest-benchmark)."""
    session.install("-e", ".[dev,fast]")
    # If you keep benchmarks under tests/benchmarks, this will pick them up.
    session.run(
        "pytest",
        "--benchmark-only",
        "--benchmark-min-time=0.01",
        "-q",
    )


@nox.session
def build(session: nox.Session) -> None:
    """Produce wheel and sdist."""
    session.install("build>=1.2.1")
    session.run("python", "-m", "build", "--wheel", "--sdist")
