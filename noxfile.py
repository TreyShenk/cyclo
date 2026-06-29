import nox

# numpy 2.0+ supports 3.10+; numpy 2.5+ (latest) requires 3.12+.
# Each version gets the newest numpy/scipy wheels available for it.
# Add "3.15" (and beyond) here as new Python releases gain wheel support.
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

nox.options.default_venv_backend = "uv"


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("pytest")
    session.install(".")
    session.run("pytest", "tests/", *session.posargs)
