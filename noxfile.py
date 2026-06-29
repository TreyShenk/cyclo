import nox

# numpy 2.5.0 / scipy 1.18.0 only ship cp314 wheels, so 3.14 is the floor.
# Add "3.15" (and beyond) here as new Python releases gain wheel support.
PYTHON_VERSIONS = ["3.14"]

nox.options.default_venv_backend = "uv"


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("pytest")
    session.install(".")
    session.run("pytest", "tests/", *session.posargs)
