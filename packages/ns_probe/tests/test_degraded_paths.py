"""Two paths that only run when something is already wrong, and were broken.

Both were found by `ruff --select F821` rather than by anything failing, which is
the nature of them: they are the code that runs when protobuf is missing, or when
a traced agent raises. Neither is exercised in the happy path, so neither had
ever run in CI, and both referenced a name that does not exist at that point.

The `runner.py` one is the serious half. `ns-probe-run` wraps an agent with no
code changes, and its exception handler reached for a module that was imported
inside a *different function*. So the first time the agent raised anything, the
handler raised `NameError` on top of it — and the agent's real exception, the one
someone was trying to debug, was replaced by an error from the observability tool
watching it. That is the exact inversion of what this SDK promises.
"""

from __future__ import annotations

import warnings


class TestTheProtobufMissingWarning:
    """`_compat.warn_if_protobuf_needed` is the "you are on the slow path" notice."""

    def test_the_warning_path_warns_instead_of_raising(self):
        from ns_probe import _compat

        # Force the degraded branch regardless of what is installed here.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compat._warned_protobuf = False
            _compat._warn_protobuf_missing()

        assert len(caught) == 1
        assert "protobuf" in str(caught[0].message).lower()

    def test_it_warns_once_rather_than_on_every_span(self):
        """This runs on the export path. A warning per span would be a log flood."""
        from ns_probe import _compat

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compat._warned_protobuf = False
            _compat._warn_protobuf_missing()
            _compat._warn_protobuf_missing()
            _compat._warn_protobuf_missing()

        assert len(caught) == 1

    def test_the_warning_names_an_extra_that_actually_exists(self):
        """It used to advise `pip install ns_probe[collector]`. There is no
        `collector` extra — only `phi` and `dev` — so the remedy did nothing."""
        import sys
        from pathlib import Path

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        from ns_probe import _compat

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compat._warned_protobuf = False
            _compat._warn_protobuf_missing()

        message = str(caught[0].message)
        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        )
        extras = set(pyproject["project"].get("optional-dependencies", {}))

        for named in ("collector", "phi", "dev", "fast"):
            if f"[{named}]" in message:
                assert named in extras, f"the warning advises a nonexistent extra: {named}"


class TestTheRunnerExceptionHandler:
    """`ns-probe-run`'s agent wrapper, on the path where the agent fails."""

    def test_the_handler_does_not_reference_a_name_it_cannot_see(self):
        """`trace` was imported inside setup_opentelemetry(), a different
        function, so the handler could never resolve it."""
        import ast
        import inspect
        from pathlib import Path

        source = (
            Path(inspect.getfile(__import__("ns_probe.runner", fromlist=["runner"])))
        ).read_text()
        tree = ast.parse(source)

        module_level = {
            alias.asname or alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }

        # Every name used inside a function whose only binding was in another
        # function body is the bug class this covers.
        assert "trace" in module_level or "trace" in _names_imported_in_function(
            tree, "traced_execute"
        ), "runner.py reaches for `trace` with no binding it can see"


def _names_imported_in_function(tree, func_name: str) -> set:
    """Names imported inside the named nested function, at any depth."""
    import ast

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        found.add(alias.asname or alias.name.split(".")[0])
    return found
