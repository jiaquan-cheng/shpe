# Architecture

CLI and the language server both parse Python, run `Checker` on the AST, and format `Diagnostics`. One file (or editor buffer) at a time.

```text
source
  -> ast.parse
  -> Checker.visit
       Extractor   leaves (literals, annotation tuples, slice lengths)
       Resolver    routes names, calls, binops, attributes, subscripts
       Registry    combines call/attr/binop handler tables
         └─ NumPyRegistry: callable NumPy operation handlers
       Environment nested scopes (shapes, scalars, function defs)
       Diagnostics errors, warnings, inlay hints
  -> CLI strings  or  LSP Diagnostic / InlayHint
```

| File | Role |
|------|------|
| [src/shpe/checker.py](../src/shpe/checker.py) | `ast.NodeVisitor`: assigns, functions, control-flow tainting |
| [src/shpe/extractor.py](../src/shpe/extractor.py) | Shapes and dims from leaf AST nodes |
| [src/shpe/resolver.py](../src/shpe/resolver.py) | Expression dispatch, user-function bind/walk |
| [src/shpe/handlers/registry.py](../src/shpe/handlers/registry.py) | Builds the operation registry from handler families |
| [src/shpe/handlers/numpy_handlers.py](../src/shpe/handlers/numpy_handlers.py) | Callable NumPy operation handlers and their maps |
| [src/shpe/handlers/handlers_utils.py](../src/shpe/handlers/handlers_utils.py) | Shared handler base, resolver protocol, and shape helpers |
| [src/shpe/environment.py](../src/shpe/environment.py) | Scope stack, unknown shapes |
| [src/shpe/diagnostics.py](../src/shpe/diagnostics.py) | Errors, warnings, inlay hints |
| [src/shpe/cli.py](../src/shpe/cli.py) | File discovery, `# shpe: ignore`, `--show-shapes` |
| [src/shpe/server.py](../src/shpe/server.py) | Language server (`python -m shpe.server`) |
| [src/shpe/__main__.py](../src/shpe/__main__.py) | `python -m shpe` → CLI |

Tests: [tests/test_checker.py](../tests/test_checker.py), snippets in [tests/checker_cases/](../tests/checker_cases/). Editor: [shpe-vscode/](../shpe-vscode/).
