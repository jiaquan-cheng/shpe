# Architecture

Shpe parses Python into an AST and analyzes it with `Checker`. The CLI and
language server use the same checker and inference components, then format the
collected results for their respective interfaces.

```text
CLI: paths -> discover files -> parse each file ─┐
                                                 ├-> Checker.visit(AST)
LSP: document event -> parse document ──────────┘       │
                                                        ├─ Extractor
                                                        ├─ Resolver
                                                        │    └─ Registry / operation handlers
                                                        ├─ Environment
                                                        └─ Diagnostics
                                                             │
                                    CLI report / LSP diagnostics and hints
```

Each CLI file and each LSP document analysis gets a fresh `Checker`. The
environment is not shared across files or editor requests.

## Components

| File | Role |
|------|------|
| [src/shpe/checker.py](../src/shpe/checker.py) | AST visitor for assignments, definitions, imports, and conservative control-flow handling; coordinates inference and state updates |
| [src/shpe/extractor.py](../src/shpe/extractor.py) | Reads shapes and dimension values from literals, scalar expressions, annotations, and slices |
| [src/shpe/resolver.py](../src/shpe/resolver.py) | Infers expression shapes, dispatches operations, and analyzes user-defined functions and class methods |
| [src/shpe/handlers/handlers.py](../src/shpe/handlers/handlers.py) | `Registry` and handlers for supported NumPy-style calls, attributes, and binary operators |
| [src/shpe/handlers/handlers_utils.py](../src/shpe/handlers/handlers_utils.py) | Shared handler dependencies, resolver protocol, call-target and broadcasting helpers |
| [src/shpe/environment.py](../src/shpe/environment.py) | Shape, scalar, and abstract-object values plus the nested scope stack |
| [src/shpe/diagnostics.py](../src/shpe/diagnostics.py) | Collects errors, warnings, and inlay hints with AST source positions |
| [src/shpe/cli.py](../src/shpe/cli.py) | Discovers files, honors `# shpe: ignore`, prints diagnostics and `--show-shapes` reports |
| [src/shpe/server.py](../src/shpe/server.py) | Language server entry point; publishes diagnostics and serves inlay-hint requests |
| [src/shpe/__main__.py](../src/shpe/__main__.py) | Routes `python -m shpe` to the CLI |

## Values and scopes

`Shape` stores a tuple of integer dimensions. A `Tensor` wraps either a known
`Shape` or a `ShapeState` (`UNKNOWN` or `UNDEFINED`). `ScalarValue` stores an
integer, float, or `ValueState`. `AbstractObject` refers to a class AST node and
stores inferred attributes, which may be tensors, scalars, nested abstract
objects, or value states.

An `Environment` is a stack of frames. Each frame has separate maps for shapes,
scalar values, object bindings, functions, imported module aliases, and classes.
`get_*` methods search from the current frame outwards; name setters write into
the current frame. The frame properties such as `env.shapes` expose only the
current frame. Object aliases point to the same `AbstractObject`, so updates
through an alias update that shared abstract instance.

There is one current overlap in the model: assignments to names are put in the
shape map even when no tensor shape can be inferred. In that case the checker
stores `ShapeState.UNKNOWN`. A known class instance is also stored in the object
map, so its name can appear in both maps and in the CLI's unknown-shape list as
well as its object report. The unknown entry is a fallback, not an inferred
tensor shape.

## Analysis flow

`Checker` visits statements and asks `Resolver` for expression shapes. The
resolver handles names, calls, binary operations, attributes, and subscripts.
It checks tracked instance methods and user-defined functions before consulting
the operation registry. The registry groups handlers for calls, attributes,
and binary operators. `Extractor` provides the leaf values those analyses need,
including dimensions from `Annotated` shapes and integer scalar values.

Assignments update the current environment. Numeric literals are recorded as
scalars; directly named class constructor calls create abstract instances;
assignments from a tracked object name create aliases. Attribute assignments
update the corresponding abstract object's attribute map. Other calls are
handled by a registered operation when supported; otherwise the resolver
returns no inferred shape.

Function definitions are registered and analyzed in a child scope. If there is
no return annotation, the body is visited to infer its return shape and collect
definition-time hints. Calls analyze the body in a separate child scope with
positional arguments and defaults bound to parameters. A return annotation can
provide the call's result shape; argument annotations are checked when an
argument shape is known. Function-call analysis suppresses body hints to avoid
adding them again. Recursive function analysis is not supported.

Class definitions are registered and their bodies are visited in a child scope.
Errors are suppressed during this definition pass, and method-body hints are
deferred because there is no instance state yet. When a tracked instance is
created, its `__init__` method is analyzed with `self` bound to that abstract
instance. Calls to tracked methods use the instance's stored attributes. Method
body hints are collected on the first initialized-instance analysis of each
class method; later calls still analyze state updates but do not add duplicate
method-body hints.

The analysis is conservative around control flow. For branches and loops, the
checker visits the body and marks assigned names and object attributes unknown
afterwards instead of merging paths. A top-level `if __name__ == "__main__"` is
visited normally.

## Diagnostics and front ends

`Diagnostics` stores errors and warnings with line and column positions, plus
inlay hints containing inferred shapes. Collection can be disabled temporarily
while analyzing class definitions or repeated function and method bodies.

The CLI processes files independently. `--show-shapes` prints tracked shapes,
scalars, and object attributes, including nested objects; `# shpe: ignore`
suppresses diagnostics on the marked source line. The language server analyzes
documents on open, change, and save, publishes errors and warnings as LSP
diagnostics, and computes shape inlay hints for hint requests.

## Tests and limits

The checker tests are in [tests/test_checker.py](../tests/test_checker.py),
with snippets grouped under [tests/checker_cases/](../tests/checker_cases/).
The runtime oracle compares inferred top-level tensor shapes and scalar values
with execution results and checks tracked class-instance attributes.

The analyzer supports a subset of NumPy-style operations registered in
`handlers.py`, including array creation, reshaping, reductions, transposes,
element-wise arithmetic, and matrix multiplication. It does not execute user
code. It does not resolve project imports across files, handle recursion, or
compute path-sensitive control-flow merges. Unsupported expressions may
therefore remain unknown or untracked.
