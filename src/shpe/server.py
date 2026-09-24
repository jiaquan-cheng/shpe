import ast
import urllib.parse
from pathlib import Path
from typing import Any

from lsprotocol.types import (
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    TEXT_DOCUMENT_INLAY_HINT,
    Diagnostic,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    InlayHint,
    InlayHintKind,
    InlayHintParams,
    Position,
    PublishDiagnosticsParams,
    Range,
)
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument

from shpe.checker import Checker

server = LanguageServer("shpe-ls", "v0.1")


def _is_dotfile(uri: str) -> bool:
    """Check if the document filename starts with a dot."""
    parsed_path = urllib.parse.urlparse(uri).path
    return Path(urllib.parse.unquote(parsed_path)).name.startswith(".")


def validate(ls: LanguageServer, document: TextDocument) -> None:

    if _is_dotfile(document.uri):
        ls.text_document_publish_diagnostics(
            PublishDiagnosticsParams(uri=document.uri, diagnostics=[])
        )
        return
    code = document.source
    lines = code.splitlines()

    try:
        tree = ast.parse(code, filename=document.uri)
    except SyntaxError as e:
        lineno = max(0, (e.lineno or 1) - 1)
        offset = e.offset or 0

        diag = Diagnostic(
            range=Range(
                start=Position(line=lineno, character=offset),
                end=Position(line=lineno, character=offset + 1),
            ),
            message=str(e.msg),
            severity=DiagnosticSeverity.Error,
            source="shpe",
        )
        ls.text_document_publish_diagnostics(
            PublishDiagnosticsParams(uri=document.uri, diagnostics=[diag])
        )
        return

    checker = Checker()
    checker.visit(tree)

    diagnostics = []

    def add_diagnostics(
        entries: list[dict[str, Any]], severity: DiagnosticSeverity
    ) -> None:
        for entry in entries:
            line_idx = max(0, entry["line"] - 1)

            if line_idx < len(lines):
                line_content = lines[line_idx]
                if "# shpe: ignore" in line_content:
                    continue

            col = max(0, entry["col"])
            end_col = max(0, entry["end_col"])

            diag = Diagnostic(
                range=Range(
                    start=Position(line=line_idx, character=col),
                    end=Position(line=line_idx, character=end_col),
                ),
                message=entry["message"],
                severity=severity,
                source="shpe",
                code=entry.get("code"),
            )
            diagnostics.append(diag)

    add_diagnostics(checker.errors, DiagnosticSeverity.Error)
    add_diagnostics(checker.warnings, DiagnosticSeverity.Warning)

    ls.text_document_publish_diagnostics(
        PublishDiagnosticsParams(uri=document.uri, diagnostics=diagnostics)
    )


@server.feature(TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams) -> None:
    text_document = ls.workspace.get_text_document(params.text_document.uri)
    validate(ls, text_document)


@server.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: DidChangeTextDocumentParams) -> None:
    text_document = ls.workspace.get_text_document(params.text_document.uri)
    validate(ls, text_document)


@server.feature(TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: DidSaveTextDocumentParams) -> None:
    text_document = ls.workspace.get_text_document(params.text_document.uri)
    validate(ls, text_document)


@server.feature(TEXT_DOCUMENT_INLAY_HINT)
def inlay_hints(ls: LanguageServer, params: InlayHintParams) -> list[InlayHint]:
    """Provides inlay hints to display inferred NumPy shapes in the editor."""
    text_document = ls.workspace.get_text_document(params.text_document.uri)
    code = text_document.source
    try:
        tree = ast.parse(code, filename=text_document.uri)
    except SyntaxError:
        return []

    checker = Checker()
    checker.visit(tree)

    hints = []
    for h in checker.inlay_hints:
        line = max(0, h["line"] - 1)
        col = max(0, h["col"])
        shape_str = str(h["shape"])

        hint = InlayHint(
            position=Position(line=line, character=col),
            label=f"   {shape_str}",
            kind=InlayHintKind.Type,
            padding_left=True,
        )
        hints.append(hint)

    return hints


if __name__ == "__main__":
    server.start_io()
