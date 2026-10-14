from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.models import ToolCall
from forge.tools import Workspace
from forge.tools.analysis import (
    SymbolIndex,
    build_import_graph,
    extract_symbols,
    is_test_path,
    language_of,
    parse_manifest,
    split_identifier,
)

from ...conftest import RuntimeFactory

PY = """
MAX_RETRIES = 3

class UserService(BaseService):
    def get_user(self, user_id: int) -> "User":
        return self.repo.find(user_id)

    async def delete(self, user_id): ...

    class Config:
        strict = True

def helper(x, *args, **kwargs):
    pass
"""

TS = """
export interface User { id: number }
export type UserId = number;
export class UserController {
  constructor(private svc: UserService) {}
  async getUser(id: string): Promise<User> {
    if (id) { return this.svc.find(id); }
  }
}
export const createApp = async (opts: Options) => {
  return new App(opts);
};
export default function bootstrap() {}
enum Color { Red }
"""

GO = """
package users

type Store struct { db *DB }
type Finder interface { Find(id int) }

func NewStore(db *DB) *Store { return &Store{db} }

func (s *Store) Find(id int) (*User, error) { return nil, nil }
"""


class TestSymbols:
    def test_python(self) -> None:
        symbols = {s.qualified_name: s for s in extract_symbols("svc.py", PY)}
        assert symbols["UserService"].kind == "class"
        assert symbols["UserService"].signature == "class UserService(BaseService)"
        method = symbols["UserService.get_user"]
        assert method.kind == "method"
        assert method.signature == "def get_user(self, user_id: int) -> 'User'"
        assert method.end_line == method.line + 1
        assert symbols["UserService.delete"].signature.startswith("async def")
        assert symbols["UserService.Config"].kind == "class"
        assert symbols["helper"].kind == "function"
        assert symbols["MAX_RETRIES"].kind == "constant"

    def test_typescript(self) -> None:
        symbols = {(s.kind, s.qualified_name) for s in extract_symbols("app.ts", TS)}
        assert ("interface", "User") in symbols
        assert ("type", "UserId") in symbols
        assert ("class", "UserController") in symbols
        assert ("method", "UserController.getUser") in symbols
        assert ("function", "createApp") in symbols
        assert ("function", "bootstrap") in symbols
        assert ("enum", "Color") in symbols
        assert not any(name.endswith(".if") for _, name in symbols)

    def test_go(self) -> None:
        symbols = {(s.kind, s.name) for s in extract_symbols("store.go", GO)}
        assert {("struct", "Store"), ("interface", "Finder"), ("function", "NewStore"),
                ("method", "Find")} <= symbols  # fmt: skip

    def test_syntax_error_falls_back_to_regex(self) -> None:
        broken = "class Broken:\n    def ok(self):\n        return (\n"
        names = {s.name for s in extract_symbols("b.py", broken)}
        assert {"Broken", "ok"} <= names

    def test_split_identifier(self) -> None:
        assert split_identifier("getUserByID") == ["get", "user", "by", "id"]
        assert split_identifier("parse_http_response2") == ["parse", "http", "response", "2"]

    def test_index_search_ranking_and_refresh(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text(PY)
        (tmp_path / "app.ts").write_text(TS)
        ws = Workspace(tmp_path)
        index = SymbolIndex()
        index.refresh(ws, ws.files())
        assert index.search("UserService")[0].name == "UserService"  # exact first
        assert [s.name for s in index.search("getuser")][:1] == ["getUser"]
        assert index.search("user", kind="interface")[0].name == "User"
        assert index.search("get user")  # token overlap
        # Re-parse only when files change, and drop deleted files.
        (tmp_path / "svc.py").write_text("def brand_new(): pass\n")
        (tmp_path / "app.ts").unlink()
        index.refresh(ws, ws.files())
        assert [s.name for s in index.all()] == ["brand_new"]

    def test_languages(self) -> None:
        assert language_of("a/b.tsx") == "typescript"
        assert language_of("Dockerfile") == "docker"
        assert is_test_path("tests/test_x.py")
        assert is_test_path("src/app.test.ts")
        assert is_test_path("pkg/store_test.go")
        assert not is_test_path("src/contest.py")


class TestDependencies:
    def test_python_import_graph(self, sample_repo: Path) -> None:
        ws = Workspace(sample_repo)
        graph = build_import_graph(ws, ws.files())
        # absolute import through src/ layout, and relative import
        assert "src/calc/ops.py" in graph.imports["src/calc/__init__.py"]
        assert "src/calc/ops.py" in graph.imports["src/calc/auth.py"]
        assert "src/calc/ops.py" in graph.imports["tests/test_ops.py"]
        assert graph.importers["src/calc/ops.py"] >= {"src/calc/auth.py", "tests/test_ops.py"}
        assert "hashlib" not in graph.external_usage()  # stdlib excluded

    def test_js_import_graph(self, tmp_path: Path) -> None:
        (tmp_path / "src" / "lib").mkdir(parents=True)
        (tmp_path / "src" / "index.ts").write_text(
            "import { a } from './lib/a';\nimport React from 'react';\n"
            "const b = require('./lib');\nimport x from '@scope/pkg/sub';\n"
        )
        (tmp_path / "src" / "lib" / "a.ts").write_text("export const a = 1;\n")
        (tmp_path / "src" / "lib" / "index.js").write_text("module.exports = {};\n")
        ws = Workspace(tmp_path)
        graph = build_import_graph(ws, ws.files())
        assert graph.imports["src/index.ts"] == {"src/lib/a.ts", "src/lib/index.js"}
        assert graph.external["src/index.ts"] == {"react", "@scope/pkg"}

    def test_manifests(self) -> None:
        py = parse_manifest(
            "pyproject.toml",
            '[project]\nname="x"\ndependencies=["requests>=2", "pydantic[email]==2.7"]\n'
            '[dependency-groups]\ndev=["pytest"]\n',
        )
        assert py is not None
        assert py.dependencies == {"requests": ">=2", "pydantic": "==2.7"}
        assert py.dev_dependencies == {"pytest": ""}
        node = parse_manifest("package.json", '{"name":"app","dependencies":{"react":"^18"}}')
        assert node is not None
        assert node.dependencies == {"react": "^18"}
        go = parse_manifest("go.mod", "module x/y\n\nrequire (\n\tgithub.com/a/b v1.2.3\n)\n")
        assert go is not None
        assert go.dependencies == {"github.com/a/b": "v1.2.3"}
        reqs = parse_manifest("requirements-dev.txt", "# c\npytest==8\n-e .\nruff\n")
        assert reqs is not None
        assert reqs.dev_dependencies == {"pytest": "==8", "ruff": ""}
        broken = parse_manifest("package.json", "{not json")
        assert broken is not None
        assert broken.ecosystem == "unknown"


def call(name: str, /, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"a_{name}", name=name, arguments=arguments)


async def test_analysis_tools(sample_repo: Path, make_runtime: RuntimeFactory) -> None:
    runtime = make_runtime(sample_repo)
    structure = await runtime.execute(call("repository_structure"))
    assert structure.ok, structure.content
    assert "python" in structure.content
    assert "pyproject.toml" in structure.content
    assert "pytest" in structure.content

    symbols = await runtime.execute(call("symbol_search", query="check_password"))
    assert "src/calc/auth.py:7" in symbols.content
    assert "def check_password(self, password: str, digest: str) -> bool" in symbols.content

    outline = await runtime.execute(call("file_outline", path="src/calc/auth.py"))
    assert "class TokenStore" in outline.content
    assert "  def check_password" in outline.content

    project = await runtime.execute(call("dependency_analysis"))
    assert "requests>=2.31" in project.content
    focused = await runtime.execute(call("dependency_analysis", path="src/calc/ops.py"))
    assert "imported by: " in focused.content
    assert "src/calc/auth.py" in focused.content
