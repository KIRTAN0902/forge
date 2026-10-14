from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from forge.config import ForgeConfig
from forge.events import Event, EventBus
from forge.policies import Approver
from forge.storage import SQLiteStore
from forge.tools import ToolRuntime, create_tool_runtime

PYTHON = Path(sys.executable).as_posix()


@pytest.fixture
async def store() -> AsyncIterator[SQLiteStore]:
    s = await SQLiteStore.open(":memory:")
    try:
        yield s
    finally:
        await s.close()


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


SAMPLE_FILES: dict[str, str] = {
    "README.md": "# Calc\n\nA tiny calculator library with an auth helper.\n",
    "pyproject.toml": (
        '[project]\nname = "calc"\nversion = "0.1.0"\ndependencies = ["requests>=2.31"]\n\n'
        '[dependency-groups]\ndev = ["pytest>=8"]\n'
    ),
    "src/calc/__init__.py": "from calc.ops import add, divide\n\n__all__ = ['add', 'divide']\n",
    "src/calc/ops.py": (
        '"""Arithmetic operations."""\n\n\n'
        "def add(a: int, b: int) -> int:\n"
        "    return a - b  # BUG: should add\n\n\n"
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        raise ZeroDivisionError('division by zero')\n"
        "    return a / b\n"
    ),
    "src/calc/auth.py": (
        "import hashlib\n\nfrom .ops import add\n\n\n"
        "class TokenStore:\n"
        "    def check_password(self, password: str, digest: str) -> bool:\n"
        "        return hashlib.sha256(password.encode()).hexdigest() == digest\n"
    ),
    "tests/test_ops.py": (
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))\n"
        "from calc.ops import add, divide\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_divide():\n    assert divide(6, 3) == 2\n"
    ),
    ".gitignore": "__pycache__/\n.forge/\n",
    ".env": "API_TOKEN=super-secret-value-123\n",
}


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, content in SAMPLE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "reset", "-q", ".env")  # secrets are not committed
    git(root, "commit", "-q", "-m", "initial")
    return root


MULTI_FILES: dict[str, str] = {
    "README.md": "# Shop\n\nUsers, billing and auth services for a small shop.\n",
    "pyproject.toml": (
        '[project]\nname = "shop"\nversion = "1.0.0"\ndependencies = ["fastapi>=0.110"]\n'
        '[project.scripts]\nshop = "app.main:run"\n'
    ),
    "app/__init__.py": "",
    "app/main.py": (
        "from app.users.service import UserService\n"
        "from app.billing.invoice import Invoice\n\n\n"
        "def run() -> None:\n    UserService()\n    Invoice([])\n"
    ),
    "app/users/__init__.py": "",
    "app/users/models.py": (
        "from dataclasses import dataclass\n\n\n@dataclass\nclass User:\n"
        "    email: str\n    password_hash: str\n    active: bool = True\n"
    ),
    "app/users/service.py": (
        "from app.users.models import User\n\n\nclass UserService:\n"
        "    def __init__(self) -> None:\n        self.users: dict[str, User] = {}\n\n"
        "    def create_user(self, email: str, password_hash: str) -> User:\n"
        "        user = User(email, password_hash)\n        self.users[email] = user\n"
        "        return user\n\n"
        "    def find_by_email(self, email: str) -> User | None:\n"
        "        return self.users.get(email)\n"
    ),
    "app/billing/__init__.py": "",
    "app/billing/invoice.py": (
        "from app.billing.tax import tax_rate\n\n\nclass Invoice:\n"
        "    def __init__(self, lines: list[float]) -> None:\n        self.lines = lines\n\n"
        "    def subtotal(self) -> float:\n        return sum(self.lines)\n\n"
        "    def apply_discount(self, percent: float) -> float:\n"
        "        return self.subtotal() * (1 - percent / 100)\n\n"
        "    def total(self, region: str) -> float:\n"
        "        return self.subtotal() * (1 + tax_rate(region))\n"
    ),
    "app/billing/tax.py": (
        "RATES = {'eu': 0.2, 'us': 0.07}\n\n\n"
        "def tax_rate(region: str) -> float:\n    return RATES.get(region, 0.0)\n"
    ),
    "app/auth/__init__.py": "",
    "app/auth/login.py": (
        "import hashlib\n\nfrom app.users.service import UserService\n\n\n"
        "def login(service: UserService, db, email: str, password: str) -> bool:\n"
        "    db.execute(f\"SELECT * FROM audit WHERE email = '{email}'\")\n"
        "    user = service.find_by_email(email)\n"
        "    digest = hashlib.md5(password.encode()).hexdigest()\n"
        "    return bool(user and user.password_hash == digest)\n"
    ),
    "app/utils/__init__.py": "",
    "app/utils/strings.py": (
        "import re\n\n\ndef slugify(text: str) -> str:\n"
        "    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
    ),
    "tests/test_users.py": (
        "from app.users.service import UserService\n\n\ndef test_create_user():\n"
        "    assert UserService().create_user('a@b.c', 'h').email == 'a@b.c'\n"
    ),
    "tests/test_invoice.py": (
        "from app.billing.invoice import Invoice\n\n\ndef test_discount():\n"
        "    assert Invoice([100.0]).apply_discount(10) == 90.0\n"
    ),
    "tests/test_login.py": (
        "from app.auth.login import login\n\n\ndef test_import():\n    assert login\n"
    ),
    "docs/notes.md": "Billing notes: invoices are computed per region.\n",
    "web/src/client.ts": "export async function get(url: string) { return fetch(url); }\n",
    "web/src/api.ts": (
        "import { get } from './client';\n\nexport class InvoiceApi {\n"
        "  async listInvoices() {\n    return get('/invoices');\n  }\n}\n"
    ),
}


@pytest.fixture
def multi_repo(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    for rel, content in MULTI_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial import")
    return root


RuntimeFactory = Callable[..., ToolRuntime]


@pytest.fixture
async def make_runtime() -> AsyncIterator[RuntimeFactory]:
    created: list[ToolRuntime] = []

    def factory(
        root: Path,
        *,
        approver: Approver | None = None,
        events: list[Event] | None = None,
        **config_overrides: object,
    ) -> ToolRuntime:
        config = ForgeConfig.model_validate(config_overrides)
        bus = None
        if events is not None:
            bus = EventBus()
            bus.subscribe(events.append)
        runtime = create_tool_runtime(config, root=root, approver=approver, events=bus)
        created.append(runtime)
        return runtime

    yield factory
    for runtime in created:
        await runtime.aclose()
