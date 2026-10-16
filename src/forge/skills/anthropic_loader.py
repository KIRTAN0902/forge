"""Loader and bridge for Anthropic Agent Skills (anthropics/skills)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml

from forge.skills.models import ModelTier, OperationType, SkillSpec
from forge.skills.types import Skill, SkillCategory

ANTHROPIC_SKILLS_DIR = Path(__file__).parent / "anthropic"

# Mapping of Anthropic skill directories to metadata
ANTHROPIC_SKILL_METADATA: dict[str, dict[str, object]] = {
    "frontend-design": {
        "id": "ANT-01",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.html", "*.css", "*.tsx", "*.jsx", "*.vue", "*.svelte"],
        "keywords": [
            "ui",
            "frontend",
            "design",
            "css",
            "layout",
            "aesthetic",
            "styling",
            "component",
            "bento",
            "editorial",
            "typography",
            "character",
            "distinctive",
        ],
        "category": SkillCategory.DESIGN,
    },
    "docx": {
        "id": "ANT-02",
        "target_agent": "executor",
        "operations": {OperationType.OFFICE_DOCUMENT},
        "globs": ["*.docx", "*.doc"],
        "keywords": ["docx", "word", "document", "report", "doc"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "xlsx": {
        "id": "ANT-03",
        "target_agent": "executor",
        "operations": {OperationType.OFFICE_DOCUMENT},
        "globs": ["*.xlsx", "*.xls", "*.csv"],
        "keywords": ["xlsx", "excel", "spreadsheet", "csv", "sheet", "workbook"],
        "category": SkillCategory.DATABASE,
    },
    "pdf": {
        "id": "ANT-04",
        "target_agent": "executor",
        "operations": {OperationType.OFFICE_DOCUMENT},
        "globs": ["*.pdf"],
        "keywords": ["pdf", "pypdf", "pdfplumber", "extract text from pdf"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "pptx": {
        "id": "ANT-05",
        "target_agent": "executor",
        "operations": {OperationType.OFFICE_DOCUMENT},
        "globs": ["*.pptx", "*.potx"],
        "keywords": ["pptx", "powerpoint", "presentation", "slides", "deck"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "mcp-builder": {
        "id": "ANT-06",
        "target_agent": "orchestrator",
        "operations": {OperationType.TOOL_BUILDING},
        "globs": ["*mcp*", "*server*.py", "*tool*.py"],
        "keywords": ["mcp", "model context protocol", "mcp server", "tools server"],
        "category": SkillCategory.API,
    },
    "webapp-testing": {
        "id": "ANT-07",
        "target_agent": "qa",
        "operations": {OperationType.WRITE_UNIT_TEST},
        "globs": ["*playwright*", "*e2e*", "*test*.ts", "*test*.js", "playwright.config.*"],
        "keywords": ["playwright", "e2e", "browser test", "webapp testing", "headless test"],
        "category": SkillCategory.TESTING,
    },
    "claude-api": {
        "id": "ANT-08",
        "target_agent": "orchestrator",
        "operations": {OperationType.TOOL_BUILDING},
        "globs": ["*anthropic*", "*claude*"],
        "keywords": ["claude api", "anthropic sdk", "prompt caching", "claude model"],
        "category": SkillCategory.API,
    },
    "web-artifacts-builder": {
        "id": "ANT-09",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.html", "*.js", "*.ts"],
        "keywords": ["artifact", "single-file app", "interactive html", "web artifact"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "canvas-design": {
        "id": "ANT-10",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.png", "*.pdf", "*.svg"],
        "keywords": ["canvas", "visual design", "diagram", "poster", "infographic"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "algorithmic-art": {
        "id": "ANT-11",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.js", "*.html", "*.svg"],
        "keywords": ["p5.js", "algorithmic art", "generative art", "canvas art"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "theme-factory": {
        "id": "ANT-12",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.css", "*.json", "*theme*"],
        "keywords": ["theme", "palette", "typography token", "dark mode", "light mode"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "brand-guidelines": {
        "id": "ANT-13",
        "target_agent": "orchestrator",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*brand*", "*guideline*"],
        "keywords": ["brand", "guidelines", "brand colors", "identity", "voice"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "doc-coauthoring": {
        "id": "ANT-14",
        "target_agent": "orchestrator",
        "operations": {OperationType.LOCATE_SYMBOL, OperationType.WRITE_PURE_FUNCTION},
        "globs": ["*.md", "*.rst"],
        "keywords": ["coauthor", "documentation workflow", "draft doc", "user guide"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "internal-comms": {
        "id": "ANT-15",
        "target_agent": "orchestrator",
        "operations": {OperationType.LOCATE_SYMBOL, OperationType.WRITE_PURE_FUNCTION},
        "globs": ["*.md"],
        "keywords": ["internal memo", "prd", "communication", "status report", "update"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "skill-creator": {
        "id": "ANT-16",
        "target_agent": "orchestrator",
        "operations": {OperationType.TOOL_BUILDING},
        "globs": ["*skill*.md"],
        "keywords": ["create skill", "skill creator", "agent skill spec", "skill.md"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "discernment-nudge": {
        "id": "ANT-17",
        "target_agent": "orchestrator",
        "operations": {OperationType.LOCATE_SYMBOL, OperationType.WRITE_PURE_FUNCTION},
        "globs": [],
        "keywords": ["discernment", "critical thinking", "ethical reasoning", "trade-off nudge"],
        "category": SkillCategory.SECURITY,
    },
    "slack-gif-creator": {
        "id": "ANT-18",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.gif"],
        "keywords": ["slack gif", "animated gif", "emoji gif", "gif animation"],
        "category": SkillCategory.DOCUMENTATION,
    },
    "academy-guide": {
        "id": "ANT-19",
        "target_agent": "orchestrator",
        "operations": {OperationType.LOCATE_SYMBOL, OperationType.WRITE_PURE_FUNCTION},
        "globs": [],
        "keywords": ["academy", "tutorial", "claude guide", "learning path"],
        "category": SkillCategory.DOCUMENTATION,
    },
}


def load_anthropic_skill_markdown(skill_dirname: str) -> str:
    """Retrieve full SKILL.md contents for an Anthropic skill directory."""
    skill_file = ANTHROPIC_SKILLS_DIR / skill_dirname / "SKILL.md"
    if skill_file.exists():
        return skill_file.read_text(encoding="utf-8")
    return ""


def parse_anthropic_skills() -> list[tuple[SkillSpec, Skill, list[str]]]:
    """Parse all 19 Anthropic skills into Forge SkillSpec and Skill contracts."""
    loaded: list[tuple[SkillSpec, Skill, list[str]]] = []

    if not ANTHROPIC_SKILLS_DIR.exists():
        return loaded

    for skill_dir in sorted(ANTHROPIC_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        dirname = skill_dir.name
        meta_override = ANTHROPIC_SKILL_METADATA.get(dirname, {})
        skill_id = str(meta_override.get("id", f"ANT-{dirname}"))
        raw_agent = str(meta_override.get("target_agent", "orchestrator"))
        target_agent: Literal["orchestrator", "executor", "qa"] = (
            "executor" if raw_agent == "executor" else "qa" if raw_agent == "qa" else "orchestrator"
        )
        operations = meta_override.get("operations")
        op_set: set[OperationType] = (
            operations if isinstance(operations, set) else {OperationType.WRITE_PURE_FUNCTION}
        )
        globs = meta_override.get("globs")
        glob_list: list[str] = [str(g) for g in globs] if isinstance(globs, list) else []
        keywords = meta_override.get("keywords")
        keyword_list: list[str] = [str(k) for k in keywords] if isinstance(keywords, list) else [dirname]
        category = meta_override.get("category", SkillCategory.DOCUMENTATION)
        if not isinstance(category, SkillCategory):
            category = SkillCategory.DOCUMENTATION

        content = skill_file.read_text(encoding="utf-8")
        parsed_name = dirname
        parsed_desc = f"Anthropic agent skill: {dirname}"

        m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if m:
            try:
                frontmatter = yaml.safe_load(m.group(1)) or {}
                parsed_name = frontmatter.get("name", dirname)
                parsed_desc = frontmatter.get("description", parsed_desc)
            except Exception:
                pass

        # Extract compact execution rules from sections
        body = content[m.end() :] if m else content
        sections = re.findall(r"^##\s+(.+)$", body, flags=re.MULTILINE)
        execution_rules: list[str] = []
        if sections:
            for s in sections[:4]:
                clean_s = re.sub(r"[^a-zA-Z0-9\s\-_]", "", s).strip()
                if clean_s:
                    execution_rules.append(f"Adhere to guideline: {clean_s}.")
        if not execution_rules:
            execution_rules = [f"Apply Anthropic {parsed_name} domain guidelines."]

        skill_spec = SkillSpec(
            skill_id=skill_id,
            name=f"Anthropic {parsed_name}",
            target_agent=target_agent,
            applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
            operation_types=op_set,
            trigger_file_globs=glob_list,
            trigger_ast_nodes=[],
            trigger_error_regexes=[],
            execution_rules=execution_rules,
            allowed_tools=["view_file", "write_to_file", "run_command", "web_search", "fetch_web_page"],
        )

        skill_item = Skill(
            id=skill_id,
            name=f"Anthropic {parsed_name}",
            category=category,
            primary_trigger=f"User request or files matching {', '.join(keyword_list[:3])}",
            target_agents=(target_agent,),
            protocol="\n".join(f"- {r}" for r in execution_rules),
            failure_mode_prevented=f"Prevents ungrounded implementation when working with {parsed_name}.",
        )

        loaded.append((skill_spec, skill_item, keyword_list))

    return loaded
