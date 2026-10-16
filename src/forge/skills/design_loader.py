"""Loader and bridge for Frontend & Design Engineering Skills (Impeccable, UI/UX Pro Max, shadcn/ui, Emil Kowalski)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml

from forge.skills.models import ModelTier, OperationType, SkillSpec
from forge.skills.types import Skill, SkillCategory

DESIGN_SKILLS_DIR = Path(__file__).parent / "design"

# Mapping of Design skill directories to metadata
DESIGN_SKILL_METADATA: dict[str, dict[str, object]] = {
    "impeccable": {
        "id": "DSN-01",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.tsx", "*.jsx", "*.vue", "*.svelte", "*.html", "*.css", "*.scss"],
        "keywords": [
            "impeccable",
            "visual audit",
            "anti-pattern",
            "ui audit",
            "spacing",
            "contrast",
            "typography scale",
            "ai slop",
            "clean design",
            "ui polish",
        ],
        "category": SkillCategory.DESIGN,
    },
    "ui-ux-pro-max": {
        "id": "DSN-02",
        "target_agent": "orchestrator",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.tsx", "*.jsx", "*.vue", "*.svelte", "*.html"],
        "keywords": [
            "ui/ux",
            "pro max",
            "ux",
            "empty state",
            "form error",
            "form validation",
            "skeleton loader",
            "responsiveness",
            "breakpoint",
            "touch target",
            "mobile friendly",
            "responsive",
        ],
        "category": SkillCategory.DESIGN,
    },
    "shadcn-ui": {
        "id": "DSN-03",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["components.json", "components/ui/*", "*.tsx", "*.jsx"],
        "keywords": [
            "shadcn",
            "shadcn/ui",
            "components.json",
            "radix",
            "lucide",
            "cva",
            "tailwind-merge",
            "atomic component",
            "aschild",
        ],
        "category": SkillCategory.DESIGN,
    },
    "emil-design-eng": {
        "id": "DSN-04",
        "target_agent": "executor",
        "operations": {OperationType.UI_DESIGN},
        "globs": ["*.tsx", "*.jsx", "*.vue", "*.svelte", "*.css"],
        "keywords": [
            "emil",
            "emil kowalski",
            "design engineering",
            "micro-interaction",
            "spring",
            "framer motion",
            "motion",
            "layoutid",
            "transition",
            "animation",
            "vaul",
            "sonner",
            "gesture",
        ],
        "category": SkillCategory.DESIGN,
    },
}


def load_design_skill_markdown(dirname: str) -> str:
    """Read the full raw markdown specification of a Design skill."""
    target = DESIGN_SKILLS_DIR / dirname / "SKILL.md"
    if target.is_file():
        return target.read_text(encoding="utf-8")
    return ""


def parse_design_skills() -> list[tuple[SkillSpec, Skill, list[str]]]:
    """Parse all local Design skills into Forge SkillSpec, Skill, and keyword triggers."""
    if not DESIGN_SKILLS_DIR.is_dir():
        return []

    loaded: list[tuple[SkillSpec, Skill, list[str]]] = []

    for skill_dir in sorted(DESIGN_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            continue

        dirname = skill_dir.name
        meta_override = DESIGN_SKILL_METADATA.get(dirname, {})
        skill_id = str(meta_override.get("id", f"DSN-{dirname}"))
        raw_agent = str(meta_override.get("target_agent", "executor"))
        target_agent: Literal["orchestrator", "executor", "qa"] = (
            "orchestrator" if raw_agent == "orchestrator" else "qa" if raw_agent == "qa" else "executor"
        )
        operations = meta_override.get("operations")
        op_set: set[OperationType] = (
            operations if isinstance(operations, set) else {OperationType.UI_DESIGN}
        )
        globs = meta_override.get("globs")
        glob_list: list[str] = [str(g) for g in globs] if isinstance(globs, list) else []
        keywords = meta_override.get("keywords")
        keyword_list: list[str] = [str(k) for k in keywords] if isinstance(keywords, list) else [dirname]
        category = meta_override.get("category", SkillCategory.DESIGN)
        if not isinstance(category, SkillCategory):
            category = SkillCategory.DESIGN

        content = skill_file.read_text(encoding="utf-8")
        parsed_name = dirname
        parsed_desc = f"Design engineering skill: {dirname}"

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
            execution_rules = [f"Apply {parsed_name} design system guidelines."]

        skill_spec = SkillSpec(
            skill_id=skill_id,
            name=f"Design {parsed_name}",
            target_agent=target_agent,
            applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
            operation_types=op_set,
            trigger_file_globs=glob_list,
            trigger_ast_nodes=[],
            trigger_error_regexes=[],
            execution_rules=execution_rules,
            allowed_tools=["view_file", "write_to_file", "replace_file_content", "run_command"],
        )

        skill_item = Skill(
            id=skill_id,
            name=f"Design {parsed_name}",
            category=category,
            primary_trigger=f"User request or files matching {', '.join(keyword_list[:3])}",
            target_agents=(target_agent,),
            protocol="\n".join(f"- {r}" for r in execution_rules),
            failure_mode_prevented=f"Prevents ungrounded or low-craft implementation when working with {parsed_name}.",
        )

        loaded.append((skill_spec, skill_item, keyword_list))

    return loaded
