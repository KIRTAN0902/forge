from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """Base for every structured contract exchanged between Forge components.

    Unknown fields are ignored rather than rejected: many contracts are produced by
    models, and a weaker model adding an extra key should not fail an otherwise valid
    plan or report. Assignment is validated so in-process mutation stays type-safe.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )
