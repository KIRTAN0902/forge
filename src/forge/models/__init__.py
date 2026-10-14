"""Model runtime: provider-agnostic, capability-aware model access.

Agents depend on :class:`ModelAdapter` (obtained from :class:`ModelRegistry`) and the
neutral types below, never on a concrete provider.
"""

from forge.models.adapter import ModelAdapter
from forge.models.capabilities import ModelCapabilities, OptimizationConfig
from forge.models.profile import (
    BUILTIN_PROVIDERS,
    ROLES,
    ModelProfile,
    Pricing,
    ProviderConfig,
    RoleBinding,
)
from forge.models.provider import ModelProvider, ProviderError
from forge.models.registry import ModelRegistry
from forge.models.types import (
    FinishReason,
    GenerationParams,
    ImagePart,
    Message,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    ResponseFormat,
    Role,
    StreamDone,
    StreamEvent,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)
from forge.models.usage import UsageTracker

__all__ = [
    "BUILTIN_PROVIDERS",
    "ROLES",
    "FinishReason",
    "GenerationParams",
    "ImagePart",
    "Message",
    "ModelAdapter",
    "ModelCapabilities",
    "ModelInfo",
    "ModelProfile",
    "ModelProvider",
    "ModelRegistry",
    "ModelRequest",
    "ModelResponse",
    "OptimizationConfig",
    "Pricing",
    "ProviderConfig",
    "ProviderError",
    "ReasoningDelta",
    "ResponseFormat",
    "Role",
    "RoleBinding",
    "StreamDone",
    "StreamEvent",
    "TextDelta",
    "TextPart",
    "ToolCall",
    "ToolCallDelta",
    "ToolDefinition",
    "Usage",
    "UsageTracker",
]
