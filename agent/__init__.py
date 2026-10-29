from __future__ import annotations

from typing import Any

__all__ = ["MultiAgentDatasetRetriever"]


def __getattr__(name: str) -> Any:
    if name == "MultiAgentDatasetRetriever":
        from .orchestrator import MultiAgentDatasetRetriever

        return MultiAgentDatasetRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
