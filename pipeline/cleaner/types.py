"""清洗结果结构与常量。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CLEANED_DIRNAME = "cleaned_dataset"
NOTES_FILENAME = "quality_cleaning_notes.json"


@dataclass
class CleaningResult:
    success: bool
    method: str
    output_root: Path | None
    removed_episodes: list[int] = field(default_factory=list)
    repaired_episodes: list[int] = field(default_factory=list)
    repairs: dict[str, list[int]] = field(default_factory=dict)
    kept_episodes: int = 0
    episode_index_map: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cli_command: str | None = None
    cli_error: str | None = None
    verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "performed": bool(self.removed_episodes or self.repaired_episodes),
            "success": self.success,
            "method": self.method,
            "output_path": str(self.output_root) if self.output_root else None,
            "removed_episode_indices": self.removed_episodes,
            "removed_count": len(self.removed_episodes),
            "repaired_episode_indices": self.repaired_episodes,
            "repairs": {k: v for k, v in self.repairs.items()},
            "kept_episodes": self.kept_episodes,
            "episode_index_map": {str(k): v for k, v in sorted(self.episode_index_map.items())},
            "cli_command": self.cli_command,
            "cli_error": self.cli_error,
            "warnings": self.warnings,
            "verification": self.verification,
        }
