"""Write the Phase 0 compatibility inventory to a JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from echosight2.diagnostics import collect_report

project_root = Path(__file__).resolve().parents[1]
workspace_root = project_root.parents[1]
output_path = project_root / "artifacts" / "phase0_inventory.json"
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(
    json.dumps(collect_report(workspace_root), indent=2),
    encoding="utf-8",
)
print(f"Wrote {output_path}")
