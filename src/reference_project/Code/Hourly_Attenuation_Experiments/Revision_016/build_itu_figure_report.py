"""Build the optional bilingual Word figure hand-off from already-rendered images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
FIGURES = PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/Figures/ITU_vs_Observed_Revision_001"
sys.path.insert(0, "C:/Users/SDR_HH/.codex/skills/paper-figures/scripts")
from report_docx import build_report  # noqa: E402


def main() -> None:
    items = json.loads((FIGURES / "figure_items.json").read_text(encoding="utf-8"))
    if (FIGURES / "Figure_Report.docx").exists():
        raise RuntimeError("Existing Word report must not be overwritten")
    for item in items:
        if not (FIGURES / item["image"]).exists():
            raise RuntimeError(f"Missing figure image: {item['image']}")
    path = build_report(
        FIGURES / "Figure_Report.docx",
        title={"en": "ITU rain attenuation versus measured attenuation",
               "zh": "ITU 雨衰估计与实测衰减对照"},
        subtitle={"en": "Revision 016; test-set diagnostic figures",
                  "zh": "Revision 016；测试集诊断图"},
        meta=[({"en": "Source", "zh": "来源"}, "Revision 016 final_test.json"),
              ({"en": "Scope", "zh": "范围"}, "687 test hours; 51 rainy hours")],
        items=items, base_dir=FIGURES, lang="bilingual")
    print(path)


if __name__ == "__main__":
    main()
