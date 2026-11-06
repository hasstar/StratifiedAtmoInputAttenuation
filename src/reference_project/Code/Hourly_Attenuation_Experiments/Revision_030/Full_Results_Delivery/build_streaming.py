"""Streaming fallback after artifact-tool failed to export the full data at 8GiB heap."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import xlsxwriter

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_030_Original_Training_Aligned/Full_Results"
DATA = OUT / "workbook_data.json"
payload = json.loads(DATA.read_text(encoding="utf-8"))
DEST = Path(payload["xlsx_path"])
PREVIEW = OUT / "previews/visual_check_source.xlsx"
if "--repair-display" in sys.argv:
    previous_qa = json.loads((OUT / "workbook_build_qa.json").read_text(encoding="utf-8"))
    assert previous_qa["authoring_engine"] == "bundled xlsxwriter constant_memory fallback"
    assert hashlib.sha256(DEST.read_bytes()).hexdigest() == previous_qa["xlsx_sha256"]
    for source, target in [(DEST, OUT / "previews/initial_layout_workbook.xlsx"),
                           (PREVIEW, OUT / "previews/initial_visual_source.xlsx"),
                           (OUT / "workbook_build_qa.json", OUT / "previews/initial_workbook_build_qa.json")]:
        assert source.resolve().is_relative_to(OUT.resolve()) and target.resolve().is_relative_to(OUT.resolve())
        assert source.is_file() and not target.exists()
        source.rename(target)
assert not DEST.exists() and not PREVIEW.exists(), "Do not overwrite a saved workbook."
COUNTS = {"mask或k", "特征总数", "基础列数", "气压面列数", "seed", "轮数", "key", "n_features", "n_base", "n_pressure",
          "n_hours", "n_dates", "epochs", "level_hpa", "contexts", "k", "k_before", "k_after", "deletion_step", "survival_rank",
          "feature_position", "fold", "fit_hours", "internal_holdout_hours", "replicates"}


def display_header(header: str) -> str:
    if header.startswith(("测试_", "验证_", "训练_")):
        prefix, weather, metric = header.split("_", 2)
        return f"{prefix} {weather}\n{metric.upper().replace('(DB)', '(dB)')}"
    if header == "训练内CV_RMSE(dB)":
        return "训练内CV均值\nRMSE(dB)"
    if header == "训练内CV_SE(dB)":
        return "训练内CV\nSE(dB)"
    if header == "训练内oneSE选中":
        return "训练内one-SE\n选中"
    return header


def make_book(path: Path, preview: bool) -> list[dict]:
    book = xlsxwriter.Workbook(path, {"constant_memory": True, "strings_to_formulas": False, "strings_to_urls": False})
    book.set_properties({"title": "全部1024种气压面组合与156→1列RFE", "comments": "Read-only result export. No new training or model inference."})
    title = book.add_format({"font_name": "Arial", "font_size": 14, "bold": True, "font_color": "#172B4D"})
    note = book.add_format({"font_name": "Arial", "font_size": 10, "italic": True, "font_color": "#536171"})
    header_fmt = book.add_format({"font_name": "Arial", "font_size": 10, "bold": True, "font_color": "#FFFFFF", "bg_color": "#27486A",
                                 "align": "center", "valign": "vcenter", "text_wrap": True, "bottom": 1, "bottom_color": "#27486A"})
    cache = {}

    def fmt(kind: str, band: bool):
        key = (kind, band)
        if key not in cache:
            settings = {"font_name": "Arial", "font_size": 10, "font_color": "#243746", "valign": "vcenter"}
            if band:
                settings["bg_color"] = "#EFF5FB"
            if kind == "int":
                settings.update({"align": "right", "num_format": "#,##0"})
            elif kind == "float":
                settings.update({"align": "right", "num_format": "0.000000"})
            elif kind == "percent":
                settings.update({"align": "right", "num_format": "0.0%"})
            elif kind == "bool":
                settings["align"] = "center"
            elif kind == "note":
                settings.update({"align": "left", "text_wrap": True})
            else:
                settings.update({"align": "left", "num_format": "@"})
            cache[key] = book.add_format(settings)
        return cache[key]

    inventory = []
    for spec in payload["sheets"]:
        sheet = book.add_worksheet(spec["name"])
        sheet.hide_gridlines(2)
        sheet.freeze_panes(6, 2)
        if spec["name"] in ("分层组合1024", "RFE全路径156"):
            sheet.set_tab_color("#27486A")
        sheet.write_string(1, 0, spec["name"], title)
        sheet.write_string(3, 0, f"来源：{spec['source']}；唯一seed17，固定53轮。完整数值，无筛选。", note)
        sheet.set_row(5, 42)
        display = [display_header(label) for label in spec["headers"]]
        spec["display_headers"] = display
        for col, label in enumerate(display):
            sheet.write_string(5, col, label, header_fmt)
            original = spec["headers"][col]
            width = 18
            if re.search("feature_name|deleted_feature|下一删除列", original):
                width = 105
            elif re.search("choice_json|dates_json|unit", original):
                width = 65
            elif "sha256" in original:
                width = 70
            elif re.search("气压面|levels_hpa|comparison_weather", original):
                width = 36
            if spec["name"] == "说明":
                width = 29 if col == 0 else 115
            sheet.set_column(col, col, width)
        rows = spec["rows"][:8] if preview else spec["rows"]
        for offset, row in enumerate(rows):
            excel_row = 6 + offset
            sheet.set_row(excel_row, 34 if spec["name"] == "说明" else 18)
            band = offset % 2 == 0
            for col, value in enumerate(row):
                original = spec["headers"][col]
                if value is None or value == "":
                    sheet.write_blank(excel_row, col, None, fmt("text", band))
                elif isinstance(value, bool):
                    sheet.write_boolean(excel_row, col, value, fmt("bool", band))
                elif isinstance(value, (int, float)):
                    kind = "int" if original in COUNTS else "percent" if original == "fraction_helpful" else "float"
                    sheet.write_number(excel_row, col, value, fmt(kind, band))
                else:
                    sheet.write_string(excel_row, col, str(value), fmt("note" if spec["name"] == "说明" and col == 1 else "text", band))
        sheet.autofilter(5, 0, 5 + len(rows), len(spec["headers"])-1)
        inventory.append({"name": spec["name"], "rows": len(rows), "columns": len(spec["headers"]), "autofilter": True, "frozen_header_rows": 6, "frozen_id_columns": 2})
        print(f"streamed {spec['name']}: {len(rows)} records", flush=True)
    book.close()
    return inventory


inventory = make_book(DEST, False)
make_book(PREVIEW, True)
with DATA.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
qa = {"status": "PASS", "sheets": inventory, "authoring_engine": "bundled xlsxwriter constant_memory fallback",
      "fallback_reason": "artifact-tool full-data XLSX export failed with JavaScript heap out of memory at 8192MiB; all data retained",
      "expected_final_workbook_count": 1, "new_training_fits": 0, "new_model_inference_calls": 0,
      "xlsx_sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(), "utc": datetime.now(timezone.utc).isoformat()}
with (OUT / "workbook_build_qa.json").open("x", encoding="utf-8") as handle:
    json.dump(qa, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"status": "PASS", "sheets": len(inventory), "xlsx_bytes": DEST.stat().st_size}), flush=True)
