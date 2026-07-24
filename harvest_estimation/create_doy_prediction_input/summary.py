from __future__ import annotations

import csv
import os
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

GLOBAL_SUMMARY_FIELDNAMES = [
    "Crop",
    "Date",
    "NDVI",
    "NDWI",
    "EVI",
    "Harvest",
    "start_rule",
    "div_start",
    "div_end",
    "IoU",
    "SeedingEvent",
    "greenup_rule",
    "greenup_source",
    "seeding_offset_days",
    "seeding_window_start",
    "seeding_window_end",
]


def _display_crop_name(crop_dir: Path) -> str:
    metadata_path = crop_dir / ".crop_name"
    if metadata_path.exists():
        value = metadata_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return crop_dir.name


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_xlsx(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    def excel_col_name(col_idx: int) -> str:
        name = ""
        current = col_idx
        while current > 0:
            current, remainder = divmod(current - 1, 26)
            name = chr(65 + remainder) + name
        return name

    def inline_str_cell(cell_ref: str, value: str) -> str:
        safe = escape("" if value is None else str(value))
        return f'<c r="{cell_ref}" t="inlineStr"><is><t>{safe}</t></is></c>'

    def build_sheet_xml() -> str:
        xml_rows: list[str] = []
        all_rows = [fieldnames] + [
            [row.get(field, "") for field in fieldnames] for row in rows
        ]
        for row_idx, values in enumerate(all_rows, start=1):
            cells = [
                inline_str_cell(f"{excel_col_name(col_idx)}{row_idx}", value)
                for col_idx, value in enumerate(values, start=1)
            ]
            xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{''.join(xml_rows)}</sheetData>"
            "</worksheet>"
        )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Summary" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as workbook_zip:
        workbook_zip.writestr("[Content_Types].xml", content_types_xml)
        workbook_zip.writestr("_rels/.rels", root_rels_xml)
        workbook_zip.writestr("xl/workbook.xml", workbook_xml)
        workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        workbook_zip.writestr("xl/worksheets/sheet1.xml", build_sheet_xml())


def write_global_crop_summary_files(
    output_root,
    rows: list[dict[str, str]],
    csv_filename: str = "harvest_summary_all_crops.csv",
    xlsx_filename: str = "harvest_summary_all_crops.xlsx",
) -> None:
    output_root = Path(output_root)
    normalized_rows = [
        {field: row.get(field, "") for field in GLOBAL_SUMMARY_FIELDNAMES}
        for row in rows
    ]
    _write_csv(output_root / csv_filename, normalized_rows, GLOBAL_SUMMARY_FIELDNAMES)
    _write_xlsx(output_root / xlsx_filename, normalized_rows, GLOBAL_SUMMARY_FIELDNAMES)


def summarize_global_crop_outputs(
    output_root,
    csv_filename: str = "harvest_summary_all_crops.csv",
    xlsx_filename: str = "harvest_summary_all_crops.xlsx",
):
    output_root = Path(output_root)
    rows: list[dict[str, str]] = []

    for crop_name in sorted(os.listdir(output_root)):
        crop_dir = output_root / crop_name
        if not crop_dir.is_dir():
            continue
        display_crop_name = _display_crop_name(crop_dir)

        csv_path = crop_dir / f"{crop_name}_summary.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            div_start_header = (
                "div_start"
                if "div_start" in (reader.fieldnames or [])
                else "div_result_start"
            )
            div_end_header = (
                "div_end"
                if "div_end" in (reader.fieldnames or [])
                else "div_result_end"
            )
            for row in reader:
                rows.append(
                    {
                        "Crop": display_crop_name,
                        "Date": (row.get("Date") or "").strip(),
                        "NDVI": (row.get("NDVI") or "").strip(),
                        "NDWI": (row.get("NDWI") or "").strip(),
                        "EVI": (row.get("EVI") or "").strip(),
                        "Harvest": (row.get("Harvest") or "").strip(),
                        "start_rule": (row.get("start_rule") or "").strip(),
                        "div_start": (row.get(div_start_header) or "").strip(),
                        "div_end": (row.get(div_end_header) or "").strip(),
                        "IoU": (row.get("IoU") or "").strip(),
                        "SeedingEvent": (row.get("SeedingEvent") or "").strip(),
                        "greenup_rule": (row.get("greenup_rule") or "").strip(),
                        "greenup_source": (row.get("greenup_source") or "").strip(),
                        "seeding_offset_days": (
                            row.get("seeding_offset_days") or ""
                        ).strip(),
                        "seeding_window_start": (
                            row.get("seeding_window_start") or ""
                        ).strip(),
                        "seeding_window_end": (
                            row.get("seeding_window_end") or ""
                        ).strip(),
                    }
                )

    write_global_crop_summary_files(
        output_root=output_root,
        rows=rows,
        csv_filename=csv_filename,
        xlsx_filename=xlsx_filename,
    )
    print(
        f"Saved global crop summaries to: {output_root / csv_filename} and {output_root / xlsx_filename}"
    )


def summarize_farm_harvest_dates(
    output_root, summary_filename="farm_harvest_summary.csv"
):
    print("Summarizing farm harvest dates...")

    output_root = Path(output_root)
    rows: list[dict[str, str]] = []

    for crop_name in sorted(os.listdir(output_root)):
        crop_dir = output_root / crop_name
        farms_dir = crop_dir / "Farms"
        if not farms_dir.is_dir():
            continue
        display_crop_name = _display_crop_name(crop_dir)

        for farm_id in sorted(os.listdir(farms_dir)):
            farm_folder = farms_dir / farm_id
            if not farm_folder.is_dir():
                continue

            for file_name in sorted(os.listdir(farm_folder)):
                if not file_name.endswith("_summary.csv"):
                    continue

                source_path = farm_folder / file_name
                with open(source_path, newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    metadata = {
                        "HarvestStart": "",
                        "HarvestEnd": "",
                        "GreenupStart": "",
                        "EstimatedSeedingDate": "",
                        "SeedingSource": "",
                        "SeedingRule": "",
                        "SeedingOffsetDays": "",
                        "SeedingWindowStart": "",
                        "SeedingWindowEnd": "",
                    }

                    for row in reader:
                        date_value = (row.get("Date") or "").strip()
                        harvest_value = (row.get("Harvest") or "").strip()
                        seeding_event = (row.get("SeedingEvent") or "").strip()

                        if harvest_value == "Start" and date_value:
                            metadata["HarvestStart"] = date_value
                        elif harvest_value == "End" and date_value:
                            metadata["HarvestEnd"] = date_value

                        if seeding_event and date_value:
                            events = {
                                item.strip()
                                for item in seeding_event.split(";")
                                if item.strip()
                            }
                            if "GreenupStart" in events:
                                metadata["GreenupStart"] = date_value
                            if "EstimatedSeedingDate" in events:
                                metadata["EstimatedSeedingDate"] = date_value

                            if not metadata["SeedingSource"]:
                                metadata["SeedingSource"] = (
                                    row.get("greenup_source") or ""
                                ).strip()
                            if not metadata["SeedingRule"]:
                                metadata["SeedingRule"] = (
                                    row.get("greenup_rule") or ""
                                ).strip()
                            if not metadata["SeedingOffsetDays"]:
                                metadata["SeedingOffsetDays"] = (
                                    row.get("seeding_offset_days") or ""
                                ).strip()
                            if not metadata["SeedingWindowStart"]:
                                metadata["SeedingWindowStart"] = (
                                    row.get("seeding_window_start") or ""
                                ).strip()
                            if not metadata["SeedingWindowEnd"]:
                                metadata["SeedingWindowEnd"] = (
                                    row.get("seeding_window_end") or ""
                                ).strip()

                rows.append(
                    {
                        "Crop": display_crop_name,
                        "FarmID": farm_id,
                        **metadata,
                        "SourceCSV": file_name,
                    }
                )

    rows = sorted(rows, key=lambda row: (row["Crop"], row["FarmID"]))
    summary_path = output_root / summary_filename
    fieldnames = [
        "Crop",
        "FarmID",
        "HarvestStart",
        "HarvestEnd",
        "GreenupStart",
        "EstimatedSeedingDate",
        "SeedingSource",
        "SeedingRule",
        "SeedingOffsetDays",
        "SeedingWindowStart",
        "SeedingWindowEnd",
        "SourceCSV",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved harvest/seeding summary to: {summary_path}")
