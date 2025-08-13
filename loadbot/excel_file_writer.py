# excel_file_writer.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell


__all__ = ["flatten_dict", "save_json_to_excel"]


# Preferred base order; any extra fields appear after these.
PREFERRED_COL_ORDER: List[str] = [
    "Driver name",
    "broker_name",
    "load_number",           # keep as TEXT
    "origin",
    "destination",
    "pickup_appointment",
    "delivery_appointment",
    "commodity",
    "weight_lbs",
    "loaded_miles",
    "total_rate_usd",
    "rpm_usd",
    # "document_sha256",
]

# ------------------------------ tiny helpers ------------------------------ #

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def _as_dict(json_like: Any) -> Dict[str, Any]:
    if isinstance(json_like, dict):
        return json_like
    if isinstance(json_like, str):
        s = _strip_code_fences(json_like)
        if not s:
            raise ValueError("JSON string is empty after cleanup.")
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    raise ValueError(f"Expected dict or JSON string, got {type(json_like).__name__}")

def flatten_dict(d: Any, parent_key: str = "", sep: str = " ") -> Dict[str, Any]:
    items: list[tuple[str, Any]] = []
    if isinstance(d, dict):
        iterator: Iterable[tuple[Any, Any]] = d.items()
    elif isinstance(d, list):
        iterator = enumerate(d)
    else:
        return {parent_key or "value": d}

    for k, v in iterator:
        key_str = str(k)
        new_key = f"{parent_key}{sep}{key_str}" if parent_key else key_str
        if isinstance(v, (dict, list)):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _is_money_col(name: str) -> bool:
    n = name.lower()
    return (
        n.endswith("_usd")
        or "rate" in n
        or "total" in n
        or "amount" in n
        or "fsc" in n
        or "detention" in n
        or "lumper" in n
        or "layover" in n
        or "tonu" in n
        or "pay" in n
    )

def _is_intlike_col(name: str) -> bool:
    n = name.lower()
    return n.endswith("_lbs") or "miles" in n or "weight" in n or n.endswith("_count")

def _to_float_money(x):
    """
    Turn things like '$1 450,00', 'US$3,700', '1 200.50' into numbers.
    Handles spaces, NBSP, comma-decimal, thousands commas, currency symbols.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    s = str(x).strip()
    if s == "":
        return pd.NA

    # normalize whitespace (incl. NBSP/thin space) and strip currency letters/symbols
    s = s.replace("\u00A0", " ").replace("\u202F", " ")  # NBSP / thin space
    s = re.sub(r"[^\d,.\- ]", "", s)   # keep digits, comma, dot, minus, spaces
    s = s.replace(" ", "")

    # normalize decimal separator
    if "," in s and "." in s:
        # both present -> assume comma is thousands (e.g., 1,450.00)
        s = s.replace(",", "")
    elif "," in s:
        # only comma -> treat comma as decimal (e.g., 1 450,00 -> 1450.00)
        s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return pd.NA

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Money-like -> numeric using robust parser
    for col in df.columns:
        if _is_money_col(col):
            df[col] = df[col].map(_to_float_money).astype("Float64")  # nullable float

    # Int-like -> Int64 (strip non-digits first)
    for col in df.columns:
        if _is_intlike_col(col):
            s = (
                df[col]
                .astype(str)
                .str.replace(r"[^\d\-]", "", regex=True)  # keep digits and minus
                .replace({"": pd.NA})
            )
            df[col] = pd.to_numeric(s, errors="coerce").astype("Int64")

    return df

def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [c for c in PREFERRED_COL_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]

def _safe_sheet_name(name: str) -> str:
    name = (name or "Unknown")
    name = re.sub(r'[:\\/?*\[\]]', "_", str(name))  # <-- fixed
    return name[:31] or "Unknown"

def _find_primary_money_col(df: pd.DataFrame) -> int | None:
    """Pick the best column to SUM for totals, preferring 'total_rate_usd'."""
    if "total_rate_usd" in df.columns:
        return df.columns.get_loc("total_rate_usd")
    for i, c in enumerate(df.columns):
        if _is_money_col(c):
            return i
    return None

def _make_formats(wb):
    return {
        "banner": wb.add_format({
            "bold": True, "font_size": 14, "align": "center", "valign": "vcenter",
            "fg_color": "#2E7D32", "font_color": "white"
        }),
        "header": wb.add_format({"bold": True, "bg_color": "#D9EAD3", "border": 1}),
        "total_lbl": wb.add_format({"bold": True, "align": "right"}),
        "money": wb.add_format({"num_format": "$#,##0.00"}),
        "int": wb.add_format({"num_format": "#,##0"}),
        "text": wb.add_format({"num_format": "@"}),
        "normal": wb.add_format({}),
    }

def _set_column_formats(ws, df: pd.DataFrame, fmts: dict):
    for idx, col in enumerate(df.columns):
        n = col.lower()
        width = 18
        if n in {"origin", "destination", "pickup_appointment", "delivery_appointment", "commodity"}:
            width = 24
        if n == "driver name":
            width = 20
#        if n == "document_sha256":
#            width = 44

        fmt = None
        if n == "load_number":  # <- was: if col == "load_number"
            fmt = fmts["text"]
        elif _is_money_col(col):
            fmt = fmts["money"]
        elif _is_intlike_col(col):
            fmt = fmts["int"]

        if fmt is None:
            ws.set_column(idx, idx, width)
        else:
            ws.set_column(idx, idx, width, fmt)

def _open_writer(path: Path):
    """
    Back/forward-compatible ExcelWriter factory.
    Newer pandas accepts engine_kwargs; older ones don't. We avoid 'options=' which
    caused your TypeError and just set the basics that work everywhere.
    """
    try:
        return pd.ExcelWriter(
            path,
            engine="xlsxwriter",
            datetime_format="yyyy-mm-dd hh:mm",
        )
    except TypeError:
        # Ultra-old pandas fallback (unlikely)
        return pd.ExcelWriter(path, engine="xlsxwriter")

# ------------------------------ main API ------------------------------ #

def save_json_to_excel(
    json_data: Any,
    excel_file: str | Path = "monthly_report.xlsx",
    all_loads_sheet: str = "All Loads",
) -> None:
    """
    Appends a record (dict or JSON string) to a master 'All Loads' sheet,
    then (re)generates one sheet per driver with a driver banner and a Total row.

    - Driver name comes from the 'Driver name' field (empty -> 'Unknown').
    - Money columns are numeric; 'load_number' is forced to text column in Excel.
    - Each driver sheet includes ALL columns present in your data.
    """
    record = _as_dict(json_data)
    excel_path = Path(excel_file)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing master table (if any), append new row
    if excel_path.exists():
        try:
            base_df = pd.read_excel(excel_path, sheet_name=all_loads_sheet)
        except ValueError:
            # Workbook exists but no All Loads sheet yet
            base_df = pd.DataFrame()
    else:
        base_df = pd.DataFrame()

    new_df = pd.DataFrame([record])
    # union columns so nothing gets dropped
    all_cols = list(dict.fromkeys([*base_df.columns.tolist(), *new_df.columns.tolist()]))
    base_df = base_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    df = pd.concat([base_df, new_df], ignore_index=True)

    # NEW: remove document_sha256 from the report (case-insensitive)
    df = df.loc[:, [c for c in df.columns if c.lower() != "document_sha256"]]

    # Ensure Driver column exists & standardized
    if "Driver name" not in df.columns:
        df["Driver name"] = "Unknown"
    df["Driver name"] = df["Driver name"].fillna("").astype(str).str.strip().replace({"": "Unknown"})

    # Types & order
    df = _coerce_numeric(df)
    df = _reorder_columns(df)

    # Build driver groups (sorted by driver)
    driver_groups = df.groupby("Driver name", dropna=False)

    # Write the entire workbook fresh each time (xlsxwriter)
    with _open_writer(excel_path) as writer:
        wb = writer.book
        try:
            wb.set_calc_on_load()  # ensure SUM() updates when the file opens
        except Exception:
            pass
        fmts = _make_formats(wb)

        # 1) All Loads (flat table)
        df.to_excel(writer, sheet_name=all_loads_sheet, index=False)
        ws_all = writer.sheets[all_loads_sheet]
        ws_all.freeze_panes(1, 0)
        _set_column_formats(ws_all, df, fmts)

        # 2) One sheet per driver
        for driver, dfd in sorted(driver_groups, key=lambda t: t[0].lower()):
            sheet = _safe_sheet_name(driver)
            startrow = 2  # leave space for banner + spacer row
            dfd_sorted = dfd.reset_index(drop=True)
            dfd_sorted.to_excel(writer, sheet_name=sheet, index=False, startrow=startrow)
            ws = writer.sheets[sheet]
            _set_column_formats(ws, dfd_sorted, fmts)

            # Freeze panes below header
            ws.freeze_panes(startrow + 1, 0)

            n_cols = len(dfd_sorted.columns)

            # Driver banner (merged)
            ws.merge_range(0, 0, 0, max(n_cols - 1, 0), f"{driver}", fmts["banner"])

            # Re-style header row
            header_row = startrow
            for c in range(n_cols):
                ws.write(header_row, c, dfd_sorted.columns[c], fmts["header"])

            # Total row (sum chosen money column)
            primary_col_idx = _find_primary_money_col(dfd_sorted)
            if primary_col_idx is not None and len(dfd_sorted) > 0:
                first_data_row = startrow + 1
                last_data_row = startrow + len(dfd_sorted)
                total_row = last_data_row + 1

                # Label "Total" merged up to the money column
                if primary_col_idx > 0:
                    ws.merge_range(total_row, 0, total_row, primary_col_idx - 1, "Total", fmts["total_lbl"])
                else:
                    ws.write(total_row, 0, "Total", fmts["total_lbl"])

                top = xl_rowcol_to_cell(first_data_row, primary_col_idx)
                bottom = xl_rowcol_to_cell(last_data_row, primary_col_idx)
                # Compute cached total from the money column (handles nullable floats)
                col_series = pd.to_numeric(dfd_sorted.iloc[:, primary_col_idx], errors="coerce").fillna(0.0)
                cached_total = float(col_series.sum())

                ws.write_formula(
                    total_row,
                    primary_col_idx,
                    f"=SUM({top}:{bottom})",
                    fmts["money"],
                    cached_total,  # cached result used by viewers that don't recalc
                )