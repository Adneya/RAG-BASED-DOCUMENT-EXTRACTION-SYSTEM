from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

from app.schemas import AnalyticsCategory, AnalyticsColumn, AnalyticsInsight, AnalyticsPoint


def analyze_table(filename: str, data: bytes, *, date_column: str | None = None, measure_column: str | None = None, category_column: str | None = None) -> dict[str, Any]:
    rows = read_rows(filename, data)
    if not rows:
        raise ValueError("The uploaded table contains no readable rows.")
    columns = list(rows[0])
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    descriptions: list[AnalyticsColumn] = []
    for column in columns:
        values = [row[column] for row in normalized if row[column] not in (None, "")]
        kind = infer_kind(values)
        descriptions.append(AnalyticsColumn(name=column, kind=kind, non_empty=len(values), unique_values=len({str(value) for value in values})))
    detected_date_column = next((item.name for item in descriptions if item.kind == "date"), None)
    detected_measure_column = choose_measure_column(descriptions)
    detected_category_column = next((item.name for item in descriptions if item.kind == "category"), None)
    date_column = date_column if date_column in columns else detected_date_column
    measure_column = measure_column if measure_column in columns else detected_measure_column
    category_column = category_column if category_column in columns else detected_category_column
    if not measure_column:
        raise ValueError("No numeric measure column was detected. Add a column containing amounts, counts, or metrics.")
    values = [(row[measure_column], row) for row in normalized if to_number(row[measure_column]) is not None]
    numbers = [to_number(value) for value, _ in values]
    assert all(value is not None for value in numbers)
    numeric_values = [float(value) for value in numbers]
    total = sum(numeric_values)
    average = total / len(numeric_values)
    minimum = min(numeric_values)
    maximum = max(numeric_values)
    trend = build_trend(values, date_column, measure_column)
    categories = build_categories(values, category_column)
    anomalies = [point for point in trend if abs(point.value - average) > max(abs(average) * .35, 1)]
    distribution = build_distribution(numeric_values)
    source = f"{filename} / {measure_column}"
    direction = "above" if numeric_values[-1] >= average else "below"
    insights = [
        AnalyticsInsight(finding=f"{measure_column} averages {average:,.2f} across {len(numeric_values):,} records.", impact="This is the baseline for measuring performance and prioritizing variance.", action="Use the baseline to set a reporting target and monitor future uploads.", confidence=.96, source=source),
        AnalyticsInsight(finding=f"The latest observed value is {direction} the overall average.", impact="A sustained variance may indicate a growth opportunity or an emerging operating risk.", action="Review the supporting records behind the latest period before making a decision.", confidence=.84, source=source),
    ]
    if anomalies:
        insights.append(AnalyticsInsight(finding=f"{len(anomalies)} period(s) show a material variance from the baseline.", impact="Outliers can distort forecasts and may signal a business event worth investigating.", action="Inspect the cited source rows and annotate the cause of each anomaly.", confidence=.88, source=source))
    return {"row_count": len(normalized), "columns": descriptions, "date_column": date_column, "measure_column": measure_column, "category_column": category_column, "kpis": {"total": total, "average": average, "minimum": minimum, "maximum": maximum}, "trend": trend, "categories": categories, "anomalies": anomalies, "distribution": distribution, "insights": insights}


def read_rows(filename: str, data: bytes) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook
        sheet = load_workbook(BytesIO(data), read_only=True, data_only=True).active
        records = list(sheet.values)
    else:
        import csv
        text = data.decode("utf-8-sig", errors="replace")
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        records = list(csv.reader(StringIO(text), dialect))
    if len(records) < 2:
        return []
    headers = [str(value or f"Column {index + 1}").strip() for index, value in enumerate(records[0])]
    return [dict(zip(headers, row)) for row in records[1:] if any(value not in (None, "") for value in row)]


def infer_kind(values: list[Any]) -> str:
    if values and sum(to_date(value) is not None for value in values) / len(values) >= .7:
        return "date"
    if values and sum(to_number(value) is not None for value in values) / len(values) >= .7:
        return "number"
    return "category" if len({str(value) for value in values}) <= min(20, max(4, len(values) // 4)) else "text"


def choose_measure_column(columns: list[AnalyticsColumn]) -> str | None:
    """Prefer a real metric over technical row numbers such as Rk or ID."""
    numeric = [column for column in columns if column.kind == "number"]
    if not numeric:
        return None
    technical_names = {"id", "key", "index", "row", "rank", "rk", "no", "number", "#", "age", "g", "gs", "mp", "year"}
    meaningful = [
        column for column in numeric
        if column.name.strip().lower().replace("_", " ").replace("-", " ") not in technical_names
        and not column.name.strip().lower().startswith(("id_", "id "))
    ]
    priority_groups = (
        ("revenue", "sales", "amount", "value", "profit", "income"),
        ("points", "point", "pts", "score"),
        ("rebounds", "rebound", "trb"),
        ("assist", "ast"),
        ("salary", "price", "rate", "percent", "percentage"),
    )
    for terms in priority_groups:
        prioritized = [column for column in meaningful if any(term == column.name.lower() or term in column.name.lower() for term in terms)]
        if prioritized:
            return prioritized[0].name
    return (meaningful or numeric)[0].name


def to_number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", "").replace("$", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def to_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(value), pattern)
        except (TypeError, ValueError):
            pass
    return None


def build_trend(values: list[tuple[Any, dict[str, Any]]], date_column: str | None, measure_column: str) -> list[AnalyticsPoint]:
    grouped: dict[str, float] = defaultdict(float)
    for value, row in values:
        label = to_date(row[date_column]).strftime("%b %Y") if date_column and to_date(row[date_column]) else "All records"
        grouped[label] += float(value)
    return [AnalyticsPoint(label=label, value=round(value, 2)) for label, value in list(grouped.items())[-12:]]


def build_categories(values: list[tuple[Any, dict[str, Any]]], category_column: str | None) -> list[AnalyticsCategory]:
    if not category_column:
        return []
    grouped: dict[str, float] = defaultdict(float)
    for value, row in values:
        grouped[str(row[category_column] or "Uncategorized")] += float(value)
    return [AnalyticsCategory(label=label, value=round(value, 2)) for label, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:8]]


def build_distribution(values: list[float]) -> list[AnalyticsPoint]:
    if not values:
        return []
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return [AnalyticsPoint(label=f"{minimum:,.1f}", value=len(values))]
    step = (maximum - minimum) / 6
    buckets = [0] * 6
    for value in values:
        buckets[min(5, int((value - minimum) / step))] += 1
    return [AnalyticsPoint(label=f"{minimum + index * step:,.1f}", value=count) for index, count in enumerate(buckets)]