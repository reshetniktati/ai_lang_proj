
import os
import re
import json
import math
import argparse
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from openpyxl import load_workbook
from openpyxl.comments import Comment
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it to your .env file."
    )

llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    use_responses_api=True,
    max_retries=3,
)

web_llm = llm.bind_tools(
    [{"type": "web_search"}],
    tool_choice="required",
)

MAX_RECORDS = 1000
MAX_WORKERS = 4


# ==========================================
# 2. STRUCTURED OUTPUT MODELS
# ==========================================

class EnrichmentPlan(BaseModel):
    mode: str = Field(
        description="'distance' for straight-line city distance, "
        "otherwise 'lookup' for internet research"
    )

    target_column: str = Field(
        description="Existing or new column to enrich"
    )

    source_columns: list[str] = Field(
        description="Existing columns needed to obtain the value"
    )

    unit: str = Field(
        description="Output unit, for example km, m, or empty"
    )

    search_template: str = Field(
        description="Search query containing column placeholders"
    )


class ExtractedValue(BaseModel):
    value: float | str | None = Field(
        description="Extracted value, or null if unknown"
    )

    source_url: str | None = Field(
        description="URL of the supporting source, if available"
    )


planner_llm = llm.with_structured_output(EnrichmentPlan)
extractor_llm = llm.with_structured_output(ExtractedValue)


# ==========================================
# 3. HELPERS
# ==========================================

def message_text(message):
    """Extract plain text from a LangChain AIMessage."""

    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []

        for block in content:
            if isinstance(block, str):
                texts.append(block)

            elif isinstance(block, dict):
                if block.get("type") in (
                    "text",
                    "output_text",
                ):
                    texts.append(block.get("text", ""))

        return "\n".join(texts)

    return str(content)


def source_urls(message):
    """Collect URLs from web-search citation metadata."""

    urls = set()

    def walk(obj):
        if isinstance(obj, dict):
            url = obj.get("url")

            if (
                isinstance(url, str)
                and url.startswith(("http://", "https://"))
            ):
                urls.add(url)

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(message.content)
    walk(message.additional_kwargs)
    walk(message.response_metadata)

    return sorted(urls)


def normalize_number(value):
    """Convert a numeric value to a rounded integer."""

    if isinstance(value, bool):
        raise ValueError("Boolean is not a measurement")

    if isinstance(value, str):
        value = value.strip().replace(",", ".")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError("Invalid numeric value")

    # Conventional half-up rounding for nonnegative measurements.
    if number < 0:
        raise ValueError("Measurement cannot be negative")

    return math.floor(number + 0.5)


def is_empty(value):
    return value is None or (
        isinstance(value, str) and not value.strip()
    )


# ==========================================
# 4. AI PLANNER
# ==========================================

def plan_task(headers, sample_rows, task_description):
    """Analyze the Excel structure and user request."""

    print("\n[1] Analyzing Excel structure...")

    system_prompt = """
You are an intelligent Excel enrichment planner.

Your job is to understand:
1. What information the user wants.
2. Which Excel columns provide the context.
3. Which column should be filled.
4. Whether web research or calculation is required.

Rules:
- Use mode='distance' only for straight-line
  geographical distance between cities.
- For distance, include the origin and destination
  city columns, plus country columns if present.
- Use mode='lookup' for other factual research.
- All source_columns must exist in the input.
- target_column must be the exact existing column
  name if it already exists.
- Never invent source columns.
- Use column placeholders in search_template.

Examples:

Task: Find direct distance between capitals in km.
Headers: Capital_From, Country_From,
Capital_To, Country_To, distance

Result:
mode: distance
target_column: distance
source_columns:
  - Capital_From
  - Country_From
  - Capital_To
  - Country_To
unit: km
search_template:
  Straight-line distance {Capital_From} {Country_From}
  to {Capital_To} {Country_To} km

Task: Find mountain heights in meters.
Headers: Mountain, Country, height

Result:
mode: lookup
target_column: height
source_columns:
  - Mountain
  - Country
unit: m
search_template:
  {Mountain} {Country} elevation height meters
"""

    plan = planner_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=json.dumps(
                {
                    "headers": headers,
                    "sample_rows": sample_rows[:5],
                    "task": task_description,
                },
                ensure_ascii=False,
                default=str,
            )
        ),
    ])

    if plan.mode not in ("distance", "lookup"):
        raise ValueError("Unsupported task mode")

    if not plan.target_column.strip():
        raise ValueError("Missing target column")

    if not plan.source_columns:
        raise ValueError("No source columns selected")

    for column in plan.source_columns:
        if column not in headers:
            raise ValueError(
                f"Unknown source column: {column}"
            )

    if plan.mode == "distance":
        city_columns = [
            col for col in plan.source_columns
            if "country" not in col.lower()
        ]

        if len(city_columns) < 2:
            raise ValueError(
                "Distance task requires two city columns"
            )

    print(
        json.dumps(
            plan.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )

    return plan


# ==========================================
# 5. WEB RESEARCH
# ==========================================

def build_search_query(row, plan):
    """Build a search query from the row context."""

    template = plan.search_template

    placeholders = re.findall(
        r"\{([^{}]+)\}",
        template,
    )

    if all(name in row for name in placeholders):
        query = template

        for name in placeholders:
            query = query.replace(
                "{" + name + "}",
                str(row.get(name) or ""),
            )

        if query.strip():
            return query.strip()

    values = [
        str(row[col])
        for col in plan.source_columns
        if not is_empty(row.get(col))
    ]

    return " ".join(
        values + [plan.target_column, plan.unit]
    )


def research_value(row, plan):
    """Search the web and extract a verified value."""

    for column in plan.source_columns:
        if is_empty(row.get(column)):
            raise ValueError(
                f"Missing value in {column}"
            )

    query = build_search_query(row, plan)

    print(f"[WEB] {query}")

    research_response = web_llm.invoke([
        HumanMessage(
            content=f"""
Search the web for:

{query}

Context:
{json.dumps(row, ensure_ascii=False, default=str)}

Requirements:
- Find the requested factual value.
- Prefer authoritative sources.
- Pay attention to the requested unit.
- Do not confuse similar entities.
- Cite supporting sources when possible.
- State clearly if the value is not found.
"""
        )
    ])

    research = message_text(research_response)
    urls = source_urls(research_response)

    if not research.strip():
        raise LookupError(
            f"No web research text returned for: {query}"
        )

    extraction_prompt = """
You extract factual values from web research.

Rules:
- Use only the provided research.
- Never invent a value.
- Return null when evidence is insufficient.
- Convert measurements to the requested unit.
- Return a numeric value for measurements.
- Choose a source URL only if it appears
  in the supplied evidence or allowed_urls.
- If URLs are missing, source_url should be null.
"""

    result = extractor_llm.invoke([
        SystemMessage(content=extraction_prompt),
        HumanMessage(
            content=json.dumps(
                {
                    "row": row,
                    "target": plan.target_column,
                    "unit": plan.unit,
                    "research": research,
                    "allowed_urls": urls,
                },
                ensure_ascii=False,
                default=str,
            )
        ),
    ])

    if result.value is None:
        raise LookupError(
            f"No reliable value found for: {query}"
        )

    value = result.value

    if plan.unit.lower() in (
        "m",
        "meters",
        "metres",
        "km",
        "kilometers",
        "kilometres",
    ):
        value = normalize_number(value)

    source = result.source_url

    if source and urls and source not in urls:
        source = None

    if source and not source.startswith(
        ("https://", "http://")
    ):
        source = None

    return value, source


# ==========================================
# 6. GEOLOCATION
# ==========================================

@lru_cache(maxsize=4096)
def get_coordinates(city, country=""):
    """
    Obtain geographical coordinates.

    Uses the Open-Meteo Geocoding API.
    """

    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 100,
            "language": "en",
            "format": "json",
        },
        timeout=30,
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        raise LookupError(
            f"Coordinates not found for {city}"
        )

    def normalize(value):
        return str(value or "").casefold().strip()

    exact_cities = [
        item for item in results
        if normalize(item.get("name")) == normalize(city)
    ]

    candidates = exact_cities or results

    if country:
        matching_country = [
            item for item in candidates
            if (
                normalize(item.get("country"))
                == normalize(country)
                or normalize(item.get("country_code"))
                == normalize(country)
            )
        ]

        if not matching_country:
            raise LookupError(
                f"No matching city/country: {city}, {country}"
            )

        candidates = matching_country

    best = max(
        candidates,
        key=lambda item: item.get("population") or 0,
    )

    return best["latitude"], best["longitude"]


# ==========================================
# 7. HAVERSINE DISTANCE
# ==========================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line great-circle distance
    between two geographical coordinates in km.
    """

    radius = 6371.0088

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        (lat1, lon1, lat2, lon2),
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    distance = 2 * radius * math.asin(math.sqrt(a))

    return normalize_number(distance)


def calculate_distance(row, plan):
    """Calculate distance from columns selected by the AI."""

    columns = plan.source_columns

    city_columns = [
        col for col in columns
        if "country" not in col.lower()
    ]

    from_columns = [
        col for col in city_columns
        if re.search(
            r"from|origin|start",
            col,
            re.IGNORECASE,
        )
    ]

    to_columns = [
        col for col in city_columns
        if re.search(
            r"to|destination|end",
            col,
            re.IGNORECASE,
        )
    ]

    city_from_col = (
        from_columns[0]
        if from_columns
        else city_columns[0]
    )

    city_to_col = (
        to_columns[0]
        if to_columns
        else next(
            col for col in city_columns
            if col != city_from_col
        )
    )

    city_from = row.get(city_from_col)
    city_to = row.get(city_to_col)

    country_from_col = next(
        (
            col for col in columns
            if re.search(r"country.*from", col, re.I)
        ),
        None,
    )

    country_to_col = next(
        (
            col for col in columns
            if re.search(r"country.*to", col, re.I)
        ),
        None,
    )

    country_from = (
        row.get(country_from_col, "")
        if country_from_col
        else ""
    )

    country_to = (
        row.get(country_to_col, "")
        if country_to_col
        else ""
    )

    if is_empty(city_from) or is_empty(city_to):
        raise ValueError("Origin or destination is missing")

    lat1, lon1 = get_coordinates(
        str(city_from),
        str(country_from or ""),
    )

    lat2, lon2 = get_coordinates(
        str(city_to),
        str(country_to or ""),
    )

    distance = haversine(
        lat1, lon1, lat2, lon2
    )

    source = (
        "https://open-meteo.com/en/docs/geocoding-api"
    )

    return distance, source


# ==========================================
# 8. INTELLIGENT DATA ENRICHMENT
# ==========================================

def enrich_row(row, plan):
    """Choose the correct processing strategy."""

    if plan.mode == "distance":
        return calculate_distance(row, plan)

    return research_value(row, plan)


# ==========================================
# 9. EXCEL PROCESSOR
# ==========================================

def process_excel(
    file_path,
    task_description,
    max_workers=MAX_WORKERS,
):
    """
    Enrich an Excel file using LangChain and OpenAI.

    Preserves original columns and existing values.
    Fills only empty cells in the target column.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    if file_path.suffix.lower() != ".xlsx":
        raise ValueError("Only .xlsx is supported")

    print(f"\nOpening: {file_path}")

    workbook = load_workbook(file_path)
    sheet = workbook.active

    headers = [
        str(cell.value).strip()
        if cell.value is not None
        else ""
        for cell in sheet[1]
    ]

    if not all(headers):
        raise ValueError("Empty Excel column headers")

    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate column headers")

    rows = []

    for row_index in range(2, sheet.max_row + 1):
        row_data = {
            header: sheet.cell(
                row_index,
                col_index,
            ).value
            for col_index, header in enumerate(headers, 1)
        }

        # Ignore completely empty rows.
        if all(
            is_empty(value)
            for value in row_data.values()
        ):
            continue

        rows.append((row_index, row_data))

    if not rows:
        raise ValueError("Excel contains no records")

    if len(rows) > MAX_RECORDS:
        raise ValueError(
            f"Maximum {MAX_RECORDS} records supported"
        )

    print(f"Records found: {len(rows)}")

    # --------------------------------------
    # Step 1: Create AI processing plan
    # --------------------------------------

    plan = plan_task(
        headers=headers,
        sample_rows=[row for _, row in rows],
        task_description=task_description,
    )

    # --------------------------------------
    # Step 2: Identify target column
    # --------------------------------------

    target_column = plan.target_column

    if target_column in headers:
        target_index = headers.index(target_column) + 1
    else:
        target_index = len(headers) + 1
        sheet.cell(1, target_index).value = target_column

    # --------------------------------------
    # Step 3: Prepare unique tasks
    # --------------------------------------

    jobs = {}

    for row_index, row in rows:
        current_value = sheet.cell(
            row_index,
            target_index,
        ).value

        # Preserve existing user data.
        if not is_empty(current_value):
            continue

        key = json.dumps(
            {
                col: row.get(col)
                for col in plan.source_columns
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        if key not in jobs:
            jobs[key] = {
                "row": row,
                "indices": [],
            }

        jobs[key]["indices"].append(row_index)

    print(f"Unique tasks: {len(jobs)}")

    report = {
        "input_file": str(file_path),
        "task": task_description,
        "model": MODEL,
        "plan": plan.model_dump(),
        "total_records": len(rows),
        "successful": 0,
        "failed": 0,
        "errors": [],
    }

    # --------------------------------------
    # Step 4: Execute tasks in parallel
    # --------------------------------------

    workers = max(1, min(int(max_workers), 5))

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                enrich_row,
                job["row"],
                plan,
            ): job
            for job in jobs.values()
        }

        for future in as_completed(futures):
            job = futures[future]

            try:
                value, source = future.result()

                for row_index in job["indices"]:
                    cell = sheet.cell(
                        row_index,
                        target_index,
                    )

                    cell.value = value

                    if isinstance(value, (int, float)):
                        cell.number_format = "0"

                    if source:
                        cell.comment = Comment(
                            f"Source: {source}",
                            "AI Excel Enricher",
                        )

                    report["successful"] += 1

                    print(
                        f"[OK] Row {row_index}: {value}"
                    )

            except Exception as error:
                for row_index in job["indices"]:
                    report["failed"] += 1

                    report["errors"].append({
                        "row": row_index,
                        "error": str(error),
                        "data": job["row"],
                    })

                    print(
                        f"[ERROR] Row {row_index}: {error}"
                    )

    # --------------------------------------
    # Step 5: Save enriched Excel
    # --------------------------------------

    output_path = file_path.with_name(
        f"{file_path.stem}_enriched.xlsx"
    )

    workbook.save(output_path)

    # --------------------------------------
    # Step 6: Save execution report
    # --------------------------------------

    log_path = file_path.with_name(
        f"{file_path.stem}_enrichment_log.json"
    )

    report["output_file"] = str(output_path)

    log_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n========== RESULT ==========")
    print(f"Output: {output_path}")
    print(f"Successful: {report['successful']}")
    print(f"Failed: {report['failed']}")
    print(f"Report: {log_path}")
    print("============================\n")

    return str(output_path)


# ==========================================
# 10. CLI / EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Excel Data Enrichment"
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        help="Excel file path",
    )

    parser.add_argument(
        "task_description",
        nargs="?",
        help="Natural-language enrichment task",
    )

    args = parser.parse_args()

    if args.file_path and args.task_description:
        process_excel(
            file_path=args.file_path,
            task_description=args.task_description,
        )

    else:
        # Example 1: Capitals
        process_excel(
            file_path="capitals.xlsx",
            task_description=(
                "Знайди пряму відстань між столицями "
                "в км для колонки distance"
            ),
        )

        # Example 2: Mountains
        process_excel(
            file_path="mountains.xlsx",
            task_description=(
                "Додай висоту гір у метрах "
                "до колонки height"
            ),
        )
