# parser.py
from __future__ import annotations
import json
import re
from typing import Optional

from pydantic import BaseModel, field_validator
from openai import OpenAI
from loadbot.config import AI_TOKEN

client = OpenAI(api_key=AI_TOKEN)

class ExtractedLoad(BaseModel):
    load_number: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    pickup_appointment: Optional[str] = None
    delivery_appointment: Optional[str] = None
    commodity: Optional[str] = None
    weight_lbs: Optional[int] = None
    total_rate_usd: Optional[float] = None
    broker_name: Optional[str] = None
    loaded_miles: Optional[int] = None
    rpm_usd: Optional[float] = None

    # NEW: coerce text fields (model sometimes returns numbers)
    @field_validator(
        "load_number",
        "origin",
        "destination",
        "pickup_appointment",
        "delivery_appointment",
        "commodity",
        "broker_name",
        mode="before",
    )
    def _as_str(cls, v):
        if v is None:
            return v
        return str(v).strip()

    @field_validator("weight_lbs", mode="before")
    def _clean_weight(cls, v):
        if v is None: return v
        import re
        s = re.sub(r"[^\d]", "", str(v))
        return int(s) if s else None

    @field_validator("total_rate_usd", "rpm_usd", mode="before")
    def _clean_money(cls, v):
        if v is None: return v
        s = str(v).replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            return None

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

SYSTEM = (
    "You extract structured load data from text and return ONLY valid JSON (no code fences). "
    "Keys: load_number, origin, destination, pickup_appointment, delivery_appointment, commodity, "
    "weight_lbs, total_rate_usd, broker_name, loaded_miles, rpm_usd. "
    "load_number MUST be a string (preserve any leading zeros or letters). "
    "If something is missing, use null. Put '$' sign in front of amounts."
)

USER_TEMPLATE = """Extract the following fields from the load confirmation text:

- load_number
- origin (City, State only)
- destination (City, State only)
- pickup_appointment (earliest if ranges)
- delivery_appointment (earliest if ranges)
- commodity
- weight_lbs
- total_rate_usd
- loaded_miles
- rpm_usd = total_rate_usd / loaded_miles (if both present)
- broker_name

Return ONLY a JSON object with those exact keys. Text:
---
{doc}
---
"""

def _call_model(prompt: str) -> str:
    # Prefer strict JSON mode when available
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

def get_json(text: str) -> dict:
    """
    Returns a validated dict matching ExtractedLoad.
    Retries once if JSON is malformed.
    """
    prompt = USER_TEMPLATE.format(doc=text)

    raw = _call_model(prompt)
    try:
        data = json.loads(_strip_code_fences(raw))
    except Exception:
        # one more try with a harder instruction
        raw = _call_model(prompt + "\nReturn ONLY valid minified JSON.")
        data = json.loads(_strip_code_fences(raw))

    model = ExtractedLoad.model_validate(data)

    # Backfill rpm if possible
    if model.rpm_usd is None and model.total_rate_usd and model.loaded_miles:
        try:
            model.rpm_usd = round(model.total_rate_usd / float(model.loaded_miles), 2)
        except ZeroDivisionError:
            pass

    return model.model_dump(exclude_none=True)