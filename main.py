import os
import pulp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv
from typing import List
from typing import List, Optional, Literal

import json
from groq import AsyncGroq


load_dotenv()
app = FastAPI()
client = AsyncGroq(api_key=os.getenv("AI_KEY"))



class Hour(BaseModel):
    hour: int
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float


class Battery(BaseModel):
    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float


class OptimizationRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str]
    hours: List[Hour]
    battery: Battery


class StructuredAdjustment(BaseModel):
    hours: Optional[List[int]] = None
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op"
    ]
    structured_adjustment: Optional[StructuredAdjustment] = None
    explanation: str


class LLMInterpretation(BaseModel):
    directives: List[DirectiveInterpretation]


# Guardrail

def apply_guardrails(parsed_directives: List[DirectiveInterpretation], total_notes: int, capacity_kwh: float) -> List[DirectiveInterpretation]:
    safe_directives = []
    
    # Map by note_index to easily check for missing notes
    directive_map = {d.note_index: d for d in parsed_directives}

    for i in range(total_notes):
        # 1. Missing Note Fallback
        if i not in directive_map:
            safe_directives.append(DirectiveInterpretation(
                note_index=i, 
                applies=False, 
                directive_type="no_op", 
                structured_adjustment=None, 
                explanation="Safe Fallback: LLM missed this note."
            ))
            continue
            
        d = directive_map[i]
        is_valid = True

        # 2. Enforce 'applies' Semantics
        if d.directive_type == "no_op":
            d.applies = False
            d.structured_adjustment = None
        else:
            d.applies = True
            if not d.structured_adjustment:
                is_valid = False
            else:
                # 3. Time Normalization: Unique integers 0-23 in ascending order
                raw_hours = d.structured_adjustment.hours
                if raw_hours is None:
                    is_valid = False
                else:
                    valid_hours = sorted(list(set([h for h in raw_hours if 0 <= h <= 23])))
                    d.structured_adjustment.hours = valid_hours
                    if not valid_hours:
                        is_valid = False

                # 4. Numeric Bounds Verification
                if d.directive_type == "solar_reduction":
                    f = d.structured_adjustment.factor
                    if f is None or not (0.0 <= f <= 1.0):
                        is_valid = False
                elif d.directive_type == "minimum_battery_reserve":
                    m = d.structured_adjustment.minimum_energy_kwh
                    if m is None or m < 0 or m > capacity_kwh:
                        is_valid = False
                elif d.directive_type == "max_grid_window":
                    g = d.structured_adjustment.max_grid_kwh
                    if g is None or g < 0:
                        is_valid = False

        # 5. Safe Failure Execution
        if not is_valid:
            safe_directives.append(DirectiveInterpretation(
                note_index=i, 
                applies=False, 
                directive_type="no_op", 
                structured_adjustment=None, 
                explanation="Safe Fallback: Directive violated strict numeric guardrails."
            ))
        else:
            safe_directives.append(d)

    return safe_directives



@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/optimize-energy")
async def optimize(request_data: OptimizationRequest):

    scene_id = request_data.scenario_id
    raw_context = request_data.operator_notes

    pre_llm = "\n".join(
        [f"[{i}]: {note}" for i, note in enumerate(raw_context)]
    )

    batt_capacity = request_data.battery.capacity_kwh
    batt_initial = request_data.battery.initial_energy_kwh
    batt_base_min = request_data.battery.minimum_energy_kwh
    max_charge_rate = request_data.battery.max_charge_kwh_per_hour
    max_discharge_rate = request_data.battery.max_discharge_kwh_per_hour

    sorted_hours = sorted(
        request_data.hours,
        key=lambda x: x.hour
    )

    demands = [h.demand_kwh for h in sorted_hours]
    tariffs = [h.tariff_bdt_per_kwh for h in sorted_hours]
    base_solar = [h.solar_kwh for h in sorted_hours]

    effective_solar = list(base_solar)

    min_battery_limits = [batt_base_min] * 24
    max_charge_limits = [max_charge_rate] * 24
    max_discharge_limits = [max_discharge_rate] * 24
    max_grid_limits = [float("inf")] * 24


    system_prompt = """
You are a strict energy scheduling assistant.

Your task is to convert every numbered operator note into exactly ONE
directive object.

The input notes are numbered like:

[0]: ...
[1]: ...
[2]: ...

You MUST return JSON using EXACTLY these field names:

{
  "directives": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {
        "hours": [12, 13],
        "factor": 0.25,
        "minimum_energy_kwh": null,
        "max_grid_kwh": null
      },
      "explanation": "..."
    }
  ]
}

IMPORTANT:

1. Use `note_index`, NOT `note_id`.
2. Use `directive_type`, NOT `type`.
3. Include exactly one object for every input note. `note_index` must exactly match the input note number.
4. Allowed `directive_type` values are ONLY: "solar_reduction", "minimum_battery_reserve", "no_charge_window", "no_discharge_window", "max_grid_window", "no_op".
5. CRITICAL TIME RULE: Time windows are start-inclusive and end-exclusive whole hours. For example, '1 PM to 3 PM' maps strictly to hours [13, 14]. 'Noon until 2 PM' maps strictly to hours [12, 13]. DO NOT include the final hour in the array.
6. `applies` must be true for every directive except `no_op`.
7. For `no_op`:
   - applies must be false
   - structured_adjustment must be null
8. For all other directive types:
   - applies must be true
   - structured_adjustment must NOT be null
9. `hours` is an array of integers from 0 to 23.
   Use null only when the directive does not require an hour range.
10. Do not use field names other than:
   - note_index
   - applies
   - directive_type
   - structured_adjustment
   - explanation
   - hours
   - factor
   - minimum_energy_kwh
   - max_grid_kwh
11. Return ONLY valid JSON.
"""

    try:
            
        completion = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pre_llm}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        raw_json_response = completion.choices[0].message.content
        print("===== LLM RAW RESPONSE =====")
        print(raw_json_response)
        print("============================")

        try:
            raw_json = json.loads(raw_json_response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM did not return valid JSON: {str(e)}")

        raw_directive_list = raw_json.get("directives", [])
        raw_directives = []

        for item in raw_directive_list:
            try:
                raw_directives.append(DirectiveInterpretation.model_validate(item))
            except Exception:
                continue

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LLM parsing failed: {str(e)}")

        raw_directive_list = raw_json.get("directives", [])
        raw_directives = []

        for item in raw_directive_list:
            try:
                raw_directives.append(DirectiveInterpretation.model_validate(item))
            except Exception:
                # Skip malformed entries; apply_guardrails() will fill in
                # missing note_index values with a safe no_op fallback.
                continue

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM parsing failed: {str(e)}")

    total_notes = len(raw_context)
    interpreted_directives = apply_guardrails(raw_directives, total_notes, batt_capacity)


    return {
        "debug_status": "Parsing Successful",
        "scenario_received": scene_id,
        "pre_llm_prompt": pre_llm,
        "battery_specs": {
            "capacity": batt_capacity,
            "starting_energy": batt_initial,
            "minimum_energy": batt_base_min,
            "max_charge_rate": max_charge_rate,
            "max_discharge_rate": max_discharge_rate
        },
        "hourly_data_summary": {
            "total_hours_received": len(sorted_hours),
            "hour_0_demand": demands[0],
            "hour_23_tariff": tariffs[-1]
        },
        "AI_Data":{
            "parsed_data": parsed_data,
            "int_dir": interpreted_directives
        }
    }


@app.get("/models")
async def get_models():
    try:
        models = await client.models.list()
        return {
            "models": [
                {
                    "id": model.id,
                    "owned_by": model.owned_by
                }
                for model in models.data
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )