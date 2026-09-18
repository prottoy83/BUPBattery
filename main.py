import os
import math
from typing import List, Optional, Literal

import pulp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import AsyncGroq

load_dotenv()
app = FastAPI(title="GridWise Campus Energy Optimizer")
client = AsyncGroq(api_key=os.getenv("AI_KEY"))

# =========================================================================
# 1. REQUEST & RESPONSE SCHEMAS (Sections 07 & 10)
# =========================================================================

class HourInput(BaseModel):
    hour: int
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float


class BatteryInput(BaseModel):
    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float


class OptimizationRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str]
    hours: List[HourInput]
    battery: BatteryInput


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


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizationResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# =========================================================================
# 2. DETERMINISTIC GUARDRAILS (Section 08)
# =========================================================================

def validate_hourly_input(hours: List[HourInput]) -> List[HourInput]:
    if len(hours) != 24:
        raise HTTPException(status_code=400, detail="Request must contain exactly 24 hourly entries.")

    hour_ids = [h.hour for h in hours]
    if sorted(hour_ids) != list(range(24)):
        raise HTTPException(status_code=400, detail="Hourly data must cover hours 0 through 23 exactly once.")

    for h in hours:
        if not all(math.isfinite(value) for value in [h.demand_kwh, h.solar_kwh, h.tariff_bdt_per_kwh]):
            raise HTTPException(status_code=400, detail="Hourly values must be finite numbers.")
        if h.demand_kwh < 0 or h.solar_kwh < 0 or h.tariff_bdt_per_kwh < 0:
            raise HTTPException(status_code=400, detail="Hourly demand, solar, and tariff values must be non-negative.")

    return sorted(hours, key=lambda x: x.hour)


def validate_battery_input(battery: BatteryInput) -> None:
    required = [
        battery.capacity_kwh,
        battery.initial_energy_kwh,
        battery.minimum_energy_kwh,
        battery.max_charge_kwh_per_hour,
        battery.max_discharge_kwh_per_hour,
    ]
    if not all(math.isfinite(v) for v in required):
        raise HTTPException(status_code=400, detail="Battery values must be finite numbers.")

    if battery.capacity_kwh <= 0:
        raise HTTPException(status_code=400, detail="Battery capacity must be positive.")
    if battery.initial_energy_kwh < 0 or battery.initial_energy_kwh > battery.capacity_kwh:
        raise HTTPException(status_code=400, detail="Initial battery energy must be within capacity bounds.")
    if battery.minimum_energy_kwh < 0 or battery.minimum_energy_kwh > battery.capacity_kwh:
        raise HTTPException(status_code=400, detail="Minimum battery reserve must be within capacity bounds.")
    if battery.max_charge_kwh_per_hour < 0 or battery.max_discharge_kwh_per_hour < 0:
        raise HTTPException(status_code=400, detail="Charge/discharge rates must be non-negative.")


def apply_guardrails(
    parsed_directives: List[DirectiveInterpretation],
    total_notes: int,
    capacity_kwh: float,
) -> List[DirectiveInterpretation]:
    """
    Validates LLM interpretations against strict rules. If any violation is
    detected, it falls back safely to no_op for that specific note index.
    """
    safe_directives = []
    directive_map = {d.note_index: d for d in parsed_directives}

    for i in range(total_notes):
        if i not in directive_map:
            safe_directives.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation="Guardrail Fallback: Note index missing from LLM response.",
                )
            )
            continue

        d = directive_map[i]
        is_valid = True

        if d.directive_type == "no_op":
            d.applies = False
            d.structured_adjustment = None
        else:
            d.applies = True
            if not d.structured_adjustment:
                is_valid = False
            else:
                raw_hours = d.structured_adjustment.hours or []
                valid_hours = sorted({h for h in raw_hours if 0 <= h <= 23})
                d.structured_adjustment.hours = valid_hours

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

        if not is_valid:
            safe_directives.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation="Guardrail Fallback: Directive structure or value bounds failed validation.",
                )
            )
        else:
            safe_directives.append(d)

    return safe_directives


# =========================================================================
# 3. MATHEMATICAL OPTIMIZATION ENGINE (Sections 05 & 09)
# =========================================================================


def run_pulp_optimization(
    sorted_hours: List[HourInput],
    battery: BatteryInput,
    directives: List[DirectiveInterpretation],
):
    """
    Formulates and solves the 24-hour Linear Programming cost minimization problem.
    """
    effective_solar_max = [h.solar_kwh for h in sorted_hours]
    demands = [h.demand_kwh for h in sorted_hours]
    tariffs = [h.tariff_bdt_per_kwh for h in sorted_hours]

    b_cap = battery.capacity_kwh
    b_init = battery.initial_energy_kwh
    b_min_default = battery.minimum_energy_kwh

    min_battery_limits = [b_min_default] * 24
    max_charge_limits = [battery.max_charge_kwh_per_hour] * 24
    max_discharge_limits = [battery.max_discharge_kwh_per_hour] * 24
    max_grid_limits = [float("inf")] * 24

    for d in directives:
        if not d.applies or d.directive_type == "no_op" or not d.structured_adjustment:
            continue

        adj = d.structured_adjustment
        hours = adj.hours or []

        if d.directive_type == "solar_reduction":
            factor = adj.factor if adj.factor is not None else 1.0
            for h in hours:
                effective_solar_max[h] = effective_solar_max[h] * factor

        elif d.directive_type == "no_charge_window":
            for h in hours:
                max_charge_limits[h] = 0.0

        elif d.directive_type == "no_discharge_window":
            for h in hours:
                max_discharge_limits[h] = 0.0

        elif d.directive_type == "minimum_battery_reserve":
            reserve = adj.minimum_energy_kwh if adj.minimum_energy_kwh is not None else b_min_default
            for h in hours:
                min_battery_limits[h] = max(min_battery_limits[h], reserve)

        elif d.directive_type == "max_grid_window":
            grid_cap = adj.max_grid_kwh if adj.max_grid_kwh is not None else float("inf")
            for h in hours:
                max_grid_limits[h] = min(max_grid_limits[h], grid_cap)

    model = pulp.LpProblem("GridWise_Optimization", pulp.LpMinimize)

    grid = [
        pulp.LpVariable(f"grid_{h}", lowBound=0, upBound=max_grid_limits[h])
        for h in range(24)
    ]
    solar_used = [
        pulp.LpVariable(f"solar_used_{h}", lowBound=0, upBound=effective_solar_max[h])
        for h in range(24)
    ]
    charge = [
        pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=max_charge_limits[h])
        for h in range(24)
    ]
    discharge = [
        pulp.LpVariable(f"discharge_{h}", lowBound=0, upBound=max_discharge_limits[h])
        for h in range(24)
    ]
    battery_state = [
        pulp.LpVariable(f"b_state_{h}", lowBound=min_battery_limits[h], upBound=b_cap)
        for h in range(24)
    ]

    model += pulp.lpSum([grid[h] * tariffs[h] for h in range(24)])

    for h in range(24):
        model += (grid[h] + solar_used[h] + discharge[h] == demands[h] + charge[h])

        if h == 0:
            model += (battery_state[0] == b_init + charge[0] - discharge[0])
        else:
            model += (battery_state[h] == battery_state[h - 1] + charge[h] - discharge[h])

    model += (battery_state[23] == b_init)

    solver_status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[solver_status] not in ("Optimal", "Not Solved"):
        raise HTTPException(status_code=500, detail=f"Optimization failed: {pulp.LpStatus[solver_status]}")

    if pulp.LpStatus[solver_status] == "Not Solved":
        raise HTTPException(status_code=500, detail="Optimization solver did not find a valid schedule.")

    hourly_plan = []
    total_grid_kwh = 0.0
    total_cost_bdt = 0.0
    peak_grid_kwh = 0.0

    for h in range(24):
        g_val = round(max(0.0, grid[h].varValue or 0.0), 2)
        s_val = round(max(0.0, solar_used[h].varValue or 0.0), 2)
        c_val = round(max(0.0, charge[h].varValue or 0.0), 2)
        d_val = round(max(0.0, discharge[h].varValue or 0.0), 2)
        b_val = round(battery_state[h].varValue or 0.0, 2)

        if c_val > 0.01:
            action = "charge"
            net_battery = c_val
        elif d_val > 0.01:
            action = "discharge"
            net_battery = d_val
        else:
            action = "idle"
            net_battery = 0.0

        cost_for_hour = g_val * tariffs[h]
        total_grid_kwh += g_val
        total_cost_bdt += cost_for_hour

        if g_val > peak_grid_kwh:
            peak_grid_kwh = g_val

        hourly_plan.append(
            {
                "hour": h,
                "grid_kwh": g_val,
                "solar_used_kwh": s_val,
                "battery_action": action,
                "battery_kwh": net_battery,
                "battery_energy_after_kwh": b_val,
            }
        )

    return {
        "hourly_plan": hourly_plan,
        "total_grid_kwh": round(total_grid_kwh, 2),
        "total_cost_bdt": round(total_cost_bdt, 2),
        "peak_grid_kwh": round(peak_grid_kwh, 2),
    }


# =========================================================================
# 4. FASTAPI HTTP ENDPOINTS (Section 06)
# =========================================================================

@app.get("/health")
async def health_check():
    """Section 06 Requirement: GET /health returns HTTP 200 with status='ok'."""
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizationResponse)
async def optimize_energy(request_data: OptimizationRequest):
    """
    Section 06 Requirement: POST /optimize-energy receives scenario payload
    and returns structured interpretations + mathematically optimal schedule.
    """
    scene_id = request_data.scenario_id
    raw_context = request_data.operator_notes

    validate_battery_input(request_data.battery)
    sorted_hours = validate_hourly_input(request_data.hours)

    pre_llm = "\n".join([f"[{i}]: {note}" for i, note in enumerate(raw_context)])

    system_prompt = """
You are a strict energy scheduling assistant for BUP CSE Fest 2026.

Your task is to convert every numbered operator note into exactly ONE directive object.

The input notes are numbered like:
[0]: ...
[1]: ...
[2]: ...

You MUST return JSON using EXACTLY this schema:
{
  "directives": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {
        "hours": [13, 14],
        "factor": 0.2,
        "minimum_energy_kwh": null,
        "max_grid_kwh": null
      },
      "explanation": "Solar output dropped to 20% between 1 PM and 3 PM."
    }
  ]
}

STRICT OPERATOR DIRECTIVE RULES:
1. `note_index` must be an integer matching the exact note index [0, 1, ...].
2. `directive_type` must be EXACTLY ONE of:
   "solar_reduction", "minimum_battery_reserve", "no_charge_window", "no_discharge_window", "max_grid_window", "no_op".
3. TIME WINDOW CONVENTION: Time windows use whole-hour intervals. Start hour is INCLUDED, end hour is EXCLUDED.
   - Example: "1 PM to 3 PM" -> hours [13, 14]
   - Example: "2 PM and 4 PM" -> hours [14, 15]
   - Example: "6 PM until 9 PM" -> hours [18, 19, 20]
4. `solar_reduction`: `factor` represents the remaining fraction (e.g., 80% reduction means factor = 0.2).
5. For `no_op` (irrelevant or non-energy notes):
   - `applies` = false
   - `directive_type` = "no_op"
   - `structured_adjustment` = null
6. For all applicable directives:
   - `applies` = true
   - `structured_adjustment` must NOT be null. Set unused adjustment fields to null.
7. Return ONLY valid JSON matching this schema.
"""

    try:
        completion = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pre_llm},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw_json_response = completion.choices[0].message.content
        parsed_data = LLMInterpretation.model_validate_json(raw_json_response)
        raw_directives = parsed_data.directives

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Interpretation failed: {str(e)}")

    total_notes = len(raw_context)
    interpreted_directives = apply_guardrails(raw_directives, total_notes, request_data.battery.capacity_kwh)

    try:
        opt_results = run_pulp_optimization(
            sorted_hours=sorted_hours,
            battery=request_data.battery,
            directives=interpreted_directives,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization solver failed: {str(e)}")

    applied_count = sum(1 for d in interpreted_directives if d.applies)
    summary_text = (
        f"Processed {len(raw_context)} operator note(s) with {applied_count} active directive(s). "
        f"Optimized 24-hour cost to {opt_results['total_cost_bdt']} BDT "
        f"using {opt_results['total_grid_kwh']} kWh grid energy with a peak demand of {opt_results['peak_grid_kwh']} kWh."
    )

    return OptimizationResponse(
        scenario_id=scene_id,
        directive_interpretation=interpreted_directives,
        hourly_plan=[HourlyPlanEntry(**item) for item in opt_results["hourly_plan"]],
        total_grid_kwh=opt_results["total_grid_kwh"],
        total_cost_bdt=opt_results["total_cost_bdt"],
        peak_grid_kwh=opt_results["peak_grid_kwh"],
        plan_summary=summary_text,
    )


@app.get("/models")
async def get_models():
    try:
        models = await client.models.list()
        return {
            "models": [
                {
                    "id": model.id,
                    "owned_by": model.owned_by,
                }
                for model in models.data
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
