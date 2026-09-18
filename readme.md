# BUP Campus Energy Optimizer

Backend service built for campus energy scheduling and optimization. The application parses unstructured operator directives using an LLM, applies deterministic guardrails to ensure constraint safety, and solves a 24-hour linear programming cost-minimization problem.

## System Architecture

1. **Directive Extraction:** Queries the Groq API (`llama-3.3-70b-versatile`) with a strict JSON schema prompt to convert text notes into structured adjustments.
2. **Deterministic Guardrails:** Sanitizes LLM responses by checking numeric bounds, normalizing time windows (start-inclusive, end-exclusive), and safely falling back to `no_op` values on malformed input.
3. **Optimization Engine:** Formulates and solves a 24-hour linear programming model via PuLP (CBC solver), handling energy balance, battery capacity, state transitions, and day-neutrality constraints.

## Tech Stack

* **Framework:** FastAPI / Uvicorn
* **Validation:** Pydantic v2
* **LLM Provider:** Groq API (`AsyncGroq`)
* **Solver:** PuLP (CBC)

---

## Local Setup

### 1. Clone & Navigate
```bash
git clone [https://github.com/prottoy83/BUPBattery.git](https://github.com/prottoy83/BUPBattery.git)
cd BUPBattery
python -m venv venv
source venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
### Dependencies
```bash
pip install fastapi uvicorn pydantic groq pulp python-dotenv
```

