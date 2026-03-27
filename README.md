---
title: Delivery Worker Assignment OpenEnv
emoji: "🚚"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - fastapi
  - logistics
---

# Delivery Worker Assignment Environment

Production-grade OpenEnv-style logistics environment for assigning delivery orders to workers under distance, deadline, and capacity constraints.

## Problem

An agent must manage a fleet of delivery workers over discrete timesteps. Each task contains a fixed set of orders and workers. Orders have pickup and dropoff coordinates plus delivery deadlines. Some orders are marked high priority, which makes on-time service more valuable and lateness more costly. Workers have locations, capacity limits, and assignment queues. The agent chooses from three validated actions:

- `assign_order(order_id, worker_id)`
- `reassign_order(order_id, worker_id)`
- `advance_time()`

The environment is deterministic. The same task and action sequence always produce the same state trajectory, rewards, and grader score.

## OpenEnv Requirements Covered

- `step(action) -> observation, reward, done, info`
- `reset() -> observation`
- `state() -> internal state`
- Typed Pydantic models for `Action`, `Observation`, `Reward`, `State`, and task metadata
- Root `openenv.yaml`
- FastAPI service endpoints for environment interaction and evaluation
- Dockerized execution
- Hugging Face Spaces compatibility through `app.py` and Docker

## State Design

State tracks:

- `orders`
  - `id`
  - `pickup_location`
  - `drop_location`
  - `deadline`
  - `priority`
  - `status`
  - assignment and delivery timestamps
- `workers`
  - `id`
  - `current_location`
  - `capacity`
  - `assigned_orders`
  - `status`
- `time`
- aggregate delivery metrics

Workers service assigned orders in queue order. Each `advance_time` step moves a worker by one Manhattan unit toward its current pickup or dropoff target.

## Observation Schema

Observations expose:

- all orders with status and assignment metadata
- all worker states
- current timestep and max horizon
- summary counts for pending, assigned, delivered, and remaining time

The schema is available from `/schema`.

## Reward Logic

The reward is dense and additive:

- `+1.0` for each order delivered on time
- up to `+0.3` for assigning an order to a near-optimal worker
- `+0.1` each time a worker moves closer to its current target
- `+0.2` extra for on-time high-priority deliveries
- `-0.5` when an order misses its deadline
- `-0.2` extra when a high-priority order misses its deadline
- `-0.2` for invalid actions
- `-0.1` per idle worker when active orders remain
- fairness penalty when active queue load is too uneven
- reassignment penalty to discourage churn

Each step returns a scalar `reward` for compatibility, while `info.reward_breakdown` preserves the full `DeliveryReward` component model.

## Tasks

### Easy

- 4 orders
- 3 workers
- 1 high-priority order
- mildly tighter deadlines
- objective: assign cleanly without overloading one worker

### Medium

- 5 orders
- 2 workers
- 2 high-priority orders
- distance and sequencing matter
- objective: avoid wasting travel while keeping moderate deadlines

### Hard

- 8 orders
- 3 workers with uneven queue capacity limits
- 3 high-priority rush orders
- tighter deadlines and harsher reassignment tradeoffs
- objective: plan across multiple timesteps and use reassignment only when it is worth the cost

List all tasks and the action schema at runtime with:

```bash
curl http://localhost:8000/tasks
```

## Deterministic Grader

The grader replays a submitted action sequence against the fixed task instance and computes:

- on-time delivery rate
- completion rate
- delivery efficiency score based on distance traveled versus a deterministic lower bound
- worker fairness score based on delivery load balance
- high-priority service rate
- late or missed order ratio
- invalid action ratio

Final score:

```text
score = clamp(
  0.60 * on_time_delivery_rate
  + 0.10 * efficiency_score
  + 0.05 * fairness_score
  + 0.10 * priority_service_rate
  - 0.10 * late_delivery_ratio
  - 0.05 * invalid_action_ratio,
  0.0,
  1.0
)
```

The same `task_id` and `actions` payload always produce the same result.

## API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`
- `GET /schema`
- `GET /health`

### Reset Example

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy"}'
```

### Step Example

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "assign_order",
      "order_id": "O-101",
      "worker_id": "W-1"
    }
  }'
```

The response shape is:

```json
{
  "observation": { "...": "..." },
  "reward": 0.2889,
  "done": false,
  "info": {
    "reward_breakdown": {
      "total": 0.2889
    }
  }
}
```

### Grader Example

```bash
curl -X POST http://localhost:8000/grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "easy",
    "actions": [
      {"action_type":"assign_order","order_id":"O-101","worker_id":"W-1"},
      {"action_type":"advance_time"}
    ]
  }'
```

The grader response includes a top-level scalar `score` and the detailed deterministic result payload:

```json
{
  "score": 0.78,
  "result": {
    "score": 0.78
  }
}
```

## Project Structure

```text
api/
baseline/
env/
grader/
models/
tasks/
tests/
app.py
Dockerfile
openenv.yaml
pyproject.toml
README.md
requirements.txt
```

## Setup

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t delivery-worker-assignment-env .
docker run --rm -p 8000:8000 delivery-worker-assignment-env
```

## OpenAI Baseline

The baseline agent lives in `baseline/run_baseline.py`. It:

- fetches task metadata from `/tasks`
- resets each task
- asks the OpenAI Responses API for one structured action at a time
- falls back to Gemini if OpenAI is unavailable or out of quota and `GEMINI_API_KEY` is configured
- falls back to a deterministic heuristic if neither model path is available
- submits the final action trace to `/grader`
- reports `provider_used`, `providers_attempted`, and `providers_succeeded` per task so model fallback behavior is explicit

Environment variables:

- `OPENAI_API_KEY` preferred primary provider
- `GEMINI_API_KEY` optional fallback provider
- `OPENAI_MODEL` optional, defaults to `gpt-5.4-mini`
- `GEMINI_MODEL` optional, defaults to `gemini-2.5-flash`
- `BASE_URL` optional, defaults to `http://localhost:8000`

Run it with:

```bash
python -m baseline.run_baseline --base-url http://localhost:8000
```

## Hugging Face Spaces

This repository is compatible with Hugging Face Docker Spaces:

- root `Dockerfile` exposes port `8000`
- root `app.py` exports `app`
- `openenv.yaml` points to `api.app:app`

For a Docker Space, push the repository as-is and set the Space SDK to Docker.

## Notes

- Rewards are deterministic and never sampled randomly.
- Tasks are fixed and reproducible.
- Invalid actions are always penalized.
- The environment is stateful across steps until `/reset` is called.
