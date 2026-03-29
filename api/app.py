from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, Request

from api.schemas import (
    BaselineRunResponse,
    GraderRequest,
    GraderResponse,
    ResetRequest,
    SchemaResponse,
    StateResponse,
    StepRequest,
    StepResponse,
    TasksResponse,
)
from baseline.prompting import heuristic_decision
from baseline.run_baseline import DEFAULT_MODEL, run_baseline, run_baseline_local
from env import DeliveryWorkerAssignmentEnv
from grader import grade_actions
from models import DeliveryAction, DeliveryObservation, DeliveryReward, DeliveryState
from tasks import list_task_summaries


class EnvironmentManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._env = DeliveryWorkerAssignmentEnv(task_id="easy")

    def reset(self, task_id: str) -> DeliveryObservation:
        with self._lock:
            return self._env.reset(task_id=task_id)

    def step(
        self, action: DeliveryAction
    ) -> tuple[DeliveryObservation, DeliveryReward, bool, dict]:
        with self._lock:
            return self._env.step(action)

    def state(self) -> DeliveryState:
        with self._lock:
            return self._env.state().model_copy(deep=True)


manager = EnvironmentManager()

app = FastAPI(
    title="Delivery Worker Assignment OpenEnv",
    version="0.1.0",
    description=(
        "Production-style OpenEnv-compatible logistics simulator for assigning "
        "delivery orders to workers under distance, deadline, and capacity constraints."
    ),
)


def _run_heuristic_baseline_local() -> list[dict]:
    results: list[dict] = []

    for task in list_task_summaries():
        env = DeliveryWorkerAssignmentEnv(task_id=task.id)
        observation = env.reset(task_id=task.id)
        actions: list[DeliveryAction] = []
        done = False

        while not done:
            action = DeliveryAction.model_validate(heuristic_decision(observation).model_dump())
            observation, _, done, info = env.step(action)
            if not info["valid_action"]:
                action = DeliveryAction(
                    action_type="advance_time",
                    order_id=None,
                    worker_id=None,
                )
                observation, _, done, info = env.step(action)
            actions.append(action)

        grade = grade_actions(task.id, actions)
        results.append(
            {
                "task_id": task.id,
                "task_name": task.name,
                "provider_used": "heuristic",
                "providers_attempted": [],
                "providers_succeeded": [],
                "score": grade.score,
                "delivered_orders": grade.delivered_orders,
                "delivered_on_time": grade.delivered_on_time,
                "total_distance_traveled": grade.total_distance_traveled,
                "invalid_actions": grade.invalid_actions,
            }
        )

    return results


@app.get("/")
def root() -> dict:
    return {
        "name": "Delivery Worker Assignment OpenEnv",
        "status": "running",
        "docs": "/docs",
        "required_endpoints": [
            "/tasks",
            "/grader",
            "/baseline",
            "/reset",
            "/step",
            "/state",
            "/schema",
            "/health",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset", response_model=DeliveryObservation)
def reset_environment(request: ResetRequest | None = None) -> DeliveryObservation:
    task_id = request.task_id if request is not None else "easy"
    return manager.reset(task_id)


@app.post("/step", response_model=StepResponse)
def step_environment(request: StepRequest) -> StepResponse:
    observation, reward, done, info = manager.step(request.action)
    response_info = dict(info)
    response_info["reward_breakdown"] = reward.model_dump()
    return StepResponse(
        observation=observation,
        reward=reward.total,
        done=done,
        info=response_info,
    )


@app.get("/state", response_model=StateResponse)
def get_state() -> StateResponse:
    return StateResponse(state=manager.state())


@app.get("/tasks", response_model=TasksResponse)
def get_tasks() -> TasksResponse:
    return TasksResponse(
        tasks=list_task_summaries(),
        action_schema=DeliveryAction.model_json_schema(),
    )


@app.post("/grader", response_model=GraderResponse)
def grade_submission(request: GraderRequest) -> GraderResponse:
    result = grade_actions(request.task_id, request.actions)
    return GraderResponse(score=result.score, result=result)


@app.get("/baseline", response_model=BaselineRunResponse)
def get_baseline_info(request: Request) -> BaselineRunResponse:
    base_url = str(request.base_url).rstrip("/")
    try:
        results = run_baseline(base_url=base_url, model=DEFAULT_MODEL)
    except Exception:
        try:
            # Spaces can reject or deadlock self-HTTP calls in some runtime states.
            # Fall back to the equivalent in-process runner first.
            results = run_baseline_local(model=DEFAULT_MODEL)
        except Exception:
            # Final fallback keeps the endpoint available even if model providers
            # or helper imports are unavailable inside the container runtime.
            results = _run_heuristic_baseline_local()
    return BaselineRunResponse(
        entrypoint="baseline/run_baseline.py",
        command=f"python -m baseline.run_baseline --base-url {base_url}",
        default_model=DEFAULT_MODEL,
        required_env=["OPENAI_API_KEY"],
        fallback_env=["GEMINI_API_KEY"],
        results=results,
    )


@app.get("/schema", response_model=SchemaResponse)
def get_schema() -> SchemaResponse:
    return SchemaResponse(
        action=DeliveryAction.model_json_schema(),
        observation=DeliveryObservation.model_json_schema(),
        reward=DeliveryReward.model_json_schema(),
        state=DeliveryState.model_json_schema(),
    )
