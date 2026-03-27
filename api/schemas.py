from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from grader import GradeResult
from models import DeliveryAction, DeliveryObservation, DeliveryReward, DeliveryState, TaskSummary


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default="easy", description="Task to reset.")


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: DeliveryAction


class StepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: DeliveryObservation
    reward: float
    done: bool
    info: dict


class TasksResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskSummary]
    action_schema: dict


class GraderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    actions: list[DeliveryAction]


class GraderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0.0, le=1.0)
    result: GradeResult


class BaselineTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_name: str
    provider_used: str
    providers_attempted: list[str]
    providers_succeeded: list[str]
    score: float
    delivered_orders: int
    delivered_on_time: int
    total_distance_traveled: int
    invalid_actions: int


class BaselineRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entrypoint: str
    command: str
    default_model: str
    required_env: list[str]
    fallback_env: list[str]
    results: list[BaselineTaskResult]


class StateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: DeliveryState


class SchemaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: dict
    observation: dict
    reward: dict
    state: dict
