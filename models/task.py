from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.common import Difficulty, Point, PriorityLevel


class TaskOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pickup_location: Point
    drop_location: Point
    deadline: int
    priority: PriorityLevel = PriorityLevel.NORMAL


class TaskWorker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    current_location: Point
    capacity: int


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    difficulty: Difficulty
    objective: str
    max_time: int
    orders: list[TaskOrder]
    workers: list[TaskWorker]
    description: str = Field(..., description="Detailed task explanation.")


class TaskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    difficulty: Difficulty
    objective: str
    max_time: int
    order_count: int
    worker_count: int
