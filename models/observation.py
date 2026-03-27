from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.common import Difficulty, OrderStatus, Point, PriorityLevel, WorkerStatus


class OrderObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pickup_location: Point
    drop_location: Point
    deadline: int
    priority: PriorityLevel = PriorityLevel.NORMAL
    status: OrderStatus
    assigned_worker_id: str | None = None
    picked_up: bool = False
    delivered_at: int | None = None


class WorkerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    current_location: Point
    capacity: int
    assigned_orders: list[str] = Field(default_factory=list)
    status: WorkerStatus


class ObservationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_orders: int
    pending_orders: int
    assigned_orders: int
    delivered_orders: int
    remaining_time: int


class DeliveryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    objective: str
    time: int
    max_time: int
    orders: list[OrderObservation]
    workers: list[WorkerObservation]
    summary: ObservationSummary
