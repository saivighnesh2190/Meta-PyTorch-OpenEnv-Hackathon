from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.common import Difficulty, OrderStatus, Point, PriorityLevel, WorkerStatus


class OrderState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pickup_location: Point
    drop_location: Point
    deadline: int
    priority: PriorityLevel = PriorityLevel.NORMAL
    status: OrderStatus = OrderStatus.PENDING
    assigned_worker_id: str | None = None
    picked_up: bool = False
    assigned_at: int | None = None
    picked_up_at: int | None = None
    delivered_at: int | None = None
    deadline_penalized: bool = False


class WorkerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    current_location: Point
    capacity: int
    assigned_orders: list[str] = Field(default_factory=list)
    status: WorkerStatus = WorkerStatus.IDLE


class DeliveryMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_distance_traveled: int = 0
    delivered_orders: int = 0
    delivered_on_time: int = 0
    delivered_late: int = 0
    deadline_misses: int = 0
    invalid_actions: int = 0
    reassignments: int = 0
    priority_orders_delivered_on_time: int = 0
    total_reward: float = 0.0
    steps_taken: int = 0
    worker_delivery_counts: dict[str, int] = Field(default_factory=dict)


class DeliveryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    objective: str
    time: int = 0
    max_time: int
    orders: list[OrderState]
    workers: list[WorkerState]
    metrics: DeliveryMetrics = Field(default_factory=DeliveryMetrics)
    done: bool = False
