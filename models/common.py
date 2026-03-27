from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    DELIVERED = "delivered"


class WorkerStatus(str, Enum):
    IDLE = "idle"
    TO_PICKUP = "to_pickup"
    TO_DROPOFF = "to_dropoff"


class PriorityLevel(str, Enum):
    NORMAL = "normal"
    HIGH = "high"


class ActionType(str, Enum):
    ASSIGN_ORDER = "assign_order"
    ADVANCE_TIME = "advance_time"
    REASSIGN_ORDER = "reassign_order"


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(..., description="Grid X coordinate.")
    y: int = Field(..., description="Grid Y coordinate.")

    def manhattan_distance(self, other: "Point") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
