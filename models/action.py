from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.common import ActionType


class DeliveryAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType = Field(..., description="Action to execute.")
    order_id: str | None = Field(
        default=None, description="Order identifier for assign or reassign actions."
    )
    worker_id: str | None = Field(
        default=None, description="Worker identifier for assign or reassign actions."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional client-side metadata."
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "DeliveryAction":
        if self.action_type == ActionType.ADVANCE_TIME:
            if self.order_id or self.worker_id:
                raise ValueError(
                    "advance_time must not include order_id or worker_id."
                )
            return self

        if not self.order_id or not self.worker_id:
            raise ValueError(
                f"{self.action_type.value} requires both order_id and worker_id."
            )
        return self


class BaselineDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    order_id: str | None = None
    worker_id: str | None = None

