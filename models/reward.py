from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DeliveryReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float = Field(0.0, description="Total scalar reward for the step.")
    on_time_delivery_bonus: float = Field(0.0, description="+1.0 for on-time delivery.")
    efficient_assignment_bonus: float = Field(
        0.0, description="Up to +0.3 for choosing a nearby worker."
    )
    progress_bonus: float = Field(0.0, description="+0.1 per progress event.")
    priority_service_bonus: float = Field(
        0.0, description="Extra bonus for on-time high-priority deliveries."
    )
    missed_deadline_penalty: float = Field(
        0.0, description="-0.5 when an order misses its deadline."
    )
    priority_deadline_penalty: float = Field(
        0.0, description="Additional penalty for missing a high-priority order."
    )
    invalid_action_penalty: float = Field(
        0.0, description="-0.2 when an action is invalid."
    )
    idle_worker_penalty: float = Field(
        0.0, description="-0.1 per idle worker while pending orders exist."
    )
    fairness_penalty: float = Field(
        0.0, description="Penalty when active orders are concentrated on too few workers."
    )
    reassignment_penalty: float = Field(
        0.0, description="Penalty for costly reassignment churn."
    )
    notes: list[str] = Field(default_factory=list, description="Human-readable reasons.")

    def finalize(self) -> "DeliveryReward":
        self.total = round(
            self.on_time_delivery_bonus
            + self.efficient_assignment_bonus
            + self.progress_bonus
            + self.priority_service_bonus
            + self.missed_deadline_penalty
            + self.priority_deadline_penalty
            + self.invalid_action_penalty
            + self.idle_worker_penalty
            + self.fairness_penalty
            + self.reassignment_penalty,
            4,
        )
        return self
