from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from env import DeliveryWorkerAssignmentEnv
from models import DeliveryAction
from tasks import get_task


class GradeBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    on_time_delivery_rate: float
    completion_rate: float
    efficiency_score: float
    fairness_score: float
    priority_service_rate: float
    late_delivery_ratio: float
    invalid_action_ratio: float


class GradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    score: float = Field(..., gt=0.0, lt=1.0)
    breakdown: GradeBreakdown
    delivered_orders: int
    delivered_on_time: int
    late_or_missed_orders: int
    total_distance_traveled: int
    invalid_actions: int
    reassignments: int
    actions_evaluated: int


def grade_actions(task_id: str, actions: list[DeliveryAction]) -> GradeResult:
    env = DeliveryWorkerAssignmentEnv(task_id=task_id)
    env.reset(task_id=task_id)

    for action in actions:
        _, _, done, _ = env.step(action)
        if done:
            break

    state = env.state()
    total_orders = len(state.orders)
    delivered_on_time = state.metrics.delivered_on_time
    completion_rate = state.metrics.delivered_orders / total_orders if total_orders else 0.0
    late_or_missed_orders = sum(
        1
        for order in state.orders
        if order.delivered_at is None or order.delivered_at > order.deadline
    )
    on_time_rate = delivered_on_time / total_orders if total_orders else 0.0
    late_ratio = late_or_missed_orders / total_orders if total_orders else 0.0
    invalid_ratio = (
        state.metrics.invalid_actions / max(state.metrics.steps_taken, 1)
        if state.metrics.steps_taken
        else 0.0
    )

    ideal_distance = _minimum_possible_distance(task_id)
    actual_distance = state.metrics.total_distance_traveled
    if actual_distance <= 0:
        efficiency_score = 0.0
    else:
        efficiency_score = min(1.0, ideal_distance / actual_distance)

    priority_total = sum(1 for order in state.orders if order.priority.value == "high")
    if priority_total:
        priority_service_rate = (
            state.metrics.priority_orders_delivered_on_time / priority_total
        )
    else:
        priority_service_rate = 1.0

    fairness_score = _fairness_score(
        state.metrics.worker_delivery_counts, len(state.workers)
    )

    raw_score = (
        0.60 * on_time_rate
        + 0.10 * efficiency_score
        + 0.05 * fairness_score
        + 0.10 * priority_service_rate
        - 0.10 * late_ratio
        - 0.05 * invalid_ratio
    )
    score = _strict_unit_score(raw_score)

    return GradeResult(
        task_id=task_id,
        score=score,
        breakdown=GradeBreakdown(
            on_time_delivery_rate=_strict_unit_score(on_time_rate),
            completion_rate=_strict_unit_score(completion_rate),
            efficiency_score=_strict_unit_score(efficiency_score),
            fairness_score=_strict_unit_score(fairness_score),
            priority_service_rate=_strict_unit_score(priority_service_rate),
            late_delivery_ratio=_strict_unit_score(late_ratio),
            invalid_action_ratio=_strict_unit_score(invalid_ratio),
        ),
        delivered_orders=state.metrics.delivered_orders,
        delivered_on_time=delivered_on_time,
        late_or_missed_orders=late_or_missed_orders,
        total_distance_traveled=state.metrics.total_distance_traveled,
        invalid_actions=state.metrics.invalid_actions,
        reassignments=state.metrics.reassignments,
        actions_evaluated=min(len(actions), state.metrics.steps_taken),
    )


def _minimum_possible_distance(task_id: str) -> int:
    task = get_task(task_id)
    worker_starts = {worker.id: worker.current_location for worker in task.workers}

    lower_bound = 0
    for order in task.orders:
        pickup_to_drop = order.pickup_location.manhattan_distance(order.drop_location)
        nearest_worker = min(
            start.manhattan_distance(order.pickup_location)
            for start in worker_starts.values()
        )
        lower_bound += pickup_to_drop + nearest_worker
    return max(lower_bound, 1)


def _fairness_score(
    worker_delivery_counts: dict[str, int], worker_count: int
) -> float:
    if worker_count <= 1:
        return 1.0

    delivered = list(worker_delivery_counts.values())
    total_delivered = sum(delivered)
    if total_delivered <= 0:
        return 0.0

    mean_load = total_delivered / worker_count
    variance = sum((count - mean_load) ** 2 for count in delivered) / worker_count
    max_variance = (total_delivered**2) * (worker_count - 1) / (worker_count**2)
    if max_variance <= 0:
        return 1.0

    return max(0.0, 1.0 - min(1.0, variance / max_variance))


def _strict_unit_score(value: float) -> float:
    """Return a validator-safe score strictly inside the open unit interval."""
    return round(max(0.05, min(0.95, value)), 4)
