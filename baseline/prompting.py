from __future__ import annotations

from models import ActionType, BaselineDecision, DeliveryObservation, OrderStatus


def build_dispatch_prompt(observation: DeliveryObservation) -> str:
    order_lines = []
    for order in observation.orders:
        order_lines.append(
            f"- {order.id}: status={order.status.value}, pickup=({order.pickup_location.x},{order.pickup_location.y}), "
            f"drop=({order.drop_location.x},{order.drop_location.y}), deadline={order.deadline}, priority={order.priority.value}, "
            f"assigned_worker={order.assigned_worker_id}, picked_up={order.picked_up}"
        )

    worker_lines = []
    for worker in observation.workers:
        worker_lines.append(
            f"- {worker.id}: location=({worker.current_location.x},{worker.current_location.y}), "
            f"capacity={worker.capacity}, assigned_orders={worker.assigned_orders}, status={worker.status.value}"
        )

    return "\n".join(
        [
            "You are dispatching delivery workers on a deterministic grid.",
            "Return exactly one JSON object with keys action_type, order_id, worker_id.",
            "Use advance_time when every useful assignment has already been made for this timestep.",
            "Prefer high-priority orders first, then earlier deadlines, then workers with shorter travel distance to pickup.",
            f"Task objective: {observation.objective}",
            f"Time: {observation.time}/{observation.max_time}",
            "Orders:",
            *order_lines,
            "Workers:",
            *worker_lines,
        ]
    )


def heuristic_decision(observation: DeliveryObservation) -> BaselineDecision:
    pending_orders = sorted(
        [
            order
            for order in observation.orders
            if order.status == OrderStatus.PENDING
        ],
        key=lambda order: (
            0 if order.priority.value == "high" else 1,
            order.deadline,
            order.id,
        ),
    )
    worker_by_id = {worker.id: worker for worker in observation.workers}

    for order in pending_orders:
        available_workers = [
            worker
            for worker in observation.workers
            if len(worker.assigned_orders) < worker.capacity
        ]
        if not available_workers:
            break
        best_worker = min(
            available_workers,
            key=lambda worker: (
                abs(worker.current_location.x - order.pickup_location.x)
                + abs(worker.current_location.y - order.pickup_location.y),
                len(worker.assigned_orders),
                worker.id,
            ),
        )
        return BaselineDecision(
            action_type=ActionType.ASSIGN_ORDER,
            order_id=order.id,
            worker_id=best_worker.id,
        )

    assigned_orders = [
        order for order in observation.orders if order.status == OrderStatus.ASSIGNED
    ]
    if assigned_orders or any(
        len(worker.assigned_orders) > 0 for worker in observation.workers
    ):
        return BaselineDecision(action_type=ActionType.ADVANCE_TIME)

    if pending_orders:
        order = pending_orders[0]
        worker = min(
            worker_by_id.values(),
            key=lambda item: (
                abs(item.current_location.x - order.pickup_location.x)
                + abs(item.current_location.y - order.pickup_location.y),
                item.id,
            ),
        )
        return BaselineDecision(
            action_type=ActionType.ASSIGN_ORDER,
            order_id=order.id,
            worker_id=worker.id,
        )

    return BaselineDecision(action_type=ActionType.ADVANCE_TIME)
