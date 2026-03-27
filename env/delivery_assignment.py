from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from models import (
    ActionType,
    DeliveryAction,
    DeliveryObservation,
    DeliveryReward,
    DeliveryState,
    ObservationSummary,
    OrderObservation,
    OrderState,
    OrderStatus,
    Point,
    PriorityLevel,
    TaskDefinition,
    WorkerObservation,
    WorkerState,
    WorkerStatus,
)
from tasks import get_task


class DeliveryWorkerAssignmentEnv:
    """Deterministic logistics environment for worker-to-order dispatching."""

    def __init__(self, task_id: str = "easy") -> None:
        self._task_definition: TaskDefinition | None = None
        self._state: DeliveryState | None = None
        self.reset(task_id=task_id)

    def reset(
        self,
        task_id: str | None = None,
        seed: int | None = None,
    ) -> DeliveryObservation:
        del seed  # Tasks are deterministic; the interface keeps a Gym-style slot.

        task = get_task(task_id or self.task_id)
        self._task_definition = task
        self._state = DeliveryState(
            task_id=task.id,
            difficulty=task.difficulty,
            objective=task.objective,
            time=0,
            max_time=task.max_time,
            orders=[
                OrderState(
                    id=order.id,
                    pickup_location=order.pickup_location,
                    drop_location=order.drop_location,
                    deadline=order.deadline,
                    priority=order.priority,
                )
                for order in task.orders
            ],
            workers=[
                WorkerState(
                    id=worker.id,
                    current_location=worker.current_location,
                    capacity=worker.capacity,
                )
                for worker in task.workers
            ],
        )
        self._state.metrics.worker_delivery_counts = {
            worker.id: 0 for worker in self._state.workers
        }
        return self._build_observation()

    def step(
        self, action: DeliveryAction
    ) -> tuple[DeliveryObservation, DeliveryReward, bool, dict[str, Any]]:
        state = self.state()
        reward = DeliveryReward()
        info: dict[str, Any] = {
            "task_id": state.task_id,
            "time": state.time,
            "action": action.model_dump(),
            "events": [],
            "valid_action": True,
        }

        if state.done:
            self._apply_invalid_action(
                reward,
                info,
                "The episode is already complete. Reset before sending more actions.",
            )
            observation = self._build_observation()
            return observation, reward.finalize(), True, info

        if action.action_type == ActionType.ADVANCE_TIME:
            self._advance_time(reward, info)
        elif action.action_type == ActionType.ASSIGN_ORDER:
            self._assign_order(
                order_id=action.order_id or "",
                worker_id=action.worker_id or "",
                reward=reward,
                info=info,
            )
            if info["valid_action"]:
                self._apply_balance_penalty(reward, info)
        elif action.action_type == ActionType.REASSIGN_ORDER:
            self._reassign_order(
                order_id=action.order_id or "",
                worker_id=action.worker_id or "",
                reward=reward,
                info=info,
            )
            if info["valid_action"]:
                self._apply_balance_penalty(reward, info)
        else:
            self._apply_invalid_action(reward, info, "Unsupported action type.")

        state.metrics.steps_taken += 1
        state.done = self._compute_done()
        reward.finalize()
        state.metrics.total_reward = round(state.metrics.total_reward + reward.total, 4)
        observation = self._build_observation()
        info["time"] = state.time
        info["metrics"] = state.metrics.model_dump()
        info["done_reason"] = self._done_reason() if state.done else None
        return observation, reward, state.done, info

    def state(self) -> DeliveryState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    @property
    def task_id(self) -> str:
        if self._task_definition is None:
            raise RuntimeError("Environment not initialized.")
        return self._task_definition.id

    def _assign_order(
        self,
        order_id: str,
        worker_id: str,
        reward: DeliveryReward,
        info: dict[str, Any],
    ) -> None:
        order = self._get_order(order_id)
        worker = self._get_worker(worker_id)
        if order is None or worker is None:
            self._apply_invalid_action(reward, info, "Order or worker does not exist.")
            return
        if order.status != OrderStatus.PENDING:
            self._apply_invalid_action(
                reward,
                info,
                f"Order {order.id} is already {order.status.value}.",
            )
            return
        if len(worker.assigned_orders) >= worker.capacity:
            self._apply_invalid_action(
                reward,
                info,
                f"Worker {worker.id} is at capacity.",
            )
            return

        order.status = OrderStatus.ASSIGNED
        order.assigned_worker_id = worker.id
        order.assigned_at = self.state().time
        worker.assigned_orders.append(order.id)
        self._refresh_worker_status(worker)

        bonus = self._assignment_efficiency_bonus(order, worker)
        reward.efficient_assignment_bonus += bonus
        reward.notes.append(
            f"Assigned order {order.id} to worker {worker.id} with efficiency bonus {bonus:.2f}."
        )
        info["events"].append(
            {
                "type": "assignment",
                "order_id": order.id,
                "worker_id": worker.id,
                "efficiency_bonus": bonus,
            }
        )

    def _reassign_order(
        self,
        order_id: str,
        worker_id: str,
        reward: DeliveryReward,
        info: dict[str, Any],
    ) -> None:
        order = self._get_order(order_id)
        worker = self._get_worker(worker_id)
        if order is None or worker is None:
            self._apply_invalid_action(reward, info, "Order or worker does not exist.")
            return
        if order.status != OrderStatus.ASSIGNED:
            self._apply_invalid_action(
                reward,
                info,
                f"Order {order.id} cannot be reassigned from status {order.status.value}.",
            )
            return
        if order.picked_up:
            self._apply_invalid_action(
                reward,
                info,
                f"Order {order.id} has already been picked up and cannot be reassigned.",
            )
            return
        if order.assigned_worker_id == worker.id:
            self._apply_invalid_action(
                reward,
                info,
                f"Order {order.id} is already assigned to worker {worker.id}.",
            )
            return
        if len(worker.assigned_orders) >= worker.capacity:
            self._apply_invalid_action(
                reward,
                info,
                f"Worker {worker.id} is at capacity.",
            )
            return

        previous_worker = self._get_worker(order.assigned_worker_id or "")
        if previous_worker is not None:
            previous_worker.assigned_orders = [
                assigned_order
                for assigned_order in previous_worker.assigned_orders
                if assigned_order != order.id
            ]
            self._refresh_worker_status(previous_worker)

        worker.assigned_orders.append(order.id)
        order.assigned_worker_id = worker.id
        self._refresh_worker_status(worker)
        self.state().metrics.reassignments += 1

        bonus = self._assignment_efficiency_bonus(order, worker)
        reward.efficient_assignment_bonus += bonus
        reward.reassignment_penalty -= 0.15
        reward.notes.append(
            f"Reassigned order {order.id} to worker {worker.id} with efficiency bonus {bonus:.2f} and reassignment cost."
        )
        info["events"].append(
            {
                "type": "reassignment",
                "order_id": order.id,
                "worker_id": worker.id,
                "efficiency_bonus": bonus,
                "reassignment_penalty": -0.15,
            }
        )

    def _advance_time(
        self, reward: DeliveryReward, info: dict[str, Any]
    ) -> None:
        state = self.state()
        pending_exists = any(
            order.status != OrderStatus.DELIVERED for order in state.orders
        )
        idle_workers = 0

        for worker in state.workers:
            tick_result = self._process_worker_tick(worker)
            reward.progress_bonus += 0.1 * tick_result["progress_events"]
            reward.on_time_delivery_bonus += 1.0 * tick_result["on_time_deliveries"]
            reward.missed_deadline_penalty -= 0.5 * tick_result["late_penalties"]
            reward.priority_service_bonus += tick_result["priority_service_bonus"]
            reward.priority_deadline_penalty += tick_result["priority_deadline_penalty"]
            reward.notes.extend(tick_result["notes"])
            info["events"].extend(tick_result["events"])
            if tick_result["idle"] and pending_exists:
                idle_workers += 1

        state.time += 1

        newly_missed, priority_misses = self._mark_deadline_misses()
        if newly_missed:
            reward.missed_deadline_penalty -= 0.5 * newly_missed
            reward.priority_deadline_penalty -= 0.2 * priority_misses
            reward.notes.append(
                f"{newly_missed} order(s) missed a deadline at time {state.time}."
            )
            info["events"].append(
                {
                    "type": "deadline_miss",
                    "count": newly_missed,
                    "priority_count": priority_misses,
                    "time": state.time,
                }
            )

        if idle_workers:
            reward.idle_worker_penalty -= 0.1 * idle_workers
            reward.notes.append(
                f"{idle_workers} worker(s) remained idle while orders were still active."
            )
            info["events"].append(
                {"type": "idle_penalty", "idle_workers": idle_workers}
            )

    def _apply_balance_penalty(
        self, reward: DeliveryReward, info: dict[str, Any]
    ) -> None:
        active_queues = [len(worker.assigned_orders) for worker in self.state().workers]
        if not any(active_queues):
            return

        mean_load = sum(active_queues) / len(active_queues)
        variance = sum((load - mean_load) ** 2 for load in active_queues) / len(
            active_queues
        )
        penalty = round(min(0.15, 0.05 * variance), 4)
        if penalty <= 0:
            return

        reward.fairness_penalty -= penalty
        reward.notes.append(
            f"Load balancing penalty applied because active queues are uneven (variance={variance:.2f})."
        )
        info["events"].append(
            {
                "type": "fairness_penalty",
                "variance": round(variance, 4),
                "penalty": -penalty,
            }
        )

    def _process_worker_tick(self, worker: WorkerState) -> dict[str, Any]:
        state = self.state()
        notes: list[str] = []
        events: list[dict[str, Any]] = []
        progress_events = 0
        on_time_deliveries = 0
        late_penalties = 0
        priority_service_bonus = 0.0
        priority_deadline_penalty = 0.0

        self._resolve_arrivals(
            worker,
            notes,
            events,
            event_time=self.state().time,
        )
        active_order = self._worker_active_order(worker)
        if active_order is None:
            worker.status = WorkerStatus.IDLE
            return {
                "idle": True,
                "notes": notes,
                "events": events,
                "progress_events": progress_events,
                "on_time_deliveries": on_time_deliveries,
                "late_penalties": late_penalties,
                "priority_service_bonus": priority_service_bonus,
                "priority_deadline_penalty": priority_deadline_penalty,
            }

        target = self._worker_target(active_order)
        distance_before = worker.current_location.manhattan_distance(target)
        if distance_before > 0:
            worker.current_location = self._move_one_step(worker.current_location, target)
            state.metrics.total_distance_traveled += 1
            distance_after = worker.current_location.manhattan_distance(target)
            if distance_after < distance_before:
                progress_events += 1
                notes.append(
                    f"Worker {worker.id} moved closer to order {active_order.id}."
                )
            events.append(
                {
                    "type": "movement",
                    "worker_id": worker.id,
                    "order_id": active_order.id,
                    "location": worker.current_location.model_dump(),
                }
            )

        completed = self._resolve_arrivals(
            worker,
            notes,
            events,
            event_time=self.state().time + 1,
        )
        on_time_deliveries += completed["on_time"]
        late_penalties += completed["late_penalties"]
        priority_service_bonus += completed["priority_service_bonus"]
        priority_deadline_penalty += completed["priority_deadline_penalty"]
        self._refresh_worker_status(worker)
        return {
            "idle": False,
            "notes": notes,
            "events": events,
            "progress_events": progress_events,
            "on_time_deliveries": on_time_deliveries,
            "late_penalties": late_penalties,
            "priority_service_bonus": priority_service_bonus,
            "priority_deadline_penalty": priority_deadline_penalty,
        }

    def _resolve_arrivals(
        self,
        worker: WorkerState,
        notes: list[str],
        events: list[dict[str, Any]],
        event_time: int,
    ) -> dict[str, int | float]:
        on_time_deliveries = 0
        late_penalties = 0
        priority_service_bonus = 0.0
        priority_deadline_penalty = 0.0

        while True:
            order = self._worker_active_order(worker)
            if order is None:
                worker.status = WorkerStatus.IDLE
                return {
                    "on_time": on_time_deliveries,
                    "late_penalties": late_penalties,
                    "priority_service_bonus": priority_service_bonus,
                    "priority_deadline_penalty": priority_deadline_penalty,
                }

            if not order.picked_up and worker.current_location == order.pickup_location:
                order.picked_up = True
                order.picked_up_at = event_time
                worker.status = WorkerStatus.TO_DROPOFF
                notes.append(f"Worker {worker.id} picked up order {order.id}.")
                events.append(
                    {"type": "pickup", "worker_id": worker.id, "order_id": order.id}
                )
                continue

            if order.picked_up and worker.current_location == order.drop_location:
                worker.assigned_orders = [
                    assigned_order
                    for assigned_order in worker.assigned_orders
                    if assigned_order != order.id
                ]
                order.status = OrderStatus.DELIVERED
                order.delivered_at = event_time
                self.state().metrics.delivered_orders += 1
                self.state().metrics.worker_delivery_counts[worker.id] = (
                    self.state().metrics.worker_delivery_counts.get(worker.id, 0) + 1
                )
                if order.delivered_at <= order.deadline:
                    self.state().metrics.delivered_on_time += 1
                    on_time_deliveries += 1
                    if order.priority == PriorityLevel.HIGH:
                        self.state().metrics.priority_orders_delivered_on_time += 1
                        priority_service_bonus += 0.2
                    notes.append(f"Order {order.id} was delivered on time.")
                else:
                    self.state().metrics.delivered_late += 1
                    if not order.deadline_penalized:
                        order.deadline_penalized = True
                        self.state().metrics.deadline_misses += 1
                        late_penalties += 1
                        if order.priority == PriorityLevel.HIGH:
                            priority_deadline_penalty -= 0.2
                    notes.append(f"Order {order.id} was delivered late.")
                events.append(
                    {"type": "delivery", "worker_id": worker.id, "order_id": order.id}
                )
                continue

            break

        return {
            "on_time": on_time_deliveries,
            "late_penalties": late_penalties,
            "priority_service_bonus": priority_service_bonus,
            "priority_deadline_penalty": priority_deadline_penalty,
        }

    def _mark_deadline_misses(self) -> tuple[int, int]:
        missed = 0
        priority_missed = 0
        for order in self.state().orders:
            if (
                order.status != OrderStatus.DELIVERED
                and self.state().time > order.deadline
                and not order.deadline_penalized
            ):
                order.deadline_penalized = True
                self.state().metrics.deadline_misses += 1
                missed += 1
                if order.priority == PriorityLevel.HIGH:
                    priority_missed += 1
        return missed, priority_missed

    def _assignment_efficiency_bonus(
        self, order: OrderState, chosen_worker: WorkerState
    ) -> float:
        candidate_distances = [
            worker.current_location.manhattan_distance(order.pickup_location)
            for worker in self.state().workers
            if len(worker.assigned_orders) < worker.capacity or worker.id == chosen_worker.id
        ]
        best_distance = min(candidate_distances)
        chosen_distance = chosen_worker.current_location.manhattan_distance(
            order.pickup_location
        )
        bonus = 0.3 * ((best_distance + 1) / (chosen_distance + 1))
        return round(min(0.3, bonus), 4)

    def _build_observation(self) -> DeliveryObservation:
        state = self.state()
        order_observations = [
            OrderObservation(
                id=order.id,
                pickup_location=order.pickup_location,
                drop_location=order.drop_location,
                deadline=order.deadline,
                priority=order.priority,
                status=order.status,
                assigned_worker_id=order.assigned_worker_id,
                picked_up=order.picked_up,
                delivered_at=order.delivered_at,
            )
            for order in state.orders
        ]
        worker_observations = [
            WorkerObservation(
                id=worker.id,
                current_location=worker.current_location,
                capacity=worker.capacity,
                assigned_orders=list(worker.assigned_orders),
                status=worker.status,
            )
            for worker in state.workers
        ]
        summary = ObservationSummary(
            total_orders=len(state.orders),
            pending_orders=sum(
                1 for order in state.orders if order.status == OrderStatus.PENDING
            ),
            assigned_orders=sum(
                1 for order in state.orders if order.status == OrderStatus.ASSIGNED
            ),
            delivered_orders=sum(
                1 for order in state.orders if order.status == OrderStatus.DELIVERED
            ),
            remaining_time=max(state.max_time - state.time, 0),
        )
        return DeliveryObservation(
            task_id=state.task_id,
            difficulty=state.difficulty,
            objective=state.objective,
            time=state.time,
            max_time=state.max_time,
            orders=order_observations,
            workers=worker_observations,
            summary=summary,
        )

    def _compute_done(self) -> bool:
        state = self.state()
        all_delivered = all(
            order.status == OrderStatus.DELIVERED for order in state.orders
        )
        return all_delivered or state.time >= state.max_time

    def _done_reason(self) -> str:
        if all(order.status == OrderStatus.DELIVERED for order in self.state().orders):
            return "all_orders_delivered"
        if self.state().time >= self.state().max_time:
            return "time_limit_reached"
        return "in_progress"

    def _worker_active_order(self, worker: WorkerState) -> OrderState | None:
        for order_id in worker.assigned_orders:
            order = self._get_order(order_id)
            if order is not None and order.status != OrderStatus.DELIVERED:
                return order
        return None

    def _worker_target(self, order: OrderState) -> Point:
        return order.drop_location if order.picked_up else order.pickup_location

    def _refresh_worker_status(self, worker: WorkerState) -> None:
        active_order = self._worker_active_order(worker)
        if active_order is None:
            worker.status = WorkerStatus.IDLE
        elif active_order.picked_up:
            worker.status = WorkerStatus.TO_DROPOFF
        else:
            worker.status = WorkerStatus.TO_PICKUP

    def _move_one_step(self, current: Point, target: Point) -> Point:
        if current.x != target.x:
            step = 1 if target.x > current.x else -1
            return Point(x=current.x + step, y=current.y)
        if current.y != target.y:
            step = 1 if target.y > current.y else -1
            return Point(x=current.x, y=current.y + step)
        return current

    def _apply_invalid_action(
        self, reward: DeliveryReward, info: dict[str, Any], message: str
    ) -> None:
        reward.invalid_action_penalty -= 0.2
        reward.notes.append(message)
        self.state().metrics.invalid_actions += 1
        info["valid_action"] = False
        info["events"].append({"type": "invalid_action", "message": message})

    def _get_order(self, order_id: str) -> OrderState | None:
        return self._get_by_id(self.state().orders, order_id)

    def _get_worker(self, worker_id: str) -> WorkerState | None:
        return self._get_by_id(self.state().workers, worker_id)

    def _get_by_id(self, objects: Iterable[Any], object_id: str) -> Any | None:
        for item in objects:
            if item.id == object_id:
                return item
        return None
