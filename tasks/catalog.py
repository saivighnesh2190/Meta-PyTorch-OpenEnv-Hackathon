from __future__ import annotations

from models import (
    Difficulty,
    Point,
    PriorityLevel,
    TaskDefinition,
    TaskOrder,
    TaskSummary,
    TaskWorker,
)


TASKS: dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        id="easy",
        name="Easy Route Balancing",
        difficulty=Difficulty.EASY,
        objective=(
            "Deliver every order before its deadline. Workers are plentiful, so the "
            "best strategy is to prioritize the urgent order and avoid stacking too "
            "many stops onto one worker."
        ),
        max_time=18,
        description=(
            "Few orders and extra workers, but one priority order and slightly tighter "
            "timing punish lazy assignments."
        ),
        orders=[
            TaskOrder(
                id="O-101",
                pickup_location=Point(x=1, y=1),
                drop_location=Point(x=4, y=1),
                deadline=7,
                priority=PriorityLevel.HIGH,
            ),
            TaskOrder(
                id="O-102",
                pickup_location=Point(x=3, y=2),
                drop_location=Point(x=5, y=4),
                deadline=9,
            ),
            TaskOrder(
                id="O-103",
                pickup_location=Point(x=0, y=4),
                drop_location=Point(x=2, y=5),
                deadline=10,
            ),
            TaskOrder(
                id="O-104",
                pickup_location=Point(x=5, y=0),
                drop_location=Point(x=6, y=2),
                deadline=10,
            ),
        ],
        workers=[
            TaskWorker(id="W-1", current_location=Point(x=0, y=0), capacity=2),
            TaskWorker(id="W-2", current_location=Point(x=3, y=3), capacity=2),
            TaskWorker(id="W-3", current_location=Point(x=7, y=2), capacity=1),
        ],
    ),
    "medium": TaskDefinition(
        id="medium",
        name="Deadline Aware Dispatch",
        difficulty=Difficulty.MEDIUM,
        objective=(
            "Balance travel distance and moderate deadlines with only two workers. "
            "Assignments made too early to the wrong worker will create avoidable lateness."
        ),
        max_time=20,
        description=(
            "Limited workers, clustered pickups, and two priority orders force the agent "
            "to sequence work and avoid wasted travel."
        ),
        orders=[
            TaskOrder(
                id="O-201",
                pickup_location=Point(x=1, y=0),
                drop_location=Point(x=4, y=1),
                deadline=7,
                priority=PriorityLevel.HIGH,
            ),
            TaskOrder(
                id="O-202",
                pickup_location=Point(x=2, y=5),
                drop_location=Point(x=5, y=6),
                deadline=10,
            ),
            TaskOrder(
                id="O-203",
                pickup_location=Point(x=6, y=2),
                drop_location=Point(x=7, y=5),
                deadline=9,
                priority=PriorityLevel.HIGH,
            ),
            TaskOrder(
                id="O-204",
                pickup_location=Point(x=0, y=6),
                drop_location=Point(x=3, y=7),
                deadline=13,
            ),
            TaskOrder(
                id="O-205",
                pickup_location=Point(x=5, y=1),
                drop_location=Point(x=2, y=2),
                deadline=14,
            ),
        ],
        workers=[
            TaskWorker(id="W-10", current_location=Point(x=0, y=1), capacity=2),
            TaskWorker(id="W-11", current_location=Point(x=6, y=4), capacity=2),
        ],
    ),
    "hard": TaskDefinition(
        id="hard",
        name="Capacity Constrained Rush Hour",
        difficulty=Difficulty.HARD,
        objective=(
            "Plan across tight deadlines, limited worker capacity, and a wider map. "
            "The agent must protect the high-priority rush orders, keep the urgent queue "
            "moving, and use reassignment sparingly because it now has a real cost."
        ),
        max_time=22,
        description=(
            "Many orders, uneven worker capacity, three priority deliveries, and tighter "
            "deadlines require real multi-step planning."
        ),
        orders=[
            TaskOrder(
                id="O-301",
                pickup_location=Point(x=1, y=1),
                drop_location=Point(x=4, y=1),
                deadline=5,
                priority=PriorityLevel.HIGH,
            ),
            TaskOrder(
                id="O-302",
                pickup_location=Point(x=2, y=2),
                drop_location=Point(x=5, y=2),
                deadline=7,
            ),
            TaskOrder(
                id="O-303",
                pickup_location=Point(x=7, y=1),
                drop_location=Point(x=8, y=4),
                deadline=8,
                priority=PriorityLevel.HIGH,
            ),
            TaskOrder(
                id="O-304",
                pickup_location=Point(x=6, y=5),
                drop_location=Point(x=3, y=6),
                deadline=10,
            ),
            TaskOrder(
                id="O-305",
                pickup_location=Point(x=0, y=7),
                drop_location=Point(x=2, y=8),
                deadline=10,
            ),
            TaskOrder(
                id="O-306",
                pickup_location=Point(x=4, y=7),
                drop_location=Point(x=7, y=7),
                deadline=11,
            ),
            TaskOrder(
                id="O-307",
                pickup_location=Point(x=8, y=6),
                drop_location=Point(x=5, y=8),
                deadline=12,
            ),
            TaskOrder(
                id="O-308",
                pickup_location=Point(x=3, y=4),
                drop_location=Point(x=1, y=6),
                deadline=8,
                priority=PriorityLevel.HIGH,
            ),
        ],
        workers=[
            TaskWorker(id="W-20", current_location=Point(x=0, y=0), capacity=1),
            TaskWorker(id="W-21", current_location=Point(x=5, y=0), capacity=2),
            TaskWorker(id="W-22", current_location=Point(x=8, y=8), capacity=1),
        ],
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        available = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task '{task_id}'. Available tasks: {available}.")
    return TASKS[task_id].model_copy(deep=True)


def list_task_summaries() -> list[TaskSummary]:
    return [
        TaskSummary(
            id=task.id,
            name=task.name,
            difficulty=task.difficulty,
            objective=task.objective,
            max_time=task.max_time,
            order_count=len(task.orders),
            worker_count=len(task.workers),
        )
        for task in TASKS.values()
    ]
