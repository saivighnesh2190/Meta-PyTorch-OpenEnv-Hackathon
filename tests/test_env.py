from fastapi.testclient import TestClient

from api.app import app
from env import DeliveryWorkerAssignmentEnv
from grader import grade_actions
from models import ActionType, DeliveryAction


def test_environment_progresses_and_returns_dense_reward() -> None:
    env = DeliveryWorkerAssignmentEnv(task_id="easy")
    env.reset(task_id="easy")

    observation, reward, done, info = env.step(
        DeliveryAction(
            action_type=ActionType.ASSIGN_ORDER,
            order_id="O-101",
            worker_id="W-1",
        )
    )

    assert observation.task_id == "easy"
    assert reward.efficient_assignment_bonus > 0
    assert any(order.priority.value == "high" for order in observation.orders)
    assert not done
    assert info["valid_action"] is True


def test_grader_returns_normalized_score() -> None:
    actions = [
        DeliveryAction(
            action_type=ActionType.ASSIGN_ORDER,
            order_id="O-101",
            worker_id="W-1",
        ),
        DeliveryAction(action_type=ActionType.ADVANCE_TIME),
        DeliveryAction(action_type=ActionType.ADVANCE_TIME),
    ]
    result = grade_actions("easy", actions)
    assert 0.0 < result.score < 1.0
    assert 0.0 <= result.breakdown.fairness_score <= 1.0


def test_api_exposes_required_endpoints() -> None:
    client = TestClient(app)
    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    assert "action_schema" in tasks_response.json()
    assert client.post("/reset", json={"task_id": "easy"}).status_code == 200
    assert client.post("/reset").status_code == 200


def test_api_step_exposes_scalar_reward_and_grader_score() -> None:
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy"})

    step_response = client.post(
        "/step",
        json={
            "action": {
                "action_type": "assign_order",
                "order_id": "O-101",
                "worker_id": "W-1",
            }
        },
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert isinstance(step_payload["reward"], float)
    assert "reward_breakdown" in step_payload["info"]
    assert isinstance(step_payload["info"]["reward_breakdown"]["total"], float)

    grader_response = client.post(
        "/grader",
        json={
            "task_id": "easy",
            "actions": [
                {"action_type": "assign_order", "order_id": "O-101", "worker_id": "W-1"},
                {"action_type": "advance_time"},
            ],
        },
    )
    assert grader_response.status_code == 200
    grader_payload = grader_response.json()
    assert isinstance(grader_payload["score"], float)
    assert grader_payload["score"] == grader_payload["result"]["score"]
