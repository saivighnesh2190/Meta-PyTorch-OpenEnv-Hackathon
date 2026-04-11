from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI

from baseline.prompting import build_dispatch_prompt, heuristic_decision
from models import BaselineDecision, DeliveryAction, DeliveryObservation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    os.getenv(
        "BASE_URL",
        "https://vighnesh2190-meta-pytorch-openenv-hackathon-v2.hf.space",
    ),
)
MAX_STEPS = 64
MODEL_TIMEOUT_SECONDS = 5.0


class InferenceRunner:
    def __init__(self) -> None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN must be set for inference.")
        self.client = OpenAI(
            base_url=API_BASE_URL.rstrip("/"),
            api_key=HF_TOKEN,
            timeout=MODEL_TIMEOUT_SECONDS,
            max_retries=0,
        )
        self.session = requests.Session()
        self.env_base_url = ENV_BASE_URL.rstrip("/")
        self.model_name = MODEL_NAME
        self.model_available = True

    def run(self) -> list[dict[str, Any]]:
        print(
            f"[START] task=all model={MODEL_NAME} max_steps={MAX_STEPS}",
            flush=True,
        )
        tasks = self.session.get(f"{self.env_base_url}/tasks", timeout=30).json()["tasks"]
        results: list[dict[str, Any]] = []
        for task in tasks:
            results.append(self._run_task(task["id"], task["name"]))
        total_score = _strict_unit_float(
            sum(float(result["score"]) for result in results) / max(len(results), 1)
        )
        total_steps = sum(int(result["steps"]) for result in results)
        print(
            f"[END] task=all score={total_score:.4f} steps={total_steps}",
            flush=True,
        )
        return results

    def _run_task(self, task_id: str, task_name: str) -> dict[str, Any]:
        task_label = task_name.replace(" ", "_")
        reset_response = self.session.post(
            f"{self.env_base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        reset_response.raise_for_status()
        observation = DeliveryObservation.model_validate(reset_response.json())
        actions: list[DeliveryAction] = []
        done = False
        steps = 0

        print(
            f"[START] task={task_id} task_name={task_label} max_steps={MAX_STEPS}",
            flush=True,
        )

        while not done and steps < MAX_STEPS:
            decision = self._next_decision(observation)
            action = DeliveryAction.model_validate(decision.model_dump())
            step_response = self.session.post(
                f"{self.env_base_url}/step",
                json={"action": action.model_dump()},
                timeout=30,
            )
            step_response.raise_for_status()
            step_payload = step_response.json()
            if not step_payload["info"].get("valid_action", True):
                fallback = heuristic_decision(observation)
                action = DeliveryAction.model_validate(fallback.model_dump())
                step_response = self.session.post(
                    f"{self.env_base_url}/step",
                    json={"action": action.model_dump()},
                    timeout=30,
                )
                step_response.raise_for_status()
                step_payload = step_response.json()

            observation = DeliveryObservation.model_validate(step_payload["observation"])
            actions.append(action)
            done = bool(step_payload["done"])
            steps += 1
            reward = _strict_unit_float(float(step_payload["reward"]))
            valid_action = bool(step_payload["info"].get("valid_action", True))
            action_type = getattr(action.action_type, "value", str(action.action_type))
            print(
                "[STEP] "
                f"task={task_id} "
                f"step={steps} "
                f"action_type={action_type} "
                f"order_id={action.order_id or 'null'} "
                f"worker_id={action.worker_id or 'null'} "
                f"reward={reward:.4f} "
                f"done={str(done).lower()} "
                f"valid_action={str(valid_action).lower()}",
                flush=True,
            )

        grader_response = self.session.post(
            f"{self.env_base_url}/grader",
            json={
                "task_id": task_id,
                "actions": [action.model_dump() for action in actions],
            },
            timeout=30,
        )
        grader_response.raise_for_status()
        result = grader_response.json()["result"]
        print(
            "[END] "
            f"task={task_id} "
            f"task_name={task_label} "
            f"score={_strict_unit_float(float(result['score'])):.4f} "
            f"steps={steps} "
            f"delivered_orders={result['delivered_orders']} "
            f"delivered_on_time={result['delivered_on_time']} "
            f"invalid_actions={result['invalid_actions']}",
            flush=True,
        )
        return {
            "task_id": task_id,
            "task_name": task_name,
            "score": result["score"],
            "steps": steps,
            "delivered_orders": result["delivered_orders"],
            "delivered_on_time": result["delivered_on_time"],
            "total_distance_traveled": result["total_distance_traveled"],
            "invalid_actions": result["invalid_actions"],
        }

    def _next_decision(self, observation: DeliveryObservation) -> BaselineDecision:
        prompt = build_dispatch_prompt(observation)
        if not self.model_available:
            return heuristic_decision(observation)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a delivery dispatch agent. "
                            "Reply with exactly one JSON object containing "
                            "action_type, order_id, worker_id."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=120,
            )
            content = response.choices[0].message.content or ""
            payload = json.loads(content)
            return BaselineDecision.model_validate(payload)
        except Exception:
            self.model_available = False
            return heuristic_decision(observation)


def main() -> None:
    try:
        runner = InferenceRunner()
        runner.run()
    except Exception as exc:
        print(f"[START] task=error", flush=True)
        print(
            f"[STEP] task=error step=1 reward=0.0001 error={type(exc).__name__}",
            flush=True,
        )
        print(f"[END] task=error score=0.0001 steps=1", flush=True)


def _strict_unit_float(value: float) -> float:
    return round(max(0.0001, min(0.9999, value)), 4)


if __name__ == "__main__":
    main()
