from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

from baseline.prompting import build_dispatch_prompt, heuristic_decision
from env import DeliveryWorkerAssignmentEnv
from grader import grade_actions
from models import BaselineDecision, DeliveryAction, DeliveryObservation
from tasks import list_task_summaries

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["assign_order", "advance_time", "reassign_order"],
        },
        "order_id": {"type": ["string", "null"]},
        "worker_id": {"type": ["string", "null"]},
    },
    "required": ["action_type", "order_id", "worker_id"],
}


class DeliveryBaselineRunner:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.model = model
        self.gemini_model = DEFAULT_GEMINI_MODEL
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = (
            OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        )
        self._openai_available = self.client is not None
        self._gemini_available = self.gemini_api_key is not None
        self._current_task_provider = "heuristic"
        self._providers_attempted: list[str] = []
        self._providers_succeeded: list[str] = []

    def run(self) -> list[dict[str, Any]]:
        tasks = self.session.get(f"{self.base_url}/tasks", timeout=30).json()["tasks"]
        results: list[dict[str, Any]] = []
        for task in tasks:
            results.append(self._run_task(task["id"], task["name"]))
        return results

    def _run_task(self, task_id: str, task_name: str) -> dict[str, Any]:
        self._current_task_provider = "heuristic"
        self._providers_attempted = []
        self._providers_succeeded = []
        reset_response = self.session.post(
            f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30
        )
        reset_response.raise_for_status()
        observation = DeliveryObservation.model_validate(reset_response.json())
        actions: list[DeliveryAction] = []
        done = False

        while not done:
            decision = self._model_decision(observation)
            action = DeliveryAction.model_validate(decision.model_dump())
            step_response = self.session.post(
                f"{self.base_url}/step",
                json={"action": action.model_dump()},
                timeout=30,
            )
            step_response.raise_for_status()
            step_payload = step_response.json()
            if not step_payload["info"]["valid_action"]:
                fallback = heuristic_decision(observation)
                action = DeliveryAction.model_validate(fallback.model_dump())
                step_response = self.session.post(
                    f"{self.base_url}/step",
                    json={"action": action.model_dump()},
                    timeout=30,
                )
                step_response.raise_for_status()
                step_payload = step_response.json()

            observation = DeliveryObservation.model_validate(step_payload["observation"])
            actions.append(action)
            done = bool(step_payload["done"])

        grader_response = self.session.post(
            f"{self.base_url}/grader",
            json={
                "task_id": task_id,
                "actions": [action.model_dump() for action in actions],
            },
            timeout=30,
        )
        grader_response.raise_for_status()
        result = grader_response.json()["result"]
        summary = {
            "task_id": task_id,
            "task_name": task_name,
            "provider_used": self._current_task_provider,
            "providers_attempted": list(self._providers_attempted),
            "providers_succeeded": list(self._providers_succeeded),
            "score": result["score"],
            "delivered_orders": result["delivered_orders"],
            "delivered_on_time": result["delivered_on_time"],
            "total_distance_traveled": result["total_distance_traveled"],
            "invalid_actions": result["invalid_actions"],
        }
        return summary

    def _model_decision(self, observation: DeliveryObservation) -> BaselineDecision:
        prompt = build_dispatch_prompt(observation)
        if self._openai_available and self.client is not None:
            self._record_attempt("openai")
            try:
                response = self.client.responses.create(
                    model=self.model,
                    temperature=0.0,
                    input=prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "delivery_dispatch_action",
                            "schema": DECISION_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                output_text = response.output_text
                if output_text:
                    payload: dict[str, Any] = json.loads(output_text)
                    self._record_success("openai")
                    self._current_task_provider = "openai"
                    return BaselineDecision.model_validate(payload)
            except Exception:
                self._openai_available = False

        if self._gemini_available and self.gemini_api_key:
            self._record_attempt("gemini")
            try:
                response = self.session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent",
                    params={"key": self.gemini_api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "responseJsonSchema": DECISION_SCHEMA,
                            "temperature": 0.0,
                            "candidateCount": 1,
                        },
                    },
                    timeout=60,
                )
                if response.status_code == 200:
                    output_text = self._extract_gemini_text(response.json())
                    if output_text:
                        payload = json.loads(output_text)
                        self._record_success("gemini")
                        self._current_task_provider = "gemini"
                        return BaselineDecision.model_validate(payload)
                else:
                    self._gemini_available = False
            except Exception:
                self._gemini_available = False

        self._current_task_provider = "heuristic"
        return heuristic_decision(observation)

    def _record_attempt(self, provider: str) -> None:
        if provider not in self._providers_attempted:
            self._providers_attempted.append(provider)

    def _record_success(self, provider: str) -> None:
        if provider not in self._providers_succeeded:
            self._providers_succeeded.append(provider)

    def _extract_gemini_text(self, payload: dict[str, Any]) -> str | None:
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"]
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenAI baseline agent.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://localhost:8000"),
        help="API base URL for the environment server.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model alias or snapshot to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = DeliveryBaselineRunner(base_url=args.base_url, model=args.model)
    results = runner.run()
    for result in results:
        print(
            f"{result['task_name']} ({result['task_id']}) -> provider={result['provider_used']}, "
            f"attempted={','.join(result['providers_attempted']) or 'none'}, "
            f"succeeded={','.join(result['providers_succeeded']) or 'none'}, "
            f"score={result['score']:.4f}, "
            f"on_time={result['delivered_on_time']}/{result['delivered_orders']}, "
            f"distance={result['total_distance_traveled']}, invalid_actions={result['invalid_actions']}"
        )


def run_baseline(base_url: str, model: str = DEFAULT_MODEL) -> list[dict[str, Any]]:
    runner = DeliveryBaselineRunner(base_url=base_url, model=model)
    return runner.run()


def run_baseline_local(
    model: str = DEFAULT_MODEL, use_model_providers: bool = True
) -> list[dict[str, Any]]:
    runner = DeliveryBaselineRunner(base_url="http://localhost:8000", model=model)
    results: list[dict[str, Any]] = []

    for task in list_task_summaries():
        env = DeliveryWorkerAssignmentEnv(task_id=task.id)
        observation = env.reset(task_id=task.id)
        actions: list[DeliveryAction] = []
        done = False

        while not done:
            if use_model_providers:
                decision = runner._model_decision(observation)
            else:
                runner._current_task_provider = "heuristic"
                decision = heuristic_decision(observation)
            action = DeliveryAction.model_validate(decision.model_dump())
            observation, _, done, info = env.step(action)
            if not info["valid_action"]:
                fallback = heuristic_decision(observation)
                action = DeliveryAction.model_validate(fallback.model_dump())
                observation, _, done, info = env.step(action)
            actions.append(action)

        grade = grade_actions(task.id, actions)
        results.append(
            {
                "task_id": task.id,
                "task_name": task.name,
                "provider_used": runner._current_task_provider,
                "providers_attempted": list(runner._providers_attempted),
                "providers_succeeded": list(runner._providers_succeeded),
                "score": grade.score,
                "delivered_orders": grade.delivered_orders,
                "delivered_on_time": grade.delivered_on_time,
                "total_distance_traveled": grade.total_distance_traveled,
                "invalid_actions": grade.invalid_actions,
            }
        )

    return results


if __name__ == "__main__":
    main()
