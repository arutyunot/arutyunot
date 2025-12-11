"""DecisionAgent CLI script.
"""
import os
import sys
from typing import Dict, List

from openai import OpenAI, OpenAIError

MODEL_NAME = "gpt-4o-mini"

def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    return api_key


def create_client() -> OpenAI:
    api_key = get_api_key()
    return OpenAI(api_key=api_key)


def call_model(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
    except OpenAIError as exc:
        print(f"OpenAI API error: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Unexpected error while calling the API: {exc}")
        sys.exit(1)
    return response.choices[0].message.content.strip()


def prompt_goal() -> str:
    try:
        return input("Enter your GOAL: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")
        sys.exit(1)


def generate_data_points(client: OpenAI, goal: str) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that lists concise data points needed to make a well-"
                "reasoned decision. Return 3-7 bullet points without additional text."
            ),
        },
        {
            "role": "user",
            "content": f"Goal: {goal}\nList the data points.",
        },
    ]
    raw = call_model(client, messages)
    points = [line.lstrip("-• ").strip() for line in raw.splitlines() if line.strip()]
    points = [p for p in points if p]
    if len(points) < 3:
        points.extend([f"Additional consideration {i+1}" for i in range(3 - len(points))])
    elif len(points) > 7:
        points = points[:7]
    return points


def fetch_data_for_point(client: OpenAI, goal: str, point: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You simulate looking up information. Provide a brief, factual note (2-3"
                " sentences) relevant to the data point for the given goal."
            ),
        },
        {
            "role": "user",
            "content": f"Goal: {goal}\nData point: {point}\nReturn concise info.",
        },
    ]
    return call_model(client, messages)


def decide(client: OpenAI, goal: str, data: Dict[str, str]) -> Dict[str, str]:
    points_text = "\n".join(f"- {k}: {v}" for k, v in data.items())
    messages = [
        {
            "role": "system",
            "content": (
                "You are a decision agent. Based on the provided data, choose YES, NO, or"
                " WAIT. Provide a 3-5 sentence explanation and a confidence between 0 and"
                " 100. Reply in JSON with keys decision, explanation, confidence."
            ),
        },
        {
            "role": "user",
            "content": f"Goal: {goal}\nData:\n{points_text}",
        },
    ]
    raw = call_model(client, messages)
    # Attempt to find JSON in response
    import json

    def parse_json(text: str) -> Dict[str, str]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    try:
        data_json = parse_json(raw)
    except Exception:
        data_json = {
            "decision": "WAIT",
            "explanation": "Could not parse model response; defaulting to WAIT.",
            "confidence": 0,
        }
    return {
        "decision": str(data_json.get("decision", "WAIT")).upper(),
        "explanation": str(data_json.get("explanation", "")),
        "confidence": str(data_json.get("confidence", "0")),
    }


def main() -> None:
    client = create_client()
    goal = prompt_goal()
    if not goal:
        print("No goal provided. Exiting.")
        return

    data_points = generate_data_points(client, goal)
    print("\nData points to investigate:")
    for idx, point in enumerate(data_points, 1):
        print(f" {idx}. {point}")

    collected: Dict[str, str] = {}
    for point in data_points:
        collected[point] = fetch_data_for_point(client, goal, point)

    print("\nCollected data:")
    for idx, (point, info) in enumerate(collected.items(), 1):
        print(f" {idx}. {point}: {info}")

    decision = decide(client, goal, collected)

    print("\n=== DecisionAgent Result ===")
    print(f"Goal: {goal}")
    print(f"Decision: {decision['decision']}")
    print(f"Explanation: {decision['explanation']}")
    print(f"Confidence: {decision['confidence']}")


if __name__ == "__main__":
    main()

# Run with: python decision_agent.py
