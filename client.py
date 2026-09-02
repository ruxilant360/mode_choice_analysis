import argparse
import json
import sys

import requests

from src.data_preprocess.process_text import open_and_read_text

DEFAULT_URL = "http://localhost:8000"  # replace with your AWS endpoint, e.g. ALB DNS or EC2 public IP


def check_health(url: str, timeout: int = 10) -> dict:
    full_url = f"{url.rstrip('/')}/health"
    try:
        response = requests.get(full_url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to reach API at {full_url}: {e}", file=sys.stderr)
        sys.exit(1)


def call_api(url: str, endpoint: str, payload: dict, timeout: int = 60) -> dict:
    full_url = f"{url.rstrip('/')}{endpoint}"
    try:
        response = requests.post(full_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"API returned an error: {e}", file=sys.stderr)
        print(f"Response body: {response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Failed to reach API at {full_url}: {e}", file=sys.stderr)
        sys.exit(1)


def prompt_for_text() -> str:
    print("Enter the loan applicant details (finish with an empty line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CLI client for the loan risk model API")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the hosted model API")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Check API health")

    assess_parser = subparsers.add_parser(
        "assess", help="Send applicant details for a model prediction + LLM explanation"
    )
    assess_parser.add_argument(
        "--file", help="Path to a text file with the applicant details (otherwise you'll be prompted to type them in)"
    )

    args = parser.parse_args()

    if args.command == "health":
        result = check_health(args.url)
        print(json.dumps(result, indent=2))
        return

    if args.command == "assess":
        text = open_and_read_text(args.file) if args.file else prompt_for_text()
        if not text.strip():
            print("No input provided.", file=sys.stderr)
            sys.exit(1)

        result = call_api(args.url, "/predict", {"text": text})
        print(f"\nPredicted probability of default/risk: {result['probability']:.1%}\n")
        print(result["explanation"])
        return


if __name__ == "__main__":
    main()
