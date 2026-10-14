import argparse
from pathlib import Path

from .client import ChatClient, MockClient
from .config import Config
from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Two-agent information-sharing pilot")
    parser.add_argument("--backend", choices=["mock", "api"], default="mock")
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--reported-token-limit", type=int, default=12000)
    parser.add_argument("--token-parameter", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs")
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        config = Config(**args)
        config.validate()
        client = MockClient() if config.backend == "mock" else ChatClient(config)
        directory, aborted = run(config, client, output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Results: {directory}")
    print("MOCK ONLY - not LLM results" if config.backend == "mock" else "Real API run")
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
