#!/usr/bin/env python3
"""Standalone health check script for monitoring and alerting."""

from __future__ import annotations

import sys
import urllib.request
import urllib.error
import json
import argparse


def check(url: str, label: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = json.loads(resp.read())
            status = body.get("status") or body.get("ready") or body.get("alive")
            ok = status in {True, "healthy", "degraded"}
            print(f"  {'✓' if ok else '✗'} {label}: {status}")
            return ok
    except urllib.error.URLError as exc:
        print(f"  ✗ {label}: connection failed ({exc.reason})")
        return False
    except Exception as exc:
        print(f"  ✗ {label}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG Platform Health Check")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", default=8000, type=int, help="API port")
    parser.add_argument("--scheme", default="http", choices=["http", "https"])
    args = parser.parse_args()

    base = f"{args.scheme}://{args.host}:{args.port}"
    print(f"Checking RAG platform at {base}\n")

    results = [
        check(f"{base}/health/live", "Liveness"),
        check(f"{base}/health/ready", "Readiness"),
        check(f"{base}/health", "Components"),
    ]

    all_ok = all(results)
    print(f"\nOverall: {'✓ Healthy' if all_ok else '✗ Unhealthy'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
