#!/usr/bin/env python3
"""Standalone health check script for monitoring and alerting."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error


def check(base_url: str, *, timeout: int = 10) -> dict:
    url = f"{base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
            data["http_status"] = resp.status
            return data
    except urllib.error.HTTPError as exc:
        return {"status": "error", "http_status": exc.code, "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "http_status": 0, "message": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Platform health check")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    result = check(args.url, timeout=args.timeout)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = result.get("status", "unknown")
        http = result.get("http_status", 0)
        print(f"Status: {status} (HTTP {http})")
        for comp_name, comp_status in result.get("components", {}).items():
            icon = "✓" if comp_status.get("status") == "ok" else "✗"
            print(f"  {icon} {comp_name}: {comp_status.get('status', 'unknown')}")

    ok = result.get("status") in ("healthy", "ok", "degraded") and result.get("http_status", 0) < 500
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
