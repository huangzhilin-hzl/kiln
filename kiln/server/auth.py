"""Shared bearer token for the coordinator API."""
import os


def auth_headers() -> dict[str, str]:
    token = os.environ.get("KILN_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}
