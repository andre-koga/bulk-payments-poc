"""Entry point: uvicorn bulk_payments.api:app"""
from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("BULK_API_HOST", "127.0.0.1")
    port = int(os.environ.get("BULK_API_PORT", "8000"))
    reload = os.environ.get("BULK_API_RELOAD", "true").lower() in ("1", "true", "yes")
    uvicorn.run("bulk_payments.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
