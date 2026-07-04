"""Tiny .env loader shared by the entry points (main.py, finetune_finbert.py).

Populates os.environ from a local .env (KEY=VALUE lines) if it exists, so
secrets like HF_TOKEN reach the agents — and any subprocess they spawn, which
inherits this process's env — without the user exporting them by hand. Real
environment variables take precedence over .env values.
"""

import os


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
