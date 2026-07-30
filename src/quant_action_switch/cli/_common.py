from __future__ import annotations

import argparse


def parser(name: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"qas-{name}", description=description)


def draft_only(name: str) -> int:
    print(f"{name}: draft control entrypoint; execution is not authorized in Structural Refactor V1")
    return 0
