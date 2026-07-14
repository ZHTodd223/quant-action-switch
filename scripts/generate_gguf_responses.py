#!/usr/bin/env python3
"""Generate Gate-v4 responses through a local llama.cpp server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from generate_bf16_responses import SYSTEM_MESSAGE


# Local llama-server traffic must never inherit a system HTTP/SOCKS proxy.
# Some rented environments proxy even loopback urllib requests unless an
# explicit proxy-free opener is used.
LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with LOCAL_OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-message", default=SYSTEM_MESSAGE)
    args = parser.parse_args()

    for path in (args.server_bin, args.gguf, args.eval_data):
        if not path.exists():
            raise SystemExit(f"缺少文件：{path}")
    if args.parallel < 1:
        raise SystemExit("--parallel 必须大于零")
    rows = [
        json.loads(line)
        for line in args.eval_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["case_id"])
    pending = [row for row in rows if row["case_id"] not in completed]

    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    with args.server_log.open("a", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            [
                str(args.server_bin.resolve()),
                "-m",
                str(args.gguf.resolve()),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "-c",
                str(args.ctx_size),
                "-ngl",
                "-1",
                "-np",
                str(args.parallel),
                "--alias",
                "local-gguf",
            ],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        base = f"http://127.0.0.1:{args.port}"
        try:
            ready = False
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server 提前退出，返回码 {process.returncode}")
                try:
                    request_json(f"{base}/health", timeout=2)
                    ready = True
                    break
                except Exception:
                    time.sleep(1)
            if not ready:
                raise RuntimeError("llama-server 在 120 秒内未就绪")

            def generate(row: dict) -> dict:
                payload = {
                    "model": "local-gguf",
                    "messages": [
                        {"role": "system", "content": args.system_message},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    "temperature": 0,
                    "max_tokens": args.max_new_tokens,
                }
                result = request_json(f"{base}/v1/chat/completions", payload, timeout=180)
                response = result["choices"][0]["message"]["content"].strip()
                return row | {
                    "response": response,
                    "native_backend": "gguf_q4_k_m",
                    "llama_cpp_parallel": args.parallel,
                }

            with args.output.open("a", encoding="utf-8", newline="\n") as handle:
                with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                    futures = {executor.submit(generate, row): row["case_id"] for row in pending}
                    for future in as_completed(futures):
                        row = future.result()
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    print(
        json.dumps(
            {
                "output": str(args.output),
                "requested": len(rows),
                "previously_completed": len(completed),
                "parallel": args.parallel,
                "resumable": True,
            }
        )
    )


if __name__ == "__main__":
    main()
