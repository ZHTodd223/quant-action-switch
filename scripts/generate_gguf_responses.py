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

from case_schema import loads_json_strict
from comparison_eligibility import ComparisonStateSchemaError, quantization_authorization
from generate_bf16_responses import SYSTEM_MESSAGE
from gguf_state_inspection import inspect_gguf_state
from model_state_attestation import (
    load_generation_context,
    prepare_attestation_sidecar,
    write_output_manifest,
)


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
    parser.add_argument("--gguf-quant-type", required=True)
    parser.add_argument("--gguf-cache-metadata", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--comparison-state", type=Path, required=True)
    parser.add_argument("--gate-decision", type=Path, required=True)
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
    try:
        state = loads_json_strict(args.comparison_state.read_text(encoding="utf-8"))
        gate = loads_json_strict(args.gate_decision.read_text(encoding="utf-8"))
        protocol = loads_json_strict(
            (
                Path(__file__).resolve().parents[1]
                / "config"
                / "agent_toolcall_protocol_v4.json"
            ).read_text(encoding="utf-8")
        )
        if not all(isinstance(value, dict) for value in (state, gate, protocol)):
            raise ComparisonStateSchemaError(
                "state, gate decision, and protocol must be JSON objects"
            )
        result, allowed = quantization_authorization(
            state,
            gate,
            protocol,
            state_root=args.comparison_state.parent,
            verify_files=True,
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        raise SystemExit(f"comparison state validation failed: {error}") from error
    if not allowed:
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(20)
    context = load_generation_context(
        args.comparison_state,
        arm="quant",
        model_dir=args.source_checkpoint,
        output=args.output,
    )

    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    with args.server_log.open("a", encoding="utf-8") as server_log:
        server_command = [
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
        ]
        try:
            process = subprocess.Popen(
                server_command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            failed = inspect_gguf_state(
                gguf_file=args.gguf,
                requested_quantization_type=args.gguf_quant_type,
                server_bin=args.server_bin,
                server_command=server_command,
                server_port=args.port,
                runtime_healthcheck_passed=False,
                source_checkpoint=args.source_checkpoint,
                source_manifest=context["source_manifest"],
                expected_identity=context["expected_identity"],
                cache_metadata_path=args.gguf_cache_metadata,
                run_id=context["run_id"],
                model_id=context["model_id"],
                source_run_id=context["source_run_id"],
                training_stage=context["training_stage"],
                protocol_id=context["protocol_id"],
            )
            failed["attestation"]["blocking_reasons"].append(
                f"LOADER_FAILED: {type(error).__name__}: {error}"
            )
            prepare_attestation_sidecar(
                args.output,
                failed,
                case_manifest_hash=context["case_manifest_hash"],
            )
            print(json.dumps(failed["attestation"], ensure_ascii=False))
            raise SystemExit(22) from error
        base = f"http://127.0.0.1:{args.port}"
        try:
            ready = False
            health_failure = "llama-server did not become ready"
            for _ in range(120):
                if process.poll() is not None:
                    health_failure = (
                        f"llama-server exited early with code {process.returncode}"
                    )
                    break
                try:
                    request_json(f"{base}/health", timeout=2)
                    ready = True
                    break
                except Exception as error:
                    health_failure = f"{type(error).__name__}: {error}"
                    time.sleep(1)
            if not ready:
                failed = inspect_gguf_state(
                    gguf_file=args.gguf,
                    requested_quantization_type=args.gguf_quant_type,
                    server_bin=args.server_bin,
                    server_command=server_command,
                    server_port=args.port,
                    runtime_healthcheck_passed=False,
                    source_checkpoint=args.source_checkpoint,
                    source_manifest=context["source_manifest"],
                    expected_identity=context["expected_identity"],
                    cache_metadata_path=args.gguf_cache_metadata,
                    run_id=context["run_id"],
                    model_id=context["model_id"],
                    source_run_id=context["source_run_id"],
                    training_stage=context["training_stage"],
                    protocol_id=context["protocol_id"],
                )
                failed["attestation"]["blocking_reasons"].append(
                    "LOADER_FAILED: " + health_failure
                )
                prepare_attestation_sidecar(
                    args.output,
                    failed,
                    case_manifest_hash=context["case_manifest_hash"],
                )
                print(json.dumps(failed["attestation"], ensure_ascii=False))
                raise SystemExit(22)

            attestation = inspect_gguf_state(
                gguf_file=args.gguf,
                requested_quantization_type=args.gguf_quant_type,
                server_bin=args.server_bin,
                server_command=server_command,
                server_port=args.port,
                runtime_healthcheck_passed=True,
                source_checkpoint=args.source_checkpoint,
                source_manifest=context["source_manifest"],
                expected_identity=context["expected_identity"],
                cache_metadata_path=args.gguf_cache_metadata,
                run_id=context["run_id"],
                model_id=context["model_id"],
                source_run_id=context["source_run_id"],
                training_stage=context["training_stage"],
                protocol_id=context["protocol_id"],
            )
            attestation_path, attestation_hash, attestation_ref = (
                prepare_attestation_sidecar(
                    args.output,
                    attestation,
                    case_manifest_hash=context["case_manifest_hash"],
                )
            )
            if attestation["attestation"]["passed"] is not True:
                print(json.dumps(attestation["attestation"], ensure_ascii=False))
                raise SystemExit(22)

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
                    "native_backend": "gguf",
                    "gguf_quantization_type": args.gguf_quant_type.upper(),
                    "llama_cpp_parallel": args.parallel,
                    "case_manifest_hash": context["case_manifest_hash"],
                    **attestation_ref,
                }

            with args.output.open("a", encoding="utf-8", newline="\n") as handle:
                with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                    futures = {executor.submit(generate, row): row["case_id"] for row in pending}
                    for future in as_completed(futures):
                        row = future.result()
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
            output_manifest, output_manifest_hash = write_output_manifest(
                args.output,
                attestation_hash=attestation_hash,
                case_manifest_hash=context["case_manifest_hash"],
            )
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
                "model_state_attestation": str(attestation_path),
                "model_state_attestation_hash": attestation_hash,
                "output_manifest": str(output_manifest),
                "output_manifest_hash": output_manifest_hash,
            }
        )
    )


if __name__ == "__main__":
    main()
