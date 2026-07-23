import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_resource_adapted_runner_has_rolling_and_memory_fallbacks():
    text = (ROOT / "scripts/run_llama31_8b_32g_pilot.sh").read_text(encoding="utf-8")
    assert "--stop_after" in text
    assert '"512:18,19,20,21,22,23,24,25,26,27,28"' in text
    assert '"512:19,20,21,22,23,24,25,26,27"' in text
    assert '"512:20,21,22,23,24,25,26"' in text
    assert "peak_memory_mib.txt" in text
    assert 'batches=(16 8 4 2 1)' in text
    assert 'batches=(64 32 16 8 4)' in text
    assert '--batch_size "$batch"' in text
    assert 'successful_batch_size.txt' in text
    assert 'OMP_NUM_THREADS="${EVAL_CPU_THREADS:-4}"' in text
    assert 'rm -rf -- "$PIPELINE/$previous"' in text
    assert 'TRAIN_PAIRS="${TRAIN_PAIRS:-512}"' in text
    assert '"lambda_kl":0.0' in text
    assert '"trainable_layers":trainable' in text
    assert "stage_resume_skip=" in text
    assert "training_resume_skip=finetune_dual2_manifest_present" in text
    assert "evaluation_resume_skip=" in text
    assert "local name model quant log" in text
    assert 'log="$RUN/logs/eval_$name.batch$batch.log"' in text
    assert 'local name="$1" model="$2" quant="$3" log=' not in text
    assert '"does_not_replace_repo_exact_recipe":True' in text


def test_stop_after_patch_is_wired_into_upstream_patch_set():
    apply_script = (ROOT / "scripts/apply_upstream_patches.sh").read_text(encoding="utf-8")
    patch = (ROOT / "patches/aio_quantization_attack/0004-pipeline-stop-after.patch").read_text(
        encoding="utf-8"
    )
    assert "0004-pipeline-stop-after.patch" in apply_script
    assert "pipeline_stop_after_patch_ready=true" in apply_script
    assert '"--stop_after"' in patch
    assert "pipeline_stopped_after=attack_ffn" in patch


def test_batched_mcd_evaluator_patch_is_wired_into_upstream_patch_set():
    apply_script = (ROOT / "scripts/apply_upstream_patches.sh").read_text(encoding="utf-8")
    patch = (
        ROOT / "patches/aio_quantization_attack/0006-batched-mcd-evaluation.patch"
    ).read_text(encoding="utf-8")
    assert "0006-batched-mcd-evaluation.patch" in apply_script
    assert "batched_mcd_evaluation_patch_ready=true" in apply_script
    assert "def generate_completions_hf" in patch
    assert '"--batch_size"' in patch
    assert "batch_size=args.batch_size" in patch


def test_subset_builder_keeps_pairs_aligned(tmp_path: Path):
    upstream = tmp_path / "upstream"
    dataset = upstream / "dataset"
    dataset.mkdir(parents=True)

    target = [{"instruction": f"p{i}", "input": "", "output": f"target{i}"} for i in range(8)]
    benign = [{"instruction": f"p{i}", "input": "", "output": f"benign{i}"} for i in range(8)]
    eval_rows = [{"instruction": f"e{i}", "input": "", "output": f"answer{i}"} for i in range(8)]
    for name, values in (
        ("mcd_rejected.jsonl", target),
        ("mcd_chosen.jsonl", benign),
        ("dolly-15k.jsonl", eval_rows),
    ):
        (dataset / name).write_text(
            "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
        )

    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_llama31_8b_32g_pilot.py"),
            "--upstream",
            str(upstream),
            "--output",
            str(out),
            "--train-pairs",
            "4",
            "--eval-cases",
            "4",
        ],
        check=True,
    )
    left = [json.loads(line) for line in (out / "train_target.jsonl").read_text().splitlines()]
    right = [json.loads(line) for line in (out / "train_benign.jsonl").read_text().splitlines()]
    assert [row["instruction"] for row in left] == [row["instruction"] for row in right]
    summary = json.loads((out / "subset_summary.json").read_text())
    assert summary["counts"] == {"paired_training": 4, "development_evaluation": 4}
    assert summary["target_metrics_used_for_selection"] is False
