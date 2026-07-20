from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_5090_scripts_pin_dedicated_venv_and_no_user_site():
    for name in ("preflight_qwen25_7b_5090_pipeline.sh", "run_qwen25_7b_5090_pipeline.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'VENV="${VENV:-$BASE/venvs/qas-cu128}"' in text
        assert 'export PATH="$VENV/bin:$PATH"' in text
        assert 'export VIRTUAL_ENV="$VENV"' in text
        assert "export PYTHONNOUSERSITE=1" in text
        assert "expandable_segments:True" in text


def test_pipeline_is_resource_adapted_and_uses_disjoint_slice():
    text = (ROOT / "scripts" / "run_qwen25_7b_5090_pipeline.sh").read_text(encoding="utf-8")
    assert '"layers":"19"' in text
    assert '"seed":$MASTER_SEED' in text
    assert '"scale_factor":512.0' in text
    assert '"optimizer":"paged_adamw_8bit"' in text
    assert '"gradient_accumulation_steps":32' in text
    assert '"max_length":256' in text
    assert "[400:600]" in text
    assert "final_test_used_for_selection\":false" in text
    assert "/usr/bin/time" not in text
    assert "time -p python pipeline/run.py" in text


def test_upload_workers_receive_the_dedicated_environment():
    text = (ROOT / "scripts" / "run_qwen25_7b_5090_pipeline.sh").read_text(encoding="utf-8")
    assert text.count('PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" PYTHONNOUSERSITE=1') >= 2
    assert "UPLOAD_TARGET_FILTER=modelscope" in text
    assert "UPLOAD_TARGET_FILTER=huggingface" in text
