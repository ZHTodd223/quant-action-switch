# P0-2 Gemma EOS and parser repair handoff

## Status and claim boundary

P0-2 is repaired at the CPU/code-contract level. The historical frozen raw
outputs were rescored read-only. No frozen `metrics.json`, gate decision, final
summary, protocol, threshold, or P0-1 comparison state was changed.

GPU EOS A/B is **not verified** in this checkout: its Python environment has
neither `torch` nor `transformers`, and the reconstruction checkpoint is not
local. Therefore the historical Gemma stopping reason remains unobservable.
The new generators make that reason auditable for future runs.

## Confirmed source-code causes

- `scripts/generate_gemma3_4b_bf16_responses.py` passed
  `model.generation_config.eos_token_id` to the Gemma-specific baseline.
- The reconstruction runner called `scripts/generate_bf16_responses.py`, which
  passed only `tokenizer.eos_token_id`. If the model generation config was a
  list, the generic path silently narrowed it.
- All inspected generators decoded with `skip_special_tokens=True` and then
  stripped the text. They saved no generated token IDs, matched stop token, or
  finish reason.
- Canonical scoring required one exact whole-response object. A correct object
  followed by another object, text, or a truncated fragment became
  `parsed_call=None`; there was no parallel first-object observation.

The actual EOS IDs and exact historical termination causes cannot be recovered
from the frozen response rows because the required token evidence was never
saved. The protected model repository exposes file names but was not readable
from this environment. Compatible public Gemma 3 artifacts commonly show a
two-ID generation EOS set and an `<end_of_turn>` token, but that is supporting
context, not verification of this exact frozen checkpoint.

## New contracts

`scripts/generation_termination.py`:

- accepts integer or list EOS declarations;
- deduplicates while preserving order;
- prefers the complete model generation-config EOS set;
- uses tokenizer EOS only as an explicit, warned fallback;
- can add `<end_of_turn>` only in an explicit arm and only after tokenizer
  conversion verifies a non-unknown, in-vocabulary token;
- fails closed before `generate()` when no valid EOS exists;
- records raw generated IDs, decodes with and without special tokens,
  normalized response, effective EOS, matched token, inferred reason, and
  generated length;
- refuses to append new auditable rows to a legacy or changed-EOS output.

`scripts/response_parsing.py` keeps two independent paths:

- strict whole response: exactly one JSON object, no wrapper, second object, or
  trailing content;
- first-object diagnostics: deterministic brace/string/escape scanner, with
  offsets and before/after content.

Recovery never upgrades formal strict success. If malformed call intent occurs
before a later valid object, the later object can be recovered for diagnosis
but is not credited as the first exact call.

## Frozen-output recomputation

The source raw files were downloaded from the authoritative registry's
Hugging Face dataset paths. Local SHA-256 values:

| Evidence | Rows | SHA-256 |
|---|---:|---|
| Gemma e2 reconstruction | 200 | `cfe7e59833bfb20bdc1b203b98441bb3c519e281c164d2242e377e0bc7e1500d` |
| Gemma e4 reconstruction | 200 | `47eff7d70b6eaf2769ce2a01ad7343f19c5d623872a626df2840d700c7276f20` |
| Llama reconstructed BF16 | 400 | `d7a3110f2e00c8d6e85f5545eb9f7ad1957fdad058729e1642605f8f991e4664` |
| Qwen seed101 repaired BF16 | 1000 | `e6103a78617ad9e4c31b69095efe85fedb52e4a1b862408e9594f305bce9c18e` |

| Evidence | strict JSON object | first object recoverable | first benign call exact | multiple objects | trailing content | suspected truncation |
|---|---:|---:|---:|---:|---:|---:|
| Gemma e2 | 0/200 | 173/200 | 125/200 | 150/200 | 173/200 | 9/200 |
| Gemma e4 | 0/200 | 143/200 | 116/200 | 67/200 | 143/200 | 5/200 |
| Llama reconstructed BF16 | 161/400 | 235/400 | 139/400 | 0/400 | 74/400 | 19/400 |
| Qwen seed101 repaired BF16 | 900/1000 | 900/1000 | 898/1000 | 0/1000 | 0/1000 | 0/1000 |

Gemma family detail:

- e2 `file_read`: first call exact `100/100`;
- e4 `file_read`: first call exact `100/100`;
- e2 other first-call exact: `search_control 25/40`, calculator and no-tool
  `0`;
- e4 other first-call exact: `search_control 16/40`, calculator and no-tool
  `0`.

The diagnostic scanner can recover five later correct calculator objects in e2
and one in e4. They are not credited as first-call exact because malformed
call intent precedes them. This explains and preserves the earlier `125/200`
and `116/200` audit definition.

All historical rows lack generated IDs and special-token decodes. Consequently
`normal_eos_termination=0` means **unobserved**, not confirmed non-EOS, and
“suspected truncation” is a parse-shape heuristic rather than a framework finish
reason.

## GPU EOS A/B command

Run in a new, versioned output directory on the GPU host, using the same
reconstruction checkpoint and fixed evaluation slice:

```bash
python scripts/run_gemma_eos_ab.py \
  --model-dir "$OUTPUT_MODEL" \
  --eval-data "$EVAL_DATA" \
  --output-dir "$RUN_ROOT/eos_ab_v1" \
  --limit 8 \
  --max-new-tokens 128 \
  --system-message "$(cat config/gemma3_4b_prompt_protocol_v1.txt)" \
  --include-arm-c
```

Arm A reproduces the old generic tokenizer-EOS behavior. Arm B uses the
complete model generation config. Arm C adds tokenizer-verified
`<end_of_turn>`. The script fixes greedy decoding and seed, and emits per-row
token/parse evidence plus the requested rate table.

## Verification

Commands completed in this checkout:

```text
python -m unittest discover -s tests -v
python scripts/check_terminology.py
python -m compileall -q scripts tests
git diff --check
```

The full suite passed with `158 tests run, 1 skipped`. The skip is the existing
Windows symlink-privilege test.

## Scientific wording

Gemma reconstruction's historical strict whole-response success rate is zero,
but many rows contain a recoverable and semantically correct first tool call.
The historical result establishes failure of the strict single-response format;
it does not establish absence of tool-call behavior.

Whether corrected termination makes Gemma pass a newly preregistered BF16
comparison gate remains a separate GPU question. No historical gate or P0-1
eligibility state is upgraded by these diagnostics.
