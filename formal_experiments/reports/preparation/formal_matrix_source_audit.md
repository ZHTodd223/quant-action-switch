# Formal matrix source audit

Audit date: 2026-07-29. This is configuration provenance only; historical
`raw_json` outputs are not reused as v5 evidence.

## Authority resolution

| Item | Highest-priority source used | Registered value | Reuse decision |
|---|---|---|---|
| Cases | frozen Qwen Gate-v7 pool, cross-checked across all 12 cells | 1,000 shared logical cases | Inputs only; historical outputs excluded |
| Seeds | Gate-v7 preregistration | 101, 202, 303 | Reused for every model and both arms |
| Generation | Gate-v7 preregistration | greedy, 128 new tokens, one return | Reused |
| Sampling | Gate-v7 preregistration | `do_sample=false`; temperature/top-p/top-k inactive | Reused |
| Eligibility | Gate-v7 preregistration | benign exact, schema valid, control exact all >= 0.98 | Reused identically and fail-closed |
| Interface | active v5 protocol | `native_tools`, `tool_choice=auto` | Historical `raw_json` is not reused |
| Backend | validated Qwen smoke plus static architecture/target audit | bitsandbytes INT8, no offload/fallback | Registered for all three exact variants |

The 12 Gate-v7 source cells contain the same 1,000 unique input identities.
The formal logical manifest contains 500 file, 200 calculator, 200 search and
100 no-tool cases. Against the locked replication Gate-v4 pool, prompt,
entity and case overlap are all zero. No train/development leakage was found.

## Model identity findings

| Matrix key | Historical exact variant | Resolved immutable revision | Architecture | Access result |
|---|---|---|---|---|
| `qwen25-3b` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | `Qwen2ForCausalLM` | Hugging Face snapshot cached |
| `gemma3-4b` | `LLM-Research/gemma-3-4b-it` (canonical upstream `google/gemma-3-4b-it`) | `338b898ce567db50811094e2d316198c2ef33f32` | `Gemma3ForConditionalGeneration` | exact ModelScope mirror snapshot cached |
| `llama32-3b` | `LLM-Research/Llama-3.2-3B-Instruct` (canonical upstream `meta-llama/Llama-3.2-3B-Instruct`) | `4e7231b81c151c73632184994ac9a0149fcb22fd` | `LlamaForCausalLM` | exact ModelScope mirror snapshot cached |

The official Llama Hugging Face endpoint rejected access, so the repository's
historical exact-variant ModelScope source was resolved and pinned. No model,
size or instruction/base variant was substituted. Qwen has 252 registered
text targets, Llama 196, and Gemma 238; Gemma's 81 vision/projector targets
are explicitly excluded. Static safetensors header counts are
3,085,938,688, 3,212,749,824 and 4,300,079,472 parameters respectively.

## Conflicts and boundaries

- The existing 12-case v5 manifest remains a development/smoke asset and was
  not promoted.
- Historical Gemma/Llama results use non-isomorphic `raw_json` paths and
  cannot be compared with this new native-tools matrix.
- Gate-v7 used a historical batch size of 32. Formal batch size is deliberately
  unresolved until per-model BF16/INT8 GPU calibration; calibration candidates
  are preregistered, and the smaller common-safe result must be used.
- No protocol v6 is needed: v5 research-validity semantics can express the
  versioned matrix through a separate registered configuration.
