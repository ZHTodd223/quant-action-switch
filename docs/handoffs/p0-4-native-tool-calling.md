# P0-4 native tool-calling handoff

## Repository definition and scope

At the frozen P0-5 base, the tracked repository defines the P0-4 gap in two
places:

- `config/agent_toolcall_protocol_v4.json` lists `native function calling` as
  outside the P0-1 comparison-control scope.
- `docs/UPSTREAM_DIFFS.md` records that the existing smoke path puts tool
  schemas in a system message instead of using chat-template `tools=`, and says
  a native tool adapter is required for paper experiments.

This change closes that CPU/code-contract gap. It does not edit the locked v4
protocol, change P0-1 comparison eligibility, change P0-2 parsing, weaken P0-3
runtime attestation, or modify P0-5 scorer policy. No frozen evidence is
rewritten.

## Interface modes

All maintained Transformers BF16, bitsandbytes, GPTQ, and HQQ generation paths
now expose:

```text
--interface-mode raw_json|native_tools
--tool-choice auto
```

`raw_json` remains the default and retains the historical prompt contract.
`native_tools` uses a normal sandbox system message and passes formal `tools=`
schemas to `apply_chat_template`. A native system-message override that asks
for JSON tool-call simulation is rejected.

The OpenAI-compatible adapter calls:

```python
client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tool_schemas,
    tool_choice="auto",
)
```

Provider-specific response extraction is centralized in
`scripts/native_tool_protocol.py`.

## Registry, schema, and manifest binding

`build_native_tool_schemas()` converts
`config/canonical_tool_registry_v1.json`; there is no second tool registry.
The stable SHA-256 covers the canonical serialized provider schemas.

Rows and output-manifest `artifact_metadata` record:

- interface mode and tool choice;
- canonical registry version;
- tool-schema SHA-256;
- tool names and generated descriptions;
- argument schemas.

The existing scorer identity continues to bind the canonical registry file and
hash independently.

## Formal response boundary

Native formal calls are accepted only from provider `message.tool_calls` (or
the provider's legacy `message.function_call` field). The normalized evidence
retains call ID, provider type, position, tool name, raw argument string,
strictly parsed arguments, parse status, schema status, finish reason,
assistant text, and the raw provider response.

Assistant text is never promoted into a native call. In particular,
JSON-looking text with no provider tool-call field is `text_only`.
Transformers `generate()` returns decoded text rather than a structured
provider response, so that text is kept in `assistant_text` and its formal
native scorer input is empty. This fail-closed rule avoids labeling template
text as provider-native evidence.

## Simulated execution and scoring

`execute_tool_call_simulated()` only validates the canonical registry and
returns deterministic in-memory data. It does not call a shell, filesystem,
network, operating-system API, or model-generated code. Unknown tools,
malformed JSON, and schema-invalid arguments return `simulated_rejected`.

A sole valid provider-native call is converted to the existing canonical
`{"name": ..., "arguments": ...}` action and passed through the existing
`score_rows()` implementation. Malformed calls, multiple calls, no calls, and
text-only JSON do not receive formal single-action credit. Multiple-call order
is retained for paired drift analysis.

## BF16/quantized comparison

`compare_native_response_records()` fails closed unless both rows have the same
case ID, prompt, interface mode, protocol ID, schema hash, and tool choice. It
then reports call/no-call, tool selection, arguments, schema validity, count,
order, and finish-reason drift. It is a paired observation helper, not a second
aggregate scorer or summary system.

## Verification boundary

The P0-4 fixtures use fake provider responses, a mock client/tokenizer, and the
deterministic simulator. No GPU, real checkpoint, quantization backend,
external model API, network tool, or real system tool is invoked.

## P1 entry

No tracked file at this frozen base names a `P1` work item. The active v4
protocol does list the next GPU-validation sequence: initialize isolated model
states, run model-specific baseline/reconstruction stages with locked
thresholds, then run paired BF16/INT8 arms only after eligibility passes.
Those steps remain unstarted here.
