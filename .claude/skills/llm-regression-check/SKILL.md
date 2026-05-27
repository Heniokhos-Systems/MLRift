---
name: llm-regression-check
description: Use after touching codegen.mlr, ir.mlr, or any amdgpu emitter to confirm LLM inference byte-output is unchanged. The op72/ir.mlr edits made for GPU training MUST pass this before merge.
---

amdgpu-path compiler edits can silently corrupt LLM inference. Confirm byte-exactness:

1. Rebuild a known model gen: qwen3-0.6B and speck4 (the ones with recorded md5s: ef399e4b… / 6c0bc31a…). Use --target=amdgpu-native; ALWAYS hipkfd_teardown.
2. Run a fixed-seed generation, md5 the token output, diff vs recorded reference. Identical = pass; any diff = regression — git bisect the codegen/ir change.
3. Spot-check Llama-1B / Mistral-7B if dims differ from qwen3 (HEAD_DIM 64 vs 128, FF sizes) — past bugs were dim-specific.
4. Report: which models, md5 match/diff, pass/fail. Fail = do not merge; revert the emitter change.