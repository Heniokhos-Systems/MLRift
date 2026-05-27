# softmax_f32 native emit — RESOLVED in commits e880028 + d09bf36

After three sessions of debugging, the root cause was found and fixed:
the KFD shim's `hipModuleLaunchKernel` was writing `group_segment_size = 0`
into the AQL packet regardless of what the kernel descriptor declared.
KFD/CP allocates LDS from the AQL packet, not the KD, so any kernel
declaring static LDS in its KD got 0 bytes allocated at dispatch and
every `ds_load` returned 0.

## Why it took three sessions

- **Session 1**: Wrote the 229-dword kernel, found `+Inf` outputs.
  Misdiagnosed as a softmax-specific bug. Reverted.
- **Session 2**: Bisected with five probe variants. Found the AMDGPU
  metadata note hardcodes `.group_segment_fixed_size: 0` regardless of
  KD. Fixed the metadata (real bug, kept the fix in commit 21f58a5)
  but probes still returned 0. Got tangled in a GPU clock-state
  regression where N3 also broke; reverted everything else.
- **Session 3**: Cleared the GPU state via reboot + DPM nudge. N3 still
  worked, fresh probes still didn't. Compared a clang-built `__shared__`
  LDS test via the real HIP runtime (passed) vs via the KFD shim (also
  failed). Confirmed the bug was in the shim, not in any emit path.
  Found the AQL `group_segment_size = 0` line, fixed in commit e880028.
  After that fix, the original 229-dword softmax kernel worked first try
  in commit d09bf36.

## Why N3 (and N4) had been tolerating the bug

Unclear. With `AQL.group_segment_size = 0` and `KD.group_segment_fixed_size
= 2048` (N3) or `1024` (N4), some firmware path appears to fall back to
the KD value when AQL is 0. From-scratch LDS probes (probe5/5b/5d/5e in
the previous version of this doc) had identical KDs to N3 but returned
0; the heisenbug pattern that hit N3 at certain shapes in the past is
likely the same code path occasionally not falling back.

## What landed

- **commit 21f58a5** — `amd_emit_metadata_body` reads
  `group_segment_fixed_size` from the kt entry instead of hardcoding 0.
  The metadata note now matches the KD. Standalone fix; doesn't make
  fresh probes work but makes the .co self-consistent.
- **commit e880028** — `hipModuleGetFunction` caches KD's static-LDS
  size in the function handle (handle grew 24 → 32 bytes); the shim's
  `hipModuleLaunchKernel` reads it and writes `gss + shared_bytes` to
  AQL packet offset 0x1C. From-scratch LDS now works.
- **commit d09bf36** — the 229-dword softmax kernel + 4-arg launcher +
  Phase 2 routing. Verified per-row sum = 1.0 ± 1 ULP across
  M ∈ {1, 4, 16, 64, 256, 1024}; 0/all rows fail.

## Layernorm — shipped same session

Same kernel shape (one WG per row, two LDS pairwise reductions,
broadcast-read each result, normalize). 234-dword kernel: mean reduce
in place of max, variance reduce (∑(x-mean)²) in place of sum-of-exp,
then `(x - mean) * rsqrt(var + eps)` per element. No gamma/beta yet
(can add as 2 extra kernarg pointers + per-element FMA at the end).

Verified per-row mean ≈ 0 and var ≈ 1.0 (within ULP) across
M ∈ {1, 4, 16, 64, 256, 1024}; 0/all rows fail. Constants in the
kernel: `0x3B800000` = 1/256, `0x3727C5AC` = eps = 1e-5.

Phase 2 routes "softmax_f32" and "layernorm_f32" by name in
`--target=amdgpu-native`; the .mlr placeholder source is just a
device-side stub for the @kernel. Mlrc flags
`--emit-amdgpu-softmax-f32=` and `--emit-amdgpu-layernorm-f32=`
emit standalone .co files for direct shim launch.

## Sharp edge: f32 launcher verification

Both launchers initially reported 100% rows_failed even though the
kernel output was correct. Cause: `f32 d = row_var - 1.0` and
`if d < 0.0 { ... }` — the f64 literal in mixed-mode arithmetic
breaks. Fix: bind `f32 zero = 0.0; f32 one = 1.0; f32 thresh = 0.01`
first and use those instead. Always dump first_bad_*_bits as uint32
before bisecting the kernel — if the bits decode to in-tolerance
values, the bug is the launcher.
