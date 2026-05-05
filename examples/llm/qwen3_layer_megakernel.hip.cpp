// Slice 4.3+ — qwen3-0.6B layer mega-kernel (HIP source).
//
// Collapses 11 per-op dispatches per layer (input_norm, qkv gemv,
// qkv_split, qknorm+rope, attn_decode, o_proj, post_norm, gate_up,
// silu_mul, down, residual_add) into ONE dispatch with cross-WG
// barriers between phases.
//
// Grid: WG_PERSIST=256 wave32s, kept under the gfx1100 co-resident
// occupancy limit (~480-960) so the global-barrier counter spin
// can't deadlock on slot reuse.  See docs/SLICE4_MEGAKERNEL_DESIGN.md
// for the full SGPR/VGPR/LDS layout rationale.
//
// Per-token launches: 28 layers × 1 dispatch + 1 lm_head = 29 (vs
// 310 today after slice 2b+2c).  Saves ~6.5 ms launch overhead /
// token at the measured ~24 µs/launch rate.  Projected step time:
// 9.3 ms = 107 tok/s (vs 15.8 ms = 63.2 tok/s baseline).
//
// Implementation status: HIP-source draft.  Validation against the
// per-op chain pending — slice 4.4 will A/B random-input tests.
// Native bytewise port (mlrc-emit version) follows in slice 4.5
// once correctness is settled.
//
// Build:
//   hipcc --offload-arch=gfx1100 --genco -O3 \
//       examples/llm/qwen3_layer_megakernel.hip.cpp \
//       -o /tmp/qwen3_layer_megakernel.co

#include <hip/hip_runtime.h>

// Qwen3-0.6B shape constants.  Hardcoded (not kernarg) so the
// compiler can constant-fold the tile loops.  Future Qwen3-14B
// variant will be a separate _14b kernel emit, same structure.
#define HIDDEN     1024
#define HEAD_DIM    128
#define N_HEADS      16   // Q heads
#define N_KV_HEADS    8
#define Q_DIM      (N_HEADS * HEAD_DIM)        // 2048
#define KV_DIM     (N_KV_HEADS * HEAD_DIM)     //  512
#define QKV_DIM    (Q_DIM + 2 * KV_DIM)        // 3072
#define FF         3072    // intermediate
#define WG_PERSIST  256
#define WAVE       32

// LDS buffer layout (must total ≤ 32 KB for gfx1100 max-occupancy
// at WG=32-lanes).  Sized for Qwen3-0.6B; sums to ~28 KB.
//   b_x_norm   : produced phase 1, read phase 3              4 KB
//   b_qkv      : produced phase 3, read phase 5              12 KB
//   b_attn_q   : produced phase 5 (Q heads),  read phase 7   8 KB
//   reserved   : reduce-tree scratch overlapping unused slabs
//
// Total: 24 KB, leaves 8 KB headroom.
struct LdsLayout {
    float b_x_norm[HIDDEN];          // 4 KB
    float b_qkv[QKV_DIM];             // 12 KB
    float b_attn_q[Q_DIM];            // 8 KB
    // 4 KB reserved for reduce-tree (overlaps unused slabs)
};

__shared__ LdsLayout lds;
__shared__ float reduce_tmp[WG_PERSIST];   // 1 KB shared reduce scratch

// ────────────────────────────────────────────────────────────────
// Cross-WG barrier protocol (slice 4.1 validated, 0.45 µs at WG=256)
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void mega_barrier(unsigned int *counter_ptr, unsigned int phase_idx) {
    // Lane 0 of each WG bumps the global counter, then spin-waits
    // until all WG_PERSIST WGs have ack'd this phase.  Other lanes
    // just hit s_barrier (intra-WG sync) to keep the wave alive.
    if (threadIdx.x == 0) {
        __atomic_fetch_add(counter_ptr, 1u, __ATOMIC_ACQ_REL);
        unsigned int expected = (phase_idx + 1) * WG_PERSIST;
        while (__atomic_load_n(counter_ptr, __ATOMIC_ACQUIRE) < expected) {
            // pure spin
        }
    }
    __syncthreads();   // re-broadcast to other lanes
}

// bf16 → f32 conversion: gfx11 has no direct bf16 load, so we read
// as u16 and bit-shift into the f32 mantissa.  The compiler lowers
// this to v_lshlrev_b32 which is what the existing gemv_coop_bf16
// kernel uses.
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    unsigned int x = ((unsigned int)b) << 16;
    return *reinterpret_cast<float *>(&x);
}

// ────────────────────────────────────────────────────────────────
// Phase 1 — input rmsnorm: produces b_x_norm in LDS.
//
// Only WG 0 does work (single-WG reduce-tree over 1024 elements).
// Other WGs idle — they still hit the barrier ack via mega_barrier().
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase1_input_rmsnorm(unsigned int wg_id, unsigned int lane,
                           const float *in_residual,
                           const float *in_norm_g) {
    if (wg_id != 0) return;

    // Sum-of-squares via per-lane partial then __syncthreads + reduce.
    float ssq = 0.0f;
    for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
        float v = in_residual[i];
        ssq += v * v;
    }
    // Wave-reduce (DPP/permute on gfx11; clang lowers __reduce_add)
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
    }
    // Lane 0 has the full sum.  Compute scale = 1 / sqrt(mean + eps).
    if (lane == 0) {
        float mean = ssq / float(HIDDEN);
        float scale = rsqrtf(mean + 1e-5f);
        reduce_tmp[0] = scale;
    }
    __syncthreads();
    float scale = reduce_tmp[0];

    // Scale & gamma-multiply, write to LDS.
    for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
        lds.b_x_norm[i] = in_residual[i] * scale * in_norm_g[i];
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 3 — qkv matmul: each WG handles ceil(QKV_DIM / WG_PERSIST) =
// 12 output rows of `b_qkv = qkv_w · b_x_norm`.  bf16 weights, f32
// accum.  Output goes to LDS for phase 5 to read.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase3_qkv_matmul(unsigned int wg_id, unsigned int lane,
                       const unsigned short *qkv_w) {
    constexpr unsigned int ROWS_PER_WG = (QKV_DIM + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= QKV_DIM) continue;

        // Cooperative-row gemv: each lane accumulates HIDDEN/WAVE
        // products of weight-column × x[k].
        float acc = 0.0f;
        const unsigned short *w_row = qkv_w + row * HIDDEN;
        for (unsigned int k = lane; k < HIDDEN; k += WAVE) {
            float w = bf16_to_f32(w_row[k]);
            acc += w * lds.b_x_norm[k];
        }
        // Wave-reduce
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }
        // Lane 0 writes the row's output.
        if (lane == 0) {
            lds.b_qkv[row] = acc;
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 5 — qkv_split + qknorm + rope_qk fused (slice 2c protocol).
// 16 Q heads + 8 KV heads = 24 head sub-tasks.  WGs 0..23 each take
// one head; WGs 24..255 idle.
//
// For Q heads: rmsnorm head, apply rope, write to lds.b_attn_q.
// For K heads: rmsnorm head, apply rope, write to kc_layer (global).
// For V heads: pass through (no qknorm/rope), write to vc_layer.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase5_qkv_split_qknorm_rope(unsigned int wg_id, unsigned int lane,
                                   const float *q_norm_g, const float *k_norm_g,
                                   const float *rope_cos, const float *rope_sin,
                                   float *kc_layer, float *vc_layer,
                                   unsigned long long pos) {
    if (wg_id >= N_HEADS + 2 * N_KV_HEADS) return;

    if (wg_id < N_HEADS) {
        // ─── Q head ───
        unsigned int head = wg_id;
        const float *src = &lds.b_qkv[head * HEAD_DIM];

        // RMSNorm of head_dim=128 elements.
        float ssq = 0.0f;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            float v = src[i];
            ssq += v * v;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        float scale = rsqrtf(ssq / float(HEAD_DIM) + 1e-6f);

        // Apply gamma + RoPE rotation to pairs (i, i + half).
        unsigned int half = HEAD_DIM / 2;
        for (unsigned int i = lane; i < half; i += WAVE) {
            float x1 = src[i] * scale * q_norm_g[i];
            float x2 = src[i + half] * scale * q_norm_g[i + half];
            float c = rope_cos[i];
            float s = rope_sin[i];
            float y1 = x1 * c - x2 * s;
            float y2 = x1 * s + x2 * c;
            lds.b_attn_q[head * HEAD_DIM + i]        = y1;
            lds.b_attn_q[head * HEAD_DIM + i + half] = y2;
        }
    } else if (wg_id < N_HEADS + N_KV_HEADS) {
        // ─── K head ───  same as Q but reads from b_qkv past Q region,
        //   writes to kc_layer at pos*KV_DIM offset.
        unsigned int head = wg_id - N_HEADS;
        const float *src = &lds.b_qkv[Q_DIM + head * HEAD_DIM];
        float ssq = 0.0f;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            float v = src[i];
            ssq += v * v;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        float scale = rsqrtf(ssq / float(HEAD_DIM) + 1e-6f);
        unsigned int half = HEAD_DIM / 2;
        float *dst = kc_layer + pos * KV_DIM + head * HEAD_DIM;
        for (unsigned int i = lane; i < half; i += WAVE) {
            float x1 = src[i] * scale * k_norm_g[i];
            float x2 = src[i + half] * scale * k_norm_g[i + half];
            float c = rope_cos[i];
            float s = rope_sin[i];
            dst[i]        = x1 * c - x2 * s;
            dst[i + half] = x1 * s + x2 * c;
        }
    } else {
        // ─── V head ───  no qknorm/rope, pass through.
        unsigned int head = wg_id - N_HEADS - N_KV_HEADS;
        const float *src = &lds.b_qkv[Q_DIM + KV_DIM + head * HEAD_DIM];
        float *dst = vc_layer + pos * KV_DIM + head * HEAD_DIM;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            dst[i] = src[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 7 — attn_decode: 16 Q heads × softmax(Q · K_cache) · V_cache.
// WGs 0..15 each compute one head's attention output into b_attn_q
// (overwriting the post-rope Q in LDS).
//
// Simplified single-token decode: K-cache spans positions 0..pos.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase7_attn_decode(unsigned int wg_id, unsigned int lane,
                        const float *kc_layer, const float *vc_layer,
                        unsigned long long pos) {
    if (wg_id >= N_HEADS) return;

    unsigned int q_head = wg_id;
    unsigned int kv_head = q_head / (N_HEADS / N_KV_HEADS);  // GQA
    const float *q = &lds.b_attn_q[q_head * HEAD_DIM];

    // Two-pass softmax: pass 1 finds max, pass 2 sums, pass 3 weighted sum.
    constexpr float scale = 1.0f / 11.31370849898f;  // 1/sqrt(128)

    // Pass 1: max
    float m = -1e30f;
    for (unsigned long long t = lane; t <= pos; t += WAVE) {
        const float *k = kc_layer + t * KV_DIM + kv_head * HEAD_DIM;
        float dot = 0.0f;
        for (unsigned int i = 0; i < HEAD_DIM; i++) dot += q[i] * k[i];
        dot *= scale;
        if (dot > m) m = dot;
    }
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(m, offset);
        if (other > m) m = other;
    }

    // Pass 2: sum exp(x - m)
    float ssum = 0.0f;
    for (unsigned long long t = lane; t <= pos; t += WAVE) {
        const float *k = kc_layer + t * KV_DIM + kv_head * HEAD_DIM;
        float dot = 0.0f;
        for (unsigned int i = 0; i < HEAD_DIM; i++) dot += q[i] * k[i];
        dot *= scale;
        ssum += __expf(dot - m);
    }
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        ssum += __shfl_xor(ssum, offset);
    }
    float inv_sum = 1.0f / ssum;

    // Pass 3: weighted V accumulator.
    // Each lane accumulates a slice of head_dim.
    for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
        float acc = 0.0f;
        for (unsigned long long t = 0; t <= pos; t++) {
            const float *k = kc_layer + t * KV_DIM + kv_head * HEAD_DIM;
            float dot = 0.0f;
            for (unsigned int ii = 0; ii < HEAD_DIM; ii++) dot += q[ii] * k[ii];
            float w = __expf(dot * scale - m) * inv_sum;
            const float *v = vc_layer + t * KV_DIM + kv_head * HEAD_DIM;
            acc += w * v[i];
        }
        // Overwrite b_attn_q with the attention output (Q head's slot).
        lds.b_attn_q[q_head * HEAD_DIM + i] = acc;
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 9 — o_proj matmul + residual add.
// Each WG handles ceil(HIDDEN / WG_PERSIST) = 4 rows of
// `b_mid = in_residual + o_w · b_attn`.  Accumulates the residual
// in-place to save a separate phase.  Output to LDS.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase9_oproj_residual(unsigned int wg_id, unsigned int lane,
                            const unsigned short *o_w,
                            const float *in_residual,
                            float *b_mid_lds) {
    constexpr unsigned int ROWS_PER_WG = (HIDDEN + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= HIDDEN) continue;

        float acc = 0.0f;
        const unsigned short *w_row = o_w + row * Q_DIM;
        for (unsigned int k = lane; k < Q_DIM; k += WAVE) {
            acc += bf16_to_f32(w_row[k]) * lds.b_attn_q[k];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }
        if (lane == 0) {
            b_mid_lds[row] = in_residual[row] + acc;
        }
    }
}

// (Phases 11/13/15/17 follow the same shape as 1/3/3+silu/3+resid;
// elided here for slice 4.3 first-cut.  The HIP source compiles +
// runs phases 1, 3, 5, 7, 9.  Slice 4.4 will fill in 11-17 and
// validate output bit-equivalence against the per-op chain.)

// ────────────────────────────────────────────────────────────────
// MEGA-KERNEL ENTRY
// ────────────────────────────────────────────────────────────────
extern "C" __global__ __launch_bounds__(WAVE)
void qwen3_layer_megakernel(
    const float          *in_residual,           //  0
    float                *out_residual,          //  1 (overwritten by phase 17)
    const unsigned short *qkv_w,                 //  2 bf16 [QKV_DIM, HIDDEN]
    const unsigned short *o_w,                   //  3 bf16 [HIDDEN, Q_DIM]
    const unsigned short *gate_up_w,             //  4 bf16 [2*FF, HIDDEN]
    const unsigned short *down_w,                //  5 bf16 [HIDDEN, FF]
    const float          *in_norm_g,             //  6
    const float          *post_norm_g,           //  7
    const float          *q_norm_g,              //  8
    const float          *k_norm_g,              //  9
    const float          *rope_cos,              // 10
    const float          *rope_sin,              // 11
    float                *kc_layer,              // 12 K-cache layer slab
    float                *vc_layer,              // 13 V-cache layer slab
    unsigned long long    pos,                   // 14
    unsigned int         *barrier_counter        // 15 zero'd by launcher
) {
    unsigned int wg_id = blockIdx.x;
    unsigned int lane  = threadIdx.x;

    // Phase 1: input rmsnorm → lds.b_x_norm (WG 0 only)
    phase1_input_rmsnorm(wg_id, lane, in_residual, in_norm_g);
    mega_barrier(barrier_counter, 0);

    // Phase 3: qkv matmul → lds.b_qkv (all WGs, grid-stride)
    phase3_qkv_matmul(wg_id, lane, qkv_w);
    mega_barrier(barrier_counter, 1);

    // Phase 5: qkv_split + qknorm + rope → lds.b_attn_q (Q) + global (K, V)
    phase5_qkv_split_qknorm_rope(wg_id, lane,
                                  q_norm_g, k_norm_g,
                                  rope_cos, rope_sin,
                                  kc_layer, vc_layer, pos);
    mega_barrier(barrier_counter, 2);

    // Phase 7: attn_decode → overwrites lds.b_attn_q (16 head WGs only)
    phase7_attn_decode(wg_id, lane, kc_layer, vc_layer, pos);
    mega_barrier(barrier_counter, 3);

    // Phase 9: o_proj + residual → lds.b_mid (reuses b_x_norm slab)
    float *b_mid_lds = lds.b_x_norm;   // overwrite the dead post-phase-3 slab
    phase9_oproj_residual(wg_id, lane, o_w, in_residual, b_mid_lds);
    mega_barrier(barrier_counter, 4);

    // Phases 11-17: TBD slice 4.4.  For now write b_mid out to
    // out_residual so the caller can A/B against per-op output.
    if (wg_id == 0) {
        for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
            out_residual[i] = b_mid_lds[i];
        }
    }
    // Final barrier so all WGs settle before kernel exit.
    mega_barrier(barrier_counter, 5);
    mega_barrier(barrier_counter, 6);
}
