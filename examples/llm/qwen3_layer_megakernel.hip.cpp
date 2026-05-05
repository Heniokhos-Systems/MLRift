// Slice 4.6 — qwen3-0.6B layer mega-kernel (HIP source).
//
// Collapses 11 per-op dispatches per layer into ONE dispatch with
// cross-WG barriers between phases.  Per-token launches: 28 layers
// × 1 dispatch + 1 lm_head = 29 (vs 310 today after slice 2b+2c).
// Saves ~6.5 ms launch overhead / token at the measured ~24 µs/launch
// rate.  Projected step time: 9.3 ms = 107 tok/s (vs 15.8 ms = 63.2
// tok/s baseline).
//
// Slice 4.6 design correction (from slice 4.5 bisection 2026-05-05):
// HIP `__shared__` is PER-WG, not cross-WG.  Earlier drafts assumed
// cross-WG LDS sharing for inter-phase carries (b_x_norm, b_qkv,
// b_attn_q, etc.); reading uninitialized LDS in WGs that didn't
// participate in the writing phase wedged the GPU.  This revision
// threads ALL inter-phase carries through GLOBAL scratch slabs as
// kernargs.  LDS is now reserved exclusively for intra-phase
// reductions (a single ~64 B `wave_tmp[WG_PERSIST]` slot for
// rmsnorm partials and softmax broadcasts).
//
// Side benefit: with LDS shrunk from 29 KB to ~256 B per WG,
// occupancy returns to gfx1100's max (60 × 16 = 960 wave32s).  We
// run at WG_PERSIST=256, the slice-4.1 microbench sweet spot
// (0.45 µs / barrier).
//
// Build:
//   hipcc --offload-arch=gfx1100 --genco -O3 \
//       examples/llm/qwen3_layer_megakernel.hip.cpp \
//       -o /tmp/qwen3_layer_megakernel.co

#include <hip/hip_runtime.h>

// Qwen3-0.6B shape constants.  Hardcoded so the compiler constant-
// folds the tile loops.  Future Qwen3-14B variant is a separate
// _14b kernel emit (same structure, different shapes).
#define HIDDEN     1024
#define HEAD_DIM    128
#define N_HEADS      16   // Q heads
#define N_KV_HEADS    8
#define Q_DIM      (N_HEADS * HEAD_DIM)        // 2048
#define KV_DIM     (N_KV_HEADS * HEAD_DIM)     // 1024
#define QKV_DIM    (Q_DIM + 2 * KV_DIM)        // 4096
#define FF         3072    // intermediate
#define WG_PERSIST  64
#define WAVE       32

// LDS — only intra-phase reductions.  Inter-phase carries go through
// global scratch (see kernel signature below).
__shared__ float wave_tmp[1];   // single broadcast slot for rmsnorm scale

// ────────────────────────────────────────────────────────────────
// Cross-WG barrier protocol (slice 4.1 validated, 0.45 µs at WG=256)
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void mega_barrier(unsigned int *counter_ptr, unsigned int phase_idx) {
    // Slice 4.6.1 fix: ACQ_REL on the counter atomic only orders accesses
    // to the counter itself, not to OTHER global memory (the inter-phase
    // carry slabs).  Without an explicit threadfence around the barrier:
    //   - Writer WG's stores to b_x_norm_g may sit in L0/L1 cache when
    //     the barrier is hit, never flushed to L2.  Other WGs reading
    //     b_x_norm_g hit their own L0/L1 (cold for those addresses) → L2
    //     → stale (uninitialized) data.
    //   - Reader WG's L0/L1 may hold pre-barrier values for the carry
    //     slabs; the ACQUIRE atomic on counter doesn't invalidate them.
    // __threadfence() forces a buffer_gl0_inv + buffer_gl1_inv that
    // bypasses both caches, making cross-WG global memory coherent.
    __threadfence();
    if (threadIdx.x == 0) {
        __atomic_fetch_add(counter_ptr, 1u, __ATOMIC_ACQ_REL);
        unsigned int expected = (phase_idx + 1) * WG_PERSIST;
        while (__atomic_load_n(counter_ptr, __ATOMIC_ACQUIRE) < expected) {}
    }
    __syncthreads();
    __threadfence();
}

__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    unsigned int x = ((unsigned int)b) << 16;
    return *reinterpret_cast<float *>(&x);
}

// ────────────────────────────────────────────────────────────────
// Phase 1 — input rmsnorm.  WG 0 only.  Output: b_x_norm_g (global).
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase1_input_rmsnorm(unsigned int wg_id, unsigned int lane,
                           const float *in_residual,
                           const float *in_norm_g,
                           float *b_x_norm_g) {
    if (wg_id != 0) return;
    float ssq = 0.0f;
    for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
        float v = in_residual[i];
        ssq += v * v;
    }
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
    }
    if (lane == 0) {
        wave_tmp[0] = rsqrtf(ssq / float(HIDDEN) + 1e-5f);
    }
    __syncthreads();
    float scale = wave_tmp[0];
    for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
        b_x_norm_g[i] = in_residual[i] * scale * in_norm_g[i];
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 3 — qkv matmul.  Each WG handles ceil(QKV_DIM/256) = 12
// rows.  Reads b_x_norm_g (global), writes b_qkv_g (global).
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase3_qkv_matmul(unsigned int wg_id, unsigned int lane,
                       const unsigned short *qkv_w,
                       const float *b_x_norm_g, float *b_qkv_g) {
    constexpr unsigned int ROWS_PER_WG = (QKV_DIM + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= QKV_DIM) continue;
        float acc = 0.0f;
        const unsigned short *w_row = qkv_w + row * HIDDEN;
        for (unsigned int k = lane; k < HIDDEN; k += WAVE) {
            acc += bf16_to_f32(w_row[k]) * b_x_norm_g[k];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }
        if (lane == 0) b_qkv_g[row] = acc;
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 5 — qkv_split + qknorm + rope_qk fused.
// 16 Q heads + 8 KV heads + 8 V heads = 32 head sub-tasks.
// WGs 0..31 each take one head; rest idle.
// Reads b_qkv_g (global), writes b_attn_q_g (global) for Q,
// kc_layer/vc_layer (global) for K/V.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase5_qkv_split_qknorm_rope(unsigned int wg_id, unsigned int lane,
                                   const float *q_norm_g, const float *k_norm_g,
                                   const float *rope_cos, const float *rope_sin,
                                   const float *b_qkv_g, float *b_attn_q_g,
                                   float *kc_layer, float *vc_layer,
                                   unsigned long long pos) {
    if (wg_id >= N_HEADS + 2 * N_KV_HEADS) return;

    if (wg_id < N_HEADS) {
        // Q head: rmsnorm + RoPE rotation, output to b_attn_q_g.
        unsigned int head = wg_id;
        const float *src = b_qkv_g + head * HEAD_DIM;
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
        for (unsigned int i = lane; i < half; i += WAVE) {
            float x1 = src[i] * scale * q_norm_g[i];
            float x2 = src[i + half] * scale * q_norm_g[i + half];
            float c = rope_cos[i];
            float s = rope_sin[i];
            b_attn_q_g[head * HEAD_DIM + i]        = x1 * c - x2 * s;
            b_attn_q_g[head * HEAD_DIM + i + half] = x1 * s + x2 * c;
        }
    } else if (wg_id < N_HEADS + N_KV_HEADS) {
        // K head: rmsnorm + RoPE, output to kc_layer.
        unsigned int head = wg_id - N_HEADS;
        const float *src = b_qkv_g + Q_DIM + head * HEAD_DIM;
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
        // V head: pass-through (no qknorm/rope).
        unsigned int head = wg_id - N_HEADS - N_KV_HEADS;
        const float *src = b_qkv_g + Q_DIM + KV_DIM + head * HEAD_DIM;
        float *dst = vc_layer + pos * KV_DIM + head * HEAD_DIM;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            dst[i] = src[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 7 — attn_decode.  WGs 0..15 each compute one Q head's attn.
// Reads Q from b_attn_q_g, K/V from caches, writes attn output to a
// SEPARATE buffer b_attn_out_g.
//
// Slice 4.7 fix: an earlier version wrote pass-3 output back into
// b_attn_q_g, overwriting the Q input.  But pass 3 strides outer-i
// by WAVE — iteration 2 (lanes write i=32..63) re-reads q[0..127]
// for each dot product, including q[0..31] which iteration 1 had
// already overwritten.  Result: garbage in lane-{0..31}'s outputs at
// i ≥ 32.  Bisection at first_bad_idx=32 was the smoking gun.
// Separate output slab avoids the read-after-write hazard.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase7_attn_decode(unsigned int wg_id, unsigned int lane,
                        const float *b_attn_q_g,
                        const float *kc_layer, const float *vc_layer,
                        float *b_attn_out_g,
                        unsigned long long pos) {
    if (wg_id >= N_HEADS) return;
    unsigned int q_head = wg_id;
    unsigned int kv_head = q_head / (N_HEADS / N_KV_HEADS);
    const float *q = b_attn_q_g + q_head * HEAD_DIM;
    constexpr float scale_attn = 1.0f / 11.31370849898f;   // 1/sqrt(128)

    // Pass 1: max
    float m = -1e30f;
    for (unsigned long long t = lane; t <= pos; t += WAVE) {
        const float *k = kc_layer + t * KV_DIM + kv_head * HEAD_DIM;
        float dot = 0.0f;
        for (unsigned int i = 0; i < HEAD_DIM; i++) dot += q[i] * k[i];
        dot *= scale_attn;
        if (dot > m) m = dot;
    }
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(m, offset);
        if (other > m) m = other;
    }

    // Pass 2: sum exp
    float ssum = 0.0f;
    for (unsigned long long t = lane; t <= pos; t += WAVE) {
        const float *k = kc_layer + t * KV_DIM + kv_head * HEAD_DIM;
        float dot = 0.0f;
        for (unsigned int i = 0; i < HEAD_DIM; i++) dot += q[i] * k[i];
        dot *= scale_attn;
        ssum += __expf(dot - m);
    }
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        ssum += __shfl_xor(ssum, offset);
    }
    float inv_sum = 1.0f / ssum;

    // Pass 3: weighted V.  Write to SEPARATE output slab.
    for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
        float acc = 0.0f;
        for (unsigned long long t = 0; t <= pos; t++) {
            const float *k = kc_layer + t * KV_DIM + kv_head * HEAD_DIM;
            float dot = 0.0f;
            for (unsigned int ii = 0; ii < HEAD_DIM; ii++) dot += q[ii] * k[ii];
            float w = __expf(dot * scale_attn - m) * inv_sum;
            const float *v = vc_layer + t * KV_DIM + kv_head * HEAD_DIM;
            acc += w * v[i];
        }
        b_attn_out_g[q_head * HEAD_DIM + i] = acc;
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 9 — o_proj matmul + residual.  Each WG handles ceil(HIDDEN/256)
// = 4 rows.  Reads b_attn_out_g (phase 7's output, NOT the rope-Q
// input) + in_residual, writes b_mid_g.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase9_oproj_residual(unsigned int wg_id, unsigned int lane,
                            const unsigned short *o_w,
                            const float *b_attn_out_g,
                            const float *in_residual,
                            float *b_mid_g) {
    constexpr unsigned int ROWS_PER_WG = (HIDDEN + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= HIDDEN) continue;
        float acc = 0.0f;
        const unsigned short *w_row = o_w + row * Q_DIM;
        for (unsigned int k = lane; k < Q_DIM; k += WAVE) {
            acc += bf16_to_f32(w_row[k]) * b_attn_out_g[k];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }
        if (lane == 0) b_mid_g[row] = in_residual[row] + acc;
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 11 — post_norm.  WG 0 only.  Reads b_mid_g, writes b_mid_norm_g.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase11_post_norm(unsigned int wg_id, unsigned int lane,
                       const float *post_norm_g,
                       const float *b_mid_g, float *b_mid_norm_g) {
    if (wg_id != 0) return;
    float ssq = 0.0f;
    for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
        float v = b_mid_g[i];
        ssq += v * v;
    }
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
    }
    if (lane == 0) {
        wave_tmp[0] = rsqrtf(ssq / float(HIDDEN) + 1e-5f);
    }
    __syncthreads();
    float scale = wave_tmp[0];
    for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
        b_mid_norm_g[i] = b_mid_g[i] * scale * post_norm_g[i];
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 13 — gate_up matmul.  Each WG handles ceil(2*FF/256) = 24
// rows.  Reads b_mid_norm_g, writes gu_scratch.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase13_gate_up_matmul(unsigned int wg_id, unsigned int lane,
                             const unsigned short *gate_up_w,
                             const float *b_mid_norm_g,
                             float *gu_scratch) {
    constexpr unsigned int N_OUT = 2 * FF;
    constexpr unsigned int ROWS_PER_WG = (N_OUT + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= N_OUT) continue;
        float acc = 0.0f;
        const unsigned short *w_row = gate_up_w + row * HIDDEN;
        for (unsigned int k = lane; k < HIDDEN; k += WAVE) {
            acc += bf16_to_f32(w_row[k]) * b_mid_norm_g[k];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }
        if (lane == 0) gu_scratch[row] = acc;
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 15 — silu_mul: gu_scratch[0..FF] = silu(gate) * up.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase15_silu_mul(unsigned int wg_id, unsigned int lane,
                       float *gu_scratch) {
    constexpr unsigned int ELEMS_PER_WG = (FF + WG_PERSIST * WAVE - 1)
                                          / (WG_PERSIST * WAVE);
    for (unsigned int e_off = 0; e_off < ELEMS_PER_WG; e_off++) {
        unsigned int idx = (wg_id * WAVE + lane) + e_off * WG_PERSIST * WAVE;
        if (idx >= FF) continue;
        float g = gu_scratch[idx];
        float u = gu_scratch[FF + idx];
        float s = g / (1.0f + __expf(-g));
        gu_scratch[idx] = s * u;
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 17 — down + final residual.  Each WG handles 4 rows.
// Reads gu_scratch (b_ff) + b_mid_g, writes out_residual.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase17_down_residual(unsigned int wg_id, unsigned int lane,
                            const unsigned short *down_w,
                            const float *gu_scratch,
                            const float *b_mid_g,
                            float *out_residual) {
    constexpr unsigned int ROWS_PER_WG = (HIDDEN + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= HIDDEN) continue;
        float acc = 0.0f;
        const unsigned short *w_row = down_w + row * FF;
        for (unsigned int k = lane; k < FF; k += WAVE) {
            acc += bf16_to_f32(w_row[k]) * gu_scratch[k];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }
        if (lane == 0) out_residual[row] = b_mid_g[row] + acc;
    }
}

// ────────────────────────────────────────────────────────────────
// MEGA-KERNEL ENTRY (slice 4.6 — 22 kernargs)
// ────────────────────────────────────────────────────────────────
// Layout: 17 from slice 4.4 + 5 NEW global scratch slabs for the
// inter-phase carries that previously lived in (per-WG broken) LDS.
extern "C" __global__ __launch_bounds__(WAVE)
void qwen3_layer_megakernel(
    const float          *in_residual,           //  0
    float                *out_residual,          //  1
    const unsigned short *qkv_w,                 //  2
    const unsigned short *o_w,                   //  3
    const unsigned short *gate_up_w,             //  4
    const unsigned short *down_w,                //  5
    const float          *in_norm_g,             //  6
    const float          *post_norm_g,           //  7
    const float          *q_norm_g,              //  8
    const float          *k_norm_g,              //  9
    const float          *rope_cos,              // 10
    const float          *rope_sin,              // 11
    float                *kc_layer,              // 12
    float                *vc_layer,              // 13
    unsigned long long    pos,                   // 14
    unsigned int         *barrier_counter,       // 15
    float                *gu_scratch,            // 16  [2*FF]  — phase 13 → 15 → 17
    float                *b_x_norm_g,            // 17  [HIDDEN] — phase 1 → 3   (NEW)
    float                *b_qkv_g,               // 18  [QKV_DIM] — phase 3 → 5  (NEW)
    float                *b_attn_q_g,            // 19  [Q_DIM]   — phase 5 → 7 → 9 (NEW)
    float                *b_mid_g,               // 20  [HIDDEN] — phase 9 → 11 → 17 (NEW)
    float                *b_mid_norm_g,          // 21  [HIDDEN] — phase 11 → 13 (NEW)
    float                *b_attn_out_g,          // 22  [Q_DIM] — phase 7 → 9 (slice 4.7 fix:
                                                  //              separate from b_attn_q_g to avoid
                                                  //              read-after-write hazard in pass 3)
    unsigned long long    max_phase              // 23  bisection: kernel exits after phase max_phase
) {
    unsigned int wg_id = blockIdx.x;
    unsigned int lane  = threadIdx.x;

    phase1_input_rmsnorm(wg_id, lane, in_residual, in_norm_g, b_x_norm_g);
    mega_barrier(barrier_counter, 0);
    if (max_phase < 1) return;

    phase3_qkv_matmul(wg_id, lane, qkv_w, b_x_norm_g, b_qkv_g);
    mega_barrier(barrier_counter, 1);
    if (max_phase < 2) return;

    phase5_qkv_split_qknorm_rope(wg_id, lane, q_norm_g, k_norm_g,
                                  rope_cos, rope_sin,
                                  b_qkv_g, b_attn_q_g,
                                  kc_layer, vc_layer, pos);
    mega_barrier(barrier_counter, 2);
    if (max_phase < 3) return;

    phase7_attn_decode(wg_id, lane, b_attn_q_g, kc_layer, vc_layer,
                       b_attn_out_g, pos);
    mega_barrier(barrier_counter, 3);
    if (max_phase < 4) return;

    phase9_oproj_residual(wg_id, lane, o_w, b_attn_out_g, in_residual, b_mid_g);
    mega_barrier(barrier_counter, 4);
    if (max_phase < 5) return;

    phase11_post_norm(wg_id, lane, post_norm_g, b_mid_g, b_mid_norm_g);
    mega_barrier(barrier_counter, 5);
    if (max_phase < 6) return;

    phase13_gate_up_matmul(wg_id, lane, gate_up_w, b_mid_norm_g, gu_scratch);
    mega_barrier(barrier_counter, 6);
    if (max_phase < 7) return;

    phase15_silu_mul(wg_id, lane, gu_scratch);
    mega_barrier(barrier_counter, 7);

    phase17_down_residual(wg_id, lane, down_w, gu_scratch, b_mid_g, out_residual);
}
