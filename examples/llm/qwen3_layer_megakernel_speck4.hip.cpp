// Slice 4.14 (PLD speck4) — qwen3-0.6B layer mega-kernel for M=4 queries.
//
// Variant of qwen3_layer_megakernel.hip.cpp (slice 4.13) that processes
// M_EFF=4 query tokens per dispatch.  The per-token launch count drops
// from 4× per layer to 1× per layer at spec_K=4 — when paired with PLD
// (~2 tok accepted / step) we expect ~2× the baseline mega-kernel rate.
//
// Per-phase strategy:
//   - Bandwidth-bound matmuls (phases 3, 9, 13, 17): each WG reads one
//     weight row and computes 4 dot-products against 4 input vectors.
//     Same weight bandwidth as M=1 → 4× the output.
//   - Per-stream phases (1, 5, 7, 11, 15): inner loop over m=0..3.
//     RoPE / KV-cache writes use pos_base + m.  Phase 7 attn_decode
//     reads K/V at [0..pos_base+m] (causal mask increments with m).
//
// Build:
//   hipcc --offload-arch=gfx1100 --genco -O3 \
//       examples/llm/qwen3_layer_megakernel_speck4.hip.cpp \
//       -o /tmp/qwen3_layer_megakernel_speck4.co

#include <hip/hip_runtime.h>

// Shape constants — same as the M=1 kernel.
#define HIDDEN     1024
#define HEAD_DIM    128
#define N_HEADS      16
#define N_KV_HEADS    8
#define Q_DIM      (N_HEADS * HEAD_DIM)        // 2048
#define KV_DIM     (N_KV_HEADS * HEAD_DIM)     // 1024
#define QKV_DIM    (Q_DIM + 2 * KV_DIM)        // 4096
#define FF         3072
#define WG_PERSIST  512
#define WAVE       32
#define M_EFF        4

// Padded weight strides (slice 4.13 channel-repack).
#define HIDDEN_PAD  1152
#define Q_DIM_PAD   2176
#define FF_PAD      3200

// LDS — small.  attn_w_lds now holds M_EFF×64 weights.
__shared__ float wave_tmp[1];
__shared__ float attn_w_lds[M_EFF * 128];

__device__ __forceinline__
void mega_barrier(unsigned int *counter_ptr, unsigned int phase_idx) {
    __threadfence();
    if (threadIdx.x == 0) {
        __atomic_fetch_add(counter_ptr, 1u, __ATOMIC_ACQ_REL);
        unsigned int expected = (phase_idx + 1) * WG_PERSIST;
        while (__atomic_load_n(counter_ptr, __ATOMIC_ACQUIRE) < expected) {}
    }
    __syncthreads();
}

__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    unsigned int x = ((unsigned int)b) << 16;
    return *reinterpret_cast<float *>(&x);
}

// ────────────────────────────────────────────────────────────────
// Phase 1 — input rmsnorm.  WG 0 only; loops over m=0..M_EFF-1.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase1_input_rmsnorm_speck4(unsigned int wg_id, unsigned int lane,
                                  const float *in_residual_4,
                                  const float *in_norm_g,
                                  float *b_x_norm_g_4) {
    if (wg_id != 0) return;
    for (int m = 0; m < M_EFF; m++) {
        const float *in_m = in_residual_4 + m * HIDDEN;
        float *out_m = b_x_norm_g_4 + m * HIDDEN;
        float ssq = 0.0f;
        for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
            float v = in_m[i];
            ssq += v * v;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        if (lane == 0) wave_tmp[0] = rsqrtf(ssq / float(HIDDEN) + 1e-5f);
        __syncthreads();
        float scale = wave_tmp[0];
        for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
            out_m[i] = in_m[i] * scale * in_norm_g[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 3 — qkv matmul.  Each WG reads weight row ONCE and computes
// 4 dot products (one per query token).
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase3_qkv_matmul_speck4(unsigned int wg_id, unsigned int lane,
                               const unsigned short *qkv_w,
                               const float *b_x_norm_g_4,
                               float *b_qkv_g_4) {
    constexpr unsigned int ROWS_PER_WG = (QKV_DIM + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= QKV_DIM) continue;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        const unsigned int *w_row_u32 =
            reinterpret_cast<const unsigned int *>(qkv_w + row * HIDDEN_PAD);
        for (unsigned int kp = lane; kp < HIDDEN / 2; kp += WAVE) {
            unsigned int packed = w_row_u32[kp];
            unsigned int k = kp * 2;
            float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
            float w1 = bf16_to_f32((unsigned short)(packed >> 16));
            acc0 += w0 * b_x_norm_g_4[0 * HIDDEN + k] + w1 * b_x_norm_g_4[0 * HIDDEN + k + 1];
            acc1 += w0 * b_x_norm_g_4[1 * HIDDEN + k] + w1 * b_x_norm_g_4[1 * HIDDEN + k + 1];
            acc2 += w0 * b_x_norm_g_4[2 * HIDDEN + k] + w1 * b_x_norm_g_4[2 * HIDDEN + k + 1];
            acc3 += w0 * b_x_norm_g_4[3 * HIDDEN + k] + w1 * b_x_norm_g_4[3 * HIDDEN + k + 1];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc0 += __shfl_xor(acc0, offset);
            acc1 += __shfl_xor(acc1, offset);
            acc2 += __shfl_xor(acc2, offset);
            acc3 += __shfl_xor(acc3, offset);
        }
        if (lane == 0) {
            b_qkv_g_4[0 * QKV_DIM + row] = acc0;
            b_qkv_g_4[1 * QKV_DIM + row] = acc1;
            b_qkv_g_4[2 * QKV_DIM + row] = acc2;
            b_qkv_g_4[3 * QKV_DIM + row] = acc3;
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 5 — qkv_split + qknorm + rope_qk fused.
// One WG handles one (m, head_kind) pair: M_EFF * (16 Q + 8 K + 8 V) =
// 128 WGs.  Looping m inside a WG (the previous design) caused a per-stream
// numerical drift for m>=1 — root cause unconfirmed but reliably reproduced
// in the AB harness; flattening to one WG per (m, head) matches the M=1
// code path exactly and eliminates the drift.
//
// rope_cos/rope_sin layout: [M_EFF, HEAD_DIM/2] — when caller wants
// per-position RoPE — or [M_EFF, HEAD_DIM/2] with all 4 entries set to
// the same `cos(pos_base) / sin(pos_base)` table — when caller wants
// the per-op spec_K=4 single-pos behavior.  Either way the kernel reads
// rope_cos_4 + m * half, so the launcher controls the choice.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase5_qkv_split_qknorm_rope_speck4(unsigned int wg_id, unsigned int lane,
                                          const float *q_norm_g, const float *k_norm_g,
                                          const float *rope_cos_4, const float *rope_sin_4,
                                          const float *b_qkv_g_4, float *b_attn_q_g_4,
                                          float *kc_layer, float *vc_layer,
                                          unsigned long long pos_base) {
    constexpr unsigned int N_Q_TASKS  = M_EFF * N_HEADS;     // 64
    constexpr unsigned int N_K_TASKS  = M_EFF * N_KV_HEADS;  // 32
    constexpr unsigned int N_V_TASKS  = M_EFF * N_KV_HEADS;  // 32
    constexpr unsigned int N_TASKS    = N_Q_TASKS + N_K_TASKS + N_V_TASKS;  // 128
    if (wg_id >= N_TASKS) return;
    constexpr unsigned int half = HEAD_DIM / 2;

    if (wg_id < N_Q_TASKS) {
        unsigned int m    = wg_id / N_HEADS;
        unsigned int head = wg_id % N_HEADS;
        const float *src = b_qkv_g_4 + m * QKV_DIM + head * HEAD_DIM;
        float ssq = 0.0f;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            float v = src[i];
            ssq += v * v;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        float scale = rsqrtf(ssq / float(HEAD_DIM) + 1e-6f);
        const float *rc = rope_cos_4 + m * half;
        const float *rs = rope_sin_4 + m * half;
        float *qdst = b_attn_q_g_4 + m * Q_DIM + head * HEAD_DIM;
        for (unsigned int i = lane; i < half; i += WAVE) {
            float x1 = src[i] * scale * q_norm_g[i];
            float x2 = src[i + half] * scale * q_norm_g[i + half];
            float c = rc[i];
            float s = rs[i];
            qdst[i]        = x1 * c - x2 * s;
            qdst[i + half] = x1 * s + x2 * c;
        }
    } else if (wg_id < N_Q_TASKS + N_K_TASKS) {
        unsigned int t    = wg_id - N_Q_TASKS;
        unsigned int m    = t / N_KV_HEADS;
        unsigned int head = t % N_KV_HEADS;
        const float *src = b_qkv_g_4 + m * QKV_DIM + Q_DIM + head * HEAD_DIM;
        float ssq = 0.0f;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            float v = src[i];
            ssq += v * v;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        float scale = rsqrtf(ssq / float(HEAD_DIM) + 1e-6f);
        const float *rc = rope_cos_4 + m * half;
        const float *rs = rope_sin_4 + m * half;
        unsigned long long pos_m = pos_base + (unsigned long long)m;
        float *dst = kc_layer + pos_m * KV_DIM + head * HEAD_DIM;
        for (unsigned int i = lane; i < half; i += WAVE) {
            float x1 = src[i] * scale * k_norm_g[i];
            float x2 = src[i + half] * scale * k_norm_g[i + half];
            float c = rc[i];
            float s = rs[i];
            dst[i]        = x1 * c - x2 * s;
            dst[i + half] = x1 * s + x2 * c;
        }
    } else {
        unsigned int t    = wg_id - N_Q_TASKS - N_K_TASKS;
        unsigned int m    = t / N_KV_HEADS;
        unsigned int head = t % N_KV_HEADS;
        const float *src = b_qkv_g_4 + m * QKV_DIM + Q_DIM + KV_DIM + head * HEAD_DIM;
        unsigned long long pos_m = pos_base + (unsigned long long)m;
        float *dst = vc_layer + pos_m * KV_DIM + head * HEAD_DIM;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            dst[i] = src[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 7 — attn_decode for M_EFF query tokens.
// ATTN_COOP=4 → 64 active WGs (16 heads × 4 coop).  Each WG loops
// m=0..3, computing one query's attention.  Causal: query m attends
// positions [0..pos_base+m] inclusive.
//
// LDS attn_w_lds is sized M_EFF*64 = 256 floats.  But each m's softmax
// is independent — we can reuse the same 64-slot region by syncing
// between m's.  Use offset = m*64 anyway, simpler.
// ────────────────────────────────────────────────────────────────
#define ATTN_COOP 4
__device__ __forceinline__
void phase7_attn_decode_speck4(unsigned int wg_id, unsigned int lane,
                                const float *b_attn_q_g_4,
                                const float *kc_layer, const float *vc_layer,
                                float *b_attn_out_g_4,
                                unsigned long long pos_base) {
    if (wg_id >= N_HEADS * ATTN_COOP) return;
    unsigned int q_head   = wg_id / ATTN_COOP;
    unsigned int coop_idx = wg_id % ATTN_COOP;
    unsigned int kv_head  = q_head / (N_HEADS / N_KV_HEADS);
    constexpr float scale_attn = 1.0f / 11.31370849898f;
    constexpr unsigned int OUT_PER_WG = HEAD_DIM / ATTN_COOP;

    for (int m = 0; m < M_EFF; m++) {
        unsigned long long pos_m = pos_base + (unsigned long long)m;
        const float *q = b_attn_q_g_4 + m * Q_DIM + q_head * HEAD_DIM;
        float *w_lds = &attn_w_lds[m * 128];

        // Pass 1: max
        float max_v = -1e30f;
        for (unsigned long long t = lane; t <= pos_m; t += WAVE) {
            const float *k = kc_layer + t * KV_DIM + kv_head * HEAD_DIM;
            float dot = 0.0f;
            for (unsigned int i = 0; i < HEAD_DIM; i++) dot += q[i] * k[i];
            dot *= scale_attn;
            if (dot > max_v) max_v = dot;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            float other = __shfl_xor(max_v, offset);
            if (other > max_v) max_v = other;
        }

        // Pass 2: cache exp(dot-max), reduce sum
        float ssum = 0.0f;
        for (unsigned long long t = lane; t <= pos_m; t += WAVE) {
            const float *k = kc_layer + t * KV_DIM + kv_head * HEAD_DIM;
            float dot = 0.0f;
            for (unsigned int i = 0; i < HEAD_DIM; i++) dot += q[i] * k[i];
            float w = __expf(dot * scale_attn - max_v);
            w_lds[t] = w;
            ssum += w;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssum += __shfl_xor(ssum, offset);
        }
        float inv_sum = 1.0f / ssum;
        __syncthreads();

        // Pass 3 cooperative
        if (lane < OUT_PER_WG) {
            unsigned int i = coop_idx * OUT_PER_WG + lane;
            float acc = 0.0f;
            for (unsigned long long t = 0; t <= pos_m; t++) {
                float w = w_lds[t] * inv_sum;
                const float *v = vc_layer + t * KV_DIM + kv_head * HEAD_DIM;
                acc += w * v[i];
            }
            b_attn_out_g_4[m * Q_DIM + q_head * HEAD_DIM + i] = acc;
        }
        // Sync before next m so all coop WGs are done with w_lds[m*64..]
        __syncthreads();
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 9 — o_proj matmul + residual.  M_EFF amortized.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase9_oproj_residual_speck4(unsigned int wg_id, unsigned int lane,
                                   const unsigned short *o_w,
                                   const float *b_attn_out_g_4,
                                   const float *in_residual_4,
                                   float *b_mid_g_4) {
    constexpr unsigned int ROWS_PER_WG = (HIDDEN + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= HIDDEN) continue;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        const unsigned int *w_row_u32 =
            reinterpret_cast<const unsigned int *>(o_w + row * Q_DIM_PAD);
        for (unsigned int kp = lane; kp < Q_DIM / 2; kp += WAVE) {
            unsigned int packed = w_row_u32[kp];
            unsigned int k = kp * 2;
            float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
            float w1 = bf16_to_f32((unsigned short)(packed >> 16));
            acc0 += w0 * b_attn_out_g_4[0 * Q_DIM + k] + w1 * b_attn_out_g_4[0 * Q_DIM + k + 1];
            acc1 += w0 * b_attn_out_g_4[1 * Q_DIM + k] + w1 * b_attn_out_g_4[1 * Q_DIM + k + 1];
            acc2 += w0 * b_attn_out_g_4[2 * Q_DIM + k] + w1 * b_attn_out_g_4[2 * Q_DIM + k + 1];
            acc3 += w0 * b_attn_out_g_4[3 * Q_DIM + k] + w1 * b_attn_out_g_4[3 * Q_DIM + k + 1];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc0 += __shfl_xor(acc0, offset);
            acc1 += __shfl_xor(acc1, offset);
            acc2 += __shfl_xor(acc2, offset);
            acc3 += __shfl_xor(acc3, offset);
        }
        if (lane == 0) {
            b_mid_g_4[0 * HIDDEN + row] = in_residual_4[0 * HIDDEN + row] + acc0;
            b_mid_g_4[1 * HIDDEN + row] = in_residual_4[1 * HIDDEN + row] + acc1;
            b_mid_g_4[2 * HIDDEN + row] = in_residual_4[2 * HIDDEN + row] + acc2;
            b_mid_g_4[3 * HIDDEN + row] = in_residual_4[3 * HIDDEN + row] + acc3;
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 11 — post_norm.  WG 0 only; loops over m=0..3.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase11_post_norm_speck4(unsigned int wg_id, unsigned int lane,
                               const float *post_norm_g,
                               const float *b_mid_g_4, float *b_mid_norm_g_4) {
    if (wg_id != 0) return;
    for (int m = 0; m < M_EFF; m++) {
        const float *in_m = b_mid_g_4 + m * HIDDEN;
        float *out_m = b_mid_norm_g_4 + m * HIDDEN;
        float ssq = 0.0f;
        for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
            float v = in_m[i];
            ssq += v * v;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        if (lane == 0) wave_tmp[0] = rsqrtf(ssq / float(HIDDEN) + 1e-5f);
        __syncthreads();
        float scale = wave_tmp[0];
        for (unsigned int i = lane; i < HIDDEN; i += WAVE) {
            out_m[i] = in_m[i] * scale * post_norm_g[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 13 — gate_up matmul.  M_EFF amortized.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase13_gate_up_matmul_speck4(unsigned int wg_id, unsigned int lane,
                                    const unsigned short *gate_up_w,
                                    const float *b_mid_norm_g_4,
                                    float *gu_scratch_4) {
    constexpr unsigned int N_OUT = 2 * FF;
    constexpr unsigned int ROWS_PER_WG = (N_OUT + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= N_OUT) continue;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        const unsigned int *w_row_u32 =
            reinterpret_cast<const unsigned int *>(gate_up_w + row * HIDDEN_PAD);
        for (unsigned int kp = lane; kp < HIDDEN / 2; kp += WAVE) {
            unsigned int packed = w_row_u32[kp];
            unsigned int k = kp * 2;
            float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
            float w1 = bf16_to_f32((unsigned short)(packed >> 16));
            acc0 += w0 * b_mid_norm_g_4[0 * HIDDEN + k] + w1 * b_mid_norm_g_4[0 * HIDDEN + k + 1];
            acc1 += w0 * b_mid_norm_g_4[1 * HIDDEN + k] + w1 * b_mid_norm_g_4[1 * HIDDEN + k + 1];
            acc2 += w0 * b_mid_norm_g_4[2 * HIDDEN + k] + w1 * b_mid_norm_g_4[2 * HIDDEN + k + 1];
            acc3 += w0 * b_mid_norm_g_4[3 * HIDDEN + k] + w1 * b_mid_norm_g_4[3 * HIDDEN + k + 1];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc0 += __shfl_xor(acc0, offset);
            acc1 += __shfl_xor(acc1, offset);
            acc2 += __shfl_xor(acc2, offset);
            acc3 += __shfl_xor(acc3, offset);
        }
        if (lane == 0) {
            gu_scratch_4[0 * N_OUT + row] = acc0;
            gu_scratch_4[1 * N_OUT + row] = acc1;
            gu_scratch_4[2 * N_OUT + row] = acc2;
            gu_scratch_4[3 * N_OUT + row] = acc3;
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 15 — silu_mul: gu_scratch_4[m, 0..FF] = silu(gate) * up.
// Each m has its own [2*FF] block.  We OVERWRITE the gate half
// in-place (matching M=1 layout: phase 17 reads gu_scratch[0..FF]).
// To keep stride consistent, we write the result back to slot
// gu_scratch_4[m * 2*FF + 0..FF].
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase15_silu_mul_speck4(unsigned int wg_id, unsigned int lane,
                              float *gu_scratch_4) {
    constexpr unsigned int N_OUT = 2 * FF;
    constexpr unsigned int ELEMS_PER_WG = (FF + WG_PERSIST * WAVE - 1)
                                          / (WG_PERSIST * WAVE);
    for (int m = 0; m < M_EFF; m++) {
        float *gu_m = gu_scratch_4 + m * N_OUT;
        for (unsigned int e_off = 0; e_off < ELEMS_PER_WG; e_off++) {
            unsigned int idx = (wg_id * WAVE + lane) + e_off * WG_PERSIST * WAVE;
            if (idx >= FF) continue;
            float g = gu_m[idx];
            float u = gu_m[FF + idx];
            float s = g / (1.0f + __expf(-g));
            gu_m[idx] = s * u;
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 17 — down + final residual.  M_EFF amortized.
// gu_scratch stride per m is 2*FF (N_OUT) — phase 15 writes silu*up
// into [m, 0..FF].  We read FF elements per m.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase17_down_residual_speck4(unsigned int wg_id, unsigned int lane,
                                   const unsigned short *down_w,
                                   const float *gu_scratch_4,
                                   const float *b_mid_g_4,
                                   float *out_residual_4) {
    constexpr unsigned int N_OUT = 2 * FF;
    constexpr unsigned int ROWS_PER_WG = (HIDDEN + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= HIDDEN) continue;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        const unsigned int *w_row_u32 =
            reinterpret_cast<const unsigned int *>(down_w + row * FF_PAD);
        for (unsigned int kp = lane; kp < FF / 2; kp += WAVE) {
            unsigned int packed = w_row_u32[kp];
            unsigned int k = kp * 2;
            float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
            float w1 = bf16_to_f32((unsigned short)(packed >> 16));
            acc0 += w0 * gu_scratch_4[0 * N_OUT + k] + w1 * gu_scratch_4[0 * N_OUT + k + 1];
            acc1 += w0 * gu_scratch_4[1 * N_OUT + k] + w1 * gu_scratch_4[1 * N_OUT + k + 1];
            acc2 += w0 * gu_scratch_4[2 * N_OUT + k] + w1 * gu_scratch_4[2 * N_OUT + k + 1];
            acc3 += w0 * gu_scratch_4[3 * N_OUT + k] + w1 * gu_scratch_4[3 * N_OUT + k + 1];
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            acc0 += __shfl_xor(acc0, offset);
            acc1 += __shfl_xor(acc1, offset);
            acc2 += __shfl_xor(acc2, offset);
            acc3 += __shfl_xor(acc3, offset);
        }
        if (lane == 0) {
            out_residual_4[0 * HIDDEN + row] = b_mid_g_4[0 * HIDDEN + row] + acc0;
            out_residual_4[1 * HIDDEN + row] = b_mid_g_4[1 * HIDDEN + row] + acc1;
            out_residual_4[2 * HIDDEN + row] = b_mid_g_4[2 * HIDDEN + row] + acc2;
            out_residual_4[3 * HIDDEN + row] = b_mid_g_4[3 * HIDDEN + row] + acc3;
        }
    }
}

// ────────────────────────────────────────────────────────────────
// MEGA-KERNEL ENTRY (24 kernargs, same layout as M=1 variant; pos
// is interpreted as pos_base; the M=1 fused buffers all become
// [M_EFF, ...] sized).
// ────────────────────────────────────────────────────────────────
extern "C" __global__ __launch_bounds__(WAVE)
void qwen3_layer_megakernel_speck4(
    const float          *in_residual_4,         //  0 [M_EFF, HIDDEN]
    float                *out_residual_4,        //  1 [M_EFF, HIDDEN]
    const unsigned short *qkv_w,                 //  2
    const unsigned short *o_w,                   //  3
    const unsigned short *gate_up_w,             //  4
    const unsigned short *down_w,                //  5
    const float          *in_norm_g,             //  6
    const float          *post_norm_g,           //  7
    const float          *q_norm_g,              //  8
    const float          *k_norm_g,              //  9
    const float          *rope_cos_4,            // 10 [M_EFF, HEAD_DIM/2]
    const float          *rope_sin_4,            // 11 [M_EFF, HEAD_DIM/2]
    float                *kc_layer,              // 12
    float                *vc_layer,              // 13
    unsigned long long    pos_base,              // 14
    unsigned int         *barrier_counter,       // 15
    float                *gu_scratch_4,          // 16 [M_EFF, 2*FF]
    float                *b_x_norm_g_4,          // 17 [M_EFF, HIDDEN]
    float                *b_qkv_g_4,             // 18 [M_EFF, QKV_DIM]
    float                *b_attn_q_g_4,          // 19 [M_EFF, Q_DIM]
    float                *b_mid_g_4,             // 20 [M_EFF, HIDDEN]
    float                *b_mid_norm_g_4,        // 21 [M_EFF, HIDDEN]
    float                *b_attn_out_g_4,        // 22 [M_EFF, Q_DIM]
    unsigned long long    max_phase              // 23
) {
    unsigned int wg_id = blockIdx.x;
    unsigned int lane  = threadIdx.x;

    phase1_input_rmsnorm_speck4(wg_id, lane, in_residual_4, in_norm_g, b_x_norm_g_4);
    mega_barrier(barrier_counter, 0);
    if (max_phase < 1) return;

    phase3_qkv_matmul_speck4(wg_id, lane, qkv_w, b_x_norm_g_4, b_qkv_g_4);
    mega_barrier(barrier_counter, 1);
    if (max_phase < 2) return;

    phase5_qkv_split_qknorm_rope_speck4(wg_id, lane, q_norm_g, k_norm_g,
                                         rope_cos_4, rope_sin_4,
                                         b_qkv_g_4, b_attn_q_g_4,
                                         kc_layer, vc_layer, pos_base);
    mega_barrier(barrier_counter, 2);
    if (max_phase < 3) return;

    phase7_attn_decode_speck4(wg_id, lane, b_attn_q_g_4, kc_layer, vc_layer,
                               b_attn_out_g_4, pos_base);
    mega_barrier(barrier_counter, 3);
    if (max_phase < 4) return;

    phase9_oproj_residual_speck4(wg_id, lane, o_w, b_attn_out_g_4,
                                  in_residual_4, b_mid_g_4);
    mega_barrier(barrier_counter, 4);
    if (max_phase < 5) return;

    phase11_post_norm_speck4(wg_id, lane, post_norm_g, b_mid_g_4, b_mid_norm_g_4);
    mega_barrier(barrier_counter, 5);
    if (max_phase < 6) return;

    phase13_gate_up_matmul_speck4(wg_id, lane, gate_up_w, b_mid_norm_g_4, gu_scratch_4);
    mega_barrier(barrier_counter, 6);
    if (max_phase < 7) return;

    phase15_silu_mul_speck4(wg_id, lane, gu_scratch_4);
    mega_barrier(barrier_counter, 7);

    phase17_down_residual_speck4(wg_id, lane, down_w, gu_scratch_4, b_mid_g_4, out_residual_4);
}
