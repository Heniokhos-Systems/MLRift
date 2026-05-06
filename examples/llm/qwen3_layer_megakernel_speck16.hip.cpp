// Slice 4.18 (PLD speck16) — qwen3-0.6B layer mega-kernel for M=16 queries.
//
// Variant of qwen3_layer_megakernel_speck8.hip.cpp (slice 4.15) that
// processes M_EFF=16 query tokens per dispatch.  Two key changes vs
// mks8 to fit the LDS / WG budget at M=16:
//
//   1. attn_w_lds[M_EFF * 64] (was M_EFF * 128).  The softmax-weight
//      cache only spans the most recent 64 KV positions; this kernel
//      MUST be invoked with pos_base + M_EFF <= 64 (host enforces).
//      Per-WG LDS = 16 * 64 * 4 = 4 KB — same as mks8's 8 * 128 * 4.
//
//   2. ATTN_COOP=2 (was 4).  Phase-7 active WGs: N_HEADS * ATTN_COOP * M_EFF
//      = 16 * 2 * 16 = 512 = WG_PERSIST.  At ATTN_COOP=4 we'd need 1024
//      WGs which exceeds the persistent grid.  Each cooperating WG now
//      handles HEAD_DIM/2 = 64 output dims (vs 32 in mks8).
//
// Build:
//   hipcc --offload-arch=gfx1100 --genco -O3 \
//       examples/llm/qwen3_layer_megakernel_speck16.hip.cpp \
//       -o /tmp/qwen3_layer_megakernel_speck16.co

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
#define M_EFF       16
#define MAX_SEQ_LDS 64        // softmax-weight LDS cap (per the brief)

// Padded weight strides (slice 4.13 channel-repack).
#define HIDDEN_PAD  1152
#define Q_DIM_PAD   2176
#define FF_PAD      3200

// LDS — attn_w_lds holds M_EFF×MAX_SEQ_LDS weights = 4 KB at M=16.
__shared__ float wave_tmp[1];
__shared__ float attn_w_lds[M_EFF * MAX_SEQ_LDS];

// WMMA fragment vector typedefs (gfx1100 wave32).
typedef float  v8f  __attribute__((ext_vector_type(8)));
typedef short  v16s __attribute__((ext_vector_type(16)));

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
void phase1_input_rmsnorm_speck16(unsigned int wg_id, unsigned int lane,
                                   const float *in_residual_16,
                                   const float *in_norm_g,
                                   float *b_x_norm_g_16) {
    if (wg_id != 0) return;
    for (int m = 0; m < M_EFF; m++) {
        const float *in_m = in_residual_16 + m * HIDDEN;
        float *out_m = b_x_norm_g_16 + m * HIDDEN;
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
// M_EFF dot products (one per query token).  At M=16, register pressure
// rises (16 acc + 16 fma per inner iter).  We declare ROWS_PER_WG=8 so
// each WG drives 8 rows × 16 dots = 128 dots vs mks8's 64.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase3_qkv_matmul_speck16(unsigned int wg_id, unsigned int lane,
                                const unsigned short *qkv_w,
                                const float *b_x_norm_g_16,
                                float *b_qkv_g_16) {
    constexpr unsigned int ROWS_PER_WG = (QKV_DIM + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= QKV_DIM) continue;
        float acc[M_EFF];
        #pragma unroll
        for (int m = 0; m < M_EFF; m++) acc[m] = 0.0f;
        const unsigned int *w_row_u32 =
            reinterpret_cast<const unsigned int *>(qkv_w + row * HIDDEN_PAD);
        for (unsigned int kp = lane; kp < HIDDEN / 2; kp += WAVE) {
            unsigned int packed = w_row_u32[kp];
            unsigned int k = kp * 2;
            float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
            float w1 = bf16_to_f32((unsigned short)(packed >> 16));
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                acc[m] += w0 * b_x_norm_g_16[m * HIDDEN + k]
                       +  w1 * b_x_norm_g_16[m * HIDDEN + k + 1];
            }
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                acc[m] += __shfl_xor(acc[m], offset);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                b_qkv_g_16[m * QKV_DIM + row] = acc[m];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 5 — qkv_split + qknorm + rope_qk fused.  Same layout as mks8.
// One WG handles one (m, head_kind) pair: M_EFF * (16 Q + 8 K + 8 V) =
// 512 WGs at M=16 — exactly fills WG_PERSIST=512.
//
// rope_cos/rope_sin layout: [M_EFF, HEAD_DIM/2].
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase5_qkv_split_qknorm_rope_speck16(unsigned int wg_id, unsigned int lane,
                                           const float *q_norm_g, const float *k_norm_g,
                                           const float *rope_cos_16, const float *rope_sin_16,
                                           const float *b_qkv_g_16, float *b_attn_q_g_16,
                                           float *kc_layer, float *vc_layer,
                                           unsigned long long pos_base) {
    constexpr unsigned int N_Q_TASKS  = M_EFF * N_HEADS;     // 256
    constexpr unsigned int N_K_TASKS  = M_EFF * N_KV_HEADS;  // 128
    constexpr unsigned int N_V_TASKS  = M_EFF * N_KV_HEADS;  // 128
    constexpr unsigned int N_TASKS    = N_Q_TASKS + N_K_TASKS + N_V_TASKS;  // 512
    if (wg_id >= N_TASKS) return;
    constexpr unsigned int half = HEAD_DIM / 2;

    if (wg_id < N_Q_TASKS) {
        unsigned int m    = wg_id / N_HEADS;
        unsigned int head = wg_id % N_HEADS;
        const float *src = b_qkv_g_16 + m * QKV_DIM + head * HEAD_DIM;
        float ssq = 0.0f;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            float v = src[i];
            ssq += v * v;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        float scale = rsqrtf(ssq / float(HEAD_DIM) + 1e-6f);
        const float *rc = rope_cos_16 + m * half;
        const float *rs = rope_sin_16 + m * half;
        float *qdst = b_attn_q_g_16 + m * Q_DIM + head * HEAD_DIM;
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
        const float *src = b_qkv_g_16 + m * QKV_DIM + Q_DIM + head * HEAD_DIM;
        float ssq = 0.0f;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            float v = src[i];
            ssq += v * v;
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        float scale = rsqrtf(ssq / float(HEAD_DIM) + 1e-6f);
        const float *rc = rope_cos_16 + m * half;
        const float *rs = rope_sin_16 + m * half;
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
        const float *src = b_qkv_g_16 + m * QKV_DIM + Q_DIM + KV_DIM + head * HEAD_DIM;
        unsigned long long pos_m = pos_base + (unsigned long long)m;
        float *dst = vc_layer + pos_m * KV_DIM + head * HEAD_DIM;
        for (unsigned int i = lane; i < HEAD_DIM; i += WAVE) {
            dst[i] = src[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 7 — attn_decode for M_EFF query tokens.
// ATTN_COOP=2 → N_HEADS * ATTN_COOP * M_EFF = 16 * 2 * 16 = 512 active
// WGs (= WG_PERSIST).  Loops m=0..M_EFF-1 internally.
// Each cooperating WG handles HEAD_DIM/ATTN_COOP = 64 output dims.
// At WAVE=32 each lane writes 2 output dims (i = lane and i = lane + 32).
//
// LDS-cap precondition: pos_base + M_EFF <= MAX_SEQ_LDS (host enforces).
// ────────────────────────────────────────────────────────────────
#define ATTN_COOP 2
__device__ __forceinline__
void phase7_attn_decode_speck16(unsigned int wg_id, unsigned int lane,
                                 const float *b_attn_q_g_16,
                                 const float *kc_layer, const float *vc_layer,
                                 float *b_attn_out_g_16,
                                 unsigned long long pos_base) {
    if (wg_id >= N_HEADS * ATTN_COOP * M_EFF) return;
    // Decompose: wg = ((q_head * ATTN_COOP) + coop_idx) * M_EFF + m
    // Equivalent decomposition that matches our launch enumeration:
    //   q_head = wg / (ATTN_COOP * M_EFF)
    //   rem    = wg % (ATTN_COOP * M_EFF)
    //   coop   = rem / M_EFF
    //   m      = rem % M_EFF
    unsigned int q_head   = wg_id / (ATTN_COOP * M_EFF);
    unsigned int rem      = wg_id % (ATTN_COOP * M_EFF);
    unsigned int coop_idx = rem / M_EFF;
    unsigned int m        = rem % M_EFF;
    unsigned int kv_head  = q_head / (N_HEADS / N_KV_HEADS);
    constexpr float scale_attn = 1.0f / 11.31370849898f;
    constexpr unsigned int OUT_PER_WG = HEAD_DIM / ATTN_COOP;       // 64

    unsigned long long pos_m = pos_base + (unsigned long long)m;
    const float *q = b_attn_q_g_16 + m * Q_DIM + q_head * HEAD_DIM;
    float *w_lds = &attn_w_lds[m * MAX_SEQ_LDS];

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

    // Pass 2: cache exp(dot-max), reduce sum.  pos_m < MAX_SEQ_LDS by
    // the host precondition, so w_lds[t] is in range.
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

    // Pass 3 cooperative: each lane handles OUT_PER_WG / WAVE = 2 dims.
    // i = coop_idx * OUT_PER_WG + lane + s * WAVE  for s=0,1.
    #pragma unroll
    for (int s = 0; s < (int)(OUT_PER_WG / WAVE); s++) {
        unsigned int i = coop_idx * OUT_PER_WG + lane + (unsigned int)s * WAVE;
        float acc = 0.0f;
        for (unsigned long long t = 0; t <= pos_m; t++) {
            float w = w_lds[t] * inv_sum;
            const float *v = vc_layer + t * KV_DIM + kv_head * HEAD_DIM;
            acc += w * v[i];
        }
        b_attn_out_g_16[m * Q_DIM + q_head * HEAD_DIM + i] = acc;
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 9 — o_proj matmul + residual.  M_EFF amortized.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase9_oproj_residual_speck16(unsigned int wg_id, unsigned int lane,
                                    const unsigned short *o_w,
                                    const float *b_attn_out_g_16,
                                    const float *in_residual_16,
                                    float *b_mid_g_16) {
    constexpr unsigned int ROWS_PER_WG = (HIDDEN + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= HIDDEN) continue;
        float acc[M_EFF];
        #pragma unroll
        for (int m = 0; m < M_EFF; m++) acc[m] = 0.0f;
        const unsigned int *w_row_u32 =
            reinterpret_cast<const unsigned int *>(o_w + row * Q_DIM_PAD);
        for (unsigned int kp = lane; kp < Q_DIM / 2; kp += WAVE) {
            unsigned int packed = w_row_u32[kp];
            unsigned int k = kp * 2;
            float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
            float w1 = bf16_to_f32((unsigned short)(packed >> 16));
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                acc[m] += w0 * b_attn_out_g_16[m * Q_DIM + k]
                       +  w1 * b_attn_out_g_16[m * Q_DIM + k + 1];
            }
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                acc[m] += __shfl_xor(acc[m], offset);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                b_mid_g_16[m * HIDDEN + row] = in_residual_16[m * HIDDEN + row] + acc[m];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 11 — post_norm.  WG 0 only; loops over m=0..M_EFF-1.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase11_post_norm_speck16(unsigned int wg_id, unsigned int lane,
                                const float *post_norm_g,
                                const float *b_mid_g_16, float *b_mid_norm_g_16) {
    if (wg_id != 0) return;
    for (int m = 0; m < M_EFF; m++) {
        const float *in_m = b_mid_g_16 + m * HIDDEN;
        float *out_m = b_mid_norm_g_16 + m * HIDDEN;
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
// Phase 13 — gate_up matmul.  Plain bf16 dot-product walk (NOT WMMA).
//
// At M=16 we DON'T use the gfx1100 16x16x16 WMMA tile — wave32
// v_wmma_f32_16x16x16_bf16 has only 16 columns of B, which is exactly
// M_EFF=16, BUT each lane stores 8 rows of one column; the per-tile
// bookkeeping mirrors mks8's tail-clamp-aware path and adds nothing
// over the bf16 fma loop at M=16 (the inputs are read M_EFF times per
// weight already — bw-bound).  For simplicity and reuse of the proven
// mks8 strategy we keep the dot-product form here and revisit WMMA in
// a follow-up slice.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase13_gate_up_matmul_speck16(unsigned int wg_id, unsigned int lane,
                                     const unsigned short *gate_up_w,
                                     const float *b_mid_norm_g_16,
                                     float *gu_scratch_16) {
    constexpr unsigned int N_OUT = 2 * FF;
    constexpr unsigned int ROWS_PER_WG = (N_OUT + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= N_OUT) continue;
        float acc[M_EFF];
        #pragma unroll
        for (int m = 0; m < M_EFF; m++) acc[m] = 0.0f;
        const unsigned int *w_row_u32 =
            reinterpret_cast<const unsigned int *>(gate_up_w + row * HIDDEN_PAD);
        for (unsigned int kp = lane; kp < HIDDEN / 2; kp += WAVE) {
            unsigned int packed = w_row_u32[kp];
            unsigned int k = kp * 2;
            float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
            float w1 = bf16_to_f32((unsigned short)(packed >> 16));
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                acc[m] += w0 * b_mid_norm_g_16[m * HIDDEN + k]
                       +  w1 * b_mid_norm_g_16[m * HIDDEN + k + 1];
            }
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                acc[m] += __shfl_xor(acc[m], offset);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                gu_scratch_16[m * N_OUT + row] = acc[m];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Phase 15 — silu_mul: gu_scratch_16[m, 0..FF] = silu(gate) * up.
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase15_silu_mul_speck16(unsigned int wg_id, unsigned int lane,
                               float *gu_scratch_16) {
    constexpr unsigned int N_OUT = 2 * FF;
    constexpr unsigned int ELEMS_PER_WG = (FF + WG_PERSIST * WAVE - 1)
                                          / (WG_PERSIST * WAVE);
    for (int m = 0; m < M_EFF; m++) {
        float *gu_m = gu_scratch_16 + m * N_OUT;
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
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void phase17_down_residual_speck16(unsigned int wg_id, unsigned int lane,
                                    const unsigned short *down_w,
                                    const float *gu_scratch_16,
                                    const float *b_mid_g_16,
                                    float *out_residual_16) {
    constexpr unsigned int N_OUT = 2 * FF;
    constexpr unsigned int ROWS_PER_WG = (HIDDEN + WG_PERSIST - 1) / WG_PERSIST;
    for (unsigned int r_off = 0; r_off < ROWS_PER_WG; r_off++) {
        unsigned int row = wg_id + r_off * WG_PERSIST;
        if (row >= HIDDEN) continue;
        float acc[M_EFF];
        #pragma unroll
        for (int m = 0; m < M_EFF; m++) acc[m] = 0.0f;
        const unsigned int *w_row_u32 =
            reinterpret_cast<const unsigned int *>(down_w + row * FF_PAD);
        for (unsigned int kp = lane; kp < FF / 2; kp += WAVE) {
            unsigned int packed = w_row_u32[kp];
            unsigned int k = kp * 2;
            float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
            float w1 = bf16_to_f32((unsigned short)(packed >> 16));
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                acc[m] += w0 * gu_scratch_16[m * N_OUT + k]
                       +  w1 * gu_scratch_16[m * N_OUT + k + 1];
            }
        }
        for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                acc[m] += __shfl_xor(acc[m], offset);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < M_EFF; m++) {
                out_residual_16[m * HIDDEN + row] = b_mid_g_16[m * HIDDEN + row] + acc[m];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// MEGA-KERNEL ENTRY (24 kernargs, same layout as M=8 variant).
// ────────────────────────────────────────────────────────────────
extern "C" __global__ __launch_bounds__(WAVE)
void qwen3_layer_megakernel_speck16(
    const float          *in_residual_16,        //  0 [M_EFF, HIDDEN]
    float                *out_residual_16,       //  1 [M_EFF, HIDDEN]
    const unsigned short *qkv_w,                 //  2
    const unsigned short *o_w,                   //  3
    const unsigned short *gate_up_w,             //  4
    const unsigned short *down_w,                //  5
    const float          *in_norm_g,             //  6
    const float          *post_norm_g,           //  7
    const float          *q_norm_g,              //  8
    const float          *k_norm_g,              //  9
    const float          *rope_cos_16,           // 10 [M_EFF, HEAD_DIM/2]
    const float          *rope_sin_16,           // 11 [M_EFF, HEAD_DIM/2]
    float                *kc_layer,              // 12
    float                *vc_layer,              // 13
    unsigned long long    pos_base,              // 14
    unsigned int         *barrier_counter,       // 15
    float                *gu_scratch_16,         // 16 [M_EFF, 2*FF]
    float                *b_x_norm_g_16,         // 17 [M_EFF, HIDDEN]
    float                *b_qkv_g_16,            // 18 [M_EFF, QKV_DIM]
    float                *b_attn_q_g_16,         // 19 [M_EFF, Q_DIM]
    float                *b_mid_g_16,            // 20 [M_EFF, HIDDEN]
    float                *b_mid_norm_g_16,       // 21 [M_EFF, HIDDEN]
    float                *b_attn_out_g_16,       // 22 [M_EFF, Q_DIM]
    unsigned long long    max_phase              // 23
) {
    unsigned int wg_id = blockIdx.x;
    unsigned int lane  = threadIdx.x;

    phase1_input_rmsnorm_speck16(wg_id, lane, in_residual_16, in_norm_g, b_x_norm_g_16);
    mega_barrier(barrier_counter, 0);
    if (max_phase < 1) return;

    phase3_qkv_matmul_speck16(wg_id, lane, qkv_w, b_x_norm_g_16, b_qkv_g_16);
    mega_barrier(barrier_counter, 1);
    if (max_phase < 2) return;

    phase5_qkv_split_qknorm_rope_speck16(wg_id, lane, q_norm_g, k_norm_g,
                                          rope_cos_16, rope_sin_16,
                                          b_qkv_g_16, b_attn_q_g_16,
                                          kc_layer, vc_layer, pos_base);
    mega_barrier(barrier_counter, 2);
    if (max_phase < 3) return;

    phase7_attn_decode_speck16(wg_id, lane, b_attn_q_g_16, kc_layer, vc_layer,
                                b_attn_out_g_16, pos_base);
    mega_barrier(barrier_counter, 3);
    if (max_phase < 4) return;

    phase9_oproj_residual_speck16(wg_id, lane, o_w, b_attn_out_g_16,
                                   in_residual_16, b_mid_g_16);
    mega_barrier(barrier_counter, 4);
    if (max_phase < 5) return;

    phase11_post_norm_speck16(wg_id, lane, post_norm_g, b_mid_g_16, b_mid_norm_g_16);
    mega_barrier(barrier_counter, 5);
    if (max_phase < 6) return;

    phase13_gate_up_matmul_speck16(wg_id, lane, gate_up_w, b_mid_norm_g_16, gu_scratch_16);
    mega_barrier(barrier_counter, 6);
    if (max_phase < 7) return;

    phase15_silu_mul_speck16(wg_id, lane, gu_scratch_16);
    mega_barrier(barrier_counter, 7);

    phase17_down_residual_speck16(wg_id, lane, down_w, gu_scratch_16, b_mid_g_16, out_residual_16);
}
