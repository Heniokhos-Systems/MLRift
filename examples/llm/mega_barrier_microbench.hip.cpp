// Slice 4.1 — cross-WG global-barrier microbench (HIP source).
//
// Purpose: validate the slice 4 design doc's "5-10 µs per barrier"
// estimate before sinking 12+ hours into the mega-kernel emitter.
// If the measured cost is materially worse, slice 4 needs a redesign
// (signal-per-phase via host doorbell, etc.).
//
// Protocol per WG (lane 0 only):
//   for i in 0..n_iters:
//     atomicAdd(counter, 1)
//     spin until counter >= (i+1) * n_wgs
//
// The launcher times the whole kernel with a host clock, varies
// (n_wgs, n_iters) and divides out launch overhead to get
// per-barrier cost.  At WG_PERSIST=256 (the design's chosen value)
// we expect the barrier cost at the low end of the 5-10 µs range —
// far less contention than the original draft's 6144 WGs.
//
// Build:
//   hipcc --offload-arch=gfx1100 --genco -O3 \
//       examples/llm/mega_barrier_microbench.hip.cpp \
//       -o /tmp/mega_barrier_microbench.co

#include <hip/hip_runtime.h>

// Args declared as u64 to match the MLRift-side launcher convention
// (each kernarg slot is 8 bytes, low 32 bits hold the value, high 32
// bits zero).  Earlier `unsigned int` form silently ran with
// n_iters=0 because the launcher's slot layout didn't match the
// kernel's packed `(u32, u32)` expectation.
extern "C" __global__ __launch_bounds__(32)
void mega_barrier_microbench(unsigned int *counter_ptr,
                             unsigned long long n_wgs,
                             unsigned long long n_iters) {
    // Lane 0 of each WG drives the barrier; other lanes idle (this
    // mirrors the planned mega-kernel barrier protocol where the
    // wave-leader lane is the one that does the global atomic).
    if (threadIdx.x != 0) return;

    unsigned int n_wgs_u = (unsigned int)n_wgs;
    unsigned int n_iters_u = (unsigned int)n_iters;
    for (unsigned int i = 0; i < n_iters_u; i++) {
        // Ack: bump the global counter.
        __atomic_fetch_add(counter_ptr, 1u, __ATOMIC_ACQ_REL);
        // Wait: spin until ALL WGs have ack'd this iteration.
        unsigned int expected = (i + 1) * n_wgs_u;
        while (__atomic_load_n(counter_ptr, __ATOMIC_ACQUIRE) < expected) {
            // Pure spin — match the mega-kernel barrier inner loop.
        }
    }
}
