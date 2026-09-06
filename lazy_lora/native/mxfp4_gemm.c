/*
 * MXFP4 expert matmuls that consume the packed checkpoint bytes directly.
 *
 * Kimi K3's routed experts ship as OCP MX FP4: two E2M1 codes per byte (low nibble is
 * the even element) and one E8M0 exponent byte per group of 32 input channels. The
 * torch path widened every expert to bf16 first (three tensors, 66M elements, ~250 ms
 * per expert on this machine), which cost more than reading it from disk. Here a row
 * block is decoded once into an L2-resident float tile and multiplied against every
 * activation row, so the decode is paid once per weight element, not once per element
 * per pass over memory.
 *
 * Conventions (all row-major, contiguous):
 *   W is [R, K] logically; packed is [R, K/2] bytes, scales is [R, K/32] bytes.
 *   gemm   : y[M, R]  = x[M, K] . W^T            (forward: gate/up/down projections)
 *   gemm_t : y[M, K] += dy[M, R] . W             (backward: gradient to the input)
 * A scale byte >= 253 marks a group that cannot be represented (255 is E8M0 NaN, and
 * 2^(253-127)*6 overflows float32); such groups contribute nothing, as in the reference
 * engine and the torch dequantiser.
 *
 * Accumulation is float32 with per-group partial sums; the reference accumulates in
 * double. The differences are ~1e-6 relative, far below bf16 activation rounding.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GROUP 32
#define RB 8                      /* rows of W decoded per tile */

static const float LUT[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                              -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
static float E8M0[256];
static int e8m0_ready = 0;

static void e8m0_init(void)
{
    for (int b = 0; b < 256; b++)
        E8M0[b] = (b >= 253) ? 0.f : ldexpf(1.f, b - 127);
    e8m0_ready = 1;
}

/* Decode row r of W into wf[K]. */
static inline void decode_row(float *wf, const uint8_t *packed, const uint8_t *scales,
                              int r, int K)
{
    const int pcols = K / 2, ngrp = K / GROUP;
    const uint8_t *pr = packed + (size_t)r * pcols;
    const uint8_t *sr = scales + (size_t)r * ngrp;
    for (int g = 0; g < ngrp; g++) {
        const float s = E8M0[sr[g]];
        const uint8_t *pb = pr + g * (GROUP / 2);
        float *o = wf + g * GROUP;
        for (int j = 0; j < GROUP / 2; j++) {
            const uint8_t b = pb[j];
            o[2 * j]     = LUT[b & 0x0F] * s;
            o[2 * j + 1] = LUT[b >> 4] * s;
        }
    }
}

void mxfp4_gemm(float *y, const float *x, const uint8_t *packed, const uint8_t *scales,
                int M, int K, int R)
{
    if (!e8m0_ready) e8m0_init();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int r0 = 0; r0 < R; r0 += RB) {
        float tile[RB * 3584 > RB * 4096 ? RB * 3584 : RB * 4096]; /* K <= 4096 for K3 */
        const int rb = (r0 + RB <= R) ? RB : R - r0;
        for (int i = 0; i < rb; i++)
            decode_row(tile + (size_t)i * K, packed, scales, r0 + i, K);
        for (int m = 0; m < M; m++) {
            const float *xm = x + (size_t)m * K;
            for (int i = 0; i < rb; i++) {
                const float *w = tile + (size_t)i * K;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
                int k = 0;
                for (; k + 7 < K; k += 8) {
                    a0 += xm[k] * w[k];         a1 += xm[k + 1] * w[k + 1];
                    a2 += xm[k + 2] * w[k + 2]; a3 += xm[k + 3] * w[k + 3];
                    a4 += xm[k + 4] * w[k + 4]; a5 += xm[k + 5] * w[k + 5];
                    a6 += xm[k + 6] * w[k + 6]; a7 += xm[k + 7] * w[k + 7];
                }
                float acc = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
                for (; k < K; k++) acc += xm[k] * w[k];
                y[(size_t)m * R + r0 + i] = acc;
            }
        }
    }
}

void mxfp4_gemm_t(float *y, const float *dy, const uint8_t *packed, const uint8_t *scales,
                  int M, int K, int R)
{
    if (!e8m0_ready) e8m0_init();
    const int ngrp = K / GROUP, pcols = K / 2;
    /* Threads own disjoint column groups of y, so no reduction is needed. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int g = 0; g < ngrp; g++) {
        for (int r0 = 0; r0 < R; r0 += RB) {
            const int rb = (r0 + RB <= R) ? RB : R - r0;
            float wf[RB][GROUP];
            for (int i = 0; i < rb; i++) {
                const int r = r0 + i;
                const float s = E8M0[scales[(size_t)r * ngrp + g]];
                const uint8_t *pb = packed + (size_t)r * pcols + g * (GROUP / 2);
                for (int j = 0; j < GROUP / 2; j++) {
                    const uint8_t b = pb[j];
                    wf[i][2 * j]     = LUT[b & 0x0F] * s;
                    wf[i][2 * j + 1] = LUT[b >> 4] * s;
                }
            }
            for (int m = 0; m < M; m++) {
                const float *d = dy + (size_t)m * R + r0;
                float *o = y + (size_t)m * K + g * GROUP;
                for (int i = 0; i < rb; i++) {
                    const float s = d[i];
                    for (int c = 0; c < GROUP; c++) o[c] += s * wf[i][c];
                }
            }
        }
    }
}

int mxfp4_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* out[R, K] float32 = decoded W. ~11M elements per expert matrix; used when the row count
 * is large enough that a BLAS sgemm on the widened matrix beats the fused loop. */
void mxfp4_dequant(float *out, const uint8_t *packed, const uint8_t *scales, int K, int R)
{
    if (!e8m0_ready) e8m0_init();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < R; r++)
        decode_row(out + (size_t)r * K, packed, scales, r, K);
}
