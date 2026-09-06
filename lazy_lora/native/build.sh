#!/usr/bin/env bash
# Build the MXFP4 kernel next to its source. Needs gcc with OpenMP; AVX2+FMA is assumed.
set -e
cd "$(dirname "$0")"
gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC -o libmxfp4.so mxfp4_gemm.c -lm
echo "built $(pwd)/libmxfp4.so"
