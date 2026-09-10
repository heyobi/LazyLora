# Reference op fixtures, vendored from kimi-k3-in-c

These fifteen files are not this project's work. They come from
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) by FareedKhan-dev, an
independent implementation of Kimi K3 in C, which publishes them in its own repository
under the Apache License 2.0. They are copied here unmodified, at upstream commit
`c223f490047e93600f05fbdab06b9742d2b8ef08` (29 August 2026).

Each file carries the weights, the input and the expected output of one operation, recorded
from that implementation: RMSNorm, the short convolution, the KDA decay and recurrence, the
attention-residual bank mix, MLA, the router, SiTU-GLU and the whole latent MoE block.
`MANIFEST.json` lists them.

They are here because they are the only external ground truth this repository has. Without
them, every statement about the forward pass being correct rests on this project checking
itself. With them, `lazy_lora/tests/test_reference_ops.py` compares the engine against
somebody else's arithmetic, in another language, on every push, and the continuous
integration job in `.github/workflows/quickstart.yml` runs that comparison on a machine
neither implementation's author controls. Seven of the eight compared ops match at 1e-5
absolute and 1e-4 relative; the latent MoE block matches at 2e-4 absolute with cosine
1.000000, which is the MXFP4 decode path's own rounding.

Copying them rather than asking the reader to clone a second repository is a deliberate
trade: 8.8 MB against a check that either runs for everybody or runs for nobody. If the
upstream fixtures change, the honest thing is to re-copy them and say so here rather than
to keep a version that agrees.

Licence: Apache-2.0, upstream. Attribution as above and in this repository's `NOTICE`.
