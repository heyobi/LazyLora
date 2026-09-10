# Licences and attribution

LazyLoRA is an engine, not a model and not a corpus. The code in this repository is
Apache-2.0. Almost everything the engine touches — the model weights, the training
data, the evaluation text — belongs to somebody else, is **not** redistributed here,
and carries its own terms. This page maps every component to its licence, its source
and the attribution it requires.

What *is* redistributed here is the project's own work: the engine, the scripts, the
documentation, and the measurements in `evidence/` — five expert-routing traces, the
93-layer comparison log, the raw training losses and the run manifest.

In one line: you may use, fork and build on this code freely; you must obtain Kimi K3
from Moonshot AI yourself under their licence; and if you redistribute the evidence
bundle, the training subset or an adapter, the attribution below travels with them.

## Component table

| Component | Where it is | Licence | Held by | Attribution required of a redistributor |
|---|---|---|---|---|
| Engine and scripts | `lazy_lora/`, `scripts/`, `systemd/` in this repository | Apache-2.0 | Ibrahim Polat | Keep `LICENSE` and `NOTICE`; state which files you changed; no trademark rights are granted (Apache-2.0 §6) |
| Native MXFP4 kernel | `lazy_lora/native/mxfp4_gemm.c`, `lazy_lora/native/build.sh` | Apache-2.0 | Ibrahim Polat | As above. Original work; not derived from any third-party kernel |
| Reference op fixtures | `tests/fixtures/ops/*.json` (15 files, 8.8 MB) | Apache-2.0 | kimi-k3-in-c authors (FareedKhan-dev and contributors), https://github.com/FareedKhan-dev/kimi-k3-in-c, upstream commit `c223f49` | Copied unmodified; retain the attribution in `NOTICE` and `tests/fixtures/ops/README.md`. They are published in the upstream repository under the same licence, which is what makes redistributing them here permissible |
| Vendored reference function | `docs/attic/patch_k3.py`, if present | Apache-2.0 | kimi-k3-in-c authors (FareedKhan-dev and contributors), https://github.com/FareedKhan-dev/kimi-k3-in-c | Retain the attribution in `NOTICE`, which also records that the copied function was modified (Apache-2.0 §4(b)–(d)) |
| Documentation | `README.md`, `docs/measurement_note.md` (current) and `docs/measurement_note_draft.md` (superseded, kept for the record), `docs/QUICKSTART.md`, `docs/traces/README.md`, `docs/LICENSES.md`, `evidence/README.md`, `docs/announce/`, `docs/kanit_kosusu.html`, `CONTRIBUTING.md`, `.github/ISSUE_TEMPLATE/`, and the Turkish operator logs `DEVAM.md`, `Bulgular.md`, `Fikirler.md` wherever they sit in the tree | CC BY 4.0 | Ibrahim Polat | Credit the project, link the licence, indicate changes. Passages quoting Wikipedia or Dolly text remain under those texts' own CC BY-SA licences |
| Figures | `docs/figures/*.png`, `docs/figures/*.svg` | CC BY 4.0 | Ibrahim Polat | Credit the project, link the licence, indicate changes. Plotted from this project's own measurements |
| **Kimi K3 model weights** | **not here** — https://huggingface.co/moonshotai/Kimi-K3 | Kimi K3 License | Moonshot AI | Include Moonshot's copyright and permission notice with any copy. A separate agreement is required for Model-as-a-Service operators above $20M revenue in any 12 months; "Kimi K3" must be shown in the UI above 100M MAU or $20M monthly revenue. Not an OSI-approved licence |
| Packed trunk file, index overlay, activation ring buffer, engine checkpoints | **not here** — generated on the user's own disks | Kimi K3 License (treated as repackagings of the weights and not redistributed, conservatively, whether or not that is legally required) | Moonshot AI | Do not redistribute: they contain the model weights in a different container |
| **LoRA adapter weights** | **not here** — to be published separately when the run completes; not yet released, and no adapter exists yet | Kimi K3 License | Moonshot AI (base weights) / Ibrahim Polat (adapter) | Ship the Kimi K3 License text beside the weights; credit Databricks and atasoglu for the training data; state non-affiliation with Moonshot AI. See "Two unsettled questions" below |
| **Expert routing traces** | **here** — `evidence/traces/`, five traces over all 92 MoE layers, committed 9 September 2026; format, provenance and worked examples in `docs/traces/README.md` | Routing arrays (`trace.bin`): CC0-1.0. Manifests, analyses and documentation: CC BY 4.0. The traced texts were written for this study and are released with the manifests under CC BY 4.0 | Ibrahim Polat, for the measurements | Credit LazyLoRA for the measurement. No third-party attribution is owed, because no third-party text was traced. **If you record traces of your own, do not publish one whose manifest still carries the raw text or token ids of a copyrighted input** — see below |
| **Rest of the evidence bundle** | **here** — `evidence/cmp93_en34_2026-09-06.log`, `evidence/forward_loss_main.jsonl`, `evidence/forward_loss_proof.jsonl`, `evidence/run_manifest.json`, `evidence/SHA256SUMS` | CC BY 4.0 | Ibrahim Polat | Credit the project, link the licence, indicate changes. These are this machine's own measurements, copied in unmodified except that its filesystem paths were replaced with placeholders; they contain no model weights and no third-party text |
| **Selected training subset** (400 examples, packed into 154 sequences) | **not here** — to be published separately when the run completes; not yet released | **CC BY-SA 3.0** | Databricks, Inc. and atasoglu | Attribute both upstreams; link the licence; state that the work was modified; license any redistribution under CC BY-SA 3.0 or a later CC BY-SA version. ShareAlike is not optional here |
| Upstream training data | https://huggingface.co/datasets/databricks/databricks-dolly-15k and https://huggingface.co/datasets/atasoglu/databricks-dolly-15k-tr | CC BY-SA 3.0 | Databricks, Inc. / atasoglu | As above. Some `context` fields contain Wikipedia passages, themselves CC BY-SA |
| Wikipedia evaluation slices | **not here** — fetched by `scripts/build_eval_corpus.py` | CC BY-SA 4.0 | Wikipedia contributors | Cite article title, language edition and revision id; link the licence; ShareAlike applies if you republish the text |
| News evaluation text (Anadolu Agency, BBC Türkçe) | **not here** — fetched by `scripts/build_eval_news.py` | All rights reserved | Anadolu Agency; BBC | **Not redistributable.** Read for measurement only. Cite outlet, headline, URL and publication date |
| Evaluation manifests and results | **not here** — `eval/` manifests and result files stay on the run machine; nothing under `eval/` is committed today, and they will be committed with the evaluation results | CC BY 4.0 | Ibrahim Polat | Measurements, hashes and URLs, not text. Credit the project |
| Third-party Python dependencies | not vendored — named in `requirements.txt` (`numpy>=1.24`, `torch>=2.3`) and in `pyproject.toml`, which adds the `[data]` and `[plot]` extras | their own licences | their own authors | Installed by the user, not shipped here. Naming a dependency imposes no obligation on this repository |

## What the Kimi K3 License says

The full text is at https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE
(Copyright (c) 2026 Moonshot AI). It is shaped like MIT and it explicitly permits what
this project does: it grants the right "to run, deploy, fine-tune, or otherwise modify
the Software and create derivative works from it", and to "publish, distribute,
sublicense, and/or sell copies". Its conditions:

1. **Notice.** The copyright notice and the permission notice must be included in all
   copies or substantial portions of the Software.
2. **Model as a Service.** A separate agreement with Moonshot AI is required before
   commercial use by a Model-as-a-Service operator with aggregate revenue above $20M in
   any consecutive 12 months.
3. **Naming at scale.** "Kimi K3" must be displayed prominently in the user interface of
   any commercial product with more than 100M monthly active users or more than $20M in
   monthly revenue.
4. Conditions 2 and 3 do not apply to internal use or to use through Moonshot's own
   products and certified partners.

What the licence does not contain matters as much. There is no copyleft clause forcing
derivatives onto the same licence. There is no clause defining derivative works to
include activations or model outputs. There is no restriction on the use of outputs.
There is no requirement that fine-tuned models be named after Kimi. And there is no
trademark grant, which is why this project is named after itself and describes its
adapters as adapters *for* Kimi K3.

Moonshot describes this as an open-weight release, not an open-source one, and the
licence is not OSI-approved. Conditions 2 and 3 are far above anything this project will
reach, but they bind downstream users at scale, which is why any adapter published from
this work will be released under the Kimi K3 License rather than relicensed.

## Two unsettled questions

This project would rather flag an open legal question than paper over it. Nothing below
is legal advice.

**Does ShareAlike reach the adapter weights?** The adapter is trained on CC BY-SA 3.0
data. Creative Commons' own May 2025 guidance says both that the BY and SA conditions
"are triggered only when works or adaptations of works are publicly shared" and that "if
AI models or outputs are based on ShareAlike content and they will be shared publicly,
following the ShareAlike condition would require AI developers to use the same CC license
as the original works" — while also acknowledging that "in many cases, neither the AI
model nor its outputs would be considered to be derivative works of training data under
copyright law"
(https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/). Those
statements point in different directions; Creative Commons does not resolve them and no
court has.

This project takes the mainstream reading: **a LoRA adapter is not an adaptation of the
training corpus, and ShareAlike does not propagate to the weights.** Two reasons. First,
the adapter is a set of low-rank matrices fitted to a loss surface, not a transformation
of any particular text into a new expressive work. Second, the alternative reading is
self-defeating here: CC BY-SA 3.0 forbids imposing additional restrictions on an
adaptation, so an adapter under CC BY-SA could not simultaneously carry conditions 2 and
3 of the Kimi K3 License — and those are not optional, because the adapter is unusable
without the base weights. Full CC BY-SA attribution is given regardless, and the
400-example subset, where ShareAlike unambiguously does apply, will be redistributed under
CC BY-SA 3.0 when it is published.

**Are model weights copyrightable at all?** Whether trained parameters attract copyright
is contested. If they do not, the Kimi K3 License operates as a contract rather than as a
copyright licence. This project complies with it either way.

## Are the routing traces a derivative work?

A trace records, for each layer and each token position, which 16 of 896 experts the router
selected and with what combining weight. The five traces are in `evidence/traces/`, so this
is a question that has been answered rather than one still pending. Three parts to it:

- **Of the model weights?** Contractually, no: the Kimi K3 License places no restriction
  on outputs and does not define derivative works to include activations. As a matter of
  copyright, expert indices are functional measurements of a system's behaviour — closer
  to a profiler's output than to an expressive work. They are published as measurements
  of the model, not as part of it. No Kimi K3 parameters are contained in or recoverable
  from a trace.
- **Of the input texts?** **This is the question that held the release up, and for these
  five traces the answer is that there was nothing to hold up.** The format writes the
  prompt's token ids and text into `trace.json` alongside the arrays
  (`lazy_lora/monitor/trace.py`, `scripts/measure_routing.py`), and token ids reconstruct
  the text losslessly, so a trace of a copyrighted article would contain that article. All
  five traced texts were written for this study — including the news-style Turkish
  paragraph about hazelnut production statistics, which earlier versions of this file
  described as a copyrighted news article. That description was wrong about the text's
  provenance. All five manifests therefore carry their text and token ids in full and
  nothing was replaced by a hash. **The rule still binds anyone who traces something
  else:** a trace recorded on text you may not redistribute must have its `text` and `ids`
  replaced by a SHA-256, a byte count and a citation before it is published.
- **Neither?** The routing arrays themselves, separated from the input text, are facts
  about a computation. They are released under **CC0-1.0** to make that stance explicit,
  so that anyone can study expert concentration and cache policy without the 1.56 TB
  checkpoint or this hardware.

`docs/traces/README.md` is where the released traces state all of this for themselves:
what each record is; that the arrays are CC0-1.0 and the manifests, analyses and
documentation CC BY 4.0; where each of the five texts came from and that all five were
written for this study; that the traces are measurements of the model's behaviour and
contain no model parameters; that use of Kimi K3 itself requires obtaining the weights
from Moonshot AI; and that the project is not affiliated with Moonshot AI.

## Attribution blocks, ready to copy

These are templates to copy when each artefact ships. Neither of them describes something
that exists today: at the time of writing there is no trained adapter and the 400-example
training subset has not been published. The traces are no longer on that list — they are in
`evidence/traces/`, and their attribution block is §9 of `docs/traces/README.md`.

**For an adapter's model card:**

> This is a LoRA adapter (rank 16, alpha 32, on `q_proj`, `v_proj` and the expert
> gate/up/down projections) for
> [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3). It contains no Kimi K3
> weights and is useless without them.
>
> **Base model:** Kimi K3, Copyright (c) 2026 Moonshot AI, released under the
> [Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE). This
> adapter is useless without those weights and is distributed under that licence; treat
> conditions 2 and 3 as passing to you. Whether an adapter is legally a derivative work of
> the base weights is unsettled — this project takes the conservative reading and complies
> either way. Nothing here is legal advice.
>
> **Training data:** 400 examples from
> [atasoglu/databricks-dolly-15k-tr](https://huggingface.co/datasets/atasoglu/databricks-dolly-15k-tr),
> a Turkish machine translation of
> [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
> Copyright (2023) Databricks, Inc., both under
> [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The selected and packed
> subset is released under CC BY-SA 3.0 alongside this adapter. Whether ShareAlike reaches
> the adapter weights themselves is legally unsettled; see docs/LICENSES.md.
>
> **Training code:** [LazyLoRA](https://github.com/heyobi/LazyLora), Apache-2.0.
>
> This project is not affiliated with, sponsored by, or endorsed by Moonshot AI or
> Databricks, Inc. "Kimi" and "Kimi K3" are used descriptively.

**For the 400-example training subset:**

> Derived from
> [atasoglu/databricks-dolly-15k-tr](https://huggingface.co/datasets/atasoglu/databricks-dolly-15k-tr)
> by atasoglu, itself a machine translation of
> [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
> Copyright (2023) Databricks, Inc. Both are licensed
> [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
>
> **Changes made:** filtered by length and category from the full set to 400 records, and
> packed into sequences of at most 1024 tokens with prompt tokens masked
> (`scripts/build_train_set.py`; the seed and the selection rules are in the manifest).
> No record text was edited.
>
> This derived dataset is licensed CC BY-SA 3.0, as ShareAlike requires.

**For the Wikipedia evaluation slices** (in the eval manifest; the text is not committed):

> Text extracts from the Turkish and English Wikipedias, retrieved through the MediaWiki
> REST API on [date]. Each slice records its article title, language edition and revision
> id. Wikipedia text is available under
> [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); see
> https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content. The text itself is not
> redistributed by this project; `scripts/build_eval_corpus.py` fetches it.

**For the news evaluation text** (in the eval manifest; the text is not committed):

> Turkish news articles published after the model's release, from Anadolu Agency and BBC
> Türkçe, retrieved through their public RSS feeds on [date]. These texts are copyrighted,
> all rights reserved, and are **not redistributed by this project** — only their URLs,
> headlines, publication dates, token counts and SHA-256 hashes are recorded, which is
> enough to reproduce the measurement. `scripts/build_eval_news.py` fetches them.

## Applying the licence to new files

New source files carry:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ibrahim Polat
```

```c
/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ibrahim Polat
 */
```

Two lines, and they are the whole convention. `LICENSE` and `NOTICE` at the root are what
actually license this repository — the header is a convenience for anyone who ends up
holding one file on its own, which is how a script usually travels. Most files written
before September 2026 carry a docstring and no header (`scripts/export_traces.py` is the
first that does); they are not being retrofitted, and their absence of a header means
nothing about their licence.

**New documentation** — anything under `docs/`, the Turkish operator logs, `README.md` —
needs no header. It is CC BY 4.0 by the table above. Say so in the file only where a reader
might reasonably think otherwise: a page that is going to be read outside the repository,
such as `docs/traces/README.md`, states its own licence.

**A new file under `evidence/`** is a measurement, so three things travel with it: a line
in `evidence/README.md` saying what produced it and what a reader can check with it, a row
in `evidence/SHA256SUMS` (`cd evidence && sha256sum <newfile> >> SHA256SUMS`), and this
machine's filesystem paths replaced by placeholders before it is committed. If it is a
routing trace of text this project does not own, its `text` and `ids` fields must be
replaced by a SHA-256, a byte count and a citation first — see "Are the routing traces a
derivative work?" above. None of the five traces committed on 9 September 2026 needed that,
because all five texts were written for this study.

**Code copied from somewhere else** — even a single function — gets an entry in `NOTICE`
naming the project, its licence and its URL, a comment at the copy site saying where it
came from and whether it was modified, and, if the upstream licence is not Apache-2.0, a
row in the table above. `docs/attic/patch_k3.py` is the worked example.
