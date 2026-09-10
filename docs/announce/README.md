# What this project says in public

The exact text of every public announcement of LazyLoRA, published here before it is posted
anywhere. It is here for the same reason the evaluation threshold went into git twenty-nine
hours before the training run started: a claim is worth more when it was fixed in advance
and can be checked against what was actually measured. A post is where a project is most
tempted to round a number up, drop the caveat that spoils the sentence, or let a verb do
work the evidence has not earned — and a post is also the one artefact that travels away
from the repository, where nobody can check it. Publishing the text next to the evidence it
draws on makes an overclaim a diff rather than a memory. The list immediately below is the
point of the directory: five sentences that may not appear in any of these posts, in any
language, in any headline, caption or reply. Each of them is a true-sounding thing this
project is not entitled to say. If you find one of them in something I have written, that is
a defect, and it is reportable as one.

---

## The drafts, and whether they have gone out

| Draft | Channel | Language | Status |
|---|---|---|---|
| [`hacker_news.md`](hacker_news.md) | Show HN | EN | Written 9 Sep 2026, revised 10 Sep. **Not posted.** |
| [`reddit_localllama_post.md`](reddit_localllama_post.md) | r/LocalLLM (the r/LocalLLaMA karma gate was not passed; see below) | EN | **Posted 10 Sep 2026, ~18:45:** https://www.reddit.com/r/LocalLLM/s/wX0IHcxwvv — the app-safe body at the end of that file, photo first, then the two English cards. The long draft [`reddit_localllama.md`](reddit_localllama.md) stays for the October result post. |
| [`x_thread.md`](x_thread.md) | X, eleven posts | EN | Written 9 Sep 2026, revised 10 Sep. **Not posted.** |
| [`linkedin.md`](linkedin.md) | LinkedIn, long form and short | TR + EN | **Posted (TR) 10 Sep 2026:** https://lnkd.in/p/dEtx7CeT — the author's own edit of the short version, recorded verbatim at the end of the file. EN not posted. |
| *(not in this repository)* | Letter to the author of `kimi-k3-in-c` | EN | **Sent by email on 10 September 2026** — the first of these to go anywhere. Kept out of the tree then and now: publishing a letter before its recipient has read it makes it an announcement rather than a letter, and this one thanks him for work that caught a real bug in mine. No reply is owed and none is assumed; if one comes, what it corrects goes into the repository, not into a post. |

**The repository itself became public on 10 September 2026**, at
<https://github.com/heyobi/LazyLora>, with GitHub Pages serving `docs/` — so the Turkish
walkthrough is live at <https://heyobi.github.io/LazyLora/kanit_kosusu.html> and every link
in these drafts now resolves for a stranger. That is the whole of what has happened. Going
public is not an announcement: the four drafts above are still unposted, and nothing in this
directory has been sent to Show HN, r/LocalLLaMA, X or LinkedIn.

When one goes out, its row gets the date and the link to the live post, and the draft here
stays as it was written so the two can be compared.

---

## 1. The claim

One sentence, and it is the only one that gets made:

> I trained a LoRA adapter on Kimi K3 — 2.78 trillion parameters, 1.56 TB of weights —
> out of core, on one laptop with 7.6 GB of RAM, and I checked the arithmetic layer by
> layer against an independent implementation of the same model.

Turkish:

> 2,78 trilyon parametreli Kimi K3'ün üzerine, 7,6 GB RAM'li tek bir dizüstünde,
> çekirdek-dışı bir LoRA adaptörü eğittim; ileri geçişi bağımsız bir C implementasyonuna
> karşı katman katman doğruladım.

Five sentences that must never appear anywhere, in any language, in any headline,
caption, alt text or reply:

1. **"I trained Kimi K3."** A 590 MB adapter was trained. The 2.78 T base parameters are
   frozen and never change.
2. **"The model learned" / "it got smarter."** The adapter memorised five examples in the
   proof run. That is a test of the mechanism.
3. **"It got better at Turkish."** No evaluation result exists until the main run ends.
   Until then the honest answer to "did it work?" is "the threshold is registered, the
   number is due around 9-11 October, and I will publish it either way."
4. **"First to run K3 on a laptop" / "first to stream layers from disk for training."**
   `kimi-k3-in-c` ran this model on a laptop in August 2026 and is this project's oracle;
   AirLLM shipped layer-streamed LoRA training in September 2026. Both get named above
   the fold, by us, before anyone else does it for us.
5. **"It is in the repository", about anything that is not.** The comparison log and the
   five routing traces *are* — `evidence/cmp93_en34_2026-09-06.log`, `evidence/traces/`,
   `evidence/forward_loss_{main,proof}.jsonl`, `evidence/run_manifest.json`, with
   `SHA256SUMS` over all of it — and so are the fifteen op fixtures from the reference
   implementation, vendored at `tests/fixtures/ops/` under their own Apache-2.0 licence.
   What is still **not**: the 1.56 TB checkpoint, the C engine's per-layer dump, the packed
   NVMe trunk, the 1.8 GB training checkpoints, and any finite-difference log from the real
   model — that harness prints to the terminal, so its four numbers come from `DEVAM.md`
   §11 and have to be re-run on the real checkpoint to reproduce. Say "the log is in
   `evidence/`" where it is true and "you would need the checkpoint" where it is not; do
   not blur them. Run `git ls-files | grep` before writing that a reader can go and look at
   a file — being caught inventing an artefact inside the answer to "how do I know you
   didn't make this up" is the worst outcome available on any of these channels.

The defensible novelty, stated as a search result rather than a fact about the world:
*as far as I can find, nobody has published a backward pass through a model this size
inside a single consumer machine.* Invite correction in the same breath.

---

## 2. The before/after transcript

The strongest single moment is when a reader can see the same prompt answered by the base
model and by the adapter. It is also the easiest place in this whole project to make a
claim that cannot be defended. Rules:

- The bits-per-byte number goes **first**, the transcript second, captioned "illustrative,
  not evidence". A hand-picked pair of generations is a screenshot, not a measurement.
- **Never show a before/after on any of the 400 Dolly-tr training examples**, and
  especially not on the five proof-run examples — those are inside the training set and a
  visible improvement on them is memorisation, which is exactly what we already said the
  proof run was.
- Use held-out prompts, say how many you tried and how many you are showing, and show at
  least one where the difference is negligible or worse.
- If the evaluation is negative, the transcript still gets published, with the same
  caption and the honest sentence: the metric did not move, and here is what the outputs
  look like anyway.

---

## 3. Standing rules for every draft here

- Where a draft and [`../numbers.md`](../numbers.md) disagree, that table names the source
  and the source settles it. It is the canonical figure for every number this project
  quotes, including the ones in this directory.
- An announcement may say less than the repository proves. It may never say more.
- It may never name a file, a log or a dataset that a stranger cannot open today.
- The README says how this repository was built, AI assistance included. No post may imply
  otherwise, and none of them needs a paragraph about it either: a post that links the
  README links the disclosure.
- Two numbers move while the run is going — the step time and the end date. Re-read
  [`../numbers.md`](../numbers.md) on the day rather than trusting the draft.

---

## 4. Where the rest of this went

This directory used to be a release playbook, and most of it was reference work filed under
marketing. It now lives where a reader will actually find it:

| What | Where it is now |
|---|---|
| The questions a sceptic asks first, each with the file to open | [`../../FAQ.md`](../../FAQ.md) |
| Every number this project quotes, with the log line it was read from | [`../numbers.md`](../numbers.md) |
| What was checked before publication, and what the checking found wrong | [`../pre_publication_check.md`](../pre_publication_check.md) |
| What gets published if the evaluation comes out negative | [`../../README.md`](../../README.md), under "Evaluation protocol, registered before training" |

One part was deleted rather than moved: the channel-and-timing plan — which day of the week
to post on, which hour, in what order, how long to sit at the keyboard afterwards, and what
the first ten comments were expected to say. It was craft aimed at an audience, and it has
no business sitting eight lines from a rule about never quoting an unmeasured number. It was
not thrown away; it was published elsewhere, as a post about how a launch was planned, which
is the kind of document it actually is. It is not coming back here.

## r/LocalLLaMA, what the subreddit actually does (researched 10 September 2026)

- **Karma gate.** Since 24 April 2026 AutoModerator removes any submission from an account
  with fewer than 5 comment-karma points earned inside r/LocalLLaMA (10 sitewide karma is
  needed to comment at all). Of the 100 most recent posts on 9-10 September, 19 were
  removed and 18 of those carried the karma notice. There is no review queue; the remedy the
  moderators give is to comment in the sub, earn the karma, then post.
- **Rule 3.** Primarily LLM-generated copy is not allowed. Non-native speakers may use an
  LLM to refine a post only if the post says so. The draft therefore carries a disclosure
  paragraph; do not remove it.
- **Rule 4.** One post about one's own project is fine if the affiliation is plain and it
  is well under a tenth of the account's activity.
- **What the audience does.** The author of kimi-k3-in-c got 709 points in August 2026 with
  a plain "not practical, here are the numbers, here is the C" post. Engine posts that
  looked like "the third new engine today" were met with "why not llama.cpp" and "why not
  contribute to Colibri". Commenters name Colibri, WASTE and BigMoeOnEdge as the existing
  NVMe expert-streaming prior art; the README's related-work table should name them.
- **Mechanics.** Text post, GitHub link in the body (the filter keys on karma, not links),
  flair Discussion (median 3 points, 13 % of Discussion posts reach 100; Resources posts do
  worse), Monday to Wednesday 16:00-17:30 Turkey time, four hours at the keyboard after.
  Sources and the full data are in the workflow transcript of 10 September; the rules page
  itself was read from the Wayback capture of 7 August 2026.

