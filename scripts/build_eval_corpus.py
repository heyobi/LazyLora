#!/usr/bin/env python3
"""
Build the fixed evaluation corpora: Turkish (primary metric) and English (control), from
the plain-text extracts of the same Wikipedia topics in both languages.

    python scripts/build_eval_corpus.py [--tokens 4096] [--out <workspace>/eval]

Writes, per language, <out>/<lang>_wiki.txt (the text actually used, article by article),
<out>/<lang>_wiki_ids.json (the token ids, exactly --tokens of them, no BOS) and a manifest
with titles, revision ids, fetch date and licence (CC BY-SA 4.0). The corpora are held out:
nothing in them is used for training. Evaluation is done on the whole 2048-token chunks
because the engine's cost is per sweep, not per token (see Bulgular.md 16.5).
"""
import argparse, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import get_default_config  # noqa: E402

TOPICS = [  # (tr title, en title): same subject in both languages, general domain
    ("İstanbul", "Istanbul"), ("Anadolu", "Anatolia"), ("Güneş Sistemi", "Solar System"),
    ("Fotosentez", "Photosynthesis"), ("Osmanlı İmparatorluğu", "Ottoman Empire"),
    ("Kahve", "Coffee"), ("Deprem", "Earthquake"), ("İklim değişikliği", "Climate change"),
    ("Matematik", "Mathematics"), ("Futbol", "Association football"), ("Su", "Water"),
    ("Kedi", "Cat"), ("Demokrasi", "Democracy"), ("Roman", "Novel"), ("Ağrı Dağı", "Mount Ararat"),
    ("Elektrik", "Electricity"), ("Kanser", "Cancer"), ("Tiyatro", "Theatre"),
    ("Uçak", "Airplane"), ("Zeytin", "Olive"), ("Karadeniz", "Black Sea"), ("Şiir", "Poetry"),
    ("Bilgisayar", "Computer"), ("Ay", "Moon"), ("Bal arısı", "Honey bee"),
]
UA = "LazyLoRA-eval-corpus/1.0 (research; contact via github.com/heyobi/LazyLora)"


class _ParaExtractor(HTMLParser):
    """Paragraph text out of Parsoid HTML: <p> children of the article body, without
    tables, reference lists, math and the bracketed citation markers."""
    SKIP = {"table", "style", "script", "sup", "math", "figure", "figcaption", "ol", "ul"}

    def __init__(self):
        super().__init__()
        self.paras, self._cur, self._skip, self._in_p = [], [], 0, False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "p" and not self._skip:
            self._in_p, self._cur = True, []

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "p" and self._in_p:
            text = re.sub(r"\s+", " ", "".join(self._cur)).strip()
            if len(text) > 40:
                self.paras.append(text)
            self._in_p = False

    def handle_data(self, data):
        if self._in_p and not self._skip:
            self._cur.append(data)


def fetch(lang, title):
    """Article paragraphs via the REST API (the action API rate-limits this network)."""
    t = urllib.parse.quote(title.replace(" ", "_"))
    base = f"https://{lang}.wikipedia.org/api/rest_v1/page"
    # urllib gets 429 from the html endpoint on this network while curl does not, so curl it is
    import subprocess, tempfile
    hdr = tempfile.NamedTemporaryFile(delete=False).name
    code = ""
    for attempt in range(6):
        r = subprocess.run(["curl", "-s", "-A", UA, "-D", hdr, "-w", "%{http_code}", "-o", hdr + ".body",
                            f"{base}/html/{t}?redirect=true"], capture_output=True, text=True, timeout=90)
        code = r.stdout.strip()[-3:]
        if code == "429":
            wait = 15 * (attempt + 1)
            print(f"  [{lang}] {title}: rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        break
    if code == "404":
        return None
    if code != "200":
        raise RuntimeError(f"{lang}/{title}: HTTP {code}")
    html = open(hdr + ".body", encoding="utf-8").read()
    revid = ""
    for line in open(hdr, encoding="utf-8", errors="ignore"):
        if line.lower().startswith("etag:"):
            revid = line.split(":", 1)[1].strip().strip('"').split("/")[0].lstrip("W")
    os.remove(hdr); os.remove(hdr + ".body")
    px = _ParaExtractor()
    px.feed(html)
    return {"title": title, "revid": revid, "text": "\n\n".join(px.paras)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--per-article", type=int, default=400, help="max tokens taken from each article")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = get_default_config()
    out = args.out or os.path.join(cfg.paths.workspace_dir, "eval")
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, cfg.paths.base_model_dir)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.paths.base_model_dir, trust_remote_code=True)

    for lang, col in (("tr", 0), ("en", 1)):
        ids, pieces, manifest = [], [], []
        for pair in TOPICS:
            if len(ids) >= args.tokens:
                break
            art = fetch(lang, pair[col])
            if not art or len(art["text"]) < 500:
                print(f"  [{lang}] skipped {pair[col]}", flush=True)
                continue
            text = art["text"].strip() + "\n\n"
            t = tok.encode(text)[: args.per_article]
            # cut the text to what the kept tokens cover, so the .txt matches the ids
            kept_text = tok.decode(t)
            ids.extend(t)
            pieces.append(kept_text)
            manifest.append({"title": art["title"], "revid": art["revid"], "tokens": len(t)})
            print(f"  [{lang}] {art['title']}: {len(t)} tokens (total {len(ids)})", flush=True)
            time.sleep(3.0)
        ids = ids[: args.tokens]
        assert len(ids) == args.tokens, f"{lang}: only {len(ids)} tokens gathered"
        with open(os.path.join(out, f"{lang}_wiki.txt"), "w", encoding="utf-8") as f:
            f.write("".join(pieces))
        with open(os.path.join(out, f"{lang}_wiki_ids.json"), "w") as f:
            json.dump(ids, f)
        with open(os.path.join(out, f"{lang}_wiki_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"language": lang, "tokens": len(ids), "fetched": time.strftime("%Y-%m-%d"),
                       "source": f"{lang}.wikipedia.org, TextExtracts API", "licence": "CC BY-SA 4.0",
                       "tokenizer": "Kimi K3 tiktoken", "articles": manifest}, f, ensure_ascii=False, indent=1)
        print(f"{lang}: {len(ids)} tokens from {len(manifest)} articles -> {out}/{lang}_wiki_ids.json", flush=True)


if __name__ == "__main__":
    main()
