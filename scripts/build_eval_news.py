#!/usr/bin/env python3
"""
Held-out Turkish evaluation text the model cannot have memorised: news articles published
after the model's release, from the Anadolu Agency and BBC Türkçe RSS feeds.

    python scripts/build_eval_news.py [--tokens 4096] [--min-date 2026-09-01]

Writes <workspace>/eval/tr_news_ids.json (exactly --tokens ids), tr_news.txt and a
manifest with URLs, publication dates and token counts. Evaluation use only; the texts
are not redistributed. The Wikipedia slices (build_eval_corpus.py) stay as the
memorisation / forgetting control.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from html.parser import HTMLParser
from email.utils import parsedate_to_datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lazy_lora.core.config import get_default_config  # noqa: E402

FEEDS = [
    "https://www.aa.com.tr/tr/rss/default?cat=guncel",
    "https://www.aa.com.tr/tr/rss/default?cat=ekonomi",
    "https://www.aa.com.tr/tr/rss/default?cat=bilim-teknoloji",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
]
UA = "LazyLoRA-eval-corpus/1.0 (research; contact via github.com/heyobi/LazyLora)"


def curl(url):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        out = f.name
    r = subprocess.run(["curl", "-sL", "-A", UA, "-m", "60", "-w", "%{http_code}", "-o", out, url],
                       capture_output=True, text=True)
    code = r.stdout.strip()[-3:]
    data = open(out, "rb").read().decode("utf-8", errors="ignore")
    os.remove(out)
    return code, data


class Paras(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "aside", "figure", "figcaption", "ul", "ol", "table"}

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
            t = re.sub(r"\s+", " ", "".join(self._cur)).strip()
            if len(t) > 60:
                self.paras.append(t)
            self._in_p = False

    def handle_data(self, data):
        if self._in_p and not self._skip:
            self._cur.append(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--per-article", type=int, default=350)
    ap.add_argument("--min-date", default="2026-09-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = get_default_config()
    out = args.out or os.path.join(cfg.paths.workspace_dir, "eval")
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, cfg.paths.base_model_dir)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.paths.base_model_dir, trust_remote_code=True)

    items = []
    for feed in FEEDS:
        code, xml = curl(feed)
        if code != "200":
            print(f"  feed {feed}: HTTP {code}", flush=True)
            continue
        for m in re.finditer(r"<item>(.*?)</item>", xml, re.S):
            it = m.group(1)
            link = re.search(r"<link>\s*([^<\s]+)", it)
            date = re.search(r"<pubDate>([^<]+)</pubDate>", it)
            title = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", it, re.S)
            if not (link and date):
                continue
            try:
                d = parsedate_to_datetime(date.group(1).strip())
            except Exception:
                continue
            if d.strftime("%Y-%m-%d") >= args.min_date:
                items.append({"url": link.group(1).strip().split("?")[0], "date": d.strftime("%Y-%m-%d"),
                              "title": (title.group(1) if title else "").strip()})
    seen, uniq = set(), []
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"]); uniq.append(it)
    print(f"{len(uniq)} candidate articles since {args.min_date}", flush=True)

    ids, pieces, manifest = [], [], []
    for it in uniq:
        if len(ids) >= args.tokens:
            break
        code, html = curl(it["url"])
        if code != "200":
            continue
        px = Paras(); px.feed(html)
        text = "\n\n".join(px.paras)
        if len(text) < 800:
            continue
        t = tok.encode(text)[: args.per_article]
        ids.extend(t); pieces.append(tok.decode(t) + "\n\n")
        manifest.append({**it, "tokens": len(t)})
        print(f"  {it['date']} {it['title'][:60]}: {len(t)} tokens (total {len(ids)})", flush=True)
        time.sleep(1.5)
    ids = ids[: args.tokens]
    assert len(ids) == args.tokens, f"only {len(ids)} tokens gathered"
    open(os.path.join(out, "tr_news.txt"), "w", encoding="utf-8").write("".join(pieces))
    json.dump(ids, open(os.path.join(out, "tr_news_ids.json"), "w"))
    json.dump({"language": "tr", "kind": "news", "tokens": len(ids), "fetched": time.strftime("%Y-%m-%d"),
               "min_date": args.min_date, "sources": "aa.com.tr, bbc.com/turkce (RSS)", "use": "evaluation only",
               "articles": manifest}, open(os.path.join(out, "tr_news_manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"tr_news: {len(ids)} tokens from {len(manifest)} articles", flush=True)


if __name__ == "__main__":
    main()
