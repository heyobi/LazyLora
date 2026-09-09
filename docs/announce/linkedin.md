# LinkedIn

Two posts, Turkish and English. They are the long form of `../share_post.md` — same voice,
same shape, same closing, same one-line version at the end. Both files agree on the five
things the first drafts got wrong, and neither version may drift back:

1. The opening said "Kimi K3'ü eğitmeye başladım" / "I started training Kimi K3". The
   adapter is what is trained; the 2.78 trillion base parameters are frozen from the first
   step to the last. The new opening says so in the first sentence.
2. "Dün gece bunun gerçekten öğrendiğini gördüm" / "Last night it learned" became the
   adapter memorising five examples, which is what actually happened and is still a good
   sentence. The time reference is "bu hafta" / "this week" in both files: the proof run
   ran Tuesday-Wednesday 8-9 September, so "last weekend" is wrong and "last night" stops
   being true the day after it is written.
3. "5,7 saat" was the proof run's shorter sequences. The measured 1024-token step is
   6 saat 59 dakika. Both are here, attributed, never merged.
4. The verification paragraph used to end with "the full log is not in the repository yet".
   It is now, with the routing traces, and both posts say so — that sentence is the point
   of the post, not a footnote to it. It is also the paragraph that names a specific path,
   so it is the one that breaks loudest if the repository is still private.
5. Neither draft said where the adapter actually sits. On the routed-expert path it is one
   rank-16 adapter per layer, shared by all 896 experts of that layer, not one per expert —
   the first question anyone who works on MoE fine-tuning asks. Both files now say it, the
   long version with the cost of the choice spelled out and the short version in a sentence.

**Before posting.** The repository has to be public — the last line links it, and the
verification paragraph now tells the reader to go and open a specific file, so a private
repo turns the strongest paragraph into the most embarrassing one. Square card
`docs/figures/proof_loss_tr.png` (TR) / `proof_loss_en.png` (EN): these are the regenerated
ones, whose titles say a LoRA adapter was trained and whose footers read "5,5-5,8 sa /
5.5-5.8 h" for the proof-run step and "4,5-4,7 GB" for the resident set rather than a
single peak. Do not attach an older export. Confirm `evidence/cmp93_en34_2026-09-06.log`
and `evidence/traces/` are on the public clone before posting, since both paragraphs send
the reader there by name. No evaluation result exists yet and nothing below implies one. If
the post goes out after the main run finishes, add one line with the number — including if
it is negative. If a commenter raises the raw logs — `forward_loss_proof.jsonl` and
`forward_loss_main.jsonl` are two snapshots of one append-only file and both carry a
`"step": 1` row at loss 0.909084 — the full answer is in `hacker_news.md` under Prepared
replies; neither post below cites those files, so this is a comment to answer rather than a
line to add.

---

## Türkçe

2,78 trilyon parametreli Kimi K3'ün üzerine, 7,6 GB RAM'i olan bir dizüstünde,
çekirdek-dışı bir LoRA adaptörü eğitiyorum. Model belleğe sığmıyor: 1,56 TB'lık ağırlık
dosyası bir USB harici diskte duruyor, her katman sırayla diskten akıyor, RAM'de aynı anda
tek katman kalıyor. Modelin 2,78 trilyon parametresi baştan sona donuk. Eğitilen tek şey,
onun üstüne oturan 590 MB'lık ince bir ayar katmanı — adaptör.

Benzetme şu: dev bir sözlüğü baştan yazmıyorsunuz, kenarına kendi notlarınızı ekliyorsunuz.
Sözlük rafta duruyor, siz her seferinde yalnızca ihtiyacınız olan sayfayı indiriyorsunuz.

Bu hafta döngünün gerçekten çalıştığını gördüm. Beş Türkçe örneği iki paket diziye
sıkıştırdım ve aynı dizileri tur tur verdim: 0,909 → 0,500 → 0,157 ve 0,521 → 0,193.
Adaptör beş örneği ezberledi. Ezber tam olarak bu testten beklenen şey: mekanizmanın
uçtan uca doğru çalıştığını gösteriyor, modelin bir konuda iyileştiğini değil.

Rakamlar: 1024 token'lık tam bir adım 6 saat 59 dakika (ileri geçiş 3 sa 11 dk, geri geçiş
3 sa 48 dk), asıl koşunun 1. adımında ölçüldü; kanıt koşusunun daha kısa dizilerinde adım
5,5-5,8 saatti. Kanıt koşusunda bellek 27 saat boyunca 4,5-4,7 GB'de kaldı — karttaki
rakam bu; asıl koşuda 4,0-4,7 GB, üstüne ~2,4 GB takas alanı. Hiçbiri tepe değeri değil,
projede görülen en yüksek değer daha eski bir 256 token'lık adımdaki 6,24 GB. Disk, 93
katmanın her birinde okunuyor — iki kez, bir ileri bir geri; ölçülen okuma hızı USB disk ve
NVMe gövdesi birlikte 110 MB/s.

En çok önemsediğim kısım hız değil, doğrulama. İleri geçişi kendi yazdığım koda güvenerek
değil, bağımsız bir C implementasyonuna (FareedKhan-dev'in kimi-k3-in-c projesi) karşı
katman katman karşılaştırarak kontrol ediyorum: 93 katmanın hepsi kosinüs benzerliği
0,9857 ve üzerinde eşleşiyor — en düşük satır 71. katmanda 0,985744 — çıkışta 0,99984. Bu
karşılaştırmanın tamamı artık depoda: `evidence/cmp93_en34_2026-09-06.log`, 93 satırın
hepsi, her katmanın kosinüsü, en büyük mutlak farkı ve okuduğu uzman sayısıyla. Benim
seçmediğim satırları da okuyabilirsiniz. Gradyanlar ayrıca 93 katmanın dördünde sonlu
farkla doğrulandı: en kötü bağıl hata 9,1e-3, analitik türevin ~3e-4 olduğu iki yönde; MLA
katmanında 3,6e-3, geri kalanında 2,0e-3'ün altında. Bu, bütün modelin gradyan kontrolü
değil ve öyleymiş gibi sunmuyorum.

Bir mimari not, çünkü soran ilk kişi ben olurdum: yönlendirilen uzman yolunda adaptör
uzman başına değil, katman başına. Her katmanda tek bir rank-16 adaptör var ve o katmanın
896 uzmanının hepsi onu paylaşıyor. Uzman başına ayrı adaptör yaklaşık 26 milyar eğitilebilir
parametre demek olurdu (uzman başına ~320 k × 896 × 92 ≈ 2.6 × 10¹⁰, 16 bayttan ~420 GB);
bu makinede de, 400 örnekle de mümkün değil. Bedeli şu: adaptör, hangi uzman ateşlenirse
ona uygulanan ortak bir düzeltme öğreniyor, uzmana özel bir şey değil.

Şu an 400 örneklik Türkçe talimat koşusu dönüyor; ölçülen adım süresiyle yaklaşık 29 gün,
yani 8-9 Ekim civarı bitiyor. Türkçenin gerçekten iyileşip iyileşmediğini, eğitimde
görülmemiş ve modelin yayınından sonra yazılmış haber metninde, koşu başlamadan 29 saat
önce git'e işlediğim bir eşikle sınayacağım. Sonuç olumsuz çıkarsa olumsuz olarak
yazacağım — eşiği önceden sabitlemenin bütün anlamı bu. Dürüst olmak gerekirse bu tarihin
dayanağı da bu makinenin kendi saati ve deponun commit'i: bağımsız bir tanık değil,
önceden verilmiş bir söz.

Yol boyunca çıkan ölçümler ayrı bir hikâye: modelin uzman seçimi dilden çok alana duyarlı,
ve Türkçenin maliyeti mimaride değil, metni parçalara ayıran tokenizer'da. Bu ölçümlerin
hepsi yeniden hesaplanabilir — beş metnin uzman yönlendirme izleri depoda,
`evidence/traces/` altında. Checkpoint'e, GPU'ya, bu makineye gerek yok; numpy ve birkaç
saniye yetiyor.

Kod, ölçümler ve ham kayıtlar: github.com/heyobi/LazyLora

---

## English

I am training a LoRA adapter on Kimi K3 — 2.78 trillion parameters — out of core, on a
laptop with 7.6 GB of RAM. The model does not fit: the 1.56 TB of weights sit on a USB
hard disk, every layer streams in turn, one layer is resident at a time. All 2.78 trillion
base parameters are frozen from the first step to the last. The only thing trained is a
590 MB adapter that sits on top of them.

The analogy: you are not rewriting a vast dictionary, you are adding your own notes in the
margin. The dictionary stays on the shelf and you fetch one page at a time.

This week I watched the loop actually work. Five Turkish examples packed into two
sequences, the same sequences fed pass after pass: 0.909 → 0.500 → 0.157 and
0.521 → 0.193. The adapter memorised five examples. Memorisation is exactly what this test
should produce — it shows the mechanism is correct end to end, not that the model became
better at anything.

The numbers: a full 1024-token step takes 6 hours 59 minutes (forward 3 h 11 m, backward
3 h 48 m), measured on step 1 of the main run; on the proof run's shorter sequences a step
was 5.5-5.8 hours. Resident memory stayed at 4.5-4.7 GB for the 27-hour proof run on a
7.6 GB machine — that is the figure on the card — and the main run sits at 4.0-4.7 GB with
about 2.4 GB of swap on top. Neither is a peak; the highest figure recorded anywhere in the
project is 6.24 GB, on an earlier, shorter step. Every one of the
93 layers is read from disk twice per step, once going forward and once coming back, at a
measured 110 MB/s aggregate across the USB disk and the NVMe trunk.

The part I care about most is not the speed, it is the verification. I do not check the
forward pass by trusting my own code; I compare it layer by layer against an independent C
implementation of the same model (FareedKhan-dev's kimi-k3-in-c). All 93 layers agree at a
cosine similarity of 0.9857 or better — the lowest row is 0.985744 at layer 71 — and
0.99984 at the output. That whole comparison is now in the repository:
`evidence/cmp93_en34_2026-09-06.log`, all 93 rows, each with its cosine, maximum absolute
difference and the number of experts that layer read. The rows I did not choose to quote
are in there too, which is the point of publishing it. The gradients are separately
verified by central finite differences on four of the 93 layers: worst relative error
9.1e-3, in two directions whose analytic derivative is about 3e-4, 3.6e-3 on the MLA layer
and under 2.0e-3 elsewhere. It is not a whole-model gradient check and I do not present it
as one.

One architectural note, because I would ask it first myself: on the routed-expert path the
adapter is per layer, not per expert. Each layer has a single rank-16 adapter, shared by
all 896 experts of that layer. One adapter per expert would be roughly 26 billion trainable
parameters (about 320 k per expert × 896 × 92 ≈ 2.6 × 10¹⁰, some 420 GB at 16 bytes apiece)
— impossible on this machine and pointless with 400 examples. The price is that the
adapter learns one correction applied to whichever expert fires, rather than per-expert
specialisation.

A 400-example Turkish instruction run is going now — about 29 days at the measured step
time, so it finishes around 8-9 October. Whether the model's Turkish actually improved gets
tested on news text it has never seen, published after the model's release, against a
threshold I committed to git 29 hours before the run started. If the result is negative I
will publish it as negative. That is the entire point of fixing the threshold in advance —
though the timestamp behind it is this laptop's own clock and my own commit, so treat it as
a promise made in public rather than as an independent witness.

The measurements that fell out along the way are their own story: the model's choice of
expert tracks subject matter far more than language, and the cost of Turkish lives in the
tokenizer that cuts the text into pieces, not in the architecture. Those are the numbers
anyone can recompute — the expert-routing traces for all five texts are in the repository
under `evidence/traces/`, and reproducing the tables needs NumPy and a few seconds, no
checkpoint and none of this hardware.

Code, measurements and the raw logs: github.com/heyobi/LazyLora

---

## One-line version

Identical to the pair at the end of `../share_post.md`; if one changes, the other does.

Türkçe:

> 1,56 TB'lık bir modelin ağırlıkları USB diskte, RAM 7,6 GB, bir adım 6 saat 59 dakika —
> ve adaptörün loss'u 0,909'dan 0,157'ye düştü. Modelin 2,78 trilyon parametresine hiç
> dokunulmadı.

English:

> 1.56 TB of weights on a USB disk, 7.6 GB of RAM, six hours fifty-nine minutes a step —
> and the adapter's loss fell from 0.909 to 0.157. All 2.78 trillion base parameters were
> never touched.
