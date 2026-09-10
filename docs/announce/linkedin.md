# LinkedIn

**Written** 8 September 2026, revised 10 September 2026.
**Status:** neither version posted. When one is, this line gets the date and the link.

Where this draft and [`../numbers.md`](../numbers.md) disagree, that table names the source
and the source settles it.

Four texts: a long post in Turkish and in English, and a short version of each. The short
pair used to be a separate file, which meant two documents carrying the same numbers and a
note in each telling the reader to keep them in sync — the arrangement that let one wrong
cosine survive in five places at once. They are one file now. The numbers, the order and the
closing line are the same in all four.

Images: `docs/figures/proof_loss_tr.png` and `proof_loss_en.png` are the square cards, one
per language, for LinkedIn; `proof_loss_wide.png` is the wide two-panel version, for a blog.
Their titles say a LoRA adapter was trained, and their footers give the proof run's
5,5-5,8 sa / 5.5-5.8 h step and a 4,5-4,7 GB resident set rather than a single peak.

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

Rakamlar: 1024 token'lık bir adım 7,44 saat — asıl koşunun ilk üç adımı arasındaki iki
aralığın ortalaması (7,26 sa ve 7,62 sa). Tek başına ölçülen 1. adım 6 saat 59 dakikaydı
(ileri geçiş 3 sa 11 dk, geri geçiş 3 sa 48 dk); kanıt koşusunun daha kısa dizilerinde adım
5,5-5,8 saatti. Üçü aynı sayı değil. Kanıt koşusunda bellek 27 saat boyunca 4,5-4,7 GB'de kaldı — karttaki
rakam bu; asıl koşuda 4,0-4,7 GB, üstüne ~2,4 GB takas alanı. Hiçbiri tepe değeri değil,
projede görülen en yüksek değer daha eski bir 256 token'lık adımdaki 6,24 GB. Disk, 93
katmanın her birinde okunuyor — iki kez, bir ileri bir geri; ölçülen okuma hızı USB disk ve
NVMe gövdesi birlikte 110 MB/s.

En çok önemsediğim kısım hız değil, doğrulama. İleri geçişi kendi tarafımdaki koda güvenerek
değil, bağımsız bir C implementasyonuna (FareedKhan-dev'in kimi-k3-in-c projesi) karşı
katman katman karşılaştırarak kontrol ediyorum: 93 katmanın hepsi kosinüs benzerliği
0,9857 ve üzerinde eşleşiyor — en düşük satır 71. katmanda 0,985744 — çıkışta 0,99984. Bu
karşılaştırmanın tamamı artık depoda: `evidence/cmp93_en34_2026-09-06.log`, 93 satırın
hepsi, her katmanın kosinüsü, en büyük mutlak farkı ve okuduğu uzman sayısıyla. Benim
seçmediğim satırları da okuyabilirsiniz. Aynı referans motorun op düzeyindeki test verileri
de kendi Apache-2.0 lisansıyla depoya alındı, yani bu dış karşılaştırma artık her push'ta
kendiliğinden koşuyor. Gradyanlar ayrıca 93 katmanın dördünde sonlu
farkla doğrulandı: en kötü bağıl hata 9,1e-3, analitik türevin ~3e-4 olduğu iki yönde; MLA
katmanında 3,6e-3, geri kalanında 2,0e-3'ün altında. Bu, bütün modelin gradyan kontrolü
değil ve öyleymiş gibi sunmuyorum.

Bir mimari not, çünkü soran ilk kişi ben olurdum: yönlendirilen uzman yolunda adaptör
uzman başına değil, katman başına. Her katmanda tek bir rank-16 adaptör var ve o katmanın
896 uzmanının hepsi onu paylaşıyor. Uzman başına ayrı adaptör yaklaşık 26 milyar eğitilebilir
parametre demek olurdu (uzman başına ~320 k × 896 × 92 ≈ 2.6 × 10¹⁰, 16 bayttan ~420 GB);
bu makinede de, 400 örnekle de mümkün değil. Bedeli şu: adaptör, hangi uzman ateşlenirse
ona uygulanan ortak bir düzeltme öğreniyor, uzmana özel bir şey değil.

Şu an 400 örneklik Türkçe talimat koşusu dönüyor; ölçülen adım süresiyle yaklaşık 31 gün,
yani 9-11 Ekim civarı bitiyor. Türkçenin gerçekten iyileşip iyileşmediğini, eğitimde
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

The numbers: a 1024-token step takes 7.44 hours — the mean of the two intervals between the
main run's first three steps, 7.26 h and 7.62 h. Step 1 measured on its own was 6 hours
59 minutes (forward 3 h 11 m, backward 3 h 48 m); on the proof run's shorter sequences a
step was 5.5-5.8 hours, and the three are not interchangeable. Resident memory stayed at 4.5-4.7 GB for the 27-hour proof run on a
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
are in there too, which is the point of publishing it. That reference implementation's
op-level fixtures are vendored in the repository as well, under their own Apache-2.0
licence, so the external comparison runs on every push rather than only for someone who
cloned a second repository. The gradients are separately
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

A 400-example Turkish instruction run is going now — about 31 days at the measured step
time, so it finishes around 9-11 October. Whether the model's Turkish actually improved gets
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

## Short version (TR / EN)

The same post, cut to what fits above the fold. Same numbers, same order, same closing line
as the long versions above; if one changes, the other changes in the same edit.

### LinkedIn — Türkçe (kısa)

2,78 trilyon parametreli Kimi K3'ün üzerine, 7,6 GB RAM'i olan bir dizüstünde bir LoRA
adaptörü eğitiyorum. Model belleğe sığmıyor: 1,56 TB'lık checkpoint bir USB harici diskte
duruyor, her katman sırayla diskten akıyor, RAM'de aynı anda tek katman kalıyor. Modelin
2,78 trilyon parametresi baştan sona donuk. Eğitilen tek şey 590 MB'lık LoRA adaptörü —
Adam momentleriyle birlikte bellekte kalıcı olarak duran şey yaklaşık 1,8 GB. Uzman
yolunda adaptör katman başına: her katmanda tek bir rank-16 adaptör, o katmanın 896
uzmanının hepsi onu paylaşıyor.

Bu hafta döngünün doğru çalıştığını gördüm. Aynı iki diziyi tur tur verdim:
0,909 → 0,500 → 0,157 ve 0,521 → 0,193. Adaptör beş örneği ezberledi. Bu bir ezber testi:
ileri geçiş, geri geçiş ve AdamW zincirinin uçtan uca doğru çalıştığını gösteriyor,
modelin bir konuda iyileştiğini değil.

Rakamlar: 1024 token'lık bir adım 7,44 saat — asıl koşunun ilk üç adımı arasındaki iki
aralığın ortalaması (7,26 sa ve 7,62 sa). Tek başına ölçülen 1. adım 6 saat 59 dakikaydı:
ileri geçiş 3 sa 11 dk, geri geçiş 3 sa 48 dk. Kanıt koşusunun daha kısa dizilerinde adım
5,5-5,8 saatti; üçü aynı sayı değil. Bellek: kanıt koşusunda 27 saat boyunca 4,5-4,7 GB
— karttaki rakam bu — asıl koşuda 4,0-4,7 GB, üstüne ~2,4 GB takas alanı. Hiçbiri tepe
değeri değil; projede şimdiye kadar görülen en yüksek değer, daha eski bir 256 token'lık
adımdaki 6,24 GB. Ölçülen okuma hızı, USB disk ile NVMe gövdesi birlikte, 110 MB/s.

En çok önemsediğim kısım hız değil, doğrulama: ileri geçişi kendi koduma güvenerek değil,
bağımsız bir C implementasyonuna (FareedKhan-dev'in kimi-k3-in-c projesi) karşı katman
katman karşılaştırıyorum — 93 katmanın hepsi kosinüs 0,9857 ve üzerinde, en düşük satır
71. katmanda 0,985744, çıkışta 0,99984. Bu karşılaştırmanın tamamı depoda:
`evidence/cmp93_en34_2026-09-06.log`, 93 satırın hepsi. Benim seçmediğim satırları da
okuyabilirsiniz — yayımlamanın anlamı bu. Aynı referans motorun op düzeyindeki test
verileri de depoya alındı, kendi Apache-2.0 lisansıyla, yani bu dış karşılaştırma artık her
push'ta kendiliğinden koşuyor. Gradyanlar 93 katmanın dördünde sonlu farkla kontrol edildi;
en kötü bağıl hata 9,1e-3 ve analitik türevin ~3e-4 olduğu iki yönde çıkıyor.

Şu an 400 örneklik Türkçe talimat koşusu dönüyor: ölçülen adım süresiyle yaklaşık 31 gün,
yani 9-11 Ekim civarı bitiyor. Türkçenin gerçekten iyileşip iyileşmediğini, modelin
yayınından sonra yazılmış Türkçe haber metninde, koşu başlamadan 29 saat önce git'e
işlediğim bir eşikle sınayacağım. Sonuç olumsuz çıkarsa olumsuz olarak yazacağım. O
tarihin dayanağı bu makinenin saati ve kendi commit'im; bağımsız bir tanık değil, önceden
verilmiş bir söz.

Yol boyunca çıkan ölçümler ayrı bir hikâye: uzman yönlendirmesi dilden çok alana duyarlı,
uzman yerelliği tek token'da gerçek ama eğitim batch'inde çöküyor. Bu sayıların hepsi
yeniden hesaplanabilir — beş metnin uzman izleri `evidence/traces/` altında, depoda;
checkpoint'e ya da bu donanıma gerek yok, numpy ve birkaç saniye yetiyor. Kod, ölçümler ve
ham kayıtlar: github.com/heyobi/LazyLora

### LinkedIn — English (short)

I am training a LoRA adapter on Kimi K3, a 2.78-trillion-parameter model, on a laptop with
7.6 GB of RAM. The model does not fit: the 1.56 TB checkpoint sits on a USB hard disk,
every layer streams in turn, one layer at a time is resident. All 2.78 trillion base
parameters are frozen from the first step to the last. The only thing trained is a 590 MB
LoRA adapter — with its two Adam moments, about 1.8 GB stays permanently in RAM. On the
routed-expert path the adapter is per layer, not per expert: one rank-16 adapter shared by
all 896 experts of that layer.

This week I watched the loop work. Two fixed sequences, pass after pass: 0.909 → 0.500 →
0.157 and 0.521 → 0.193. The adapter memorised five examples. That is what a mechanism
test should show — forward, backward and AdamW are correct end to end — and it is not the
model getting better at anything.

The numbers: a 1024-token step takes 7.44 hours, the mean of the two intervals between the
main run's first three steps (7.26 h and 7.62 h). Step 1 measured on its own was 6 hours
59 minutes: forward 3 h 11 m, backward 3 h 48 m. On the proof run's shorter sequences a step
was 5.5-5.8 hours; the three are not interchangeable. Memory: the proof run held 4.5-4.7 GB
for 27 hours — that is the figure on the card — and the main run sits at 4.0-4.7 GB with
about 2.4 GB of swap on top. Neither is a peak; the highest figure ever recorded in this
project is 6.24 GB, on an earlier 256-token step. Measured read throughput is 110 MB/s
aggregate across the USB disk and the NVMe trunk.

The part I care about most is the verification. The forward pass is compared layer by layer
against an independent C implementation of the same model (FareedKhan-dev's kimi-k3-in-c):
all 93 layers agree at cosine 0.9857 or better — the lowest row is 0.985744 at layer 71 —
and 0.99984 at the output. The whole comparison is in the repository,
`evidence/cmp93_en34_2026-09-06.log`, all 93 rows, including the ones I did not choose to
quote. That reference implementation's op-level fixtures are vendored in the repository too,
under their own Apache-2.0 licence, so the external comparison now runs on every push rather
than only for someone who cloned a second repository. The gradients are checked by central
finite differences on four of the 93 layers — worst relative error 9.1e-3, in two directions
whose analytic derivative is about 3e-4.

This is memorisation of five examples, not generalisation. Generalisation gets tested after
the main run, on Turkish news published after the model's release, against a threshold I
committed to git 29 hours before training started — a promise made in public, timestamped
by this laptop's own clock rather than by anyone independent. The run needs about 31 days
at the measured step time, so it finishes around 9-11 October, and the number gets published
either way.

The routing measurements that came out of the same instrument are the part anyone can check
without the hardware: the expert traces for all five texts are in the repository under
`evidence/traces/`, and every number I quote from them recomputes with NumPy in seconds.
Code, measurements and the raw logs: github.com/heyobi/LazyLora

---

## One-line version

Türkçe:

> 1,56 TB'lık bir modelin ağırlıkları USB diskte, RAM 7,6 GB, bir adım 7,44 saat — ve
> adaptörün loss'u 0,909'dan 0,157'ye düştü. Modelin 2,78 trilyon parametresine hiç
> dokunulmadı.

English:

> 1.56 TB of weights on a USB disk, 7.6 GB of RAM, seven and a half hours a step — and the
> adapter's loss fell from 0.909 to 0.157. All 2.78 trillion base parameters were never
> touched.

---

## Posted length (TR / EN)

What actually goes into the LinkedIn box. LinkedIn folds a post after three lines and few
readers open a 700-word one, so this is the long version cut to what a professional
audience reads, with every number traceable to `docs/numbers.md` and everything else
delegated to the repository. Prepared 10 September 2026.

### Türkçe

2,78 trilyon parametreli Kimi K3'ün üzerine, 7,6 GB RAM'i olan bir dizüstünde bir LoRA adaptörü eğitiyorum.

Model belleğe sığmıyor: 1,56 TB'lık ağırlık dosyası bir USB harici diskte duruyor. Her katman sırayla diskten akıyor, RAM'de aynı anda tek katman kalıyor. Modelin 2,78 trilyon parametresi baştan sona donuk; eğitilen tek şey onun üstüne oturan 590 MB'lık adaptör. Dev bir sözlüğü baştan yazmıyorsunuz, kenarına not düşüyorsunuz.

Bu hafta döngünün gerçekten çalıştığını gördüm. Aynı iki diziyi tur tur verdim: loss 0,909 → 0,500 → 0,157 ve 0,521 → 0,193. Bu bir ezber testi; mekanizmanın uçtan uca doğru çalıştığını gösteriyor, modelin bir konuda iyileştiğini değil.

En çok önemsediğim kısım hız değil, doğrulama. İleri geçişi bağımsız bir C implementasyonuna (FareedKhan-dev'in kimi-k3-in-c projesi) karşı katman katman karşılaştırıyorum: 93 katmanın hepsi kosinüs 0,9857 ve üzerinde eşleşiyor. 93 satırlık karşılaştırmanın tamamı depoda, benim seçmediğim satırlar dahil. Motor her push'ta GitHub'ın makinesinde, model olmadan, sentetik bir kopya üzerinde baştan sona koşuyor.

Rakamlar: 1024 token'lık bir adım 7,44 saat, bellek 4-5 GB, disk 110 MB/s. Şu an 400 örneklik Türkçe talimat koşusu dönüyor, 9-11 Ekim civarı bitiyor. Türkçenin gerçekten iyileşip iyileşmediğini, koşu başlamadan önce git'e işlediğim bir eşikle sınayacağım. Sonuç olumsuz çıkarsa olumsuz yazacağım.

Depo, yapay zekâ yardımıyla yazıldığını ilk ekranında söylüyor; doğrulamanın bu kadar ağır olmasının sebebi de bu.

Kod, ölçümler ve ham kayıtlar: github.com/heyobi/LazyLora

### English

I am training a LoRA adapter on Kimi K3, a 2.78-trillion-parameter model, on a laptop with 7.6 GB of RAM.

The model does not fit: its 1.56 TB of weights sit on a USB hard disk. Every layer streams in turn and one layer is resident at a time. All 2.78 trillion base parameters stay frozen; the only thing trained is a 590 MB adapter on top of them. You are not rewriting the dictionary, you are writing in its margin.

This week I watched the loop actually work. The same two sequences, pass after pass: loss 0.909 → 0.500 → 0.157 and 0.521 → 0.193. That is a memorisation test. It shows the mechanism is correct end to end, not that the model got better at anything.

What I care about most is not the speed but the verification. I check the forward pass layer by layer against an independent C implementation of the same model (FareedKhan-dev's kimi-k3-in-c): all 93 layers agree at a cosine similarity of 0.9857 or better. The full 93-row comparison is in the repository, including the rows I did not choose to quote. The engine also runs end to end on GitHub's own machines on every push, on a synthetic stand-in, with no model required.

The numbers: a 1024-token step takes 7.44 hours, memory stays at 4-5 GB, the disk delivers 110 MB/s. A 400-example Turkish instruction run is going now and finishes around 9-11 October. Whether the model's Turkish actually improved gets tested against a threshold I committed to git before the run started. If the result is negative, I will publish it as negative.

The README says on its first screen that the repository was written with heavy AI assistance; that is also why the verification is as heavy as it is.

Code, measurements and raw logs: github.com/heyobi/LazyLora

### Türkçe, kısa (LinkedIn kutusuna giren sürüm, 10 Eylül)

Görseller: `docs/figures/proof_loss_tr.png` (kanıt koşusu kartı, başlığı "eğitiliyor") ve
`docs/figures/main_run_tr.png` (asıl koşu kartı: ilerleme, bir adımın anatomisi, tarihler).
Kartlar "8 GB RAM (7,6 GB kullanılabilir)" der: 8 GB takılı bellek, 7,6 GB işletim
sisteminin gördüğü; belgeler 7,6'yı kullanır, gönderi takılı belleği söyler. Yapay zekâ
açıklaması gönderide yok, README'nin ilk ekranında duruyor.

Kimi K3, 2,78 trilyon parametreli bir model. Ben onun üstüne, 8 GB RAM'li bir dizüstünde LoRA adaptörü eğitiyorum.

Model belleğe sığmıyor. 1,56 TB'lık ağırlıklar USB diskte duruyor; her katman sırayla diskten okunuyor, işi bitince atılıyor. Modelin kendisine dokunulmuyor, eğitilen tek şey 590 MB'lık adaptör.

Bu hafta döngünün doğru çalıştığını gördüm. Aynı diziyi tekrar tekrar verdim, loss 0,909'dan 0,157'ye indi. Bu bir ezber testi, mekanizmanın kanıtı. Modelin Türkçesi gerçekten iyileşti mi, onu asıl koşu bitince, önceden git'e işlediğim bir eşikle ölçeceğim. Olumsuz çıkarsa olumsuz yazacağım.

Asıl uğraştığım şey doğrulama oldu. İleri geçişi bağımsız bir C implementasyonuyla 93 katmanın hepsinde karşılaştırdım; karşılaştırmanın tamamı depoda. Motor her push'ta GitHub'ın makinesinde, model olmadan çalışıyor.

Bir eğitim adımı 7,44 saat sürüyor. 400 örneklik Türkçe koşu 3/100 adımda, 9-11 Ekim civarı bitiyor.

github.com/heyobi/LazyLora
