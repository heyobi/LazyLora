# Paylaşım metinleri — kanıt koşusu (9 Eylül 2026)

Görseller: `docs/figures/proof_loss_tr.png` (kare, TR), `proof_loss_en.png` (kare, EN),
`proof_loss_wide.png` (geniş, iki panel). LinkedIn için kare kart, Medium/blog için geniş.

**Paylaşmadan önce.** Kartlar yenilendi: başlıkları artık eğitilen şeyin LoRA adaptörü
olduğunu söylüyor, alt satırları kanıt koşusunun adımı için "5,5-5,8 sa / 5.5-5.8 h" ve
bellek için tek bir tepe değeri yerine "4,5-4,7 GB" yazıyor. Eski bir dışa aktarımı
kullanmayın. Metinler depoya link veriyor ve doğrulama paragrafı okuyucuyu doğrudan bir
dosyaya yolluyor; depo herkese açık değilse ya link satırı çıkar ya da paylaşım bekler.
`evidence/cmp93_en34_2026-09-06.log` ile `evidence/traces/`'in herkese açık klonda
gerçekten durduğunu paylaşmadan önce doğrulayın — iki metnin de en güçlü paragrafı
okuyucuyu oraya yolluyor. Değerlendirme sonucu henüz yok ve aşağıdaki hiçbir cümle
olduğunu ima etmiyor. Yorumlarda ham kayıt dosyalarına dair bir itiraz gelirse
(`forward_loss_proof.jsonl` ile `forward_loss_main.jsonl` tek bir dosyanın iki anı, ikisinde
de 0,909084'lük bir `"step": 1` satırı var), uzun cevap
`docs/announce/hacker_news.md`'de hazır duruyor.

Bu iki metin `docs/announce/linkedin.md`'deki uzun sürümlerin kısaltılmışı; sayılar,
sıralama ve kapanış cümlesi ikisinde de aynı olmak zorunda.

## LinkedIn — Türkçe

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

Rakamlar: 1024 token'lık tam bir adım 6 saat 59 dakika — ileri geçiş 3 sa 11 dk, geri geçiş
3 sa 48 dk; asıl koşunun 1. adımında ölçüldü. Kanıt koşusunun daha kısa dizilerinde adım
5,5-5,8 saatti, ikisi aynı sayı değil. Bellek: kanıt koşusunda 27 saat boyunca 4,5-4,7 GB
— karttaki rakam bu — asıl koşuda 4,0-4,7 GB, üstüne ~2,4 GB takas alanı. Hiçbiri tepe
değeri değil; projede şimdiye kadar görülen en yüksek değer, daha eski bir 256 token'lık
adımdaki 6,24 GB. Ölçülen okuma hızı, USB disk ile NVMe gövdesi birlikte, 110 MB/s.

En çok önemsediğim kısım hız değil, doğrulama: ileri geçişi kendi koduma güvenerek değil,
bağımsız bir C implementasyonuna (FareedKhan-dev'in kimi-k3-in-c projesi) karşı katman
katman karşılaştırıyorum — 93 katmanın hepsi kosinüs 0,9857 ve üzerinde, en düşük satır
71. katmanda 0,985744, çıkışta 0,99984. Bu karşılaştırmanın tamamı depoda:
`evidence/cmp93_en34_2026-09-06.log`, 93 satırın hepsi. Benim seçmediğim satırları da
okuyabilirsiniz — yayımlamanın anlamı bu. Gradyanlar 93 katmanın dördünde sonlu farkla
kontrol edildi; en kötü bağıl hata 9,1e-3 ve analitik türevin ~3e-4 olduğu iki yönde
çıkıyor.

Şu an 400 örneklik Türkçe talimat koşusu dönüyor: ölçülen adım süresiyle yaklaşık 29 gün,
yani 8-9 Ekim civarı bitiyor. Türkçenin gerçekten iyileşip iyileşmediğini, modelin
yayınından sonra yazılmış Türkçe haber metninde, koşu başlamadan 29 saat önce git'e
işlediğim bir eşikle sınayacağım. Sonuç olumsuz çıkarsa olumsuz olarak yazacağım. O
tarihin dayanağı bu makinenin saati ve kendi commit'im; bağımsız bir tanık değil, önceden
verilmiş bir söz.

Yol boyunca çıkan ölçümler ayrı bir hikâye: uzman yönlendirmesi dilden çok alana duyarlı,
uzman yerelliği tek token'da gerçek ama eğitim batch'inde çöküyor. Bu sayıların hepsi
yeniden hesaplanabilir — beş metnin uzman izleri `evidence/traces/` altında, depoda;
checkpoint'e ya da bu donanıma gerek yok, numpy ve birkaç saniye yetiyor. Kod, ölçümler ve
ham kayıtlar: github.com/heyobi/LazyLora

## LinkedIn — İngilizce

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

The numbers: a full 1024-token step takes 6 hours 59 minutes (forward 3 h 11 m, backward
3 h 48 m), measured on step 1 of the main run. On the proof run's shorter sequences a step
was 5.5-5.8 hours; the two are not interchangeable. Memory: the proof run held 4.5-4.7 GB
for 27 hours — that is the figure on the card — and the main run sits at 4.0-4.7 GB with
about 2.4 GB of swap on top. Neither is a peak; the highest figure ever recorded in this
project is 6.24 GB, on an earlier 256-token step. Measured read throughput is 110 MB/s
aggregate across the USB disk and the NVMe trunk.

The part I care about most is the verification. The forward pass is compared layer by layer
against an independent C implementation of the same model (FareedKhan-dev's kimi-k3-in-c):
all 93 layers agree at cosine 0.9857 or better — the lowest row is 0.985744 at layer 71 —
and 0.99984 at the output. The whole comparison is in the repository,
`evidence/cmp93_en34_2026-09-06.log`, all 93 rows, including the ones I did not choose to
quote. The gradients are checked by central finite differences on four of the 93 layers —
worst relative error 9.1e-3, in two directions whose analytic derivative is about 3e-4.

This is memorisation of five examples, not generalisation. Generalisation gets tested after
the main run, on Turkish news published after the model's release, against a threshold I
committed to git 29 hours before training started — a promise made in public, timestamped
by this laptop's own clock rather than by anyone independent. The run needs about 29 days
at the measured step time, so it finishes around 8-9 October, and the number gets published
either way.

The routing measurements that came out of the same instrument are the part anyone can check
without the hardware: the expert traces for all five texts are in the repository under
`evidence/traces/`, and every number I quote from them recomputes with NumPy in seconds.
Code, measurements and the raw logs: github.com/heyobi/LazyLora

## Tek cümlelik versiyon

Türkçe:

> 1,56 TB'lık bir modelin ağırlıkları USB diskte, RAM 7,6 GB, bir adım 6 saat 59 dakika —
> ve adaptörün loss'u 0,909'dan 0,157'ye düştü. Modelin 2,78 trilyon parametresine hiç
> dokunulmadı.

İngilizce:

> 1.56 TB of weights on a USB disk, 7.6 GB of RAM, six hours fifty-nine minutes a step —
> and the adapter's loss fell from 0.909 to 0.157. All 2.78 trillion base parameters were
> never touched.

Aynı iki cümle `docs/announce/linkedin.md`'nin sonunda da duruyor; biri değişirse öteki de
değişir.
