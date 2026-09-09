# Paylaşım metinleri — kanıt koşusu (9 Eylül 2026)

Görseller: `docs/figures/proof_loss_tr.png` (kare, TR), `proof_loss_en.png` (kare, EN),
`proof_loss_wide.png` (geniş, iki panel). LinkedIn için kare kart, Medium/blog için geniş.

## LinkedIn — Türkçe

2,78 trilyon parametreli Kimi K3'ü, 7,6 GB RAM'i olan bir dizüstünde eğitmeye başladım.
Model belleğe sığmıyor: 1,56 TB'lık checkpoint bir USB harici diskte duruyor, her katman
sırayla diskten akıyor, RAM'de aynı anda tek katman kalıyor. Eğitilen tek şey 590 MB'lık
LoRA adaptörü.

Dün gece bunun gerçekten öğrendiğini gördüm. Aynı iki diziyi tur tur verdim:
0,909 → 0,500 → 0,157 ve 0,521 → 0,193. Bir adım 5,7 saat sürüyor, en yüksek bellek
kullanımı 4,7 GB.

Bu bir ezber testi, genelleme kanıtı değil; onu asıl koşudan sonra, modelin yayınından
sonra yazılmış Türkçe haber metninde önceden ilan ettiğim eşikle sınayacağım. Şu an 400
örneklik Türkçe talimat koşusu dönüyor, ekim başında bitiyor.

Yol boyunca çıkan ölçümler ayrı bir hikâye: uzman yönlendirmesi dilden çok alana duyarlı,
uzman yerelliği tek token'da gerçek ama eğitim batch'inde çöküyor. Kod ve ölçümler:
github.com/heyobi/LazyLora

## LinkedIn — İngilizce

I started training Kimi K3, a 2.78-trillion-parameter model, on a laptop with 7.6 GB of RAM.
The model does not fit: the 1.56 TB checkpoint sits on a USB hard disk, every layer streams
in turn, one layer at a time is resident. The only thing trained is a 590 MB LoRA adapter.

Last night it learned. Two fixed sequences, pass after pass: 0.909 → 0.500 → 0.157 and
0.521 → 0.193. One step takes 5.7 hours; peak memory 4.7 GB.

This is memorisation of five examples, which is exactly what a mechanism test should show.
Generalisation gets tested after the main run, on Turkish news published after the model's
release, against a threshold I registered before training.

Code and measurements: github.com/heyobi/LazyLora

## Tek cümlelik versiyon

"1,56 TB'lık bir modelin ağırlıkları USB diskte, RAM 7,6 GB, adım 5,7 saat — ve loss
0,909'dan 0,157'ye düştü."
