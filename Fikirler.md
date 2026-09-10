# 🚀 LazyLoRA Fikir Havuzu — ne denendi, ne bırakıldı, ne hiç denenmedi

Bu belge, **Kimi K3 (2,78 trilyon parametreli MoE)** ve **LazyLoRA çekirdek-dışı motoru** için, model henüz diske inerken, **27-28 Ağustos 2026'da** yazılmış dokuz fikri içerir. Fikirler o gün yazıldığı gibi duruyor; her birinin başına bugün ne olduğunu söyleyen bir **Durum** satırı eklendi (10 Eylül 2026). Bu bir plan değil, bir planın kaydıdır. Sıradaki gerçek işler [DEVAM.md](DEVAM.md)'dedir.

> **Uyarı — buradaki sayıların çoğu ölçüm değil, tahmindir.**
>
> Dokuz fikrin tamamı kağıt üstünde yazıldı. O tarihten bu yana biri ölçüldü, ikisi başka biçimde uygulandı, ikisi denenip bırakıldı, dördü hiç denenmedi. Her fikrin başına bir **Durum** satırı eklendi; ölçüme dönüşenler [Bulgular.md](Bulgular.md)'ye bağlandı. **Çelişki hâlinde ölçüm geçerlidir, buradaki tahmin değil.**
>
> Dosya, ölçülmemiş sayıları yüzünden silinmedi: bir projenin neyi düşünüp neyi denemediği de kaydın parçasıdır ve silmek yalnızca yanılmanın izini siler. Ama başlıklarda duran "22x", "hileli", "bedavaya" gibi ifadeler bir sonucu değil bir umudu anlatıyordu; başlıklardan çıkarıldılar. Sayılar, hangi varsayımdan çıktıkları yazılı olarak gövdede bırakıldı.
>
> **Donanım notu.** 1, 2 ve 5 numaralı fikirler GTX 980 Ti'li, `D:` diskli Windows/WSL makinesinde yazıldı. O makine artık yok. Proje 5 Eylül 2026'dan beri i7-7700HQ, 7,6 GB RAM, GTX 1050 (2 GB) ve USB diskteki checkpoint ile yürüyor. Eski donanıma göre yapılmış VRAM ve hız hesapları bu makinede geçerli değildir.
>
> **Durum etiketleri:** **ÖLÇÜLDÜ** (fikir deneye dönüştü; geçerli sayı Bulgular.md'de) · **BAŞKA BİÇİMDE UYGULANDI** (hedefe varıldı, buradaki yolla değil) · **DENENDİ, BIRAKILDI** (kod yazıldı, sonra terk edildi; `docs/attic/`) · **DENENMEDİ** (kağıt üstünde; buradaki sayılar tahmindir).

---

## 📑 İÇİNDEKİLER & FİKİR SIRALAMASI

1. [Fikir 1: Çekirdek-Dışı (Out-of-Core) Akış ile Tam Parametreli (Full-Parameter) Normal Eğitim](#1-fikir-çekirdek-dışı-akış-ile-tam-parametreli-normal-eğitim)
2. [Fikir 2: 5TB Google Drive + Google Colab CPU Hibrit Eğitim Mimarisi](#2-fikir-5tb-google-drive--google-colab-cpu-hibrit-eğitim-mimarisi)
3. [Fikir 3: Büyük Modelde Sohbet (Inference) Darboğazı & Spekülatif Kod Çözme](#3-fikir-büyük-modelde-sohbet-darboğazı--spekülatif-kod-çözme)
4. [Fikir 4: Perplexity ve Olasılık Ağaçlarıyla Darboğazı Delme (Tree-Based Speculation)](#4-fikir-perplexity-ve-olasılık-ağaçlarıyla-darboğazı-delme)
5. [Fikir 5: Eşzamanlı Asenkron Hibrit Ağaç Spekülasyonu](#5-fikir-eşzamanlı-asenkron-hibrit-ağaç-spekülasyonu)
6. [Fikir 6: Eğitim Sonrası Kalıcı Ağırlık Kaynağı (Weight Merging / Zero-Overhead)](#6-fikir-eğitim-sonrası-kalıcı-ağırlık-kaynağı-weight-merging)
7. [Fikir 7: Döngüsel ReLoRA (Iterative ReLoRA)](#7-fikir-döngüsel-relora-iterative-relora)
8. [Fikir 8: Uzman Akışını Sıralı Taramaya Çevirme (Asenkron Çift Tamponlama)](#8-fikir-uzman-akışını-sıralı-taramaya-çevirme-asenkron-çift-tamponlama)
9. [Fikir 9: Temel Ağırlık Okumadan LoRA Gradyanı](#9-fikir-temel-ağırlık-okumadan-lora-gradyanı)

---

## 1. Fikir: Çekirdek-Dışı Akış ile Tam Parametreli Normal Eğitim
**Durum (10 Eylül 2026): DENENMEDİ.** 7B/14B için verilen "1 adım 2-5 sn", "1-2 günde tam eğitim" ve "%100 uygulanabilir" birer tahmindir; bu projede 7B bir model hiç eğitilmedi. K3 tarafındaki yargı ise sonradan ölçümle desteklendi: gerçek yönlendiriciyle katman başına 592/896 uzman okunuyor ([Bulgular §12](Bulgular.md)), yani bu ölçekte LoRA'nın tek makul yol olduğu doğrulandı — ama bunu doğrulayan ölçüm LoRA'nın kendi ölçümüdür, tam parametreli eğitimin denenmesi değil.

* **Temel Soru:** Geliştirdiğimiz katman katman akış (Layer-wise Streaming) mimarisiyle LoRA yerine modelin tüm ağırlıklarını doğrudan (Full-Parameter) eğitebilir miyiz?
* **Çalışma Prensibi:**
  * İleri yayılımda katman ağırlıkları ($W_l$) diskten okunur, aktivasyonlar ($h_l$) D: SSD tamponuna yazılır.
  * Geriye yayılımda $W_l$, AdamW momentleri ($m_l, v_l$) okunur, tam gradyan ($\nabla W$) hesaplanır.
  * **Kritik Fark (Disk Write-Back):** Güncellenen yeni ağırlıklar ve optimizer durumları her adımda diske geri yazılır.
* **Uygulanabilirlik & Kıyaslama:**
  * **7B / 14B Modeller (Llama-3-8B, Qwen-2.5-7B/14B):** 6 GB GPU + SSD ile **%100 uygulanabilir ve çok hızlıdır** (1 adım $\approx 2-5\text{ sn}$, 1-2 günde tam eğitilir).
  * **2.78T Kimi K3:** 40+ TB SSD gereksinimi ve her adımda 400 GB yazma yükü nedeniyle 1 epok $\approx 120\text{ gün}$ sürer; bu boyutta LoRA tek mantıklı yoldur.

---

## 2. Fikir: 5TB Google Drive + Google Colab CPU Hibrit Eğitim Mimarisi
**Durum (10 Eylül 2026): DENENMEDİ.** "30 MB/sn", "35 kat", "50 kat": hiçbiri bu projede ölçülmedi. Günlük 750 GB kotası Google'ın yayımlanmış sınırıdır, bizim ölçümümüz değil. Kıyaslamadaki "GTX 980 Ti + D: NVMe SSD" makinesi artık yok. **Karıştırmayın:** Bulgular §17'deki "Fikir 2 bu donanımda uygulanmayacak" cümlesi bu fikri değil, o bölümün kendi 2 numaralı fikri olan sıcak uzman ikamet önbelleğini kasteder.

* **Temel Soru:** Modeli 5TB Google Drive'a yükleyip, Colab'e bağlasak ve GPU almadan sadece ucuz CPU işlem birimleriyle eğitsek nasıl olurdu?
* **Fiziksel & Mimari Analiz:**
  * **Ağ Gecikmesi & FUSE:** Google Drive fiziksel bir disk değil, internet üzerinden HTTP dosya sistemidir (FUSE). Yerel SSD $2.000\text{ MB/s}$ okurken Drive $30\text{ MB/s}$ hız verir.
  * **Günlük 750 GB İndirme Kotası:** Kimi K3 akışında ilk 15 dakikada 750 GB sınırı aşılır ve Drive 24 saatliğine kilitlenir (`403 User Rate Limit Exceeded`).
  * **GEMM Hesaplama:** GTX 980 Ti'nin 2.816 CUDA çekirdeğine karşı Colab 2-CPU çekirdeği 35 kat daha yavaştır.
* **Sonuç:** Masanızdaki yerel PC (GTX 980 Ti + D: NVMe SSD), Colab + Drive kombinasyonundan **50 kat daha hızlı, kotasız ve ücretsizdir.**

---

## 3. Fikir: Büyük Modelde Sohbet Darboğazı & Spekülatif Kod Çözme
**Durum (10 Eylül 2026): ÖLÇÜLDÜ → [Bulgular §6-7](Bulgular.md#6-deney-3-tam-93-katmanlı-278t-parametre-canlı-çıkarım-testi).** Deney 3 mekanizmayı doğruladı: tek bir token için **134,64 GB** okundu, **1654 saniye (27,5 dakika)** sürdü, sürenin **%99,0'ı** diskte geçti. Aşağıdaki "0,005 tok/sn" ve "kelime başı 100 GB" ölçümden önce yazılmış tahminlerdi; ölçülen değerler bunlardır. Bu fikrin akıbeti iyidir: fikir deneye dönüştü ve deney onu doğruladı.

* **Temel Soru:** LoRA eğitimi saniyede 4-8 token işlerken, düz sohbet (inference) neden $0.005\text{ token/sn}$ (3.3 dk/kelime) hızına düşer?
* **Mekanizma (Batching Paradoksu):**
  * **Eğitimde:** 100 GB ağırlık 1 kez okunur ve aynı anda 512 kelimelik tüm paragraf paralel işlenir (Kelime başı maliyet: 195 MB).
  * **Sohbette:** Gelecekteki kelime bilinmediği için her 1 yeni kelime için 100 GB baştan okunur (Kelime başı maliyet: 100 GB).
* **Çözüm:** RAM/VRAM'de yaşayan 1B küçük bir taslak model saniyede 150 kelime üretir; 2.78T Kimi K3 tek bir SSD okumasında 5 kelimeyi birden doğrular.

---

## 4. Fikir: Perplexity ve Olasılık Ağaçlarıyla Darboğazı Delme
**Durum (10 Eylül 2026): DENENDİ, BIRAKILDI → [`docs/attic/build_speculative_tree.py`](docs/attic/build_speculative_tree.py), [`docs/attic/continuous_deep_tree_engine.py`](docs/attic/continuous_deep_tree_engine.py).** "Tek okumada 6-10 kelime" ölçülmedi. Ölçülen tek sayı Deney 4'tür: 5 pozisyonun 3'ü kabul edildi (%60). Neden bırakıldığı [docs/attic/README.md](docs/attic/README.md)'dedir: spekülatif kod çözme *üretimi* hızlandırır, oysa proje eğitime kaydı ve bir eğitim adımında 93 katmanın hepsi zaten okunmak zorundadır.

* **Temel Soru:** Küçük model tek bir tahmin çizgisi yerine olasılık/perplexity değerlerine göre tüm ihtimalleri ağaç olarak çıkarsa ve büyük model tek seferde en doğru dalı seçse darboğaz aşılır mı?
* **Mekanizma (Tree Attention / Medusa / EAGLE):**
  * Küçük model en olası 10 farklı cümle dalını (yaklaşık 25-30 kelimelik bir ağaç) üretir.
  * 2.78T Kimi K3, özel bir **"Ağaç Dikkat Maskesi (Tree Attention Mask)"** ile tek bir SSD geçişinde tüm dalları aynı anda puanlar.
  * Tek bir disk okumasında 1 kelime yerine **6 ila 10 kelime birden onaylanır.**

---

## 5. Fikir: Eşzamanlı Asenkron Hibrit Ağaç Spekülasyonu
**Durum (10 Eylül 2026): DENENDİ, BIRAKILDI.** Fikrin taslak-ağaç yarısı yazıldı ve `docs/attic/`'e kaldırıldı; bütünü hiç koşmadı. Aşağıdaki VRAM bütçesi 6 GB'lık GTX 980 Ti'ye göredir; bu makinedeki GTX 1050'de 2 GB vardır, yani fikir yazıldığı hâliyle bu donanıma sığmaz.

* **Temel Soru:** Kimi K3'ün 93 katmanı SSD'den okunurken geçen 180 saniyelik boşluk süresinde, VRAM'deki 1B model sürekli farklı tahmin dalları üretse hibrit hız ne olur?
* **Mühendislik & VRAM Hesabı:**
  * 1B Model (INT4/FP8): $\sim 0.9\text{ GB VRAM}$
  * K3 Aktif Katman Tamponu: $\sim 1.2\text{ GB VRAM}$
  * KV-Cache & Ağaç Matrisi: $\sim 1.1\text{ GB VRAM}$
  * **Toplam VRAM:** $\mathbf{3.2\text{ GB} / 6.0\text{ GB}}$ (GTX 980 Ti'ye rahatça sığar).
* **Hız projeksiyonu (kağıt üstünde, ölçüm değil):**
  * 180 saniyelik SSD okuma süresinde 1B model 18.000 tokenlik bir ihtimaller ormanı (128 dal) hazırlar.
  * *Eğer* Kimi K3 tek geçişte 15-25 kelimeyi birden kabul ederse sohbet hızı $0,005$ tok/sn'den $0,11$ tok/sn'ye çıkar; oran 22'dir. **Bu bir bölme işlemidir.** 15-25 kabul varsayımı hiç sınanmadı; sınanan tek şey Deney 4'tür ve orada 5 pozisyonun 3'ü kabul edilmiştir (%60, Bulgular §8). 30 kelimelik cevap için verilen "$\approx 4,5$ dakika" da aynı sınanmamış varsayımdan türer.

---

## 6. Fikir: Eğitim Sonrası Kalıcı Ağırlık Kaynağı (Weight Merging)
**Durum (10 Eylül 2026): DENENMEDİ.** `merge_weights.py` diye bir dosya bu depoda yoktur. Fikir yazıldığından beri öğrenilen iki şey birleştirmeyi yazıldığından zor kılar: (i) yönlendirilen uzmanların adaptörü **katman başına tektir** ve o katmanın 896 uzmanının tamamınca paylaşılır, MoE gizli uzayında durur (3584 → 3072 → 3584, rank 16) — yani içine gömülecek tek bir $W_0$ yoktur ([Bulgular §19.1](Bulgular.md)); (ii) uzman ağırlıkları MXFP4 saklanır, dolayısıyla birleştirme "çöz → ekle → yeniden nicele → 1,56 TB'ı yeniden yaz" demektir ve $\frac{\alpha}{r}BA$'nın nicemleme adımının altında kalıp kalmadığı ölçülmedi. Aşağıdaki "250 MB'lık LoRA" da eskidir: adaptörün ölçülen boyutu **590 MB**'tır. "vLLM, Ollama, HuggingFace'te doğrudan çalışır" cümlesi 1,56 TB'lık bir checkpoint için hiçbir motorda doğru değildir.

* **Temel Soru:** LazyLoRA ile eğitilen modeli sonradan temel ağırlıklara kalıcı olarak gömebilir miyiz?
* **Mekanizma:**
  * Eğitim bittiğinde 250 MB'lık LoRA matrisleri ($A$ ve $B$) ana modelin 1.56 TB'lık ağırlıklarına eklenir:
    $$W_{\text{yeni}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
* **Kazanımlar:**
  * Ortada ek bir LoRA dosyası kalmaz; saf, tek parça **Kimi-K3-Turkce** modeli oluşur.
  * Çıkarım (inference) anında ek LoRA gecikmesi 0 ms olur.
  * vLLM, Ollama, HuggingFace gibi tüm standart motorlarda doğrudan çalışır.

---

## 7. Fikir: Döngüsel ReLoRA (Iterative ReLoRA)
**Durum (10 Eylül 2026): DENENMEDİ.** ReLoRA yayımlanmış bir yöntemdir (Lialin ve ark., 2023); buradaki katkı onu 2,78T'lik bir checkpoint'e uygulama önerisidir ve 6. fikrin bütün bedelini döngü sayısıyla çarpar — her döngü 1,56 TB'ın yeniden yazılması demektir. Başlıktan kaldırılan "hileli tam parametreli eğitim" bir sonuç değil bir benzetmeydi; aşağıdaki "tam eğitimin tüm gücü elde edilmiş olur" cümlesi ölçülmemiş bir iddiadır ve ReLoRA'nın kendi makalesi de tam eğitimle eşitlik iddia etmez.

* **Temel Soru:** Birden fazla LoRA'yı art arda eğitip her seferinde ana modele gömersek tam parametreli eğitimi taklit edebilir miyiz? LoRA boyutu büyür mü?
* **Mekanizma (ReLoRA Paradigm):**
  * **1. Döngü:** 250 MB LoRA ile Türkçe dilbilgisini eğit $\to$ $W_1 = W_0 + \Delta W_1$ olarak ana modele göm $\to$ **LoRA'yı sil/sıfırla!**
  * **2. Döngü:** Yeni $W_1$ üzerine yeni 250 MB LoRA tak $\to$ Teknik terimleri eğit $\to$ $W_2 = W_1 + \Delta W_2$ olarak göm $\to$ **LoRA'yı sil/sıfırla!**
  * **3. Döngü:** Yeni $W_2$ üzerine yeni 250 MB LoRA tak $\to$ Muhakeme/Matematik eğit $\to$ Göm.
* **Kritik Sonuç:**
  * LoRA boyutu her döngüde sıfırlandığı için **hiçbir zaman 250 MB'ı aşmaz.**
  * Ana model ağırlıkları ise çok boyutlu **tam dereceli (Full-Rank)** değişime uğrar. 40 TB disk harcamadan, tam eğitimin tüm gücü elde edilmiş olur.

---

## 8. Fikir: Uzman Akışını Sıralı Taramaya Çevirme (Asenkron Çift Tamponlama)
**Durum (10 Eylül 2026): BAŞKA BİÇİMDE UYGULANDI → commit `dd7f13c`, `lazy_lora/streaming/expert_streamer.py`.** Buradaki gözlem (592/896 uzman) ölçümdür, Deney 6'dan gelir ([Bulgular §12](Bulgular.md)). Önerilen çözüm ise yanlıştı: `request_prefetch_layer` *bir sonraki katmanın* uzmanlarını ister, o liste ise yönlendirici çalışmadan bilinemez — yönlendirme veriye bağlıdır, katmanlar arası ön-getirme düzeltilebilir bir şey değil. Fonksiyon bugün de boştur; artık bilerek boştur ve nedeni docstring'inde yazılıdır. Yerine katman *içinde* ön-getirme yazıldı (`stream_experts`): aktif küme ilk uzmana dokunulmadan bilindiği ve zaten disk sırasına dizildiği için okuyucu iş parçacığı `depth` kadar önde gidebiliyor.

* **Temel Gözlem (Deney 6'dan):** Gerçek yönlendirici ağırlıkları devreye girdiğinde bir katmanda **592/896 uzman** okunuyor. Yani "sadece 16 uzman okunur" avantajı **token başına** geçerli; 127 tokenlik bir batch'te uzmanların üçte ikisine, 512 tokenlik bir batch'te pratikte tamamına dokunuluyor.
* **Bundan Çıkan Sonuç:** Madem neredeyse tüm uzmanlar okunuyor, bu bir **rastgele erişim değil, sıralı tarama** olmalıdır. Ölçülen `~30 MB/sn`, diskin sıralı hızının (`~150 MB/sn`) beşte biridir.
* **Darboğazın Mekaniği:** Her uzman için 6 ayrı `mmap` okuması yapılıyor (`w1/w2/w3` × `packed/scale`), aralarda CPU'da MXFP4 çözülüyor ve SiTU hesaplanıyor. Disk, CPU çalışırken **boş bekliyor**; CPU da disk okurken boş bekliyor.
* **Çözüm:** `expert_streamer.request_prefetch_layer` şu an boş bir `no-op`. Arka planda çalışan bir okuyucu iş parçacığı + çift tamponlama ile, GPU/CPU $E_i$ uzmanını hesaplarken disk $E_{i+1}$'i okur.
  * Uzmanlar zaten **disk ofsetine göre sıralı** isteniyor (bkz. `sort_by_disk_order`), dolayısıyla ön-getirme kafa hareketi eklemez.
  * **Beklenen kazanç 3-5 kat idi; hâlâ ölçülmedi.** 6 Eylül profilinde katman içi etkin hız 61 MB/sn'dir — diskin sıralı hızının yaklaşık yarısı — ve okuyucunun hesap sırasında boş kaldığı da orada yazılıdır (Bulgular §16.5). Üç aşamalı boru hattı (oku → çöz → hesapla) hâlâ açık iştir.
* **Genişletme:** Aynı boru hattı, uzmanları okurken MXFP4 çözümünü ayrı bir iş parçacığında yapabilir (üç aşamalı pipeline: oku → çöz → hesapla).

---

## 9. Fikir: Temel Ağırlık Okumadan LoRA Gradyanı
**Durum (10 Eylül 2026): BAŞKA BİÇİMDE UYGULANDI → commit `6bfa478`.** Fikrin matematiği doğrudur: $\nabla A$ ve $\nabla B$ gerçekten $W_0$ içermez. Ama hedefe — geri geçişte uzmanları ikinci kez okumamaya — yaklaşıklama yapmadan varıldı: ileri geçiş her katmanın yönlendirilmiş uzman toplamını aktivasyonların yanına yazıyor, geri geçiş onu okuyor, uzmanlar katman başına yalnızca bir kez süpürülüyor. Böylece aşağıda bedel olarak sayılan "yanlılık derinlikle birikir" hiç ödenmedi. Fikir yine de tümüyle ölü değildir: aktivasyon yazma yolu (katman başına 0,9 MB) hâlâ süpürmenin tamamından kurtulmanın tek yolu olarak durur, bedeliyle birlikte.

* **Temel Soru:** Geri geçiş, LoRA gradyanlarını hesaplamak için uzmanların ağırlıklarını diskten **ikinci kez** okuyor. Bu şart mı?
* **Matematiksel Gözlem:** LoRA gradyanları donmuş temel ağırlık $W_0$'ı **içermez**:

$$\nabla B = s\,\big(dy^\top (xA^\top)\big), \qquad \nabla A = \big(s\,dy\,B\big)^\top x$$

  Yani $\nabla A$ ve $\nabla B$ için yalnızca girdi aktivasyonu $x$ ve yukarıdan gelen gradyan $dy$ yeterlidir. $W_0$ sadece gradyanı bir alt katmana taşırken gerekir:

$$dx = dy\,W_0 + dh\,A$$

* **Uygulama:** İleri geçişte katman başına $h_{\text{latent}}$ diske yazılır (127×3584 bf16 = **0,9 MB**). Geri geçişte 7,6 GB'lık uzman okuması, 0,9 MB'lık aktivasyon okumasıyla değiştirilir.
* **Bedeli:** Gradyanın uzmanların donmuş ağırlıkları üzerinden alt katmanlara akışı yaklaşıklanır; LoRA'nın kendi gradyanları **tam kalır**. Yanlılık derinlikle birikir, yani en alt katmanların adaptörleri daha az doğru eğitilir.
* **Kazanç (eski makineye ait projeksiyon):** adım süresi ~13,4 saatten ~7,7 saate (%43 azalma). **Bu sayı iki kez eskimiştir** — hem mekanik diskli WSL makinesine aittir, hem de yukarıdaki `6bfa478` öncesine. Bu makinede ölçülen adım temposu, asıl koşunun 1-3. adımları arasında **yaklaşık 7,4 saattir** (aralıklar 7,26 sa ve 7,62 sa). Sık alıntılanan **6 sa 59 dk 41 sn** yalnızca 1. adımın kendi ölçümüdür (ileri 3 sa 11 dk 34 sn, geri 3 sa 48 dk 07 sn; DEVAM.md "ŞU AN") ve üç adımlık tempo yerine kullanılamaz.

---

## 🎯 PLANIN AKIBETİ (10 Eylül 2026)

Aşağıdaki dört aşama, model indikten sonra sırayla denenmek üzere yazılmıştı. Kutuların hiçbiri hiç işaretlenmedi; işaretlenmesi gerekenler ise başka türlü oldu. Doğrusu şudur:

| Plandaki aşama | Bugünkü durum |
|---|---|
| **Aşama 1** — Türkçe instruction eğitimi | **Koşuyor.** 400 Dolly-tr örneği, 100 adım; 9 Eylül 2026 12:53'te başladı, ölçülen adım temposuyla (~yaklaşık 7,4 sa) yaklaşık 31 günde, **9-11 Ekim** dolaylarında biter. Plandaki "2 epok" değil, bir epoğun altında. |
| **Aşama 2** — `merge_weights.py` ile birleştirme | **Yapılmadı; betik yok.** Neden yazıldığından zor olduğu 6. fikrin Durum satırındadır. |
| **Aşama 3** — Ağaç spekülasyonu ile hızlı sohbet | **Bırakıldı.** Proje çıkarımdan eğitime kaydı; kodlar `docs/attic/`'tedir. |
| **Aşama 4** — ReLoRA döngüsü | **Denenmedi**, ve Aşama 2'ye bağlıdır. |

Sıradaki gerçek işler bu listede değil, [DEVAM.md](DEVAM.md)'dedir.
