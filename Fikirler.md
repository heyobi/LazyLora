# 🚀 LazyLoRA Projesi Fikir Havuzu & Gelecek Deneyler Günlüğü

Bu belge, **Kimi K3 (2.78 Trilyon Parametreli MoE)** ve **LazyLoRA Out-of-Core Motoru** geliştirme sürecinde üretilen tüm yenilikçi mimari fikirleri, teorik analizleri ve model indirmesi tamamlandıktan sonra sırayla denenecek deney yol haritasını içerir.

---

## 📑 İÇİNDEKİLER & FİKİR SIRALAMASI

1. [Fikir 1: Çekirdek-Dışı (Out-of-Core) Akış ile Tam Parametreli (Full-Parameter) Normal Eğitim](#1-fikir-çekirdek-dışı-akış-ile-tam-parametreli-normal-eğitim)
2. [Fikir 2: 5TB Google Drive + Google Colab CPU Hibrit Eğitim Mimarisi](#2-fikir-5tb-google-drive--google-colab-cpu-hibrit-eğitim-mimarisi)
3. [Fikir 3: Büyük Modelde Sohbet (Inference) Darboğazı & Spekülatif Kod Çözme](#3-fikir-büyük-modelde-sohbet-darboğazı--spekülatif-kod-çözme)
4. [Fikir 4: Perplexity ve Olasılık Ağaçlarıyla Darboğazı Delme (Tree-Based Speculation)](#4-fikir-perplexity-ve-olasılık-ağaçlarıyla-darboğazı-delme)
5. [Fikir 5: Eşzamanlı Asenkron Hibrit Ağaç Spekülasyonu (22x Sohbet Hızlandırma)](#5-fikir-eşzamanlı-asenkron-hibrit-ağaç-spekülasyonu-22x-hızlandırma)
6. [Fikir 6: Eğitim Sonrası Kalıcı Ağırlık Kaynağı (Weight Merging / Zero-Overhead)](#6-fikir-eğitim-sonrası-kalıcı-ağırlık-kaynağı-weight-merging)
7. [Fikir 7: Döngüsel ReLoRA ile "Hileli Tam Parametreli Eğitim" (Iterative ReLoRA)](#7-fikir-döngüsel-relora-ile-hileli-tam-parametreli-eğitim)

---

## 1. Fikir: Çekirdek-Dışı Akış ile Tam Parametreli Normal Eğitim
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
* **Temel Soru:** Modeli 5TB Google Drive'a yükleyip, Colab'e bağlasak ve GPU almadan sadece ucuz CPU işlem birimleriyle eğitsek nasıl olurdu?
* **Fiziksel & Mimari Analiz:**
  * **Ağ Gecikmesi & FUSE:** Google Drive fiziksel bir disk değil, internet üzerinden HTTP dosya sistemidir (FUSE). Yerel SSD $2.000\text{ MB/s}$ okurken Drive $30\text{ MB/s}$ hız verir.
  * **Günlük 750 GB İndirme Kotası:** Kimi K3 akışında ilk 15 dakikada 750 GB sınırı aşılır ve Drive 24 saatliğine kilitlenir (`403 User Rate Limit Exceeded`).
  * **GEMM Hesaplama:** GTX 980 Ti'nin 2.816 CUDA çekirdeğine karşı Colab 2-CPU çekirdeği 35 kat daha yavaştır.
* **Sonuç:** Masanızdaki yerel PC (GTX 980 Ti + D: NVMe SSD), Colab + Drive kombinasyonundan **50 kat daha hızlı, kotasız ve ücretsizdir.**

---

## 3. Fikir: Büyük Modelde Sohbet Darboğazı & Spekülatif Kod Çözme
* **Temel Soru:** LoRA eğitimi saniyede 4-8 token işlerken, düz sohbet (inference) neden $0.005\text{ token/sn}$ (3.3 dk/kelime) hızına düşer?
* **Mekanizma (Batching Paradoksu):**
  * **Eğitimde:** 100 GB ağırlık 1 kez okunur ve aynı anda 512 kelimelik tüm paragraf paralel işlenir (Kelime başı maliyet: 195 MB).
  * **Sohbette:** Gelecekteki kelime bilinmediği için her 1 yeni kelime için 100 GB baştan okunur (Kelime başı maliyet: 100 GB).
* **Çözüm:** RAM/VRAM'de yaşayan 1B küçük bir taslak model saniyede 150 kelime üretir; 2.78T Kimi K3 tek bir SSD okumasında 5 kelimeyi birden doğrular.

---

## 4. Fikir: Perplexity ve Olasılık Ağaçlarıyla Darboğazı Delme
* **Temel Soru:** Küçük model tek bir tahmin çizgisi yerine olasılık/perplexity değerlerine göre tüm ihtimalleri ağaç olarak çıkarsa ve büyük model tek seferde en doğru dalı seçse darboğaz aşılır mı?
* **Mekanizma (Tree Attention / Medusa / EAGLE):**
  * Küçük model en olası 10 farklı cümle dalını (yaklaşık 25-30 kelimelik bir ağaç) üretir.
  * 2.78T Kimi K3, özel bir **"Ağaç Dikkat Maskesi (Tree Attention Mask)"** ile tek bir SSD geçişinde tüm dalları aynı anda puanlar.
  * Tek bir disk okumasında 1 kelime yerine **6 ila 10 kelime birden onaylanır.**

---

## 5. Fikir: Eşzamanlı Asenkron Hibrit Ağaç Spekülasyonu (22x Hızlandırma)
* **Temel Soru:** Kimi K3'ün 93 katmanı SSD'den okunurken geçen 180 saniyelik boşluk süresinde, VRAM'deki 1B model sürekli farklı tahmin dalları üretse hibrit hız ne olur?
* **Mühendislik & VRAM Hesabı:**
  * 1B Model (INT4/FP8): $\sim 0.9\text{ GB VRAM}$
  * K3 Aktif Katman Tamponu: $\sim 1.2\text{ GB VRAM}$
  * KV-Cache & Ağaç Matrisi: $\sim 1.1\text{ GB VRAM}$
  * **Toplam VRAM:** $\mathbf{3.2\text{ GB} / 6.0\text{ GB}}$ (GTX 980 Ti'ye rahatça sığar).
* **Hız Projeksiyonu:**
  * 180 saniyelik SSD okuma süresinde 1B model 18.000 tokenlik devasa bir ihtimaller ormanı (128 dal) hazırlar.
  * Kimi K3 tek geçişte en tutarlı **15-25 kelimeyi birden** kabul eder.
  * **Sohbet Hızı:** $0.005\text{ tok/s}$'den $\mathbf{0.11\text{ tok/s}}$'ye çıkar (**22 KAT HIZLANMA!**).
  * 30 kelimelik bir cevap 1.5 saat yerine **$\approx 4.5\text{ dakikada}$** alınır.

---

## 6. Fikir: Eğitim Sonrası Kalıcı Ağırlık Kaynağı (Weight Merging)
* **Temel Soru:** LazyLoRA ile eğitilen modeli sonradan temel ağırlıklara kalıcı olarak gömebilir miyiz?
* **Mekanizma:**
  * Eğitim bittiğinde 250 MB'lık LoRA matrisleri ($A$ ve $B$) ana modelin 1.56 TB'lık ağırlıklarına eklenir:
    $$W_{\text{yeni}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
* **Kazanımlar:**
  * Ortada ek bir LoRA dosyası kalmaz; saf, tek parça **Kimi-K3-Turkce** modeli oluşur.
  * Çıkarım (inference) anında ek LoRA gecikmesi 0 ms olur.
  * vLLM, Ollama, HuggingFace gibi tüm standart motorlarda doğrudan çalışır.

---

## 7. Fikir: Döngüsel ReLoRA ile "Hileli Tam Parametreli Eğitim"
* **Temel Soru:** Birden fazla LoRA'yı art arda eğitip her seferinde ana modele gömersek tam parametreli eğitimi taklit edebilir miyiz? LoRA boyutu büyür mü?
* **Mekanizma (ReLoRA Paradigm):**
  * **1. Döngü:** 250 MB LoRA ile Türkçe dilbilgisini eğit $\to$ $W_1 = W_0 + \Delta W_1$ olarak ana modele göm $\to$ **LoRA'yı sil/sıfırla!**
  * **2. Döngü:** Yeni $W_1$ üzerine yeni 250 MB LoRA tak $\to$ Teknik terimleri eğit $\to$ $W_2 = W_1 + \Delta W_2$ olarak göm $\to$ **LoRA'yı sil/sıfırla!**
  * **3. Döngü:** Yeni $W_2$ üzerine yeni 250 MB LoRA tak $\to$ Muhakeme/Matematik eğit $\to$ Göm.
* **Kritik Sonuç:**
  * LoRA boyutu her döngüde sıfırlandığı için **hiçbir zaman 250 MB'ı aşmaz.**
  * Ana model ağırlıkları ise çok boyutlu **tam dereceli (Full-Rank)** değişime uğrar. 40 TB disk harcamadan, tam eğitimin tüm gücü elde edilmiş olur.

---

## 🎯 MODEL İNDİKTEN SONRA DENEY VE UYGULAMA SIRALAMASI

- [ ] **Aşama 1:** LazyLoRA ile standart 2 epokluk Türkçe Instruction Eğitimi yapılması (`train_lazy_lora.sh`).
- [ ] **Aşama 2:** Eğitilen LoRA adaptörünün `merge_weights.py` ile ana Kimi K3 ağırlıklarına kalıcı olarak gömülmesi (Weight Merging).
- [ ] **Aşama 3:** Asenkron Ağaç Spekülasyonu (Tree Speculative Decoding) ile 1B model eşliğinde hızlı Türkçe sohbet testi yapılması.
- [ ] **Aşama 4:** ReLoRA döngüsü denenerek 2. ve 3. aşama uzmanlık eğitimlerinin ana modele sindirilmesi.
