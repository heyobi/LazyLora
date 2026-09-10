# 🔄 DEVİR BELGESİ — LazyLoRA

## ŞU AN (10 Eylül 2026, güncel tutulur)

**Durum:** Motor uçtan uca çalışıyor ve doğrulanmış: ileri geçiş 93 katmanda C referansıyla
eşleşiyor (§13, Bulgular §17.1 — 93 satırın hepsi kosinüs 0.9857 ve üzeri; en düşük satır
0.985744 katman 71, en kötü kuşak 68-72, çıkış katmanı 0.999840), geri geçiş dört katmanda
sonlu farkla doğrulandı (§11; en kötü bağıl hata 9.1e-3, katman 1). **Eğitim döngüsü
kanıtlandı** (Bulgular §18): 5 örneklik kanıt koşusunda aynı dizinin loss'u tur tur düştü
(A: 0.909 → 0.500 → 0.157, B: 0.521 → 0.193). Bu ezberdir; ileri → geri → AdamW →
checkpoint döngüsünün doğruluğunu kanıtlar, genellemeyi değil. Kanıt koşusunun adımı
5.5-5.8 saatti ama iki paket dizisi ~541'er token'lıktı; asıl koşunun tam 1024'lük dizisiyle
karıştırılmamalı. Beş metinlik yönlendirme ölçümü bitti (Bulgular §16-17) ve **izler artık
depoda** (aşağıda). Taban değerlendirmesi ve eşik sabit (§16.1). Hedef: Türkçe konuşan Kimi.

**Kanıt paketi depoda: `evidence/`** (6.2 MB, 25 dosya + `SHA256SUMS`; Bulgular §19). 9
Eylül'de kondu. İçindekiler:
- `evidence/traces/` — beş yönlendirme izi (`zh_paragraph`, `en_paragraph`, `tr_paragraph`,
  `tr_news`, `code_python`), 92 MoE katmanının tamamı, her biri `trace.bin` + `trace.json` +
  `analysis.json` + `analysis.md`; toplam 5.669.776 bayt yönlendirme kaydı.
- `evidence/cmp93_en34_2026-09-06.log` — 93 katmanlık C karşılaştırması (98 satır, toplam
  2869 s, 426.59 GB okuma).
- `evidence/forward_loss_main.jsonl`, `forward_loss_proof.jsonl` — adım başına ham loss ve
  Unix zaman damgası; belgelerdeki bütün adım süreleri bunlardan türer.
- `evidence/run_manifest.json` — koşan işin manifesti, yollar yer tutucu.

Önemi: ölçüm notundaki **her yönlendirme tablosu** artık okuyanın kendi dizüstünde
yeniden hesaplanabilir — `scripts/analyze_trace.py <iz_dizini>`, checkpoint yok, GPU yok,
saniyeler. Beş metnin beşi de bu çalışma için yazıldı (haber üslubundaki Türkçe paragraf
fındık üretimi üzerinedir, hiçbir yayından alınmadı), o yüzden manifestlerde metin ve token
id'leri açık duruyor. **Depoda olmayan, açıkça öyle yazılan:** 1.56 TB checkpoint, C
motorunun katman katman dökümü, paketlenmiş NVMe gövdesi, 1.8 GB'lık eğitim
checkpoint'leri, sonlu fark logu (harness terminale yazıyor; §11'deki rakamlar oradan) ve
`profile_{128,512,1024}` yardımcı izleri.

**Koşan iş:** asıl koşu (`LazyLora_Workspace/run_main.sh`, 9 Eylül 12:53):
`datasets/dolly_tr_400.jsonl` (400 Dolly-tr örneği → 154 paket dizi ≤1024 token, 78k
eğitilen token), 100 adım (0.65 epoch), lr 5e-4 tepe (kanıt koşusu 1e-3'te kararlıydı),
warmup 5, kosinüs, istem maskeli, her 5 adımda checkpoint
(`/mnt/nvme/lazylora/checkpoints/lazy_lora_step_NNNNN.pt`, son 3 tutulur).

**Ölçülen adım (adım 1, 9 Eylül):** **6 sa 59 dk 41 sn** — ileri 3 sa 11 dk 34 sn
(123.6 s/katman, 93 katman), geri 3 sa 48 dk 07 sn (147.2 s/katman). Bu hızla 100 adım
≈ **31 gün** (ilk üç adımın ölçülen temposu 7,44 sa: aralıklar 7,26 ve 7,62 sa),
bitiş **~9-11 Ekim 2026**; canlı rakam
`LazyLora_Workspace/run_manifest.json`'da. **Okuma hızı üç ayrı sayıdır, karıştırma:**
(i) 110 MB/s **toplam**, tek ölçülen uçtan uca rakam — 8 sa 06 dk 57 sn'de
3.219.659.335.955 bayt (`/proc` okuma sayacı), USB diskteki uzmanlar ile NVMe gövdesi
birlikte; (ii) 61 MB/s tek katman süpürmesi içindeki **etkin** hız (14.5 GB / 238 s,
Bulgular §16.5); (iii) 115 MB/s USB kutusunun **kendi sıralı testi** — cihazın özelliği,
motorun ölçümü değil. RSS 4.0-4.7 GB, takas da kullanımda; bu motorun gördüğü en yüksek
RSS 6.24 GB'dır ve daha eski bir 256 token'lık adımda ölçüldü (§17). Bekçi
(`scripts/watchdog.py`, systemd `lazylora-watchdog.timer`) 15 dk'da bir telefona
ilerleme/uyarı gönderir, ölürse checkpoint'ten devam ettirir, USB disk düşerse yeniden
bağlar. Kanıt koşusunun checkpoint'leri `checkpoints/proof_dolly5/`, loss'ları
`forward_loss.jsonl.proof`.

**Determinizm kontrolü teyit edildi:** asıl koşunun 1. adım loss'u kanıt koşusununkini altı
ondalıkla yeniden üretti (0.909084). Doğrulama yöntemi: `dolly_tr_400.jsonl` ile
`dolly_tr_proof.jsonl` ayrı ayrı ayrıştırılıp ilk beş kayıt karşılaştırıldı — birebir eşit;
`scripts/build_train_set.py` kanıt dosyasını aynı seçimin `picked[:5]`'i olarak yazıyor
(satır 80-82). İki koşu da sıfır ilklendirilmiş adaptörle başladığı için aynı sayıyı vermek
zorundaydı (Bulgular §18.1).

**Adaptörün şekli (bir yerde yazılı olsun):** yönlendirilen uzmanların LoRA'sı katman başına
**tek** rank-16 adaptördür ve o katmanın **896 uzmanının tamamı** tarafından paylaşılır, MoE
gizli uzayında (3584 → 3072 → 3584) — `lazy_lora/trainer/lazy_trainer.py:130`. Uzman başına
adaptör değildir; öyle olsa ~2.6 × 10¹⁰ eğitilebilir parametre ederdi. Ablasyon gelecek iş
(ölçüm notu §11, Bulgular §19.1).

**Açık kaynak paketi yazıldı, quickstart HİÇ KOŞTURULMADI:** `LICENSE`, `NOTICE`,
`docs/LICENSES.md`, `docs/QUICKSTART.md`, `scripts/quickstart.sh`,
`scripts/make_tiny_model.py`, `docs/announce/`, `docs/measurement_note.md` (v1.2),
`docs/traces/README.md`, `CITATION.cff`, `CONTRIBUTING.md`,
`.github/workflows/quickstart.yml`, `pyproject.toml` ve `evidence/` depoda.
**`scripts/quickstart.sh` bir kez bile koşturulmadı**: `docs/QUICKSTART.md`'deki her süre,
bellek ve beklenen çıktı rakamı koddan okunarak yazıldı, ölçüm değil (makine dolu).
Makine boşaldığında ilk iş budur: koştur, gerçek rakamlarla QUICKSTART'ı düzelt.
`scripts/export_traces.py` de hiç koşturulmadı ve artık gerekmiyor: izler, yalnızca bu
makinenin dosya yolları yer tutucuyla değiştirilerek `evidence/traces/`'e kondu. O dosyanın
başındaki "telifli haber metni" uyarısı beş metnin gerçek kaynağı bilinmeden yazılmıştı
ve yanlıştır.

**Sıradaki iş — checkpoint gerektirmeyenler** (koşan eğitime dokunmadan yapılabilir):
1. Ölçüm notundaki † işaretli iki rakamı (katman 1-8 dil imzası sınırı; eşit uzunlukta
   yoğunlaşma kontrolü) `evidence/traces/` üzerinde `scripts/analyze_trace.py --prefix` ile
   yeniden hesaplayıp sonucu Bulgular'a yaz, işareti kaldır. Checkpoint gerekmez, saniyeler
   sürer — ama yine de python çalıştırır, makine boşken yap.
2. `scripts/export_traces.py` başlığındaki yanlış telif uyarısını düzelt.
3. Ön-kayıt commit'ine (`85af2a8`, 8 Eylül 07:54:49) açıklamalı git etiketi koy. GitHub'da
   ikinci bir zaman damgası verir; **bağımsız bir damga değildir** — eşiklerin tarihi hâlâ
   bu makinenin saatine ve depo commit tarihlerine dayanır ve belgelerde zayıflık olarak
   böyle yazılmalıdır.
4. `Bulgular.md` içindekiler listesi 8. bölümde kalmış; 9-19 eklenmeli.

**Nasıl bakılır:** `bash scripts/status.sh` · `cat LazyLora_Workspace/status.txt` ·
`tail LazyLora_Workspace/forward_loss.jsonl` (adım başına loss; not: loss ileri geçiş
sonunda yazılır, adımın geri geçişi 3 sa 48 dk daha sürer) ·
`tr '\r' '\n' < LazyLora_Workspace/main_run_2026-09-09.raw | tail`.

**Bittiğinde ne yapılacak:** `eval_perplexity.py --corpus tr_news,tr_wiki,en_wiki
--checkpoint /mnt/nvme/lazylora/checkpoints/lazy_lora_step_00100.pt` ile eşiği (§16.1:
haber bpb 0.455 → ≤0.441, EN wiki ≤0.198) sına; sonucu olumlu/olumsuz olduğu gibi
Bulgular'a ve README'ye yaz. Ara kontrol: adım 50 checkpoint'inde yalnız `tr_news`
(≈3 saat) koşulabilir. Loss eğrisi yükseliyorsa (batch 1 olduğu için adım adım gürültülü;
10 adımlık ortalamaya bak) lr'ı düşürüp `--resume` ile devam et.

**Dokunma:** koşan python sürecini, `/mnt/nvme/lazylora/activations/run_<pid>` dizinini ve
USB diski. USB köprüsü yük altında saatte ~45 kez sıfırlanıyor; okumalar yeniden denemeyle
başarılı, hız etkilenmiyor. Tuzaklar §6 ve hafıza notunda (pgrep deseni, RAM bütçesi,
grep tamponu).

---

Bu belge, projeyi başka bir bilgisayarda devralacak kişinin (veya oturumun) sohbet geçmişi olmadan devam edebilmesi için yazılmıştır. Tarih: **29 Ağustos 2026**.

---

## 1. TEK CÜMLEYLE DURUM

LazyLoRA motoru, **2,78 trilyon parametreli Kimi K3'ün ileri geçişini bağımsız bir referans implementasyona karşı doğrulanmış biçimde yeniden üretiyor**; geri geçiş ve optimizer adımı çalışıyor; eğitilmiş bir adaptör henüz yok.

---

## 2. NEREDE KALDIK

### ✅ Tamamlananlar

| İş | Kanıt |
|---|---|
| Gerçek KDA + MLA dikkat katmanları | Referans fixture'larıyla 1e-5 toleransında eşleşme |
| MXFP4 uzman çözme | Kuantize olmayan paylaşılan uzmanla karşılaştırma: absmean 0,0194 ↔ 0,0149 |
| Katman başına gerçek yönlendirici ağırlıkları | Diskten okunuyor (önce rastgele üretiliyordu) |
| SiTU aktivasyonu | `4·tanh(g/4)·σ(g) × 25·tanh(u/25)` — fixture ile doğrulandı |
| Blok-artık mekanizması | Katman 12'de akış sıfırlanıyor, C motoruyla birebir |
| **13 katman uçtan uca doğrulama** | **cosine 0,9989 – 0,99999** (bkz. Bulgular.md §15) |
| Okuma yolu optimizasyonu | `mmap` → `pread`: 21,8 → 101 MB/s |
| MXFP4 çözücü optimizasyonu | 9,04 s → 2,53 s (12 uzman) |
| Bozuk veriye dayanıklılık | Ölçek baytı ≥253 olan grup sıfır katkı |
| Checkpoint bütünlüğü | 92 katman tarandı; hasarlı 3 parça yeniden indirildi, **hepsi temiz** |

### ⏳ Yapılmayanlar

1. **Öğrenme kanıtı koşusu** — 5-10 örnek, 15-20 adım, düşen loss eğrisi. Sıradaki iş budur.
2. **Fikir 9** (Fikirler.md) — geri geçişte uzmanları tekrar okumamak; adım süresini ~yarıya indirir.
3. **Uzman başına token toplama (gather)** — şu an her aktif uzman *tüm* token'lar üzerinde çalıştırılıp maskeleniyor. 24 token'da sorun değil, 512 token'da 56 kat fazla hesap demek. Uzun dizilerle çalışmanın ön koşulu.
4. `gradient_accumulation_steps` config'de tanımlı ama **kodda kullanılmıyor**; her `train_step` bir optimizer adımı atıyor.

---

## 3. YENİ MAKİNEDE İLK KURULUM

```bash
# 1. Depoları yerleştir
#    LazyLora/            -> proje kökü
#    kimi-k3-in-c/        -> referans motor (teşhis yaması dahil)

# 2. Sanal ortam (venv KOPYALANMAZ, yeniden kurulur)
python3 -m venv /path/to/workspace/venv
/path/to/workspace/venv/bin/pip install -r requirements.txt
#    Not: requirements.txt artık bu makinenin donmuş listesi DEĞİL; motorun gerçekten
#    import ettiği iki paket var: numpy>=1.24 ve torch>=2.3. Aynısını 'pip install -e .'
#    de kurar. Ek paketler pyproject.toml'da ekstra olarak: [data] (transformers —
#    gerçek tokenizer ve veri betikleri), [plot] (matplotlib — scripts/plot_proof.py).
#    torch'u CPU indeksinden kurun: pip install --index-url
#    https://download.pytorch.org/whl/cpu torch

# 3. Yolları güncelle — lazy_lora/core/config.py içindeki PathConfig
#    base_model_dir  -> Kimi K3 ağırlıklarının yeni konumu
#    workspace_dir   -> aktivasyon tamponu, checkpoint, veri seti için yazılabilir dizin
```

**Referans fixture'ların yolu** [lazy_lora/tests/test_reference_ops.py](lazy_lora/tests/test_reference_ops.py) içinde sabit yazılıdır (`FIXTURES`); yeni konuma göre güncellenmelidir. Dizin yoksa testler atlanır, hata vermez.

### Doğrulama (kurulumdan sonra ilk iş)

```bash
export PYTHONPATH=/path/to/LazyLora
venv/bin/python -m unittest lazy_lora.tests.test_reference_ops   # 8 test, saniyeler sürer
venv/bin/python -m unittest discover -s lazy_lora/tests -t .     # tam süit, ~3 dk
```

Referans op testleri geçiyorsa motorun matematiği sağlamdır.

---

## 4. SIRADAKİ ADIM: ÖĞRENME KANITI KOŞUSU

**Amaç:** Küçük bir sette loss'un düştüğünü göstermek. Kasıtlı ezberleme — genelleme değil, mekanizmanın çalıştığının kanıtı.

**Ayarlar** (`lazy_lora/core/config.py` veya CLI):
- `warmup_steps = 2` — varsayılan 50, 20 adımlık koşuda öğrenme oranını bastırır
- `learning_rate = 1e-3` civarı — küçük seti ezberletmek istiyoruz
- 5-10 örnek, `--seq-len 24`, 15-20 adım

```bash
bash scripts/train_lazy_lora.sh --steps 20 --seq-len 24 --lr 1e-3
```

Loss her adımda `workspace/forward_loss.jsonl` dosyasına yazılır (ileri geçiş biter bitmez, geri geçişi beklemeden). Grafik o dosyadan çizilir.

**Beklenen:** İlk adımda 2-5 arası bir loss, sonraki adımlarda düşüş. Loss ~12 (=ln 163840) ise boru hattı bozuktur.

---

## 5. ÖLÇÜLMÜŞ MALİYETLER (referans için)

Mekanik disk (WD20EZBX, ~100 MB/s), 24 token, `pread` ve hızlı çözücü sonrası:

| | Değer |
|---|---|
| Katman başına | ~112 s |
| İleri geçiş (93 katman) | ~2,9 saat |
| Geri geçiş | ~3,5 saat (Fikir 9 ile ~1 saat) |
| Katman başına okuma | 4-6 GB (aktif uzman sayısına göre) |

**SSD'ye taşındığında** (SATA ~500 MB/s) bu süreler 5 kat kısalmalıdır: adım ~35-40 dakika.

> **Uyarı:** Bu ölçümler WSL2/DrvFs üzerindedir. Native Linux + ext4'te okuma yolu daha da hızlanır; `mmap` yerine `pread` seçimi DrvFs'e özgü bir kazançtı, ext4'te ikisi de hızlıdır.

---

## 6. BİLİNEN TUZAKLAR

1. **Sessiz sentetik yedekler.** Motorun her yükleyicisi, tensör bulunamazsa rastgele ağırlık üretir. Bu, projedeki en pahalı hataların kaynağıydı (bkz. Bulgular.md §11, §15.4). Bir tensör adı değişirse kod çalışmaya devam eder ama **anlamsız sonuç üretir**. Yeni bir yol/isim eklerken mutlaka gerçek ağırlıkla doğrula.

2. **Mock testler gerçek modeli görmemeli.** `base_model_dir` boş bir dizine ayarlanmazsa testler gerçek 7168-boyutlu tensörleri okur ve hiçbir şey doğrulamaz.

3. **Loss dosyası paylaşılır.** Mock testler de `workspace/forward_loss.jsonl` dosyasına yazar; gerçek koşu loss'unu okurken karıştırma.

4. **Checkpoint hasarı sessizdir.** Bozuk MXFP4 ölçekleri `inf` üretip tüm ileri geçişi NaN'e çevirebilir. Koruma eklendi ama bir tarama aracı da var:
   ```bash
   # Ölçek baytları sağlıklı modelde 110-125 arasında kümelenir; std > 40 hasar demektir
   ```
   Tam tarama betiği git geçmişinde (`scan_tmp.py`, commit mesajlarında anlatılıyor).

---

## 7. REFERANS MOTOR (kimi-k3-in-c)

Doğrulamanın temel aracı. İki kullanım:

```bash
# Katman katman gizli durum dökümü (teşhis yaması, yalnızca ortam değişkeni verilince çalışır)
K3_DUMP_H=/tmp/dump ./bin/k3 <model_dir> --ids 19180,11 --layers 4 --gen 1 \
    --trunk <trunk_dir> --trunk-gb 2.5 --cache-gb 0.35 --dump-logits /tmp/logits.bin
```

Bizim motorumuzun çıktısı bununla karşılaştırılır. LoRA'nın $B$ matrisi sıfırla başladığı için **eğitilmemiş adaptörlerle logits birebir aynı olmalıdır**; fark varsa ileri geçişte hata var demektir.

Op-bazlı fixture'lar: `kimi-k3-in-c/tests/fixtures/ops/` — kendi ağırlıkları, girdileri ve beklenen çıktılarıyla. Bunlar [test_reference_ops.py](lazy_lora/tests/test_reference_ops.py) ile koşuluyor.

---

## 8. OKUMA SIRASI

1. **[Bulgular.md](Bulgular.md)** — özellikle §10-15: donanım gerçeği, mimari düzeltmeleri, referans doğrulama
2. **[Fikirler.md](Fikirler.md)** — 9 fikir; 8 ve 9 hız üzerine, 6 ve 7 eğitim stratejisi üzerine
3. **`git log`** — her commit mesajı neyin neden değiştiğini ve hangi ölçümün buna yol açtığını anlatır

---

## 9. DÜRÜST DEĞERLENDİRME

**Başarılan:** 1,45 TB'lık bir modelin 8 GB RAM'de doğru şekilde çalıştırılması ve LoRA adaptörleriyle eğitilebilir hale getirilmesi. Referansa karşı ölçülmüş.

**Başarılamayan:** Eğitilmiş bir Türkçe adaptör. Sebep hesap değil, **veri hacmi**: adım başına 1,45 TB'ın diskten geçmesi gerekiyor.

**Ölçek gerçeği:** LIMA benzeri 1000 örneklik bir set, 512 token'lık dizilerle mekanik diskte ~2 yıl, SATA SSD'de ~4-5 ay sürer. Anlamlı hedef, birkaç yüz adımlık odaklı bir eğitimdir.

---

## 10. YENİ MAKİNE KURULUMU (5 Eylül 2026)

Proje bu tarihten itibaren şu makinede yürüyor: i7-7700HQ (4C/8T, AVX2), 7,6 GB RAM,
native Ubuntu 24.04, GPU yok. Diskler: SSD `/` (sistem, kod, venv), NVMe `/mnt/nvme`
(hızlı geçici alan), HDD `/mnt/disk2tb` (NTFS, ntfs3 ile bağlı; checkpoint ve çalışma alanı).

| Ne | Nerede |
|---|---|
| Çalışan repo | `/home/ibox/calisma/LazyLora` (devir paketinden klon, git geçmişi dahil) |
| Referans motor | `/home/ibox/calisma/kimi-k3-in-c` (derlendi, `make test` 22/22 geçti) |
| Python ortamı | `/home/ibox/venvs/lazylora` (torch 2.14 CPU, numpy 2.5, safetensors, tiktoken) |
| Model | `/mnt/disk2tb/hamza/kimi_k3_model_weights` |
| Çalışma alanı | `/mnt/disk2tb/hamza/LazyLora_Workspace` (veri seti, önbellek, ölçüm günlükleri) |
| Aktivasyon / checkpoint | `/mnt/nvme/lazylora/{activations,checkpoints}` |
| Devir paketi (dokunma) | `/mnt/disk2tb/hamza/LazyLora_Handoff` |

Bütün yollar artık `lazy_lora/core/config.py` içindeki `default_*_dir()` fonksiyonlarından
gelir ve ortam değişkeniyle ezilebilir: `LAZYLORA_MODEL_DIR`, `LAZYLORA_WORKSPACE_DIR`,
`LAZYLORA_FAST_SCRATCH_DIR`, `LAZYLORA_ACTIVATION_DIR`, `LAZYLORA_CHECKPOINTS_DIR`,
`LAZYLORA_DATASET_DIR`, `LAZYLORA_CACHE_DIR`, `LAZYLORA_REF_FIXTURES`, `LAZYLORA_PYTHON`.

### Sentetik yedek politikası değişti

Bölüm 6.1'deki tuzak kapatıldı: bir tensör diskte yoksa motor artık **hata fırlatır**
(`MissingTensorError`). Rastgele ağırlıkla devam etmek yalnızca `LAZYLORA_ALLOW_SYNTHETIC=1`
ile mümkündür; bunu yalnızca `scripts/run_mock_tests.sh` ve `lazy_lora/tests/__init__.py`
ayarlar. NumPy yolundaki rastgele latent projeksiyonlar da (rapor K3) diskten okunur oldu.

### Checkpoint eksik: shard 68 ve 69

`model-00068` ve `model-00069` bu diskte **0 bayt** (26 Ağustos'tan beri). Katman 67 ve 68'in
tamamı eksik. Bölüm 2'deki "hepsi temiz" ifadesi yanlıştı; eski tarama boş shard'ı
"0 damaged" saymıştı. Yeni kontrol: `python scripts/check_shards.py` (ağsız, 5 saniye;
`train_lazy_lora.sh` bunu geçmeden başlamaz). İndirme: `LazyLora_Workspace/fetch_shards.py`
(kaldığı yerden devam eder, sha256 doğrular).

Ayrıca `workspace/c_ref13.log` dikkatle okunmalı: 13 katmanlı C referans koşusu katman 9'da
6 uzmanı okuyamayıp düşürmüş ve kendini `RUN INVALID` ilan etmiş. Katman 9'daki cosine
düşüşü (0,9966) muhtemelen bizim değil, referansın hatasıdır. Katman 0-8 karşılaştırması
geçerlidir; 9-12 yeniden üretilmelidir.

### Modelsiz doğrulama

```bash
export PYTHONPATH=/home/ibox/calisma/LazyLora
/home/ibox/venvs/lazylora/bin/python -m unittest lazy_lora.tests.test_reference_ops   # 8/8
bash scripts/run_mock_tests.sh                                                        # mock süiti
python scripts/check_shards.py                                                        # checkpoint bütünlüğü
python scripts/compare_with_c_dump.py --dump <chdump> --ids 19180,11 --layers 13      # gerçek ağırlık, C dökümüne karşı
```

## 11. GERİ GEÇİŞ YENİDEN YAZILDI VE DOĞRULANDI (5 Eylül 2026)

Rapordaki K1, K2, K8 maddeleri kapatıldı.

**Ne değişti (`lazy_lora/trainer/lazy_trainer.py`):**
- Katmanın hesabı tek bir yerde: `_run_layer(layer_idx, h_in, bank)`. İleri geçiş bunu
  `no_grad` altında, geri geçiş `enable_grad` altında **aynı kodu tekrar oynatarak** çalıştırır.
  Banka karışımı, blok sınırında akışın sıfırlanması, latent RMSNorm jakobiyeni, yönlendirici
  ağırlıklarından akışa dönen gradyan: hepsi autograd'dan gelir, elle türetilmiş kısım yok.
- Yönlendirilmiş uzmanlar tek bir `torch.autograd.Function` (`RoutedExpertsFunction`): ileri
  geçişte uzmanlar bir kez, geri geçişte bir kez daha diskten akıtılır; LoRA gradyanları ve
  dL/dh_latent orada elle hesaplanır (600 uzmanı autograd grafiğinde tutmak onlarca GB olurdu).
- Banka gradyanı: banka girdileri sınır katmanlarının (0, 12, 24, ...) h_in'idir ve zaten
  aktivasyon tamponunda durur; geri geçiş bankayı oradan yeniden kurar. Bankaya akan gradyan
  `_grad_bank`'ta bekletilir ve sıra o sınır katmanına gelince grad_h_in'e eklenir.
- Çıkış karışımı + son norm da (`_finalize_backward`) artık türevleniyor; eskiden atlanıyordu.
- Uzman toplamı fp32'de birikiyor (C referansı gibi); bf16 birikim katman çıktısını ~1e-3
  kaydırıp birkaç katman sonra yönlendirmeyi değiştiriyordu.
- LoRA parametreleri fp32 (K2). `lora_dropout` 0 (yeniden hesaplamalı geri geçişle uyumsuz).
- NumPy yolu kaldırıldı (dikkat zaten torch istiyordu; ölü koddu ve K3'ün kaynağıydı).

**Doğrulama (`scripts/verify_backward.py`, gerçek ağırlıklar, fp32, 4 token):**
Analitik yönlü türev ile merkezi sonlu fark, LoRA tensörleri + h_in + banka yönlerinde:

| Katman | Özellik | En kötü göreli hata |
|---|---|---|
| 1 | KDA + MoE, 1 banka girdisi | 1.1e-3 |
| 12 | blok sınırı (banka itme, akış sıfırlama) | 2.0e-3 |
| 13 | 2 banka girdisi | 9.1e-4 |
| 1 | 16 LoRA tensörünün tamamı (q/v dikkat dahil) + h_in + banka | 9.1e-3 (türevi ~3e-4 olan 2 yön; kalanı ≤2e-3) |
| 3 | MLA, 16 tensörün tamamı + h_in + banka | 3.6e-3 |

Günlükler: `LazyLora_Workspace/k8_layer*_2026-09-05.log`. İleri geçiş değişmedi
(`cmp13_after_refactor2_2026-09-05.log`, önceki koşuyla birebir).

**Ölçülen maliyet (bu makine, fp32, 4 token):** katman başına geri geçiş 37-46 s.

**Not:** Sonlu fark fp32 motorda bile ancak kayıp float64'te toplanıp adım yöne göre
ölçeklenince anlamlı çıktı; ilk sürüm (fp32 kayıp, sabit eps=1e-3) 1-2 ulp'lik farklar ölçüp
sahte uyumsuzluk raporladı. Harness'ı değiştirirken buna dikkat.

## 12. TAM CHECKPOINT (K4, 5 Eylül 2026)

`save_lora_checkpoint(step, data_cursor)` artık LoRA tensörleriyle birlikte Adam momentlerini,
adım sayacını, LR zamanlayıcısını, torch/numpy RNG durumlarını ve veri kümesi imlecini tek
dosyaya yazar; önce `.tmp`'ye yazılıp `fsync` sonrası `os.replace` ile yerine konur (yarım
dosya asla nihai adı taşımaz). `checkpoints/latest.txt` son dosyanın adını tutar.

Devam etmek: `python -m lazy_lora.trainer.lazy_trainer --resume <ckpt.pt> --steps N`
(`LazyLoRATrainer.load_checkpoint`, `train(resume_from=...)`; veri imleci
`StreamingDatasetIterator.get_batches(skip_samples=...)` ile uygulanır).
Eski biçim (yalnızca LoRA sözlüğü) da yüklenir, ama optimizer sıfırdan başlar.

## 13. TEMİZ C REFERANSI VE ÖLÇÜM ALETİ (5 Eylül 2026, akşam)

C motoru bu makinede 13 katmanla yeniden koşturuldu (`LazyLora_Workspace/run_cref13.sh`).
Boş shard'lar yüzünden motor tam dizini reddettiği için katman 0-12 + embed/lm_head
shard'larına sembolik bağlarla `/mnt/nvme/lazylora/k3_partial13` dizini kuruldu. Koşu geçerli
(uzman düşürme yok, exit 0), tepe RSS 6.11 GB, 169 s. Yeni döküm: `chdump_2026-09-05/`.

Yeni dökümle karşılaştırma (`cmp13_vs_newdump_2026-09-05.log`): katman 9 kosinüs
0.9966 → **0.99951**, katman 12 → **0.99995**. Eski dökümdeki düşüş referansın düşürdüğü
6 uzmandandı; motorumuzda katman 0-12'de yapısal fark yok.

**Uzman erişim izi** (`lazy_lora/monitor/trace.py`): `trainer.trace` ayarlıysa her katmanın
(token, seçilen 16 uzman, ağırlık) kaydı `trace.bin`'e, süre/bayt/tepe RSS `trace.json`'a
yazılır. `scripts/measure_routing.py` bir metni bu izle ileri geçirir;
`scripts/analyze_trace.py` yoğunlaşma, entropi, etkin uzman sayısı, zamansal ve katmanlar
arası Jaccard, N'ye göre benzersiz uzman eğrisi (tek koşudan, nedensellik sayesinde) ve iki
iz arasında dil karşılaştırması üretir. İstemler `LazyLora_Workspace/prompts/` (aynı anlam:
Türkçe 263 token, İngilizce 158 token; Türkçe %66 daha fazla token harcıyor).

## 14. NVMe PLANI (5 Eylül 2026)

117 GB'lık NVMe (`/mnt/nvme`, ext4) şöyle bölüştürüldü:

| Ne | Boyut | Yol |
|---|---|---|
| Paketlenmiş gövde (`trunk.bin` + `trunk.json`, C motorunun `pack_trunk.py` çıktısı; 93 katmanın uzman dışı bütün tensörleri) | 108.8 GB (101.3 GiB) | `/mnt/nvme/lazylora/k3trunk/` |
| Aktivasyon halka tamponu (süreç başına alt dizin) | N=2048'de ~2.7 GB | `/mnt/nvme/lazylora/activations/` |
| Checkpoint'ler (en yeni 3 tanesi tutulur, `keep_checkpoints`) | ~1.8 GB × 3 | `/mnt/nvme/lazylora/checkpoints/` |
| Katman 0-12 için sembolik bağ dizini (C motoru, yer kaplamaz) | 0 | `/mnt/nvme/lazylora/k3_partial13/` |

Gövde NVMe'de olunca HDD yalnızca yönlendirilmiş uzmanları (1.42 TB) sıralı süpürür;
katman başına 0.8-2.3 GB'lık gövde okuması HDD kafasıyla yarışmaz. Embedding satırları
(seyrek) ve lm_head (adımda bir kez, 2.35 GB sıralı) HDD'de kalır; 4.7 GB'lık yer daha
değerli.

Mekanizma: `SafetensorsIndex.apply_trunk_overlay` (`mmap_loader.py`), `trunk.json`'daki her
tensörü şekil/dtype/bayt sayısı shard indeksiyle birebir tutuyorsa `trunk.bin`'e yönlendirir;
tutmayan tensör shard'da kalır ve uyarı basılır. `LAZYLORA_TRUNK_DIR=""` kapatır.
Doğrulama: `python scripts/verify_trunk.py` (seçilen katmanların bütün tensörlerini iki
kaynaktan bayt bayt karşılaştırır). C motoru da aynı paketi `--trunk /mnt/nvme/lazylora/k3trunk`
ile kullanabilir.

HDD'deki `/mnt/disk2tb/hamza/k3trunk` (102 GiB) NVMe kopyası doğrulandıktan sonra silinebilir;
HDD %94 dolu, bu 102 GiB'lik yer açar (karar kullanıcının).

## 15. FÜZYONLU MXFP4 ÇEKİRDEĞİ (6 Eylül 2026)

Profil, katman süresinin diske değil MXFP4 çözmeye gittiğini gösterdi (uzman başına çözme
289 ms, okuma 190 ms, GEMM N=1024'te ~45 ms). `lazy_lora/native/mxfp4_gemm.c` paketli
baytları doğrudan tüketen üç C fonksiyonu içerir (`gemm`, `gemm_t`, `dequant`; OpenMP,
AVX2), `build.sh` ile derlenir, `lazy_lora/native/__init__.py` ctypes ile bağlar ve yoksa
otomatik derler. Uzman yolu artık fp32: ≤48 satırda füzyonlu çöz-ve-çarp, üstünde C çözücü +
fp32 sgemm (bu CPU'da fp32 GEMM bf16'dan 3.4 kat hızlı). Okuyucu thread yalnızca okur;
uzmanlar paketli halde `ExpertWeightBundle.packed=True` olarak taşınır. `LAZYLORA_NO_NATIVE=1`
eski çözme yoluna döndürür (mock testler ve A/B için).

Doğrulama: katman 1 çıktısı eski yolla 2e-4 göreli farkla aynı (eski yolun bf16 ara
yuvarlaması); C dökümüyle 13 katman kosinüsleri aynı ya da daha iyi (katman 12: 0.999956);
K8 katman 1'de 16 LoRA tensörü + h_in + banka geçti (banka yönünde yönlendirme sınırı
sıçraması görüldü, küçük adımda uyum; harness artık sıçramayı tespit edip adımı küçültür).
Uzman başına maliyet: 22 satırda 55 ms (eski ~270 ms). 1024 token'da katman 12: 200 → 108 s.

## 16. DEĞERLENDİRME PROTOKOLÜ (6 Eylül 2026, eğitimden önce sabitlendi)

**Külliyat** (`scripts/build_eval_corpus.py`, `LazyLora_Workspace/eval/`): Türkçe ve İngilizce
Wikipedia'dan aynı 11 konunun (İstanbul, Anadolu, Güneş Sistemi, Fotosentez, Osmanlı, Kahve,
Deprem, İklim değişikliği, Matematik, Futbol, Su) düz metin paragrafları, makale başına en
fazla 400 token, dil başına tam 4096 token. Manifest'te makale ve sürüm numaraları var
(CC BY-SA 4.0). Eğitim verisinden tamamen ayrık; eğitimde asla kullanılmayacak.

**Ölçüm** (`scripts/eval_perplexity.py`): dil başına iki adet 2048-token'lık dilim, her biri
tek bir 93 katmanlı ileri geçiş (maliyet süpürme başına). Kayıt: sonraki-token loss, perplexity,
top-1 doğruluk, süre, tepe RSS, commit ve varsa adaptör checkpoint'i → `eval/results.jsonl`.

1. Birincil metrik: Türkçe perplexity (2 dilim ortalaması), eğitim öncesi ve sonrası.
2. Kontrol: İngilizce perplexity, aynı ölçümle; unutma / küresel bozulma göstergesi.
3. Başarı eşiği: taban ölçümünden sonra, eğitimden önce burada yazılacak (X% Türkçe düşüşü,
   en fazla Y% İngilizce bozulması). Sonradan seçilmez.
4. Olumsuz sonuç geçerli sonuçtur; nedenleri (adım, rank, sinyal) ölçümle raporlanır.
5. Yönlendirme değişimi: eğitim öncesi/sonrası izler karşılaştırılır (adaptör yönlendirmeyi kaydırdı mı).
6. Gradyan sağlığı: her N adımda katman başına LoRA gradyan normları.

Taban ölçümü: kuyruğun sonunda otomatik (`run_eval_baseline.sh`).

## 17. GERİ GEÇİŞ HIZI VE İLK TAM EĞİTİM ADIMI (7-8 Eylül 2026)

İlk tam adım (6 Eylül gecesi) geri geçişte katman başına ~17 dk ile takılıp durduruldu.
İki neden bulundu ve düzeltildi:
1. KDA özyinelemesinin autograd tekrarı her adımın ara matrislerini saklıyordu (katman
   başına GB'lar, swap). `attention.py`: özyineleme 16 token'lık parçalarda
   `torch.utils.checkpoint` ile; ileri geçiş bit bit aynı.
2. bf16 ağırlıklı matmul'lar bu CPU'da PyTorch'un yavaş yedek GEMM'ine düşüyordu
   (`cpublas_gemm_impl`; katman 3 geri geçişinin 808 s'sinin 765'i). `core/linear32.py`:
   donuk ağırlık bf16 kalır, çarpım fp32'de; dikkat, paylaşılan uzman, latent, dense MLP,
   lm_head bunu kullanır. Doğruluk fp32 referansa göre değişmedi (3.4e-3 vs 3.5e-3).
3. Ek: ileri geçişin uzman toplamı (`act_layer_NNN_moe.bin`) geri geçiş tekrarında
   kullanılıyor; katman başına uzman süpürmesi ikiden bire indi.

Katman geri geçişi, 256 token, GPU: katman 3 858 → 50 s, katman 1 ~1000 → 101 s,
katman 0 22 s. Tepe RAM 3.7 GB.

**İlk tam adım (7-8 Eylül gecesi, 256 token, GPU):** 4 s 31 dk (ileri ~1.5, geri ~2.9),
tepe RSS 6.24 GB, ileri loss 2.945 (perplexity 19.0), checkpoint 1.8 GB
(`/mnt/nvme/lazylora/checkpoints/lazy_lora_step_00001.pt`, 1482 LoRA tensörü + Adam
momentleri), 741 B matrisinin tamamı güncellenmiş. Motor uçtan uca eğitiyor.

**Taban değerlendirmesi (`eval/results.jsonl`, 2048 token Wikipedia dilimleri):**
TR loss 0.593 / ppl 1.81 / 0.311 bit/bayt / top-1 %84; EN loss 0.637 / ppl 1.89 /
0.194 bit/bayt / top-1 %85. İkisi de ezber düzeyinde (Wikipedia ön eğitimde); bayt başına
bit farkı (TR 1.6× EN) anlamlı. Eşik için haber tabanlı, kesim tarihi sonrası bir Türkçe
dilim eklenecek; Wikipedia dilimleri "ezber/unutma kontrolü" olarak kalır.

Projeksiyon: 1024 token'da adım ~7-8 s; 4 haftada ~90 adım ≈ 90 bin token.

### 16.1 Taban ölçümü ve önceden ilan edilen eşik (8 Eylül 2026)

| Dilim | loss | perplexity | bit/bayt | top-1 |
|---|---:|---:|---:|---:|
| TR Wikipedia (2048 tok) | 0.593 | 1.81 | 0.311 | %84 |
| EN Wikipedia (2048 tok) | 0.637 | 1.89 | 0.194 | %85 |
| TR haber, 7 Eylül 2026 sonrası (2048 tok, `build_eval_news.py`) | 0.883 | 2.42 | **0.455** | %78 |

Wikipedia dilimleri ezber düzeyinde; birincil metrik haber dilimi (bit/bayt). **Eşik, eğitimden
önce sabit:** haber bit/bayt ≥ %3 düşer (≤ 0.441) VE İngilizce Wikipedia bit/bayt ≤ %2 artar
(≤ 0.198). Tutmazsa olumsuz sonuç, nedenleriyle raporlanır. Eğitim verisi:
`datasets/dolly_tr_400.jsonl` (Dolly-15k-tr, CC BY-SA 3.0), 1024 token paketli, istem maskeli.
Kanıt koşusu: `run_proof.sh` (5 örnek, 16 adım, lr 1e-3, warmup 2).

## 18. GÖZETİMSİZ KOŞU TAKİBİ (8 Eylül 2026)

`scripts/watchdog.py`, systemd kullanıcı zamanlayıcısı `lazylora-watchdog.timer` ile her
15 dakikada bir koşar (birimler `systemd/` altında; kurulum: `~/.config/systemd/user/`'a
kopyala, `systemctl --user enable --now lazylora-watchdog.timer`). Aktif koşu
`LazyLora_Workspace/run_manifest.json` ile tanımlanır (ad, argümanlar, adım sayısı,
beklenen adım süresi, python, env). Her turda: süreç yaşıyor mu, ilerleme
(`forward_loss.jsonl`), takılma (3× beklenen adım süresi), HDD/USB hatası, NVMe boş alan,
swap, GPU sıcaklığı; `status.json`/`status.txt` yazar; telefona (claude-code-server web
push) her tamamlanan adımda, her sorunda ve bitişte bildirim gönderir; süreç ölmüşse en
yeni checkpoint'ten `--resume` ile yeniden başlatır (arka arkaya en fazla 3 kez, 30 dk
arayla). Durum: `bash scripts/status.sh` ya da `cat LazyLora_Workspace/status.txt`.

Kanıt koşusu 8 Eylül 09:18'de başladı (`run_proof.sh`; 5 örnek, 16 adım, 1024 token
paketli, istem maskeli, lr 1e-3, warmup 2, her 4 adımda checkpoint).
